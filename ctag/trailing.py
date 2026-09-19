"""Trailing-event check: should the last event the transcriber emitted stay?

The phase 3 error analysis (docs/error_analysis_phase3.md) found that the
transcriber's spurious events are duplicates appended at the end of the list:
6 of 7 were the last event emitted. This module tests three ways of deciding
whether that last event is real, against the gold timelines:

    iou      drop it when it overlaps another predicted event at IoU >= thr
    silence  drop it when the audio under it is quiet (RMS ratio to the clip < thr)
    stop     drop it when the model's own P(stop) before it was >= thr
             (needs `run_transcribe --stop-probs`, see ctag.stopprob)

    python -m ctag.trailing --pred-timelines runs/.../pred_timelines.jsonl \
           --timelines data/esc50/timelines.jsonl --out runs/.../trailing_iou --rule iou --thr 0.5
    python -m ctag.trailing ... --rule stop --sweep 0.05 0.1 0.2 0.3 0.5   # threshold table only

Measured on the phase 3 val+test predictions, `iou` and `silence` remove more
correct events than spurious ones (the benchmark overlaps 45% of its events and
its quiet sounds sit under loud ones); they are kept so that negative result is
reproducible. `stop` is the candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agent import _norm
from .metrics import iou
from .transcribe import Event3, score_timeline, summarize_timelines

RULES = ("iou", "silence", "stop")


# ------------------------------------------------------------------ helpers
def emitted_order(raw: str | None) -> list[tuple[str, float, float]]:
    """Events in the order the model wrote them, from the raw answer. Empty when
    the answer is not the plain JSON list (then the parsed order is used)."""
    try:
        items = json.loads(raw or "")
    except (TypeError, ValueError):
        return []
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict) and "sound" in it and "start" in it and "end" in it:
            try:
                out.append((_norm(str(it["sound"])), float(it["start"]), float(it["end"])))
            except (TypeError, ValueError):
                pass
    return out


def _key(lab, a, b):
    return (_norm(lab), round(float(a), 2), round(float(b), 2))


def last_emitted_index(events: list[dict], raw: str | None) -> int | None:
    """Index into `events` (the parsed, sorted list) of the event the model
    emitted last. Falls back to the last parsed event."""
    if not events:
        return None
    order = emitted_order(raw)
    if order:
        want = _key(*order[-1])
        for i in range(len(events) - 1, -1, -1):
            e = events[i]
            if _key(e["label"], e["onset"], e["offset"]) == want:
                return i
    return len(events) - 1


def align_stop_probs(events: list[dict], raw: str | None, probs: list | None) -> list:
    """`probs` follow generation order; return them in the order of `events`
    (None where an event cannot be matched to the raw answer)."""
    if not probs:
        return [None] * len(events)
    order = emitted_order(raw)
    if len(order) != len(probs):
        return [None] * len(events)
    lookup: dict = {}
    for ev, p in zip(order, probs):
        lookup.setdefault(_key(*ev), p)
    return [lookup.get(_key(e["label"], e["onset"], e["offset"])) for e in events]


def rms_ratio(audio_path: str, onset: float, offset: float) -> float:
    import soundfile as sf

    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    seg = y[int(max(0.0, onset) * sr): int(max(0.0, offset) * sr)]
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))


# ------------------------------------------------------------------ rules
def should_drop(rule: str, thr: float, row: dict, idx: int, audio_root: str = "") -> bool:
    ev = row["events"][idx]
    if rule == "iou":
        others = [e for j, e in enumerate(row["events"]) if j != idx]
        return any(iou((ev["onset"], ev["offset"]), (o["onset"], o["offset"])) >= thr for o in others)
    if rule == "silence":
        return rms_ratio(str(Path(audio_root) / row["audio"]), ev["onset"], ev["offset"]) < thr
    if rule == "stop":
        p = align_stop_probs(row["events"], row.get("raw"), row.get("stop_probs"))[idx]
        return p is not None and p >= thr
    raise ValueError(f"unknown rule {rule!r}; one of {RULES}")


def matched_flags(pred: list[Event3], gold: list[Event3], thr: float) -> list[bool]:
    """Which predicted events the label-aware one-to-one matching pairs with a
    gold event (same matching as ctag.transcribe.score_timeline)."""
    if not pred or not gold:
        return [False] * len(pred)
    from scipy.optimize import linear_sum_assignment

    M = np.zeros((len(pred), len(gold)))
    for i, (pl, pa, pb) in enumerate(pred):
        for j, (gl, ga, gb) in enumerate(gold):
            if _norm(pl) == _norm(gl):
                M[i, j] = iou((pa, pb), (ga, gb))
    r, c = linear_sum_assignment(-M)
    ok = [False] * len(pred)
    for i, j in zip(r, c):
        if M[i, j] >= thr:
            ok[i] = True
    return ok


def apply_rule(rows: list[dict], rule: str, thr: float, gold: dict | None, iou_thr: float = 0.5,
               audio_root: str = "") -> tuple[list[dict], dict]:
    """Filter the last emitted event of every row by `rule`; rescore against
    `gold` when given. Returns (new rows, confusion)."""
    out, conf = [], {"fired": 0, "removed_spurious": 0, "removed_correct": 0, "spurious_last_events": 0}
    for row in rows:
        events = list(row["events"])
        idx = last_emitted_index(events, row.get("raw"))
        g = [(_norm(e["label"]), e["onset"], e["offset"]) for e in gold.get(row["clip_id"], {}).get("events", [])] if gold else None
        pred3 = [(e["label"], e["onset"], e["offset"]) for e in events]
        flags = matched_flags(pred3, g, iou_thr) if g is not None else None
        drop = idx is not None and should_drop(rule, thr, row, idx, audio_root)
        if flags is not None and idx is not None and not flags[idx]:
            conf["spurious_last_events"] += 1
        if drop:
            conf["fired"] += 1
            if flags is not None:
                conf["removed_spurious" if not flags[idx] else "removed_correct"] += 1
            events.pop(idx)
        new = dict(row)
        new["events"] = events
        new["trailing_dropped"] = drop
        if g is not None:
            new.update(score_timeline([(e["label"], e["onset"], e["offset"]) for e in events], g, iou_thr))
        out.append(new)
    return out, conf


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_gold(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {d["clip_id"]: d for d in (json.loads(l) for l in open(path, encoding="utf-8"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-timelines", required=True, help="pred_timelines.jsonl from ctag.run_transcribe")
    ap.add_argument("--timelines", default=None, help="gold timelines.jsonl, to score before/after")
    ap.add_argument("--out", default=None, help="directory for the filtered pred_timelines.jsonl + summary.json")
    ap.add_argument("--rule", choices=RULES, default="stop")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--sweep", type=float, nargs="*", default=None, help="thresholds to tabulate instead of --thr")
    ap.add_argument("--iou", type=float, default=0.5, help="match threshold for scoring")
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--tune-on", default=None,
                    help="another pred_timelines.jsonl (the val split): pick the --sweep threshold with the best "
                         "event F1 there, then apply it to --pred-timelines")
    a = ap.parse_args(argv)

    rows, gold = load_rows(Path(a.pred_timelines)), load_gold(Path(a.timelines) if a.timelines else None)
    before = summarize_timelines(rows) if gold else None
    if a.tune_on:
        if not (gold and a.sweep):
            raise SystemExit("--tune-on needs --timelines and a --sweep list")
        val_rows = load_rows(Path(a.tune_on))
        base = summarize_timelines(val_rows)["event_f1_pooled"]
        best = None
        for thr in a.sweep:
            new, conf = apply_rule(val_rows, a.rule, thr, gold, a.iou, a.audio_root)
            f1 = summarize_timelines(new)["event_f1_pooled"]
            print(f"tune thr={thr:.2f} val event_f1_pooled {base:.4f} -> {f1:.4f} fired={conf['fired']} "
                  f"rm_spur={conf['removed_spurious']} rm_corr={conf['removed_correct']}")
            # strictly better than no filter, then the highest F1, then the fewest removals
            if f1 > base and (best is None or (f1, -conf["fired"]) > (best[1], -best[2])):
                best = (thr, f1, conf["fired"])
        if best is None:
            print("no threshold improves val; the filter stays off (thr=inf)")
            a.thr = float("inf")
        else:
            a.thr = best[0]
            print(f"chosen thr={a.thr:.2f} on val")
        a.sweep = None
    if a.sweep is not None:
        if before:
            print(f"before: event_f1_pooled={before['event_f1_pooled']:.4f} count_acc={before['count_acc']:.3f}")
        print(f"{'thr':>6} {'fired':>5} {'rm_spur':>7} {'rm_corr':>7} {'f1_pooled':>9} {'count_acc':>9}")
        for thr in a.sweep:
            new, conf = apply_rule(rows, a.rule, thr, gold, a.iou, a.audio_root)
            s = summarize_timelines(new) if gold else {}
            print(f"{thr:6.2f} {conf['fired']:5d} {conf['removed_spurious']:7d} {conf['removed_correct']:7d} "
                  f"{s.get('event_f1_pooled', float('nan')):9.4f} {s.get('count_acc', float('nan')):9.3f}")
        return
    new, conf = apply_rule(rows, a.rule, a.thr, gold, a.iou, a.audio_root)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "pred_timelines.jsonl", "w", encoding="utf-8") as fo:
            for r in new:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {"rule": a.rule, "thr": a.thr, "confusion": conf,
                   "before": before, "after": summarize_timelines(new) if gold else None}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = summarize_timelines(new) if gold else None
    msg = f"rule={a.rule} thr={a.thr} fired={conf['fired']}"
    if gold:
        msg += (f" removed_spurious={conf['removed_spurious']} removed_correct={conf['removed_correct']}"
                f" event_f1_pooled {before['event_f1_pooled']:.4f} -> {after['event_f1_pooled']:.4f}"
                f" count_acc {before['count_acc']:.3f} -> {after['count_acc']:.3f}")
    print(msg)


if __name__ == "__main__":
    main()
