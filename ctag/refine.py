"""Boundary refinement: snap predicted edges to the audio's own energy.

Every fine-tuned arm predicts windows of the training set's duration
(duration_ratio_median exactly 1.000, results_kaggle_v5/v6): the model learned
that composed events last 2.5 s, which is optimal on this benchmark and wrong
on real recordings. This post-processor keeps the model's decision of *where*
and *what*, and lets the signal decide *exactly when*: each predicted onset is
moved to the nearest energy rise inside a small search window, each offset to
the nearest energy fall. It is pure signal processing, needs no model, and is
scored with the same event-level metrics as the transcriber.

    # refine whole-clip predictions and rescore them against the gold timelines
    python -m ctag.refine --pred-timelines runs/esc50/transcribe_test/pred_timelines.jsonl \
           --timelines data/esc50/timelines.jsonl --out runs/esc50/transcribe_test_refined

    # refine per-question predictions (a run_zeroshot / run_agent output); the
    # benchmark supplies each question's audio
    python -m ctag.refine --predictions runs/esc50/test_lora_text/predictions.jsonl \
           --bench data/esc50/benchmark_test.jsonl --out runs/esc50/test_lora_text_refined

Measured on 30 procedural clips (164 events), window 0.5 s: with the oracle's
edges moved by up to 0.3 s, event F1 0.872 -> 0.994; by up to 0.5 s, 0.700 ->
0.939; on exact edges 1.000 -> 0.989. The two events lost on exact edges are
both an impulsive sound (clicks) overlapping a continuous one: the energy
envelope belongs to the neighbour, so the snap adopts the neighbour's edges. A
snap is accepted only where the energy step across the edge is sharper than at
the model's position, which stops most such cases but not that one. The
runner reports before and after, so whether to keep refinement is decided on
the real test clips, not assumed.

Limits, stated: an edge that sits inside another sound (overlap) sees no energy
change and is left where the model put it, or, for an impulsive sound inside a
continuous one, may adopt the neighbour's edge; the search window bounds how
far an edge may move, so a window placed on the wrong sound stays wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Interval = tuple[float, float]


# ------------------------------------------------------------------ envelope
def log_energy(audio: np.ndarray, sr: int, hop_s: float = 0.01, win_s: float = 0.025) -> tuple[np.ndarray, float]:
    """Smoothed log RMS energy per frame (dB) and the hop in seconds."""
    hop, win = max(1, int(hop_s * sr)), max(2, int(win_s * sr))
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    n = 1 + (len(audio) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = audio[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # 3-frame moving average: enough to kill single-frame spikes, short enough
    # to keep a 50 ms fade sharp
    k = np.ones(3) / 3
    db = np.convolve(np.pad(db, 1, mode="edge"), k, mode="valid")
    return db, hop / sr


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, file_sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        idx = np.linspace(0, len(x) - 1, int(len(x) * sr / file_sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x, sr


# ------------------------------------------------------------------ snapping
def snap_interval(db: np.ndarray, hop: float, onset: float, offset: float, window: float = 0.5,
                  alpha: float = 0.5, min_dur: float = 0.05) -> tuple[Interval, dict]:
    """Move (onset, offset) to the energy rise / fall nearest to each, searching
    +/- `window` seconds. The threshold sits `alpha` of the way from the clip's
    noise floor (10th percentile) to the peak inside the predicted window.

    Returns the new interval and a small record of what happened, so the
    effect can be audited per edge rather than trusted."""
    n = len(db)
    floor = float(np.percentile(db, 10))
    a0, b0 = int(round(onset / hop)), int(round(offset / hop))
    a0, b0 = max(0, min(n - 1, a0)), max(0, min(n - 1, b0))
    lo, hi = max(0, a0 - int(window / hop)), min(n - 1, b0 + int(window / hop))
    peak = float(db[max(lo, a0 - 2): hi + 1].max()) if hi > lo else float(db[a0])
    if peak - floor < 6.0:                      # nothing audible here to snap to
        return (onset, offset), {"moved": False, "reason": "no_energy", "peak_over_floor": peak - floor}
    thr = floor + alpha * (peak - floor)
    above = db >= thr
    w = int(window / hop)

    # onset: the rising crossing nearest to the predicted onset
    cands = [i for i in range(max(1, a0 - w), min(n, a0 + w + 1)) if above[i] and not above[i - 1]]
    new_a = min(cands, key=lambda i: abs(i - a0)) if cands else None
    # offset: the falling crossing nearest to the predicted offset
    cands = [i for i in range(max(0, b0 - w), min(n - 1, b0 + w + 1)) if above[i] and not above[i + 1]]
    new_b = min(cands, key=lambda i: abs(i - b0)) + 1 if cands else None

    # Accept a snap only if the edge is sharper there than where the model put
    # it: the energy step across the edge (after minus before for an onset,
    # before minus after for an offset) must grow. On exact edges this keeps
    # them; where two sounds overlap the nearest crossing can belong to the
    # other sound, and this is what stops that snap.
    k = max(1, int(0.05 / hop))

    def contrast(i, rising):
        lo_, hi_ = max(0, i - k), min(n, i + k)
        before, after = db[lo_:i], db[i:hi_]
        if len(before) == 0 or len(after) == 0:
            return -1e9
        return float(after.mean() - before.mean()) * (1 if rising else -1)

    if new_a is not None and contrast(new_a, True) <= contrast(a0, True):
        new_a = None
    if new_b is not None and contrast(new_b - 1, False) <= contrast(b0, False):
        new_b = None

    rec = {"moved": False, "onset_from": onset, "offset_from": offset, "thr_db": thr, "peak_over_floor": peak - floor}
    a = new_a * hop if new_a is not None else onset
    b = new_b * hop if new_b is not None else offset
    if b - a < min_dur:                          # a snap that collapses the window is a wrong snap
        return (onset, offset), {**rec, "reason": "collapsed"}
    rec.update({"moved": (new_a is not None) or (new_b is not None), "onset_snapped": new_a is not None,
                "offset_snapped": new_b is not None})
    return (round(float(a), 3), round(float(b), 3)), rec


def refine_intervals(audio_path: str, intervals: list[Interval], window: float = 0.5, alpha: float = 0.5,
                     min_dur: float = 0.05, _cache: dict | None = None) -> tuple[list[Interval], list[dict]]:
    if not intervals:
        return [], []
    if _cache is not None and audio_path in _cache:
        db, hop = _cache[audio_path]
    else:
        x, sr = load_audio(audio_path)
        db, hop = log_energy(x, sr)
        if _cache is not None:
            _cache[audio_path] = (db, hop)
    out, recs = [], []
    for a, b in intervals:
        iv, rec = snap_interval(db, hop, float(a), float(b), window, alpha, min_dur)
        out.append(iv)
        recs.append(rec)
    return out, recs


# ------------------------------------------------------------------ files
def refine_timelines(pred_path: Path, out_dir: Path, gold_path: Path | None, window: float, alpha: float,
                     min_dur: float, iou: float = 0.5) -> dict:
    from .agent import _norm
    from .transcribe import score_timeline, summarize_timelines

    gold = {}
    if gold_path:
        for line in open(gold_path, encoding="utf-8"):
            d = json.loads(line)
            gold[d["clip_id"]] = d
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    rows_before, rows_after, moved = [], [], 0
    with open(pred_path, encoding="utf-8") as f, open(out_dir / "pred_timelines.jsonl", "w", encoding="utf-8") as fo:
        for line in f:
            d = json.loads(line)
            evs = d.get("events", [])
            ivs = [(e["onset"], e["offset"]) for e in evs]
            new, recs = refine_intervals(d["audio"], ivs, window, alpha, min_dur, cache) if d.get("audio") else (ivs, [])
            moved += sum(1 for r in recs if r.get("moved"))
            new_evs = [{"label": e["label"], "onset": a, "offset": b} for e, (a, b) in zip(evs, new)]
            row = {**{k: v for k, v in d.items() if k not in ("events",)}, "events": new_evs, "refined": True,
                   "events_before": evs}
            if d["clip_id"] in gold:
                g = [(_norm(e["label"]), e["onset"], e["offset"]) for e in gold[d["clip_id"]]["events"]]
                rows_before.append(score_timeline([(_norm(e["label"]), e["onset"], e["offset"]) for e in evs], g, iou))
                s = score_timeline([(_norm(e["label"]), e["onset"], e["offset"]) for e in new_evs], g, iou)
                rows_after.append(s)
                row.update(s)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"refined": True, "window": window, "alpha": alpha, "edges_moved": moved}
    if rows_after:
        summary["before"] = summarize_timelines(rows_before)
        summary["after"] = summarize_timelines(rows_after)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def refine_predictions(pred_path: Path, bench_path: Path, out_dir: Path, window: float, alpha: float,
                       min_dur: float) -> dict:
    """Per-question predictions (run_zeroshot / run_agent layout); the benchmark
    maps each qid to its audio, and every row is rescored."""
    from .metrics import score_query, summarize

    audio_of = {}
    for line in open(bench_path, encoding="utf-8"):
        d = json.loads(line)
        audio_of[d["qid"]] = d["audio"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    rows, moved = [], 0
    with open(pred_path, encoding="utf-8") as f, open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fo:
        for line in f:
            d = json.loads(line)
            pred = d.get("pred")
            if isinstance(pred, str):
                pred = json.loads(pred) if pred else []
            audio = audio_of.get(d["qid"])
            if pred and audio:
                new, recs = refine_intervals(audio, [tuple(p) for p in pred], window, alpha, min_dur, cache)
                moved += sum(1 for r in recs if r.get("moved"))
            else:
                new = pred
            gt = [tuple(a) for a in d["answer"]]
            s = score_query(None if d.get("parse_fail") else [tuple(p) for p in (new or [])], gt, bool(d["expects_empty"]))
            row = {**d, "pred_before": pred, "pred": [list(p) for p in (new or [])], **s}
            rows.append(row)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"model": "refined", "edges_moved": moved, "by_type": summarize(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-timelines", default=None, help="pred_timelines.jsonl from ctag.run_transcribe")
    ap.add_argument("--timelines", default=None, help="gold timelines, to score before/after")
    ap.add_argument("--predictions", default=None, help="predictions.jsonl from run_zeroshot / run_agent")
    ap.add_argument("--bench", default=None, help="benchmark split (audio per question) for --predictions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=0.5, help="+/- seconds an edge may move")
    ap.add_argument("--alpha", type=float, default=0.5, help="threshold position between noise floor and peak")
    ap.add_argument("--min-dur", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args(argv)
    if bool(a.pred_timelines) == bool(a.predictions):
        raise SystemExit("give exactly one of --pred-timelines or --predictions")
    if a.pred_timelines:
        s = refine_timelines(Path(a.pred_timelines), Path(a.out), Path(a.timelines) if a.timelines else None,
                             a.window, a.alpha, a.min_dur, a.iou)
        if "after" in s:
            b, c = s["before"], s["after"]
            print(f"edges moved {s['edges_moved']}  event F1 (pooled) {b['event_f1_pooled']:.3f} -> {c['event_f1_pooled']:.3f}  "
                  f"centre error {b['centre_error_median']} -> {c['centre_error_median']}  "
                  f"duration ratio {b['duration_ratio_median']} -> {c['duration_ratio_median']}")
        else:
            print(f"edges moved {s['edges_moved']} (no gold timelines given, not scored)")
    else:
        if not a.bench:
            raise SystemExit("--bench is required with --predictions")
        s = refine_predictions(Path(a.predictions), Path(a.bench), Path(a.out), a.window, a.alpha, a.min_dur)
        from .run_zeroshot import _print_table
        print(f"edges moved {s['edges_moved']}")
        _print_table(s["by_type"])


if __name__ == "__main__":
    main()
