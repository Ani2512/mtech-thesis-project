"""Whole-timeline transcription: one answer per clip, every question in code.

Phase 2 showed that the remaining errors of the best arm are decisions rather
than perception: answering "nothing" when there is an answer (19% of the
non-rejection test queries) and listing fewer occurrences than there are (8%).
Every one of the seven query types is a deterministic function of the event
timeline (ctag.timeline), so if the model is asked for the whole timeline once,
the conditions cost nothing and the "nothing" decision is never the model's.

This module owns that target:

    target_timeline(events)    the string the model is trained to emit
    parse_timeline(text)       the inverse, tolerant of the shapes models emit
    score_timeline(pred, gold) event-level P/R/F1 with label-aware matching
    timeline_grounder(preds)   a Grounder for ctag.agent built from predictions,
                               so run_agent scores every query type from them

The prompt goes through ctag.models.prompt_for so training and inference are
byte-identical, exactly as for the query-level targets.
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict

import numpy as np

from .agent import Grounder, _norm, _sorted
from .metrics import iou, localisation

Event3 = tuple[str, float, float]      # (label, onset, offset)

TRANSCRIBE_QUERY = (
    "every sound event in the recording, as a JSON list of objects "
    '{"sound": name, "start": seconds, "end": seconds} sorted by start'
)


def transcribe_query(vocab: list[str] | None = None) -> str:
    """The query text. Naming the closed label set keeps the emitted names
    consistent with the ones the scorer matches against."""
    if not vocab:
        return TRANSCRIBE_QUERY
    names = ", ".join(_norm(v) for v in vocab)
    return TRANSCRIBE_QUERY + f", using only these sound names: {names}"


def target_timeline(events) -> str:
    """Exactly what parse_timeline reads back. `events` are dicts with
    label/onset/offset (timelines.jsonl) or ctag.timeline.Event objects."""
    rows = []
    for e in events:
        if isinstance(e, dict):
            lab, a, b = e["label"], e["onset"], e["offset"]
        else:
            lab, a, b = e.label, e.onset, e.offset
        rows.append({"sound": _norm(lab), "start": round(float(a), 2), "end": round(float(b), 2)})
    rows.sort(key=lambda r: (r["start"], r["end"], r["sound"]))
    return json.dumps(rows, ensure_ascii=False)


_LABEL_KEYS = ("sound", "label", "name", "event", "class")
_START_KEYS = ("start", "onset", "from", "begin")
_END_KEYS = ("end", "offset", "to", "stop")


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().rstrip("s"))
        except ValueError:
            return None
    return None


def _event_from_obj(it) -> Event3 | None:
    if isinstance(it, dict):
        lab = next((str(it[k]) for k in _LABEL_KEYS if k in it), None)
        a = next((_num(it[k]) for k in _START_KEYS if k in it and _num(it[k]) is not None), None)
        b = next((_num(it[k]) for k in _END_KEYS if k in it and _num(it[k]) is not None), None)
        if a is None or b is None:
            # {"sound": "dog", "time": [1.0, 2.0]} or {"dog": "1.0-2.0"}
            for k, v in it.items():
                if isinstance(v, (list, tuple)) and len(v) == 2 and _num(v[0]) is not None and _num(v[1]) is not None:
                    a, b = _num(v[0]), _num(v[1]); break
                if isinstance(v, str):
                    mm = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)", v)
                    if mm:
                        a, b = float(mm.group(1)), float(mm.group(2))
                        if lab is None and k not in _LABEL_KEYS:
                            lab = k
                        break
            if a is None or b is None:
                return None
        return (_norm(lab) if lab is not None else "", a, b)
    if isinstance(it, (list, tuple)):
        if len(it) == 3 and isinstance(it[0], str) and _num(it[1]) is not None and _num(it[2]) is not None:
            return (_norm(it[0]), _num(it[1]), _num(it[2]))
        if len(it) == 2 and _num(it[0]) is not None and _num(it[1]) is not None:
            return ("", _num(it[0]), _num(it[1]))
    return None


def parse_timeline(text: str | None) -> list[Event3] | None:
    """A list of (label, start, end); [] for an explicit empty answer; None if
    unreadable. Labels are normalised like ctag.agent._norm; an event whose
    label could not be read gets the empty string, which never matches."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    for m in re.finditer(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]|\[\s*\{.*\}\s*\]", text, re.S):
        blob = m.group(0)
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(blob)
            except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
                continue
            if isinstance(obj, list):
                if not obj:
                    return []
                evs = [_event_from_obj(it) for it in obj]
                evs = [e for e in evs if e is not None]
                if evs:
                    return _clean(evs)
            break
    if re.search(r"\[\s*\]|\bno (sound )?events?\b|\bnothing\b|\bnone\b", text, re.I):
        return []
    return None


def _clean(evs: list[Event3]) -> list[Event3]:
    out = []
    for lab, a, b in evs:
        if b < a:
            a, b = b, a
        if b > a:
            out.append((lab, a, b))
    return sorted(out, key=lambda e: (e[1], e[2], e[0]))


# ------------------------------------------------------------------ scoring
def _match(pred: list[Event3], gold: list[Event3], thr: float, label_aware: bool) -> int:
    if not pred or not gold:
        return 0
    from scipy.optimize import linear_sum_assignment

    M = np.zeros((len(pred), len(gold)))
    for i, (pl, pa, pb) in enumerate(pred):
        for j, (gl, ga, gb) in enumerate(gold):
            if label_aware and _norm(pl) != _norm(gl):
                continue
            M[i, j] = iou((pa, pb), (ga, gb))
    r, c = linear_sum_assignment(-M)
    return int(sum(M[i, j] >= thr for i, j in zip(r, c)))


def score_timeline(pred: list[Event3] | None, gold: list[Event3], thr: float = 0.5) -> dict:
    """Event-level precision/recall/F1 with one-to-one matching at IoU >= thr.
    The label must agree for a match ("f1"); "f1_any_label" ignores labels so a
    heard-but-misnamed event can be told apart from a missed one."""
    parse_fail = pred is None
    p = pred or []
    pred_iv = [(a, b) for _, a, b in p]
    gold_iv = [(a, b) for _, a, b in gold]
    tp = 0 if parse_fail else _match(p, gold, thr, True)
    tp_any = 0 if parse_fail else _match(p, gold, thr, False)

    def prf(tp_):
        if not p and not gold:
            return 1.0, 1.0, 1.0
        if not p or not gold:
            return 0.0, 0.0, 0.0
        pr, rc = tp_ / len(p), tp_ / len(gold)
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    pr, rc, f1 = prf(tp)
    _, _, f1_any = prf(tp_any)
    dists, dur_ratio = localisation(pred_iv, gold_iv)
    # per-label recall, so "which sounds get missed" is answerable
    by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for lab in {g[0] for g in gold}:
        gl = [g for g in gold if g[0] == lab]
        pl = [x for x in p if _norm(x[0]) == _norm(lab)]
        by_label[_norm(lab)] = [0 if parse_fail else _match(pl, gl, thr, True), len(gl)]
    return {
        "parse_fail": parse_fail,
        "precision": pr, "recall": rc, "f1": f1, "f1_any_label": f1_any,
        "n_pred": len(p), "n_gold": len(gold), "count_acc": (not parse_fail) and len(p) == len(gold),
        "centre_errors": dists, "duration_ratio": dur_ratio,
        "label_hits": {k: v for k, v in by_label.items()},
    }


def summarize_timelines(rows: list[dict]) -> dict:
    n = len(rows) or 1
    label_tp: dict[str, int] = defaultdict(int)
    label_n: dict[str, int] = defaultdict(int)
    for r in rows:
        for lab, (tp, tot) in r.get("label_hits", {}).items():
            label_tp[lab] += tp
            label_n[lab] += tot
    dists = [d for r in rows for d in r.get("centre_errors", [])]
    durs = [r["duration_ratio"] for r in rows if r.get("duration_ratio") is not None]
    # pooled over events, which is the number the phase 3 gate is set on: a
    # clip-level mean over-weights clips with few events
    tp = sum(label_tp.values()); n_pred = sum(r["n_pred"] for r in rows); n_gold = sum(label_n.values())
    pp = tp / n_pred if n_pred else None
    pr_ = tp / n_gold if n_gold else None
    pf = (2 * pp * pr_ / (pp + pr_)) if pp is not None and pr_ is not None and (pp + pr_) else None
    return {
        "n_clips": len(rows),
        "parse_fail_rate": sum(r["parse_fail"] for r in rows) / n,
        "precision": float(np.mean([r["precision"] for r in rows])) if rows else None,
        "recall": float(np.mean([r["recall"] for r in rows])) if rows else None,
        "f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
        "f1_any_label": float(np.mean([r["f1_any_label"] for r in rows])) if rows else None,
        "event_precision_pooled": pp, "event_recall_pooled": pr_, "event_f1_pooled": pf,
        "count_acc": sum(r["count_acc"] for r in rows) / n,
        "under_report_rate": sum(r["n_pred"] < r["n_gold"] for r in rows) / n,
        "centre_error_median": sorted(dists)[len(dists) // 2] if dists else None,
        "centre_within_1s": (sum(d < 1.0 for d in dists) / len(dists)) if dists else None,
        "duration_ratio_median": sorted(durs)[len(durs) // 2] if durs else None,
        "recall_by_label": {k: label_tp[k] / label_n[k] for k in sorted(label_n)},
    }


# ------------------------------------------------------------------ grounder
def timeline_grounder(pred_timelines: dict) -> Grounder:
    """Answer "when does X occur?" from predicted timelines, so ctag.run_agent
    can score every query type with no further model calls. `pred_timelines`
    maps clip_id -> {"events": [{"label","onset","offset"}, ...]}; the clip id
    is the audio file's stem, as for the oracle grounder."""
    def g(audio_path: str, sound: str):
        clip = audio_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        evs = pred_timelines.get(clip, {}).get("events", [])
        want = _norm(sound)
        return _sorted([(round(e["onset"], 3), round(e["offset"], 3)) for e in evs if _norm(e["label"]) == want])

    return g


def events_to_dicts(evs: list[Event3]) -> list[dict]:
    return [{"label": lab, "onset": round(a, 3), "offset": round(b, 3)} for lab, a, b in evs]
