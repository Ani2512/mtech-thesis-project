"""Recall-biased grounding: train and decode for the asymmetry we measured.

Downstream composition punishes a missed event far harder than a spurious one.
Holding everything else perfect and degrading only one failure mode at a time
over the full 4404-query benchmark (docs/recall_bias.md):

    cost per unit miss rate        1.411
    cost per unit false-alarm rate 0.252
    asymmetry                      5.60x

The reason is structural, not incidental. A missed occurrence removes a correct
interval *and* can silently change a relational answer -- lose the car horn and
"every bark after the horn" becomes unanswerable. A spurious detection usually
adds one wrong interval and leaves the rest intact.

So the objective should not be F1, which weights precision and recall equally.
It should be F-beta with beta^2 set to the measured asymmetry. We do not tune
beta; it is read off the degradation slopes.

Two mechanisms follow, one for training and one for decoding:

* `make_preference_pairs` builds preference data whose *rejected* side is always
  an under-detection. Over-detections are deliberately never used as negatives,
  because penalising them would train away the very behaviour that is cheap.
* `union_decode` merges several samples into one answer. Sampling k times and
  taking the union raises recall at a small precision cost, which the 5.6x
  asymmetry makes a favourable trade.
"""
from __future__ import annotations

import random

Interval = tuple[float, float]

# Measured, not tuned: the ratio of downstream cost per unit miss rate to cost
# per unit false-alarm rate, over the ESC-50 benchmark. See docs/recall_bias.md.
MEASURED_ASYMMETRY = 5.60
MEASURED_BETA = MEASURED_ASYMMETRY ** 0.5      # 2.37


def _iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def f_beta(pred: list[Interval], gt: list[Interval], beta: float = MEASURED_BETA,
           thr: float = 0.5) -> float:
    """F-beta over one-to-one matched intervals. beta > 1 weights recall.

    beta = 1 reproduces the f1@0.5 used elsewhere, so the two are comparable.
    """
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    M = np.array([[_iou(p, g) for g in gt] for p in pred])
    r, c = linear_sum_assignment(-M)
    tp = int(sum(M[i, j] >= thr for i, j in zip(r, c)))
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gt)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def union_decode(samples: list[list[Interval] | None], min_votes: int = 1,
                 merge_iou: float = 0.5) -> list[Interval]:
    """Merge k sampled answers into one, biased towards recall.

    Intervals are clustered by IoU across samples; a cluster is kept if it
    appears in at least `min_votes` distinct samples, and is reported as the
    span covering its members. min_votes=1 is pure union, the most recall-heavy
    setting and the one the measured asymmetry favours. Raising it trades recall
    back for precision and is the knob a paper should sweep.
    """
    items = []
    for si, s in enumerate(samples):
        for iv in (s or []):
            items.append((si, (float(iv[0]), float(iv[1]))))
    if not items:
        return []

    clusters: list[dict] = []
    for si, iv in sorted(items, key=lambda x: x[1][0]):
        placed = False
        for cl in clusters:
            if _iou(cl["span"], iv) >= merge_iou:
                cl["span"] = (min(cl["span"][0], iv[0]), max(cl["span"][1], iv[1]))
                cl["votes"].add(si)
                placed = True
                break
        if not placed:
            clusters.append({"span": iv, "votes": {si}})

    out = [cl["span"] for cl in clusters if len(cl["votes"]) >= min_votes]
    return sorted(out)


def make_preference_pairs(examples: list[dict], rng: random.Random,
                          drop_min: int = 1, max_pairs_per_example: int = 1) -> list[dict]:
    """Preference data whose rejected side always under-detects.

    Each pair keeps the true answer as `chosen` and removes at least one
    interval to form `rejected`. Over-detection is never used as a negative:
    the measured cost of a false alarm is a fifth that of a miss, so teaching
    the model to avoid over-calling would optimise the wrong direction.

    Examples with fewer than two intervals cannot form an under-detection that
    is still non-empty, so they yield a pair against the empty answer instead,
    which is the most severe miss.
    """
    from .metrics import parse_intervals

    pairs = []
    for ex in examples:
        gold = parse_intervals(ex["target"])
        if not gold:
            continue                                   # nothing to under-detect
        for _ in range(max_pairs_per_example):
            if len(gold) == 1:
                rejected = []
            else:
                k = rng.randint(drop_min, max(drop_min, len(gold) - 1))
                keep = sorted(rng.sample(range(len(gold)), len(gold) - k))
                rejected = [gold[i] for i in keep]
            if rejected == gold:
                continue
            pairs.append({
                "audio": ex["audio"],
                "messages": ex["messages"],
                "chosen": ex["target"],
                "rejected": _render_like(ex["target"], rejected),
                "n_gold": len(gold),
                "n_rejected": len(rejected),
            })
    return pairs


def _render_like(reference_target: str, intervals) -> str:
    """Render in whatever format the reference target uses, so chosen and
    rejected differ only in content."""
    if "<t=" in reference_target:
        from .timetokens import TimeVocab
        return TimeVocab().encode(intervals)
    import json
    return json.dumps([[round(float(a), 2), round(float(b), 2)] for a, b in intervals])
