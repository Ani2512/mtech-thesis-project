"""Training-free decompose-and-combine grounding.

A conditional query names a target X and, usually, a reference Y. Instead of
asking the model to honour the condition, ask it only for plain groundings of
X and of Y, then apply the condition ourselves with the same predicates that
define the ground truth.

This is the comparison arm for phase 2: if it closes the gap without any
training, the contribution is the decomposition rather than the fine-tuning,
and we would rather know that before spending GPU hours.

Substituting a perfect grounder (`oracle_grounder`) turns it into a diagnostic:
it measures how much of the conditional failure is bad grounding and how much
is the composition itself. That runs on CPU.
"""
from __future__ import annotations

from typing import Callable, Protocol

Interval = tuple[float, float]

# A grounder answers "when does this sound occur?" and nothing more.
Grounder = Callable[[str, str], list[Interval]]   # (audio_path, sound) -> intervals


def _norm(label: str) -> str:
    """Queries carry raw labels ('glass_breaking'); prompts carry phrases
    ('glass breaking'). Normalise both sides identically or multi-word sounds
    silently never match and the oracle stops being an oracle."""
    return label.strip().lower().replace("_", " ")


def _centre(iv: Interval) -> float:
    return (iv[0] + iv[1]) / 2


def _sorted(xs: list[Interval]) -> list[Interval]:
    return sorted(xs, key=lambda p: (p[0], p[1]))


def _unique_reference(ys: list[Interval]) -> Interval | None:
    """AFTER/BEFORE/NEXT_AFTER need one unambiguous reference. If the grounder
    returns several, take the longest: on a correct grounding of a
    single-occurrence event the extras are spurious fragments."""
    if not ys:
        return None
    return max(ys, key=lambda p: p[1] - p[0])


def combine(qtype: str, xs: list[Interval], ys: list[Interval],
            k: int | str | None = None, window: float | None = None) -> list[Interval]:
    """Apply the condition to grounded intervals. Mirrors ctag.timeline."""
    xs, ys = _sorted(xs), _sorted(ys)
    if qtype in ("PLAIN", "ABSENT"):
        return xs
    if qtype == "ORDINAL":
        if k == "last":
            return xs[-1:]
        if isinstance(k, int) and k >= 1:
            return [xs[k - 1]] if len(xs) >= k else []
        return []
    if qtype in ("AFTER", "BEFORE", "NEXT_AFTER"):
        ref = _unique_reference(ys)
        if ref is None:
            return []
        if qtype == "BEFORE":
            return [x for x in xs if x[1] <= ref[0]]
        after = [x for x in xs if x[0] >= ref[1]]
        return after[:1] if qtype == "NEXT_AFTER" else after
    if qtype == "WHILE":
        return [x for x in xs if any(min(x[1], y[1]) - max(x[0], y[0]) > 0 for y in ys)]
    if qtype == "NOT_FOLLOWED":
        w = window if window is not None else 3.0
        return [x for x in xs if not any(x[1] <= y[0] <= x[1] + w for y in ys)]
    raise KeyError(qtype)


def ground_query(query, audio_path: str, grounder: Grounder) -> list[Interval]:
    """Two plain grounding calls at most, then local composition."""
    xs = grounder(audio_path, query.x)
    ys = grounder(audio_path, query.y) if query.y else []
    return combine(query.qtype, xs, ys, query.k, query.window)


def oracle_grounder(timelines: dict) -> Grounder:
    """A perfect grounder, for measuring the ceiling of decomposition.

    Any residual error is then the composition's fault, not perception's.
    `timelines` maps clip_id -> {"events": [{"label","onset","offset"}, ...]}.
    """
    by_path: dict[str, list[dict]] = {}

    def g(audio_path: str, sound: str) -> list[Interval]:
        evs = by_path.get(audio_path)
        if evs is None:
            clip = audio_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            evs = timelines.get(clip, {}).get("events", [])
            by_path[audio_path] = evs
        want = _norm(sound)
        out = []
        for e in evs:
            if _norm(e["label"]) == want:
                out.append((round(e["onset"], 3), round(e["offset"], 3)))
        return _sorted(out)

    return g


def noisy_grounder(base: Grounder, jitter: float = 0.0, drop: float = 0.0,
                   spurious: float = 0.0, seed: int = 0) -> Grounder:
    """Degrade a perfect grounder in controlled ways, to trace how decomposition
    accuracy falls as grounding quality falls."""
    import random

    rng = random.Random(seed)

    def g(audio_path: str, sound: str) -> list[Interval]:
        out = []
        for a, b in base(audio_path, sound):
            if rng.random() < drop:
                continue
            if jitter:
                a += rng.uniform(-jitter, jitter)
                b += rng.uniform(-jitter, jitter)
                a, b = min(a, b), max(a, b)
                if b - a < 0.05:
                    b = a + 0.05
            out.append((max(0.0, a), b))
        if spurious and rng.random() < spurious:
            s = rng.uniform(0, 18)
            out.append((s, s + rng.uniform(0.3, 2.0)))
        return _sorted(out)

    return g
