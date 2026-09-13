"""Query generation. Each Query carries its condition type, the parameters,
natural-language text, and the exact answer intervals from the timeline."""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from .timeline import Timeline

TYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ABSENT"]

ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}

# Several surface forms per type so the failure curve is not a template artefact.
TEMPLATES = {
    "PLAIN": ["every {X}", "all occurrences of {X}", "each time there is {X}"],
    "ORDINAL": ["the {K} {X}", "the {K} time {X} occurs", "{X}, {K} occurrence only"],
    "AFTER": ["every {X} after the {Y}", "all {X} that happen once the {Y} has ended", "each {X} following the {Y}"],
    "BEFORE": ["every {X} before the {Y}", "all {X} that finish before the {Y} starts", "each {X} preceding the {Y}"],
    "NEXT_AFTER": ["the first {X} after the {Y}", "the next {X} once the {Y} has ended", "the {X} that comes right after the {Y}"],
    "WHILE": ["every {X} while {Y} is happening", "all {X} that overlap with {Y}", "each {X} during {Y}"],
    "NOT_FOLLOWED": ["every {X} that is not followed by {Y} within {W} seconds", "all {X} with no {Y} in the {W} seconds after it ends"],
    "ABSENT": ["every {X}", "all occurrences of {X}"],
}


@dataclass
class Query:
    qid: str
    clip_id: str
    qtype: str
    text: str
    x: str
    y: str | None
    k: int | str | None
    window: float | None
    answer: list[tuple[float, float]]
    expects_empty: bool
    plain_intervals: list[tuple[float, float]] = field(default_factory=list)  # occ(X), for diagnostic baselines
    template_idx: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Query":
        d = dict(d)
        d["answer"] = [tuple(a) for a in d["answer"]]
        d["plain_intervals"] = [tuple(a) for a in d.get("plain_intervals", [])]
        return cls(**d)


def _render(qtype: str, idx: int, **kw) -> str:
    t = TEMPLATES[qtype][idx]
    if "K" in kw and isinstance(kw["K"], int):
        kw["K"] = ORDINAL_WORDS.get(kw["K"], f"{kw['K']}th")
    return t.format(**kw)


def _pick(cands: list, answer_fn, rng: random.Random, k: int, empty_frac: float) -> list:
    """Choose up to k candidates, preferring non-empty answers so that about
    empty_frac of the chosen ones are rejection queries."""
    rng.shuffle(cands)
    scored = [(c, answer_fn(*c)) for c in cands]
    nonempty = [sc for sc in scored if sc[1]]
    empty = [sc for sc in scored if not sc[1]]
    n_empty = min(len(empty), sum(rng.random() < empty_frac for _ in range(k)))  # per-slot draw, unbiased for small k
    chosen = nonempty[: k - n_empty] + empty[:n_empty]
    if len(chosen) < k:  # top up with whatever is left
        chosen += (nonempty[k - n_empty:] + empty[n_empty:])[: k - len(chosen)]
    rng.shuffle(chosen)
    return chosen


def generate(tl: Timeline, clip_id: str, vocab: list[str], rng: random.Random, max_per_type: int = 2,
             window: float = 3.0, empty_frac: float = 0.25) -> list[Query]:
    """Generate up to max_per_type queries per condition type for one timeline.
    About empty_frac of the relational/ordinal queries are rejection queries."""
    out: list[Query] = []
    labels = tl.labels()
    counts = {l: len(tl.occ(l)) for l in labels}
    uniques = [l for l in labels if counts[l] == 1]
    repeated = [l for l in labels if counts[l] >= 2]
    absent = [l for l in vocab if l not in labels]
    n = 0

    def add(qtype, text, x, y=None, k=None, w=None, answer=None, tidx=0):
        nonlocal n
        ans = [e.interval for e in answer]
        out.append(Query(f"{clip_id}_q{n}", clip_id, qtype, text, x, y, k, w, ans, len(ans) == 0,
                         [e.interval for e in tl.occ(x)], tidx))
        n += 1

    # PLAIN
    for x in rng.sample(labels, min(max_per_type, len(labels))):
        i = rng.randrange(len(TEMPLATES["PLAIN"]))
        add("PLAIN", _render("PLAIN", i, X=x), x, answer=tl.plain(x), tidx=i)

    # ORDINAL: on repeated labels; sometimes k beyond count (rejection)
    cands = [(x, k) for x in repeated for k in [*range(1, counts[x] + 1), "last", counts[x] + 1]]
    for (x, k), ans in _pick(cands, tl.ordinal, rng, max_per_type, empty_frac):
        i = rng.randrange(len(TEMPLATES["ORDINAL"]))
        add("ORDINAL", _render("ORDINAL", i, X=x, K=k), x, k=k, answer=ans, tidx=i)

    # AFTER / BEFORE / NEXT_AFTER: Y unique, X != Y
    for qtype in ("AFTER", "BEFORE", "NEXT_AFTER"):
        fn = {"AFTER": tl.after, "BEFORE": tl.before, "NEXT_AFTER": tl.next_after}[qtype]
        cands = [(x, y) for y in uniques for x in labels if x != y]
        for (x, y), ans in _pick(cands, fn, rng, max_per_type, empty_frac):
            i = rng.randrange(len(TEMPLATES[qtype]))
            add(qtype, _render(qtype, i, X=x, Y=y), x, y=y, answer=ans, tidx=i)

    # WHILE: X != Y, Y present
    cands = [(x, y) for x in labels for y in labels if x != y]
    for (x, y), ans in _pick(cands, tl.while_, rng, max_per_type, empty_frac):
        i = rng.randrange(len(TEMPLATES["WHILE"]))
        add("WHILE", _render("WHILE", i, X=x, Y=y), x, y=y, answer=ans, tidx=i)

    # NOT_FOLLOWED
    cands = [(x, y) for x in labels for y in labels if x != y]
    for (x, y), ans in _pick(cands, lambda a, b: tl.not_followed(a, b, window), rng, max_per_type, empty_frac):
        i = rng.randrange(len(TEMPLATES["NOT_FOLLOWED"]))
        add("NOT_FOLLOWED", _render("NOT_FOLLOWED", i, X=x, Y=y, W=int(window)), x, y=y, w=window, answer=ans, tidx=i)

    # ABSENT
    for x in rng.sample(absent, min(1, len(absent))):
        i = rng.randrange(len(TEMPLATES["ABSENT"]))
        add("ABSENT", _render("ABSENT", i, X=x), x, answer=tl.absent(x), tidx=i)

    return out
