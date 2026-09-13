"""Clip-level train/val/test split.

Splitting by query would leak: queries from one clip share the same audio and
the same event timeline, so a model could memorise a clip during training and
be scored on a different question about that same audio. Splits are therefore
by clip id, and the split is a deterministic function of (clip_id, seed) so it
survives regenerating the benchmark.

    python -m ctag.split --bench data/esc50/benchmark.jsonl --out data/esc50
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def assign(clip_id: str, seed: int, frac_train: float, frac_val: float) -> str:
    """Stable hash rather than a shuffle: adding clips later does not reshuffle
    the ones already assigned, so a model trained earlier is never silently
    evaluated on its own training clips."""
    h = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / 2 ** 64
    if x < frac_train:
        return "train"
    if x < frac_train + frac_val:
        return "val"
    return "test"


def split_benchmark(bench: Path, out: Path, seed: int = 0,
                    frac_train: float = 0.7, frac_val: float = 0.15) -> dict:
    rows = [json.loads(l) for l in open(bench, encoding="utf-8")]
    buckets: dict[str, list] = defaultdict(list)
    clips: dict[str, set] = defaultdict(set)
    for r in rows:
        s = assign(r["clip_id"], seed, frac_train, frac_val)
        buckets[s].append(r)
        clips[s].add(r["clip_id"])

    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for s in ("train", "val", "test"):
        p = out / f"benchmark_{s}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in buckets[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[s] = {"clips": len(clips[s]), "queries": len(buckets[s]),
                    "by_type": dict(Counter(r["qtype"] for r in buckets[s]))}

    overlap = (clips["train"] & clips["val"]) | (clips["train"] & clips["test"]) | (clips["val"] & clips["test"])
    assert not overlap, f"clip leak across splits: {sorted(overlap)[:5]}"
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--frac-val", type=float, default=0.15)
    a = ap.parse_args(argv)
    stats = split_benchmark(Path(a.bench), Path(a.out), a.seed, a.frac_train, a.frac_val)
    for s, d in stats.items():
        print(f"{s:<6} {d['clips']:4d} clips  {d['queries']:5d} queries  {d['by_type']}")


if __name__ == "__main__":
    main()
