"""Mix supervised fine-tuning files, each repeated a whole number of times.

TEMPO-style second stage on real recordings: a few hundred real clips next to
thousands of generated ones. Repeating the real part keeps it from drowning,
the generated part keeps the model from forgetting what it learned there.

    python -m ctag.sft_mix --part data/desed/validation/sft_real_train.jsonl:4 \
           --part data/gen_varied/sft_transcribe.jsonl:1 --out runs/real/sft_mix.jsonl

Every example keeps a "source" field (the file's stem) so the mix can be
audited; ctag.train_lora ignores extra fields.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_part(spec: str) -> tuple[Path, int]:
    path, _, rep = spec.rpartition(":")
    if not path or not rep.isdigit():          # no ":N" suffix -> repeat once
        path, rep = spec, "1"
    if int(rep) < 1:
        raise ValueError(f"repeat must be >= 1 in {spec!r}")
    return Path(path), int(rep)


def mix(parts: list[tuple[Path, int]], out: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rows, per_part = [], {}
    for path, rep in parts:
        if not path.exists():
            raise FileNotFoundError(path)
        src = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not src:
            raise ValueError(f"{path} has no examples")
        for ex in src:
            ex.setdefault("source", path.stem)
        rows += src * rep
        per_part[str(path)] = {"examples": len(src), "repeat": rep, "in_mix": len(src) * rep}
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"examples": len(rows), "parts": per_part,
            "by_source": dict(Counter(ex["source"] for ex in rows)),
            "empty_targets": sum(1 for ex in rows if ex.get("target") in ("[]", "<t=none>"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True, metavar="FILE[:REPEAT]",
                    help="an SFT jsonl and how many times to include it (default 1); repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s = mix([parse_part(p) for p in a.part], Path(a.out), a.seed)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
