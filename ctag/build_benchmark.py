"""Compose clips and generate queries -> <out>/benchmark.jsonl + <out>/wav/*.wav + <out>/timelines.jsonl

    python -m ctag.build_benchmark --source procedural --n-clips 60 --out data/proc
    python -m ctag.build_benchmark --source esc50 --n-clips 300 --out data/esc50_bench [--esc50-root data/esc50]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import soundfile as sf

from .compose import SR, ESC50Bank, ProceduralBank, compose_clip
from .queries import generate


def build(source: str, n_clips: int, out: Path, seed: int = 0, duration: float = 20.0, n_events: int = 6,
          n_labels: int = 3, p_overlap: float = 0.35, esc50_root: Path | None = None, max_per_type: int = 2):
    rng = random.Random(seed)
    bank = ProceduralBank(rng) if source == "procedural" else ESC50Bank(esc50_root or out.parent / "esc50", rng)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav").mkdir(exist_ok=True)
    vocab = bank.labels()
    n_q = Counter()
    with open(out / "benchmark.jsonl", "w", encoding="utf-8") as fq, open(out / "timelines.jsonl", "w", encoding="utf-8") as ft:
        for i in range(n_clips):
            clip_id = f"{source}_{i:05d}"
            audio, tl = compose_clip(bank, rng, duration, n_events, n_labels, p_overlap)
            wav = out / "wav" / f"{clip_id}.wav"
            sf.write(wav, audio, SR)
            ft.write(json.dumps({"clip_id": clip_id, "audio": str(wav), **tl.to_dict()}) + "\n")
            # query text uses human phrases; predicates use raw labels
            for q in generate(tl, clip_id, vocab, rng, max_per_type):
                q.text = _humanize(q.text, bank)
                d = q.to_dict()
                d["audio"], d["duration"] = str(wav), duration
                fq.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_q[q.qtype] += 1
    print(f"{n_clips} clips, {sum(n_q.values())} queries: {dict(n_q)}")
    return n_q


def _humanize(text: str, bank) -> str:
    for lab in bank.labels():
        text = text.replace(lab, bank.phrase(lab))
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["procedural", "esc50"], required=True)
    ap.add_argument("--n-clips", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--n-events", type=int, default=6)
    ap.add_argument("--n-labels", type=int, default=3)
    ap.add_argument("--p-overlap", type=float, default=0.35)
    ap.add_argument("--esc50-root", default=None)
    ap.add_argument("--max-per-type", type=int, default=2)
    a = ap.parse_args(argv)
    build(a.source, a.n_clips, Path(a.out), a.seed, a.duration, a.n_events, a.n_labels, a.p_overlap,
          Path(a.esc50_root) if a.esc50_root else None, a.max_per_type)


if __name__ == "__main__":
    main()
