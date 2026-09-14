"""Generate a large composed training set: timelines (+ audio, + optional queries).

The benchmark generator (ctag.build_benchmark) makes a few hundred clips with
one fixed recipe. Training the whole-timeline target wants tens of thousands
of clips and, deliberately, the cases the phase 2 error budget points at:
overlapping sounds, the same sound repeated, short gaps between events, low
signal-to-noise. `--hard` draws every recipe knob per clip from a wide range.

    # 20,000 ESC-50 clips, 8 processes, PCM-16 wav (~0.6 MB per 20 s clip)
    python -m ctag.gen_train --source esc50 --n-clips 20000 --hard --workers 8 \
           --out data/gen_esc50 --esc50-root data/esc50_raw

    # timelines only (no audio) to size and inspect a recipe first
    python -m ctag.gen_train --source procedural --n-clips 200 --hard --no-audio --out /tmp/gen

Every clip is a pure function of (seed, index): the same command regenerates
the same audio, so timelines.jsonl is the durable artefact and audio can be
re-rendered anywhere with --render-only. Clip ids are `gen<seed>_<i>` and never
collide with the benchmark's `esc50_*`, so a split hash keeps them apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from .compose import SR, ESC50Bank, ProceduralBank, compose_clip
from .queries import generate

_BANK = None
_ARGS = None


def clip_seed(seed: int, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{i}".encode()).digest()[:8], "big")


def draw_recipe(rng: random.Random, hard: bool, duration: float) -> dict:
    """The per-clip recipe. Easy = the benchmark's fixed recipe. Hard = wide ranges
    biased towards the failure modes measured in phase 2."""
    if not hard:
        return {"duration": duration, "n_events": 6, "n_labels": 3, "p_overlap": 0.45,
                "min_overlap": 0.3, "snr_db": 20.0, "gap": (0.4, 2.5)}
    mode = rng.random()
    if mode < 0.35:        # crowded: many events, short gaps, more repeats of few labels
        rec = {"n_events": rng.randint(7, 10), "n_labels": rng.randint(2, 3),
               "p_overlap": rng.uniform(0.3, 0.6), "gap": (0.1, 0.8)}
    elif mode < 0.70:      # overlap-heavy: WHILE cases with long shared time
        rec = {"n_events": rng.randint(5, 8), "n_labels": rng.randint(3, 5),
               "p_overlap": rng.uniform(0.55, 0.85), "gap": (0.2, 1.5)}
    else:                  # benchmark-like with mild variation
        rec = {"n_events": rng.randint(4, 7), "n_labels": rng.randint(2, 4),
               "p_overlap": rng.uniform(0.2, 0.5), "gap": (0.3, 2.5)}
    rec.update({"duration": duration, "min_overlap": rng.uniform(0.2, 0.6),
                "snr_db": rng.uniform(6.0, 30.0)})
    return rec


def _init(source: str, esc50_root: str | None, classes: list[str] | None):
    global _BANK
    rng = random.Random(0)
    _BANK = ProceduralBank(rng) if source == "procedural" else ESC50Bank(Path(esc50_root), rng, classes)


def _one(task: tuple) -> dict:
    """Compose one clip deterministically. Runs in a worker process."""
    import soundfile as sf

    i, seed, hard, duration, out, fmt, write_audio, max_per_type, want_queries = task
    rng = random.Random(clip_seed(seed, i))
    _BANK.rng = rng                       # the bank draws source files from the same stream
    rec = draw_recipe(rng, hard, duration)
    audio, tl = compose_clip(_BANK, rng, **rec)
    clip_id = f"gen{seed}_{i:06d}"
    ext = "flac" if fmt == "flac" else "wav"
    wav = Path(out) / "audio" / f"{clip_id}.{ext}"
    if write_audio:
        sf.write(wav, audio, SR, subtype="PCM_16")
    row = {"clip_id": clip_id, "audio": str(wav), "recipe": {**rec, "gap": list(rec["gap"])}, **tl.to_dict()}
    queries = []
    if want_queries:
        vocab = _BANK.labels()
        qrng = random.Random(clip_seed(seed, i) ^ 0x5EED)
        for q in generate(tl, clip_id, vocab, qrng, max_per_type):
            for lab in vocab:
                q.text = q.text.replace(lab, _BANK.phrase(lab))
            d = q.to_dict(); d["audio"], d["duration"] = str(wav), duration
            queries.append(d)
    return {"timeline": row, "queries": queries}


def stats(rows: list[dict]) -> dict:
    n = len(rows) or 1
    ev = [len(r["events"]) for r in rows]
    overlaps = repeats = 0
    for r in rows:
        evs = sorted(r["events"], key=lambda e: e["onset"])
        for a, b in zip(evs, evs[1:]):
            if b["onset"] < a["offset"]:
                overlaps += 1
        c = Counter(e["label"] for e in evs)
        repeats += sum(1 for v in c.values() if v >= 2)
    labels = Counter(e["label"] for r in rows for e in r["events"])
    return {"clips": len(rows), "events_total": sum(ev), "events_per_clip_mean": round(sum(ev) / n, 2),
            "clips_with_overlap": sum(1 for r in rows if any(
                b["onset"] < a["offset"] for a, b in zip(sorted(r["events"], key=lambda e: e["onset"]),
                                                          sorted(r["events"], key=lambda e: e["onset"])[1:]))),
            "overlapping_pairs": overlaps, "repeated_labels": repeats,
            "label_counts": dict(sorted(labels.items()))}


def build(source: str, n_clips: int, out: Path, seed: int = 0, hard: bool = False, duration: float = 20.0,
          workers: int = 1, fmt: str = "wav", write_audio: bool = True, queries: bool = False,
          max_per_type: int = 2, esc50_root: Path | None = None, classes: list[str] | None = None,
          start: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if write_audio:
        (out / "audio").mkdir(exist_ok=True)
    tasks = [(i, seed, hard, duration, str(out), fmt, write_audio, max_per_type, queries)
             for i in range(start, start + n_clips)]
    init_args = (source, str(esc50_root) if esc50_root else None, classes)
    results = []
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=init_args) as pool:
            for k, r in enumerate(pool.imap(_one, tasks, chunksize=8)):
                results.append(r)
                if (k + 1) % 500 == 0:
                    print(f"{k+1}/{n_clips} clips", flush=True)
    else:
        _init(*init_args)
        results = [_one(t) for t in tasks]
    mode = "a" if start > 0 else "w"
    with open(out / "timelines.jsonl", mode, encoding="utf-8") as ft:
        for r in results:
            ft.write(json.dumps(r["timeline"]) + "\n")
    if queries:
        with open(out / "benchmark.jsonl", mode, encoding="utf-8") as fq:
            for r in results:
                for d in r["queries"]:
                    fq.write(json.dumps(d, ensure_ascii=False) + "\n")
    s = stats([r["timeline"] for r in results])
    s.update({"source": source, "seed": seed, "hard": hard, "audio": write_audio, "format": fmt,
              "queries": sum(len(r["queries"]) for r in results)})
    (out / "gen_stats.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["procedural", "esc50"], required=True)
    ap.add_argument("--n-clips", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="first clip index (to extend a set in chunks)")
    ap.add_argument("--hard", action="store_true", help="wide, failure-mode-biased recipes")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--format", choices=["wav", "flac"], default="wav")
    ap.add_argument("--no-audio", action="store_true", help="timelines only")
    ap.add_argument("--queries", action="store_true", help="also emit per-question benchmark rows")
    ap.add_argument("--max-per-type", type=int, default=2)
    ap.add_argument("--esc50-root", default=None)
    ap.add_argument("--classes", default=None, help="comma-separated ESC-50 classes (default: the benchmark's 14)")
    a = ap.parse_args(argv)
    s = build(a.source, a.n_clips, Path(a.out), a.seed, a.hard, a.duration, a.workers, a.format,
              not a.no_audio, a.queries, a.max_per_type, Path(a.esc50_root) if a.esc50_root else None,
              a.classes.split(",") if a.classes else None, a.start)
    print(json.dumps({k: v for k, v in s.items() if k != "label_counts"}, indent=2))


if __name__ == "__main__":
    main()
