"""Transcribe whole clips and score the timelines.

    # CPU sanity check: the oracle reads the gold timelines back (scores 1.0),
    # optionally degraded, to exercise the whole path without a GPU
    python -m ctag.run_transcribe --model mock:oracle --bench data/proc/benchmark_test.jsonl \
           --timelines data/proc/timelines.jsonl --out runs/proc/transcribe_oracle [--drop 0.2 --jitter 0.3]

    # a real model, base or with the trained weights from ctag.train_lora
    python -m ctag.run_transcribe --model qwen2.5-omni [--adapter runs/lora_tt] \
           --bench data/esc50/benchmark_test.jsonl --timelines data/esc50/timelines.jsonl \
           --out runs/esc50/transcribe_lora

Outputs <out>/pred_timelines.jsonl (one row per clip, timelines.jsonl layout plus
the raw answer and per-clip scores) and <out>/summary.json. Feed the predictions
to `ctag.run_agent --grounder timeline` to score every query type from them.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .agent import _norm
from .transcribe import (events_to_dicts, parse_timeline, score_timeline, summarize_timelines,
                         transcribe_query)


def clips_of(bench: Path) -> list[tuple[str, str, float | None]]:
    """Unique (clip_id, audio, duration) in first-seen order."""
    seen, out = set(), []
    for line in open(bench, encoding="utf-8"):
        d = json.loads(line)
        if d["clip_id"] in seen:
            continue
        seen.add(d["clip_id"])
        out.append((d["clip_id"], d["audio"], d.get("duration")))
    return out


def load_timelines(path: Path) -> dict:
    tl = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        tl[d["clip_id"]] = d
    return tl


def mock_transcriber(gold: dict, jitter: float = 0.0, drop: float = 0.0, spurious: float = 0.0,
                     relabel: float = 0.0, seed: int = 0):
    """Reads the gold timeline and degrades it: dropped events, moved edges,
    invented events, and wrong names. Emits the exact target string so the
    parser is exercised too."""
    from .transcribe import target_timeline

    rng = random.Random(seed)
    vocab = sorted({e["label"] for d in gold.values() for e in d["events"]})

    def t(clip_id: str, audio: str, duration):
        evs = []
        for e in gold.get(clip_id, {}).get("events", []):
            if rng.random() < drop:
                continue
            a, b = e["onset"], e["offset"]
            if jitter:
                a += rng.uniform(-jitter, jitter); b += rng.uniform(-jitter, jitter)
                a, b = max(0.0, min(a, b)), max(a, b) + (0.05 if b <= a else 0.0)
            lab = e["label"]
            if relabel and rng.random() < relabel:
                lab = rng.choice([v for v in vocab if v != lab] or [lab])
            evs.append({"label": lab, "onset": a, "offset": b})
        if spurious and rng.random() < spurious:
            s = rng.uniform(0, max(1.0, (duration or 20.0) - 2))
            evs.append({"label": rng.choice(vocab), "onset": s, "offset": s + rng.uniform(0.3, 2.0)})
        return target_timeline(evs)

    return t


def model_transcriber(name: str, vocab: list[str] | None, **kw):
    from .models import get_backend

    backend = get_backend(name, **kw)
    q = transcribe_query(vocab)

    def t(clip_id: str, audio: str, duration):
        return backend.ground(audio, q, query=None, duration=duration)

    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="'mock:oracle' or a model name from ctag.models")
    ap.add_argument("--bench", required=True, help="a benchmark split; its unique clips are transcribed")
    ap.add_argument("--timelines", required=True, help="gold timelines.jsonl, for scoring")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="trained weights from ctag.train_lora")
    ap.add_argument("--n", type=int, default=None, help="only the first n clips")
    ap.add_argument("--no-vocab-in-prompt", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.0, help="mock: +/- seconds on each edge")
    ap.add_argument("--drop", type=float, default=0.0, help="mock: probability of dropping an event")
    ap.add_argument("--spurious", type=float, default=0.0, help="mock: probability of one invented event")
    ap.add_argument("--relabel", type=float, default=0.0, help="mock: probability of a wrong name")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    gold = load_timelines(Path(a.timelines))
    vocab = None if a.no_vocab_in_prompt else sorted({e["label"] for d in gold.values() for e in d["events"]})
    if a.model == "mock:oracle":
        t = mock_transcriber(gold, a.jitter, a.drop, a.spurious, a.relabel, a.seed)
        label = f"transcribe:mock(j={a.jitter},d={a.drop},s={a.spurious},r={a.relabel})"
    else:
        kw = {}
        if a.adapter:
            if a.model != "qwen2.5-omni":
                raise SystemExit("--adapter is only wired for the qwen2.5-omni backend")
            kw["adapter"] = a.adapter
        t = model_transcriber(a.model, vocab, **kw)
        label = f"transcribe:{a.model}" + ("+lora" if a.adapter else "")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    clips = clips_of(Path(a.bench))
    if a.n is not None:
        clips = clips[: a.n]
    with open(out / "pred_timelines.jsonl", "w", encoding="utf-8") as fo:
        for k, (cid, audio, duration) in enumerate(clips):
            raw = t(cid, audio, duration)
            pred = parse_timeline(raw)
            gold_ev = [(_norm(e["label"]), e["onset"], e["offset"]) for e in gold.get(cid, {}).get("events", [])]
            s = score_timeline(pred, gold_ev, a.iou)
            row = {"clip_id": cid, "audio": audio, "duration": duration,
                   "events": events_to_dicts(pred or []), "raw": raw, **s}
            rows.append(row)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 50 == 0:
                print(f"{k+1} clips | {time.time()-t0:.0f}s", flush=True)

    summary = {"model": label, "bench": a.bench, "iou": a.iou, **summarize_timelines(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = ["n_clips", "precision", "recall", "f1", "f1_any_label", "count_acc", "under_report_rate",
            "parse_fail_rate", "centre_error_median", "duration_ratio_median"]
    print("  ".join(f"{k}={summary[k]:.3f}" if isinstance(summary[k], float) else f"{k}={summary[k]}" for k in keys))
    worst = sorted(summary["recall_by_label"].items(), key=lambda kv: kv[1])[:5]
    print("lowest recall by sound:", ", ".join(f"{k} {v:.2f}" for k, v in worst))


if __name__ == "__main__":
    main()
