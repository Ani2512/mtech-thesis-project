"""Build supervised fine-tuning examples from a benchmark split.

The prompt format here is byte-identical to inference (`ctag.models.SYSTEM`
and `prompt_for`), so the model is never trained on one phrasing and evaluated
on another.

Mix: docs/phase2_decomposition.md shows grounding quality, not condition
handling, is the binding constraint -- with perfect events the conditions are
trivial (1.000), while at the model's real grounding quality even flawless
condition logic reaches only 0.263. Training is therefore weighted towards
plain grounding. Extra PLAIN examples are synthesised for every label in every
training clip, which costs nothing because the timeline already has the answer.

    python -m ctag.sft_data --bench data/esc50/benchmark_train.jsonl \
           --timelines data/esc50/timelines.jsonl --out data/esc50/sft_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .models import SYSTEM, prompt_for


_TIME_VOCAB = None


def target_string(intervals, time_tokens: bool = False) -> str:
    """Exactly the format the parser expects and the metric scores.

    With time_tokens, emit atomic timestamp tokens instead of digit strings;
    see ctag/timetokens.py for why and for the prior work it follows.
    """
    if time_tokens:
        global _TIME_VOCAB
        if _TIME_VOCAB is None:
            from .timetokens import TimeVocab
            _TIME_VOCAB = TimeVocab()
        return _TIME_VOCAB.encode(intervals)
    return json.dumps([[round(float(a), 2), round(float(b), 2)] for a, b in intervals])


def example(audio: str, query_text: str, answer, duration: float | None = None,
            kind: str = "conditional", time_tokens: bool = False) -> dict:
    return {
        "audio": audio,
        "kind": kind,          # 'plain' or 'conditional'; lets the mix be audited
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_for(query_text, duration)},
        ],
        "target": target_string(answer, time_tokens),
    }


PLAIN_TEMPLATES = ["every {X}", "all occurrences of {X}", "each time there is {X}",
                   "when does {X} occur", "locate every {X}"]


def synth_plain(timelines: dict, clip_ids: set[str], audio_of: dict[str, str],
                rng: random.Random, per_clip_absent: int = 1,
                vocab: list[str] | None = None, time_tokens: bool = False) -> list[dict]:
    """One PLAIN example per (clip, label present), plus a few absent-sound
    examples so the model learns that an empty answer is legitimate."""
    out = []
    for cid in sorted(clip_ids):
        tl = timelines.get(cid)
        if not tl:
            continue
        audio = audio_of.get(cid)
        if not audio:
            continue
        labels = []
        for e in tl["events"]:
            if e["label"] not in labels:
                labels.append(e["label"])
        for lab in labels:
            iv = [(e["onset"], e["offset"]) for e in tl["events"] if e["label"] == lab]
            phrase = lab.replace("_", " ")
            t = rng.choice(PLAIN_TEMPLATES).format(X=phrase)
            out.append(example(audio, t, sorted(iv), tl.get("duration"), kind="plain",
                               time_tokens=time_tokens))
        if vocab:
            absent = [l for l in vocab if l not in labels]
            for lab in rng.sample(absent, min(per_clip_absent, len(absent))):
                t = rng.choice(PLAIN_TEMPLATES).format(X=lab.replace("_", " "))
                out.append(example(audio, t, [], tl.get("duration"), kind="plain",
                                   time_tokens=time_tokens))
    return out


def build(bench: Path, timelines: Path, out: Path, plain_ratio: float = 0.5,
          seed: int = 0, augment_plain: bool = True, time_tokens: bool = False) -> dict:
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(bench, encoding="utf-8")]
    tl, audio_of = {}, {}
    for line in open(timelines, encoding="utf-8"):
        d = json.loads(line)
        tl[d["clip_id"]] = d
        audio_of[d["clip_id"]] = d.get("audio", "")

    clip_ids = {r["clip_id"] for r in rows}
    vocab = sorted({e["label"] for d in tl.values() for e in d["events"]})

    from_bench = [example(r["audio"], r["text"], r["answer"], r.get("duration"),
                          kind="plain" if r["qtype"] in ("PLAIN", "ABSENT") else "conditional",
                          time_tokens=time_tokens)
                  for r in rows]
    plain_bench = [e for e in from_bench if e["kind"] == "plain"]
    cond_bench = [e for e in from_bench if e["kind"] == "conditional"]

    plain = list(plain_bench)
    if augment_plain:
        plain += synth_plain(tl, clip_ids, audio_of, rng, vocab=vocab, time_tokens=time_tokens)

    # hit the requested plain share by trimming whichever side is over-represented
    if plain_ratio <= 0:
        chosen = cond_bench
    elif plain_ratio >= 1:
        chosen = plain
    else:
        want_cond = int(len(plain) * (1 - plain_ratio) / plain_ratio)
        if want_cond <= len(cond_bench):
            chosen = plain + rng.sample(cond_bench, want_cond)
        else:
            want_plain = int(len(cond_bench) * plain_ratio / (1 - plain_ratio))
            chosen = rng.sample(plain, min(want_plain, len(plain))) + cond_bench

    rng.shuffle(chosen)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in chosen:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_plain = sum(1 for e in chosen if e["kind"] == "plain")
    stats = {
        "examples": len(chosen),
        "plain": n_plain,
        "conditional": len(chosen) - n_plain,
        "plain_share": round(n_plain / len(chosen), 3) if chosen else 0.0,
        "plain_pool": len(plain),
        "conditional_pool": len(cond_bench),
        "synthesised_plain": len(plain) - len(plain_bench),
        "empty_targets": sum(1 for e in chosen if e["target"] in ("[]", "<t=none>")),
        "time_tokens": time_tokens,
        "clips": len(clip_ids),
    }
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--timelines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain-ratio", type=float, default=0.5,
                    help="share of training examples that are plain grounding (default 0.5)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--time-tokens", action="store_true",
                    help="emit atomic timestamp tokens instead of digit strings")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s = build(Path(a.bench), Path(a.timelines), Path(a.out), a.plain_ratio, a.seed,
              not a.no_augment, a.time_tokens)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
