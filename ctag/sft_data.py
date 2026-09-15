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
from pathlib import Path

from .models import SYSTEM, prompt_for
from .transcribe import target_timeline, transcribe_query


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


def transcribe_example(audio: str, events, duration: float | None, vocab: list[str] | None) -> dict:
    """One example per clip: the whole timeline as the answer. Same SYSTEM and
    prompt_for as inference, so the trained prompt is byte-identical to the one
    ctag.run_transcribe sends."""
    return {
        "audio": audio,
        "kind": "transcribe",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_for(transcribe_query(vocab), duration)},
        ],
        "target": target_timeline(events),
    }


def build_transcribe(timelines: Path, out: Path, bench: Path | None = None, seed: int = 0,
                     vocab_in_prompt: bool = True, vocab: list[str] | None = None) -> dict:
    """Timeline-transcription training set. With --bench, only the clips of that
    split are used, which is how the train/val/test boundary is respected."""
    rng = random.Random(seed)
    keep = None
    if bench is not None:
        keep = {json.loads(l)["clip_id"] for l in open(bench, encoding="utf-8")}
    tls = [json.loads(l) for l in open(timelines, encoding="utf-8")]
    # the closed name set comes from EVERY clip in the file, not just this split:
    # the prompt must name the same sounds at training and at test time
    vocab = vocab or sorted({e["label"] for d in tls for e in d["events"]})
    if keep is not None:
        tls = [d for d in tls if d["clip_id"] in keep]
    rows = [transcribe_example(d["audio"], d["events"], d.get("duration"),
                               vocab if vocab_in_prompt else None) for d in tls if d.get("audio")]
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_ev = [len(d["events"]) for d in tls]
    return {"examples": len(rows), "clips": len(tls), "task": "transcribe",
            "events_per_clip_mean": round(sum(n_ev) / len(n_ev), 2) if n_ev else 0.0,
            "empty_targets": sum(1 for e in rows if e["target"] == "[]"),
            "vocab": vocab, "vocab_in_prompt": vocab_in_prompt}


def build(bench: Path, timelines: Path, out: Path, plain_ratio: float = 0.5,
          seed: int = 0, augment_plain: bool = True, time_tokens: bool = False,
          max_empty_share: float | None = None, max_examples: int | None = None) -> dict:
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

    # Balance the empty answer. Every rejection example is one token (<t=none>
    # or []) and the cheapest thing to emit; arm E learned it first and answered
    # it on 73% of the test queries. Cap its share of the training set.
    if max_empty_share is not None:
        empties = [e for e in chosen if e["target"] in ("[]", "<t=none>")]
        rest = [e for e in chosen if e["target"] not in ("[]", "<t=none>")]
        cap = int(max_empty_share * len(rest) / max(1e-9, 1 - max_empty_share)) if max_empty_share < 1 else len(empties)
        if len(empties) > cap:
            rng.shuffle(empties)
            empties = empties[:cap]
        chosen = rest + empties
        if max_examples is not None and len(chosen) > max_examples:
            # subsample each pool in proportion, so the cap survives the subsample
            n_e = min(len(empties), int(round(max_examples * len(empties) / len(chosen))))
            chosen = rng.sample(rest, max_examples - n_e) + rng.sample(empties, n_e)
    elif max_examples is not None and len(chosen) > max_examples:
        chosen = rng.sample(chosen, max_examples)
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
    ap.add_argument("--bench", default=None, help="split to draw clips/queries from (required for --task queries)")
    ap.add_argument("--timelines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["queries", "transcribe"], default="queries",
                    help="queries: per-question targets (phase 2); transcribe: one whole-timeline target per clip")
    ap.add_argument("--no-vocab-in-prompt", action="store_true",
                    help="transcribe only: leave the sound-name list out of the prompt")
    ap.add_argument("--plain-ratio", type=float, default=0.5,
                    help="share of training examples that are plain grounding (default 0.5)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--time-tokens", action="store_true",
                    help="emit atomic timestamp tokens instead of digit strings")
    ap.add_argument("--max-empty-share", type=float, default=None,
                    help="queries only: cap the share of examples whose answer is empty (e.g. 0.15)")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="queries only: random subsample to this many examples after mixing")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.task == "transcribe":
        s = build_transcribe(Path(a.timelines), Path(a.out), Path(a.bench) if a.bench else None,
                             a.seed, not a.no_vocab_in_prompt)
        print(json.dumps(s, indent=2)); return
    if not a.bench:
        raise SystemExit("--bench is required for --task queries")
    s = build(Path(a.bench), Path(a.timelines), Path(a.out), a.plain_ratio, a.seed,
              not a.no_augment, a.time_tokens, a.max_empty_share, a.max_examples)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
