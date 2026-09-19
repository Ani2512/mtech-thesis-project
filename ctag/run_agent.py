"""Score the decompose-and-combine agent on the benchmark.

    # ceiling: perfect grounding, so any error is the composition's fault
    python -m ctag.run_agent --grounder oracle --bench data/esc50/benchmark.jsonl \
           --timelines data/esc50/timelines.jsonl --out runs/esc50/agent_oracle

    # degradation curve: how good must grounding be for decomposition to work?
    python -m ctag.run_agent --grounder oracle --jitter 0.5 --drop 0.2 ... 

    # real model as the grounder (needs a GPU)
    python -m ctag.run_agent --grounder qwen2.5-omni ...

    # every query type answered in code from predicted whole-clip timelines
    # (the output of ctag.run_transcribe); no model calls at all
    python -m ctag.run_agent --grounder timeline --pred-timelines runs/esc50/transcribe/pred_timelines.jsonl \
           --bench data/esc50/benchmark_test.jsonl --out runs/esc50/test_from_timeline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .agent import ground_query, noisy_grounder, oracle_grounder
from .metrics import parse_intervals, score_query, summarize
from .queries import Query


def model_grounder(name: str, **kw):
    """Ask a real model only for plain groundings: 'every <sound>'."""
    from .models import get_backend

    backend = get_backend(name, **kw)
    cache: dict[tuple[str, str], list] = {}

    def g(audio_path: str, sound: str):
        key = (audio_path, sound)
        if key not in cache:          # the same clip is queried many times
            raw = backend.ground(audio_path, f"every {sound}", query=None, duration=None)
            cache[key] = parse_intervals(raw) or []
        return cache[key]

    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--timelines", default=None, help="required for --grounder oracle")
    ap.add_argument("--grounder", required=True,
                    help="'oracle', 'timeline' (predicted timelines, see --pred-timelines) "
                         "or a model name from ctag.models")
    ap.add_argument("--pred-timelines", default=None,
                    help="pred_timelines.jsonl from ctag.run_transcribe, for --grounder timeline")
    ap.add_argument("--jitter", type=float, default=0.0, help="+/- seconds of boundary noise")
    ap.add_argument("--drop", type=float, default=0.0, help="probability of missing an occurrence")
    ap.add_argument("--spurious", type=float, default=0.0, help="probability of a false detection")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default=None, choices=["fp16", "bf16", "8bit", "4bit"],
                    help="load plan for a model grounder (default: chosen from the card's memory)")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter from ctag.train_lora, so the fine-tuned model "
                         "does the plain groundings and the code does the composition")
    a = ap.parse_args(argv)

    if a.grounder == "oracle":
        assert a.timelines, "--timelines is required for the oracle grounder"
        tl = {}
        for line in open(a.timelines, encoding="utf-8"):
            d = json.loads(line)
            tl[d["clip_id"]] = d
        g = oracle_grounder(tl)
        label = "agent:oracle"
    elif a.grounder == "timeline":
        from .transcribe import timeline_grounder
        assert a.pred_timelines, "--pred-timelines is required for the timeline grounder"
        preds = {}
        for line in open(a.pred_timelines, encoding="utf-8"):
            d = json.loads(line)
            preds[d["clip_id"]] = d
        g = timeline_grounder(preds)
        label = "agent:timeline"
    else:
        kw = {}
        if a.precision:
            kw["precision"] = a.precision
        if a.adapter:
            if a.grounder != "qwen2.5-omni":
                raise SystemExit("--adapter is only wired for the qwen2.5-omni grounder")
            kw["adapter"] = a.adapter
        g = model_grounder(a.grounder, **kw)
        label = f"agent:{a.grounder}" + ("+lora" if a.adapter else "")

    if a.jitter or a.drop or a.spurious:
        g = noisy_grounder(g, a.jitter, a.drop, a.spurious, a.seed)
        label += f"(j={a.jitter},d={a.drop},s={a.spurious})"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with open(a.bench, encoding="utf-8") as f, open(out / "predictions.jsonl", "w", encoding="utf-8") as fo:
        for k, line in enumerate(f):
            if a.n is not None and k >= a.n:
                break
            d = json.loads(line)
            audio = d.pop("audio"); d.pop("duration", None)
            q = Query.from_dict(d)
            pred = ground_query(q, audio, g)
            s = score_query(pred, q.answer, q.expects_empty)
            row = {"qid": q.qid, "qtype": q.qtype, "text": q.text, "answer": q.answer,
                   "raw": json.dumps(pred), "pred": pred, **s}
            rows.append(row)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 500 == 0:
                print(f"{k+1} queries | {time.time()-t0:.0f}s", flush=True)

    summary = {"model": label, "bench": a.bench, "by_type": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from .run_zeroshot import _print_table
    _print_table(summary["by_type"])


if __name__ == "__main__":
    main()
