"""Run one backend over a benchmark -> predictions.jsonl + summary.json

    python -m ctag.run_zeroshot --model mock:first_only --bench data/proc/benchmark.jsonl --out runs/mock_first
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .metrics import parse_intervals, score_query, summarize
from .models import get_backend
from .queries import Query


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--types", default=None, help="comma-separated subset of condition types")
    ap.add_argument("--precision", default=None, choices=["fp16", "bf16", "8bit", "4bit"],
                    help="override the automatic fit (default: pick from GPU memory)")
    ap.add_argument("--max-new-tokens", type=int, default=96,
                    help="an interval list is short; 96 is ample and keeps decoding fast")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter from ctag.train_lora")
    ap.add_argument("--samples", type=int, default=1,
                    help="sample k answers and union them; raises recall, which the "
                         "measured 5.6x miss/false-alarm asymmetry makes a good trade")
    ap.add_argument("--temperature", type=float, default=0.7, help="only used when --samples > 1")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="an interval must appear in this many samples to be kept")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = {} if a.model.startswith("mock:") or a.model == "gemini" else {
        "precision": a.precision, "max_new_tokens": a.max_new_tokens}
    if a.adapter:
        assert a.model == "qwen2.5-omni", "--adapter is only wired for qwen2.5-omni"
        kw["adapter"] = a.adapter
    backend = get_backend(a.model, **kw)
    types = set(a.types.split(",")) if a.types else None
    rows, t0 = [], time.time()
    with open(a.bench, encoding="utf-8") as f, open(out / "predictions.jsonl", "w", encoding="utf-8") as fo:
        for k, line in enumerate(f):
            if a.n is not None and k >= a.n:
                break
            d = json.loads(line)
            if types and d["qtype"] not in types:
                continue
            audio, duration = d.pop("audio"), d.pop("duration")
            q = Query.from_dict(d)
            if a.samples > 1:
                raws = [backend.ground(audio, q.text, query=q, duration=duration,
                                       temperature=a.temperature) for _ in range(a.samples)]
                from .recall_bias import union_decode
                parsed = [parse_intervals(r) for r in raws]
                # all unparseable means a genuine failure; otherwise ignore the duds
                pred = None if all(p is None for p in parsed) else union_decode(parsed, a.min_votes)
                raw = json.dumps(raws)
            else:
                raw = backend.ground(audio, q.text, query=q, duration=duration)
                pred = parse_intervals(raw)
            s = score_query(pred, q.answer, q.expects_empty)
            row = {"qid": q.qid, "qtype": q.qtype, "text": q.text, "answer": q.answer, "raw": raw, "pred": pred, **s}
            rows.append(row)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 10 == 0:
                el = time.time() - t0
                print(f"{k + 1} queries | {el:.0f}s | {el / (k + 1):.1f}s per query", flush=True)
    summary = {"model": backend.name, "bench": a.bench, "by_type": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_table(summary["by_type"])


def _print_table(by_type):
    cols = ["n", "union_iou", "f1@0.5", "count_acc", "under_report_rate", "rejection_f1", "parse_fail_rate"]
    print(f"{'type':<14}" + "".join(f"{c:>18}" for c in cols))
    for t, m in by_type.items():
        print(f"{t:<14}" + "".join(f"{(m[c] if m[c] is not None else float('nan')):>18.3f}" if c != "n" else f"{m[c]:>18d}" for c in cols))


if __name__ == "__main__":
    main()
