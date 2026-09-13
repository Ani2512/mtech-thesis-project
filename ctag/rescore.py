"""Re-parse and re-score a finished run from its saved raw model outputs.

Parsing is a judgement call that improves as we see what models actually emit.
Re-running a model to test a parser change would be wasteful and would also
change the thing being measured, so predictions.jsonl keeps the raw text and
this re-scores it offline.

    python -m ctag.rescore runs/esc50/qwen2_audio
    python -m ctag.rescore runs/esc50/qwen2_audio --in-place
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import parse_intervals, score_query, summarize


def rescore(run_dir: Path, in_place: bool = False) -> dict:
    rows_in = [json.loads(l) for l in open(run_dir / "predictions.jsonl", encoding="utf-8")]
    old = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    rows, recovered = [], 0
    for r in rows_in:
        gold = [tuple(a) for a in r["answer"]]
        pred = parse_intervals(r["raw"])
        if r.get("pred") is None and pred is not None:
            recovered += 1
        s = score_query(pred, gold, r["expects_empty"])
        rows.append({**r, "pred": pred, **s})

    summary = {**{k: v for k, v in old.items() if k not in ("by_type",)},
               "rescored": True, "parse_recovered": recovered,
               "by_type": summarize(rows)}
    out = run_dir if in_place else run_dir.with_name(run_dir.name + "_rescored")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    a, b = old["by_type"]["ALL"], summary["by_type"]["ALL"]
    print(f"{run_dir}: recovered {recovered} previously unparseable answers")
    print(f"  parse_fail_rate {a['parse_fail_rate']:.3f} -> {b['parse_fail_rate']:.3f}")
    for k in ("f1@0.5", "union_iou", "count_acc"):
        av, bv = a.get(k), b.get(k)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            print(f"  {k:<12} {av:.3f} -> {bv:.3f}")
    print(f"  written to {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--in-place", action="store_true")
    a = ap.parse_args(argv)
    rescore(Path(a.run_dir), a.in_place)


if __name__ == "__main__":
    main()
