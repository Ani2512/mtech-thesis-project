"""Per-type arm selection: direct prompting or decompose-and-combine.

docs/phase2_decomposition.md shows the two arms fail in different places.
Decomposition wins on the relational types (AFTER, NEXT_AFTER, WHILE,
NOT_FOLLOWED) and loses on ORDINAL and BEFORE, which the model already handles
and where grounding a second event only adds error.

Choosing the arm per condition type is therefore worth a few points, but only
if the choice is made on validation clips and reported on test clips. Selecting
on the test set and reporting the same numbers would be choosing the maximum of
two noisy estimates and calling it a method.

Selection uses the same number that is reported: summarize()'s per-type
f1@0.5, which averages over queries whose ground truth is non-empty. An earlier
version averaged the per-row f1 over EVERY validation row, and a rejection
query (empty ground truth) scores 1.0 whenever the answer is empty. The agent
answers empty far more often than direct prompting (under-report 0.46 vs 0.23
on val), so that mean credited it with a clean sweep of all seven types on val
while the reported metric had it losing four of them. Selecting on one metric
and reporting another is not selection, it is a different question.

Selection needs validation queries. When --direct/--agent hold only test-split
queries -- which is what happens if the arms were scored on benchmark_test.jsonl
-- there is nothing to select on, and an earlier version silently fell back to
"direct" for every type, making the hybrid a byte-identical copy of arm A. Pass
the val-split runs explicitly, or let the split be derived from runs that span
both; either way an empty validation set is now an error, not a default.

    python -m ctag.hybrid --direct runs/esc50/test_direct --agent runs/esc50/test_agent \
           --direct-val runs/esc50/val_direct --agent-val runs/esc50/val_agent \
           --out runs/esc50/hybrid
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import summarize
from .split import assign


def load(run_dir: Path) -> dict[str, dict]:
    rows = {}
    for line in open(run_dir / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["qid"]] = r
    return rows


def clip_of(qid: str) -> str:
    return qid.rsplit("_q", 1)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True, help="run dir for direct prompting (test)")
    ap.add_argument("--agent", required=True, help="run dir for decompose-and-combine (test)")
    ap.add_argument("--direct-val", help="run dir for direct prompting on the val split")
    ap.add_argument("--agent-val", help="run dir for the agent on the val split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--frac-val", type=float, default=0.15)
    ap.add_argument("--metric", default="f1@0.5")
    a = ap.parse_args(argv)

    direct, agent = load(Path(a.direct)), load(Path(a.agent))
    shared = sorted(set(direct) & set(agent))
    if not shared:
        raise SystemExit("the two runs share no query ids; score them on the same benchmark")
    print(f"{len(shared)} queries scored by both arms")

    split_of = {q: assign(clip_of(q), a.seed, a.frac_train, a.frac_val) for q in shared}
    n_split = defaultdict(int)
    for s in split_of.values():
        n_split[s] += 1
    print("queries per split:", dict(n_split))

    # choose on val only -- either from dedicated val runs, or from the val
    # portion of the runs we were given
    if bool(a.direct_val) != bool(a.agent_val):
        raise SystemExit("pass both --direct-val and --agent-val, or neither")
    if a.direct_val:
        dv, gv = load(Path(a.direct_val)), load(Path(a.agent_val))
        val_shared = sorted(set(dv) & set(gv))
        if not val_shared:
            raise SystemExit("the two validation runs share no query ids")
        leak = set(val_shared) & set(shared)
        if leak:
            raise SystemExit(
                f"{len(leak)} query ids appear in both the validation and test runs, "
                "e.g. " + ", ".join(sorted(leak)[:3])
                + ". Selecting on queries that are also reported would be choosing "
                  "the maximum of two noisy estimates and calling it a method.")
        print(f"{len(val_shared)} validation queries from --direct-val/--agent-val")
    else:
        dv, gv = direct, agent
        val_shared = [q for q in shared if split_of[q] == "val"]

    # Score the validation rows exactly as the test rows are reported. The
    # per-type f1 is over non-rejection queries only, so an arm that answers
    # "nothing" often gets no credit for the rejection queries it happens to
    # get right -- that behaviour is reported separately as rejection_f1.
    val_rows_d = [dv[q] for q in val_shared]
    val_rows_g = [gv[q] for q in val_shared]
    sum_d = summarize(val_rows_d) if val_rows_d else {}
    sum_g = summarize(val_rows_g) if val_rows_g else {}
    val_scores: dict[str, dict[str, object]] = {}
    for t in sorted(set(sum_d) | set(sum_g)):
        if t == "ALL":
            continue
        md, mg = sum_d.get(t, {}).get(a.metric), sum_g.get(t, {}).get(a.metric)
        if md is None and mg is None:
            continue                      # e.g. ABSENT: every query is a rejection query
        val_scores[t] = {"direct": md, "agent": mg,
                         "n": sum_d.get(t, sum_g.get(t))["n"],
                         "n_scored": sum_d.get(t, sum_g.get(t))["n"]
                         - sum_d.get(t, sum_g.get(t))["n_rejection_queries"]}

    if not val_scores:
        raise SystemExit(
            "no validation queries, so there is nothing to select on -- the hybrid "
            "would just be a copy of arm A.\n"
            f"The runs given hold only: {dict(n_split)}.\n"
            "Score both arms on data/esc50/benchmark_val.jsonl and pass them as "
            "--direct-val/--agent-val.")

    choice = {}
    print(f"\narm chosen per type, on VAL ({a.metric}, non-rejection queries)")
    print(f"{'type':<14}{'direct':>10}{'agent':>10}{'n':>6}{'scored':>8}  chosen")
    for t in sorted(val_scores):
        md = val_scores[t]["direct"] or 0.0
        mg = val_scores[t]["agent"] or 0.0
        choice[t] = "agent" if mg > md else "direct"
        print(f"{t:<14}{md:>10.3f}{mg:>10.3f}{val_scores[t]['n']:>6}"
              f"{val_scores[t]['n_scored']:>8}  {choice[t]}")

    thin = [t for t in val_scores if val_scores[t]["n_scored"] < 20]
    if thin:
        print(f"[warn] fewer than 20 validation queries for: {', '.join(sorted(thin))} "
              "-- those per-type choices are close to a coin flip")

    # report on test only
    rows_hybrid, rows_direct, rows_agent = [], [], []
    for q in shared:
        if split_of[q] != "test":
            continue
        t = direct[q]["qtype"]
        src = agent if choice.get(t, "direct") == "agent" else direct
        rows_hybrid.append(src[q])
        rows_direct.append(direct[q])
        rows_agent.append(agent[q])

    if not rows_hybrid:
        raise SystemExit("no test-split queries in these runs; score more of the benchmark")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summaries = {"hybrid": summarize(rows_hybrid), "direct": summarize(rows_direct),
                 "agent": summarize(rows_agent)}
    (out / "summary.json").write_text(json.dumps(
        {"model": "hybrid", "choice": choice, "n_test": len(rows_hybrid),
         "by_type": summaries["hybrid"], "arms": {k: v for k, v in summaries.items() if k != "hybrid"}},
        indent=2), encoding="utf-8")

    if all(v == "direct" for v in choice.values()):
        print("\n[warn] validation picked 'direct' for every type, so the hybrid is "
              "identical to arm A by choice, not by accident")

    TY = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]
    print(f"\nTEST split, {len(rows_hybrid)} queries ({a.metric})")
    print(f"{'arm':<10}" + "".join(f"{t:>13}" for t in TY))
    print("-" * (10 + 13 * len(TY)))
    for name in ("direct", "agent", "hybrid"):
        by = summaries[name]
        print(f"{name:<10}" + "".join(
            (f"{by[t][a.metric]:>13.3f}" if isinstance(by.get(t, {}).get(a.metric), (int, float))
             else f"{'-':>13}") for t in TY))


if __name__ == "__main__":
    main()
