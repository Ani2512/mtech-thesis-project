# Kaggle run v2 — results as printed by the runner

Kernel `anirudhrangavajhala/ctag-phase2`, version 2, T4 x2, completed 2026-09-12 (~6.6 h GPU).
Extracted from the run log; the per-arm `summary.json` files did not survive Kaggle's 500-file output cap (fixed in ea20fa9).

## Step outcomes

```
--- train arm C (text timestamps): FAILED rc=1 in 176 min
--- eval arm C: FAILED rc=1 in 1 min
--- arm A (direct prompting): ok in 43 min
--- arm B (decompose and combine): ok in 13 min
--- arm A on val (for hybrid selection): ok in 27 min
--- arm B on val (for hybrid selection): ok in 8 min
--- arm D (hybrid, selection on val): ok in 0 min
--- train arm E (timestamp tokens): FAILED rc=1 in 0 min
--- eval arm E: FAILED rc=1 in 0 min
--- recall-biased decoding (k=3, 2 votes): ok in 118 min
```

Preflight: `[train] preflight OK: peak 4.3 GiB of 14.6 GiB on the longest example (675 tokens)`

Arm C trained the full epoch (176 min) and died in end-of-epoch *evaluation* (eval batch 8 vs the batch-1 preflight); adapter not saved. Arm E failed at load (base_size read from the padded embedding matrix). Both fixed in d5746f9.

## Test split, 699 queries

```
f1@0.5
arm PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL
agent 0.295 0.212 0.234 0.228 0.159 0.146 0.215 0.218
direct 0.276 0.188 0.220 0.323 0.101 0.175 0.146 0.207
hybrid 0.295 0.212 0.234 0.228 0.159 0.146 0.215 0.218
union 0.257 0.146 0.174 0.375 0.079 0.200 0.105 0.192
f_beta (recall weighted 5.6x, the measured asymmetry)
arm PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL
agent 0.291 0.212 0.227 0.227 0.159 0.148 0.217 0.217
direct 0.280 0.195 0.215 0.359 0.107 0.198 0.135 0.215
hybrid 0.291 0.212 0.227 0.227 0.159 0.148 0.217 0.217
union 0.248 0.148 0.162 0.377 0.079 0.198 0.093 0.187
count_acc
arm PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL
agent 0.490 0.781 0.607 0.607 0.798 0.396 0.396 0.567
direct 0.458 0.781 0.337 0.393 0.607 0.281 0.458 0.466
hybrid 0.490 0.781 0.607 0.607 0.798 0.396 0.396 0.564
union 0.375 0.583 0.371 0.528 0.517 0.490 0.365 0.482
```

## Vote-and-merge (k=3, 2 votes) per type — this run

```
type n union_iou f1@0.5 count_acc under_report_rate rejection_f1 parse_fail_rate
PLAIN 96 0.256 0.257 0.375 0.573 nan 0.000
ALL 699 0.203 0.192 0.482 0.522 0.479 0.000
ORDINAL 96 0.201 0.146 0.583 0.388 0.291 0.000
AFTER 89 0.203 0.174 0.371 0.603 0.421 0.000
BEFORE 89 0.350 0.375 0.528 0.371 0.519 0.000
NEXT_AFTER 89 0.091 0.079 0.517 0.492 0.394 0.000
WHILE 96 0.201 0.200 0.490 0.385 0.424 0.000
NOT_FOLLOWED 96 0.110 0.105 0.365 0.808 0.472 0.000
ABSENT 48 nan nan 0.771 nan 0.871 0.000
```

Run 1 scored 0.150 ALL on the same setting; this run 0.192. Both below direct (0.207): sampling variance at temperature 0.7, same conclusion.

## Hybrid selection on the VAL split (first time it ran with real val data)

```
type direct agent n chosen
AFTER 0.120 0.430 52 agent
BEFORE 0.330 0.418 51 agent
NEXT_AFTER 0.113 0.400 50 agent
NOT_FOLLOWED 0.125 0.144 54 agent
ORDINAL 0.250 0.268 56 agent
PLAIN 0.290 0.332 56 agent
WHILE 0.128 0.269 54 agent
```

**Correction (2026-09-13, see results_kaggle_v4.md):** this table is NOT the reported metric. The selector at the time averaged the per-row f1 over every validation row, and a rejection query (empty ground truth) scores 1.0 whenever the answer is empty. The agent answers empty about twice as often as direct prompting, so it was credited with a clean sweep. By the metric that is actually reported (f1@0.5 over non-rejection queries, exactly as in `summary.json`) the agent wins PLAIN, AFTER, NEXT_AFTER and NOT_FOLLOWED on val and loses ORDINAL, BEFORE and WHILE. The claim that "BEFORE/WHILE reversals are split noise" is withdrawn: the agent loses BEFORE and WHILE on both splits. The selector was fixed in `ctag.hybrid` and the corrected hybrid scores 0.229 on test (direct 0.207, agent 0.218).
