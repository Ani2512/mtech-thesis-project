# Kaggle run v6 — arm E re-evaluated, arm F (fine-tuned grounder inside the decomposition)

Kernel `anirudhrangavajhala/ctag-phase2`, version 6, T4 x2, ran 2026-09-13 22:25 → ~23:40 IST
(1.2 h wall clock), status COMPLETE, `ALL STEPS OK`. Code: this repository, `main` at bfa336d
(PR #1, arm F). Seeded from the Kaggle dataset `ctag-phase2-v5` (all three adapters and every
v5 result except `test_lora_tt`), so only the steps below actually ran. Output in
`runs_kaggle_v6_small/` (gitignored), same layout as v5.

## Step outcomes

```
--- eval arm E (fixed delta loader):                     ok in 24 min   ALL 0.194
--- arm F: direct lora_text on val:                       ok in 22 min   ALL 0.535
--- arm F: decompose with lora_text on test:              ok in 12 min   ALL 0.621
--- arm F: decompose with lora_text on val:               ok in  7 min   ALL 0.624
--- arm F: hybrid over lora_text (selection on val):      ok in  0 min   ALL 0.645
--- arm F (lora_tt): skipped, its direct eval is below 0.3
```

The loader logged `loaded timestamp deltas ... (base rows rebuilt from the tokenizer (302 rows,
old delta file))`, so the v5 adapter was evaluated against the same embedding rows it was
trained on. Decomposition is cheap at inference: 12 min for the test split against 45 min for
the direct pass, because each sub-query answer is short and greedy.

## Results (f1@0.5, test split, 502 non-rejection queries)

| arm | ALL | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED |
|---|---|---|---|---|---|---|---|---|
| A  direct, zero-shot                | 0.207 | 0.276 | 0.188 | 0.220 | 0.323 | 0.101 | 0.175 | 0.146 |
| D  hybrid A/B                       | 0.229 | 0.295 | 0.188 | 0.234 | 0.323 | 0.159 | 0.175 | 0.215 |
| C  QLoRA direct (v5)                | 0.530 | 0.763 | 0.438 | 0.780 | 0.746 | 0.587 | 0.250 | 0.127 |
| C-enc  QLoRA, encoders + LM (v4/v5) | 0.554 | 0.764 | 0.450 | 0.821 | 0.696 | 0.667 | 0.257 | 0.208 |
| E  timestamp tokens, fixed loader   | 0.194 | 0.368 | 0.125 | 0.141 | 0.490 | 0.143 | 0.041 | 0.018 |
| **F  decompose with C**             | **0.621** | 0.745 | 0.575 | 0.667 | 0.639 | 0.540 | 0.464 | 0.666 |
| **F-hybrid  C / decompose-C by type** | **0.645** | 0.763 | 0.575 | 0.780 | 0.639 | 0.587 | 0.464 | 0.666 |

F-beta (recall weighted 5.6x): F 0.620, F-hybrid 0.644. Other ALL diagnostics for F vs C:
count_acc 0.761 vs 0.690, under_report 0.271 vs 0.369, rejection_f1 0.777 vs 0.704,
false_rejection_rate 0.191 vs 0.293.

## Finding 1: decomposition on top of the fine-tuned grounder fixes exactly the types it was predicted to fix

The prediction recorded before launch was "agent-C >> direct-C on WHILE, NOT_FOLLOWED and
ORDINAL". All three hold, and the mechanism is the one diagnosed in v5:

- **NOT_FOLLOWED 0.127 → 0.666.** Direct C answered empty on 84 % of the non-rejection
  queries; with the decomposition doing the "is there a later event" check in code, the empty
  rate falls to 12 % and rejection_f1 rises from 0.43 to 0.69.
- **WHILE 0.250 → 0.464.** Empty answers on non-rejection queries fall from 40/65 to 26/65,
  and when the arm answers its f1 is 0.77 (direct C: 0.65). The remaining 40 % over-rejection
  is the intersection of two fixed-duration intervals coming out empty when the events only
  partly overlap, so it is a residue of the duration prior (see v5), not of the decomposition.
- **ORDINAL 0.438 → 0.575, and the ordinal-word profile flattens.** f1 by word, direct C →
  decompose C: first 0.72 → 0.67, second 0.41 → 0.59, third 0.07 → 0.36, fourth 0.00 → 0.60,
  last 0.60 → 0.65. Enumerating "every X" (PLAIN count_acc 0.72) and indexing in code counts
  further than the model does in one pass. "third" is still weak: it inherits the enumeration's
  own under-reporting (PLAIN under_report 0.24).

The cost is on the types direct C already handles: PLAIN −0.02, AFTER −0.11, BEFORE −0.11,
NEXT_AFTER −0.05. For those the decomposition's extra step (ground the anchor, then filter)
adds a failure point the single pass does not have; false_rejection on AFTER goes from 0.05 to
0.21.

## Finding 2: the per-type selector transfers, 6 of 7 choices

Selection on the validation split (400 queries) chose direct for PLAIN, AFTER and NEXT_AFTER
and decompose for ORDINAL, BEFORE, WHILE and NOT_FOLLOWED. On test every choice is right except
BEFORE (val: decompose 0.715 vs direct 0.695; test: 0.639 vs 0.746). With the oracle choice on
BEFORE the hybrid would score about 0.66, so the selector costs ~0.013 ALL. This is the same
6-of-7 transfer seen for the zero-shot hybrid in v4; BEFORE is the type that flips in both,
at n≈60 non-rejection queries per split.

## Finding 3: timestamp tokens (arm E) are a negative result as trained here

With the correct base rows the arm produces well-formed output (`<t=0.5><t=3.0><t=3.3><t=5.9>`,
parse_fail 0), so the token machinery works. It still scores 0.194, below zero-shot direct
prompting, for two reasons visible in the predictions:

1. **Over-rejection.** `<t=none>` is emitted on 513 of 699 queries, against 197 that expect it.
   On non-rejection queries the empty rate is 29 % for PLAIN and 54–95 % for the relational
   types (direct C: 5–29 %, except WHILE/NOT_FOLLOWED). The single `<t=none>` token is the
   cheapest sequence to emit and one epoch with a 0.5-weighted Gaussian time loss was not enough
   to move the model off it.
2. **Worse localisation when it does answer.** PLAIN f1 on answered queries is 0.52 against
   0.83 for the text-timestamp arm; BEFORE, where it answers most, reaches 0.72.

What can and cannot be concluded: the comparison is one epoch, 4-bit, with the new rows
trained through a delta on top of a frozen mean-of-BPE initialisation (the full-matrix
variant did not fit a T4, see `results_kaggle_v4.md`). Under those constraints text timestamps
win clearly. Whether atomic tokens catch up with more epochs, a larger time-loss weight, or a
full embedding update is untested and would need a paid GPU. `docs/timestamp_tokens.md` keeps
the motivation; this section is its empirical answer so far.

## What is now settled and what is next

Settled on the ESC-50 composed benchmark (test split, greedy):

| question | answer |
|---|---|
| Does LM-only QLoRA fix zero-shot grounding? | Yes, 0.207 → 0.530 in one epoch of 2,300 examples. |
| Is the encoder the bottleneck? | No: unfreezing it is worth ~0.02. |
| Does decomposition still help once the grounder is fine-tuned? | Yes, on the relational and counting types, 0.530 → 0.621, hybrid 0.645. |
| Are atomic timestamp tokens better than text? | Not as trained here: 0.194. |
| Do recall-biased decoding / other open models help? | No (union 0.15–0.19; AF3 0.015). |

Open, and needing the real-recording track before any of it is written up:

- every LoRA arm has duration_ratio_median exactly 1.000 (the fixed-duration prior);
- BEFORE/WHILE per-type numbers move by ±0.1 between splits at n≈60; the 1,500-clip rebuild
  in `six_month_plan.md` is what settles them;
- WHILE's residual over-rejection is an intersection-of-fixed-durations artefact and will
  either vanish or dominate on real audio.

GPU budget: Kaggle's weekly allowance resets 2026-09-19 00:00 UTC; roughly 4–5 h remain
before then. The next experiments (rebuild at scale, GRPO with the temporal reward, arm E with
more epochs) are the ones that were listed as the trigger for a paid GPU.
