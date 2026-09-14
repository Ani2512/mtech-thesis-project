# Kaggle run v5 — the first valid fine-tuning result

Kernel `anirudhrangavajhala/ctag-phase2`, version 5, T4 x2, ran 2026-09-13 09:50 → 20:58 IST
(9.4 h GPU), status COMPLETE. Code: `Ani2512/mtech-project` branch
`compositional-temporal-grounding` at 175e60e (the last commit before the repository moved
here; it corresponds to a99af79 plus the AF3 backend). Seeded from the Kaggle dataset
`ctag-phase2-v4` (the v4 adapter as `lora_text_enc`, and the five v4 runs that had finished),
so arms A, B, D and the recall-biased decoding were not rerun. Output downloaded to
`runs_kaggle_v5_small/` (gitignored): `results/<arm>/{summary.json,predictions.jsonl}`, the
adapter configs and the JSON-lines log tail. The bulk download of the checkpoints stalled on
the 600 MB `optimizer.pt` files; only `lora_text/checkpoint-288/trainer_state.json` was
recovered, so arm E has no training curve on disk.

## Step outcomes

```
--- eval arm C-enc (v4 adapter, encoders + LM): ok  ~45 min   ALL 0.554
--- train arm C (text timestamps, LM only):      ok  ~3 h      288 steps, eval_loss 0.325
--- eval arm C:                                  ok  ~45 min   ALL 0.530
--- train arm E (timestamp tokens):              ok  ~3 h      adapter + time_deltas.pt saved
--- eval arm E:                                  ok  23 min    ALL 0.003  (INVALID, loader bug, see below)
--- hybrid over A/B (corrected selector):        ok            ALL 0.229
--- arm A (audio-flamingo-3):                    ok  68 min    ALL 0.015
--- arm B (audio-flamingo-3):                    ok  18 min    ALL 0.006
```

Timings for the first four steps are approximate: the log tail returned by the API starts at
"eval arm E", and the full log is only on the Kaggle page for version 5.

Arm C's training curve (Trainer log; loss is summed over the 8-step gradient accumulation, so
divide by 8 for a per-example figure): 7.28 at step 10 → 3.83 (17 % of the epoch) → 2.57 (31 %)
→ 2.24 (87 %) → 2.00 at step 280; end-of-epoch `eval_loss` on the 600 validation examples
**0.325** per token (C-enc in v4: 0.296). Linear decay from 1e-4, 288 optimiser steps, 2,300
examples, one epoch.

## The headline: one epoch of LM-only QLoRA takes ALL from 0.207 to 0.530

f1@0.5 on the test split (699 queries, 197 of them rejection queries; f1 is over the 502
non-rejection queries, rejection behaviour is reported separately).

| arm | ALL | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED |
|---|---|---|---|---|---|---|---|---|
| A  direct, zero-shot (v2/v4)       | 0.207 | 0.276 | 0.188 | 0.220 | 0.323 | 0.101 | 0.175 | 0.146 |
| B  decompose, zero-shot (v2/v4)    | 0.218 | 0.295 | 0.212 | 0.234 | 0.228 | 0.159 | 0.146 | 0.215 |
| D  hybrid A/B, corrected selector  | 0.229 | 0.295 | 0.188 | 0.234 | 0.323 | 0.159 | 0.175 | 0.215 |
| **C  QLoRA, LM only**              | **0.530** | 0.763 | 0.438 | 0.780 | 0.746 | 0.587 | 0.250 | 0.127 |
| C-enc  QLoRA, encoders + LM (v4)   | 0.554 | 0.764 | 0.450 | 0.821 | 0.696 | 0.667 | 0.257 | 0.208 |
| E  timestamp tokens (v5 loader)    | 0.003 | — | — | — | — | — | — | — |
| AF3 direct                          | 0.015 | 0.018 | 0.029 | 0.014 | 0.019 | 0.000 | 0.019 | 0.003 |
| AF3 decompose                       | 0.006 | 0.016 | 0.013 | 0.005 | 0.003 | 0.000 | 0.000 | 0.003 |

Diagnostics for arm C (ALL): count_acc 0.690, under_report 0.369, rejection_f1 0.704
(direct zero-shot: 0.288), centre error median 0.018 s, centre within 1 s 0.847,
**duration_ratio_median exactly 1.000**.

What the per-type pattern says:

- **PLAIN, AFTER, BEFORE, NEXT_AFTER are largely solved** on this benchmark (0.59–0.78).
  Centres land within 1 s of gold on 85–93 % of answered queries.
- **WHILE (0.250) and NOT_FOLLOWED (0.127) are not.** Both fail by over-rejection: on the
  non-rejection queries the model answers empty 62 % of the time for WHILE and 84 % for
  NOT_FOLLOWED. When it does answer, its f1 is 0.65 and 0.77 respectively, so the grounding is
  fine and the decision "is there anything to report" is what fails. NOT_FOLLOWED's
  rejection_recall is 1.0 with precision 0.27: it has learned that the type is often empty.
- **ORDINAL (0.438) is a counting failure, not a grounding one.** f1 by ordinal word:
  first 0.72 (n=18), second 0.41 (22), third 0.07 (14), fourth 0.00 (5), fifth 0.00 (1),
  last 0.60 (20). The model can find the first and the last occurrence; it cannot count to
  three. PLAIN count_acc is 0.72, so "every dog" enumerates well enough that a decomposition
  which enumerates and then indexes in code should fix this. (Tested in v6 as arm F.)
- **Unfreezing the encoders buys ~2 points** (C-enc 0.554 vs C 0.530), mostly on AFTER,
  NEXT_AFTER and NOT_FOLLOWED; BEFORE goes the other way. At n≈60–90 per type this is at the
  edge of noise; the honest reading is "the encoder is not the bottleneck on ESC-50".

## Caveat that decides the next phase: the model learned the fixed-duration prior

`duration_ratio_median` is exactly 1.000 for every type in both LoRA arms. ESC-50 clips are
5 s and after silence trimming most events in the composed benchmark have the same handful of
durations, so predicting the training-set duration is optimal here and says nothing about
boundary localisation. The real-recording track must be evaluated before any boundary claim is
made; on real audio this prior will be wrong and the result may look very different.

## Arm E's 0.003 was the predicted loader bug, not the adapter

The CPU rehearsal on 2026-09-13 (before v5 finished) found that `resize_token_embeddings`
fills the new rows with fitted-normal noise, and `load_deltas` added the trained delta on top of
that instead of on the mean-of-BPE rows used in training. So the 302 timestamp tokens were
decoded against the wrong base at eval and the model emitted `<t=none>` on 99.6 % of queries.
The adapter and `time_deltas.pt` themselves are fine; the fix (`base_rows` saved with the
deltas, rebuilt from the tokenizer for old files) landed after v5 cloned, and v6 re-ran only the
eval. See `results_kaggle_v6.md`.

## Audio Flamingo 3 cannot ground on this benchmark either

First GPU run of the `audio-flamingo-3` backend (`nvidia/audio-flamingo-3-hf`, Qwen2.5-7B
base, one of TAG-Bench's 21 systems). Direct 0.015, decompose 0.006; duration_ratio_median
0.32, i.e. intervals a third of the true length, the same pattern as Qwen2-Audio in phase 1.
It joins Qwen2-Audio as a negative reference: the interval-set format with a 0.5 IoU threshold
and this prompt is beyond both. Contamination caveat as before: ESC-50 is in most audio-LLM
training mixes, which helps recognition, not localisation; the composed clips are novel.

## Hybrid A/B with the corrected selector: 0.229, as computed locally

Matches the local recomputation in `results_kaggle_v4.md` exactly (greedy decoding); nothing
new, recorded for completeness.
