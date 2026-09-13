# Kaggle run v4 — results and what it revealed

Kernel `anirudhrangavajhala/ctag-phase2`, version 4, T4 x2, ran 2026-09-12 17:15 → 23:58 IST
(6.7 h GPU, 24,188 s), status COMPLETE. Runs branch commit ea20fa9 (the AF3 backend, 31ca5af,
landed after launch and was not in this run). Output downloaded to `runs_kaggle_v4/` (gitignored):
`results/<arm>/{summary.json,predictions.jsonl}`, `adapters/lora_text/`, and the JSON-lines log.

## Step outcomes

```
--- train arm C (text timestamps): ok in 184 min          adapter saved (SaveBeforeEval fired)
--- eval arm C: FAILED rc=1 in 1 min                       ImportError: torchao 0.10.0 (peft wants >= 0.16)
--- arm A (direct prompting): ok in 43 min                 ALL 0.207, identical to v2 (greedy)
--- arm B (decompose and combine): ok in 14 min            ALL 0.218, identical to v2
--- arm A on val: ok in 27 min
--- arm B on val: ok in 9 min
--- arm D (hybrid, selection on val): ok in 0 min          selector was wrong, see below
--- train arm E (timestamp tokens): FAILED rc=1 in 0 min   RuntimeError: indices ... same device (cpu)
--- eval arm E: FAILED rc=1 in 0 min                       no adapter to load
--- recall-biased decoding (k=3, 2 votes): ok              ALL 0.151 (v2: 0.192; run 1: 0.150)
```

Arm C's training curve (Trainer log, loss summed over the 8-step gradient accumulation, so
divide by 8 for a per-example figure): 7.28 at step 10 → 4.46 (10 % of the epoch) → 2.42 (31 %)
→ 2.03 (87 %) → 1.80 at the end; 288 optimiser steps, 2,300 examples, 1.092e4 s. End-of-epoch
`eval_loss` on the 600 validation examples: **0.296** per token. Grad norm stayed between 3.5 and
21, no NaN. Preflight: peak 4.3 GiB of 14.6 GiB on the longest example (675 tokens).

## Finding 1: the arm C adapter trained the encoders too

`adapter_model.safetensors` holds 776 LoRA tensors, 103.0 M parameters:

| subtree | LoRA tensors |
|---|---|
| `model.layers.*` (language model) | 392 |
| `audio_tower.layers.*` (audio encoder) | 192 |
| `visual.blocks.*` (vision encoder) | 192 |

`target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
is matched by PEFT against the *last component* of every module path, and both encoders use the
same projection names. So the "audio encoder frozen" statement in `phase2_plan.md` was not true of
this adapter. Its attention projections received rank-32 updates; only the encoder MLPs and
convolutions stayed frozen.

This is also why the eval crashed and the training did not: in training the thinker alone was
loaded in 4-bit and every target was a bitsandbytes layer, which PEFT's bnb dispatcher handles
first. At eval the full Omni model is loaded and some targets fall through to the generic
dispatcher, whose first check is `is_torchao_available()`, which raises on the Kaggle image's
torchao 0.10.

Decisions:
- The adapter is kept and scored as its own arm, **C-enc** (LoRA on encoders + LM). It is the
  "unfreeze the encoder" variant the six-month plan said not to spend a month on, obtained for free.
- `train_lora` now targets `model.layers.N.(self_attn.[qkvo]_proj|mlp.(gate|up|down)_proj)` by
  full path and raises if any LoRA tensor lands outside `model.*`. Arm C proper is that adapter.
- The bootstrap uninstalls torchao.

## Finding 2: the hybrid selector was using a different metric from the one reported

`ctag.hybrid` averaged the per-row `f1@0.5` over *every* validation row. `summarize()` — the
number in every `summary.json` and every table — averages over rows whose ground truth is
non-empty. For a rejection query (empty ground truth) `score_query` gives f1 = 1.0 when the
answer is empty, so the row mean rewards answering "nothing". The agent answers empty roughly
twice as often as direct prompting (under-report 0.459 vs 0.226 on val; 0.376 vs 0.177 on test).

By the row mean the agent won all seven types on val (the table in `results_kaggle_v2.md`).
By the reported metric it wins four and loses three:

```
VAL (400 queries)          f1@0.5, non-rejection      under-report        rejection_f1
type           direct   agent    n  n_rej     direct   agent          direct   agent
PLAIN           0.290   0.332   56     0      0.286   0.304             -       -
ORDINAL         0.289   0.111   56    11      0.000   0.311          0.167   0.571
AFTER           0.146   0.205   52    16      0.333   0.556          0.111   0.667
BEFORE          0.230   0.186   51    17      0.235   0.412          0.692   0.857
NEXT_AFTER      0.137   0.176   50    16      0.000   0.382          0.118   0.651
WHILE           0.197   0.043   54    19      0.086   0.886          0.000   0.441
NOT_FOLLOWED    0.087   0.112   54    11      0.581   0.488          0.300   0.316
ALL             0.204   0.175  400   117      0.226   0.459          0.336   0.581

TEST (699 queries)
type           direct   agent    n  n_rej     direct   agent          direct   agent
PLAIN           0.276   0.295   96     0      0.260   0.281             -       -
ORDINAL         0.188   0.212   96    16      0.000   0.250          0.000   0.588
AFTER           0.220   0.234   89    26      0.286   0.413          0.074   0.727
BEFORE          0.323   0.228   89    27      0.113   0.371          0.303   0.716
NEXT_AFTER      0.101   0.159   89    26      0.000   0.270          0.000   0.735
WHILE           0.175   0.146   96    31      0.062   0.723          0.000   0.484
NOT_FOLLOWED    0.146   0.215   96    23      0.479   0.397          0.558   0.372
ALL             0.207   0.218  699   197      0.177   0.376          0.288   0.599
```

What is consistent across both splits: decomposition wins PLAIN, AFTER, NEXT_AFTER and
NOT_FOLLOWED and **loses BEFORE and WHILE**; ORDINAL flips (direct on val, agent on test).
The earlier reading that "BEFORE/WHILE reversals are split noise" is withdrawn.

WHILE is the clearest mechanism: the agent grounds both sounds and intersects the intervals.
With predicted durations about half the true ones (`duration_ratio_median` 0.53) two correct
but short intervals often fail to overlap, so the intersection is empty — under-report 0.72 on
test, 0.89 on val. The same short-interval bias that caps IoU on PLAIN kills intersection-based
composition outright. That is a concrete argument for recall-biased (longer, more) intervals in
the grounder, which the F-beta objective already encodes.

Second thing the table shows: the agent is far better at *rejection* (rejection_f1 0.599 vs
0.288 on test). Direct prompting almost never returns an empty answer for ORDINAL, AFTER,
NEXT_AFTER or WHILE even when nothing qualifies. Composition makes "no such event" a computed
outcome rather than a generated one.

**Corrected hybrid** (selector fixed to use `summarize()`'s per-type f1; regression test added):
chosen on val: agent for PLAIN, AFTER, NEXT_AFTER, NOT_FOLLOWED; direct for ORDINAL, BEFORE, WHILE.

```
TEST split, 699 queries (f1@0.5)
arm        PLAIN  ORDINAL  AFTER  BEFORE  NEXT_AFTER  WHILE  NOT_FOLLOWED   ALL
direct     0.276    0.188  0.220   0.323       0.101  0.175         0.146  0.207
agent      0.295    0.212  0.234   0.228       0.159  0.146         0.215  0.218
hybrid     0.295    0.188  0.234   0.323       0.159  0.175         0.215  0.229
```

Six of the seven val choices transfer to test (ORDINAL is the exception, by 0.024 at n=80).
The hybrid is now a genuine +0.022 over direct and +0.011 over the agent, selected on held-out
clips. Recomputed locally from the v4 predictions; no GPU needed.

## Finding 3: arm E died on a device mismatch, not memory

`NewRowsEmbedding` created its delta parameter on the CPU while the 4-bit base embedding sat on
the GPU; the first forward failed indexing a CPU tensor with CUDA ids. The deltas are now created
on `base.weight.device`, ids are moved to the base output's device before indexing, and the head's
tail logits are moved to the base logits' device (device_map can put `lm_head` on the second T4).
The preflight caught it in the first minute, as designed. Arm E has still never trained.

## Recall-biased decoding, third run

k=3 / 2 votes: 0.151 ALL, under-report 0.558. Three runs now: 0.150, 0.192, 0.151, every one
below direct's 0.207. The correlated-error failure is settled; the mechanism stays a negative
result unless it is moved into a reward.

## v5 plan (launched 2026-09-13)

Seeded from this run's output, attached to the kernel as the dataset `ctag-phase2-v4`
(adapter as `lora_text_enc`, plus the five finished runs). The runner copies anything with a
`summary.json` or `adapter_model.safetensors` marker into place and skips it. What runs:

1. eval C-enc (v4 adapter) — ~45 min
2. train arm C, language model only — ~3 h; eval — ~45 min
3. train arm E, timestamp tokens — ~3 h; eval — ~45 min
4. hybrid with the corrected selector — instant
5. Audio Flamingo 3, arms A and B (`CTAG_EXTRA_MODELS`) — ~1 h, first time on a GPU

About 9.5 h; the weekly quota has roughly 16 h left before the 2026-09-19 reset.

Decision rule unchanged: C > A on PLAIN means the diagnosis is actionable. C-enc vs C is the
encoder-unfreezing ablation, answered without spending the month.

## Post-launch review of v5's code paths (2026-09-13, while v5 was running)

A CPU rehearsal of arm E's full mechanics on a tiny Qwen2 language model with the real
Qwen2.5-Omni tokenizer (add tokens, resize, mean-of-BPE init, row wrappers, LoRA by path regex,
checkpointed forward/backward, time loss, save, fresh reload, generate) found **one more bug,
in the inference path**: `resize_token_embeddings` fills the new rows from a fitted normal
(`mean_resizing=True`), not from the mean-of-BPE init the training used, and `models.py` then
added the trained delta on top of those random rows. Every timestamp token would have had a
different input embedding at eval than during training. **v5 runs the old code, so v5's arm E
number is invalid regardless of how training goes.** Arm C and C-enc are unaffected (no new rows).

Fix (commit after a99af79): `save_deltas` stores the base rows the delta was trained against;
`load_deltas` restores them, or for a delta file without them rebuilds the rows deterministically
from the tokenizer, or refuses. The v5 adapter (`lora_tt` + its old-format `time_deltas.pt`) is
therefore reusable: a v6 seeded with it needs only the ~45 min arm E eval. Also removed: the
pointless shrink of the embedding matrix to the tokenizer length on the arm C eval path.

Reviewed and judged sound (with the caveat that none of it has run on a GPU): the seed copy
by marker files, the C-enc eval attaching an encoder-leaked adapter with torchao absent, the
LM-only regex at PEFT load time, the arm C eval batch and save-before-eval path, and the
Audio Flamingo 3 conversation format (transformers 5.x processor, audio by path).
