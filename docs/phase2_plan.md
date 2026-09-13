# Phase 2 — what to run, and why it is shaped this way

Phase 1 measured *where* the failure is. `docs/phase2_decomposition.md` then
narrowed it: with perfect events the conditions are trivial (the agent scores
1.000 on all 4404 queries), while at the model's real grounding quality even
flawless condition logic reaches only 0.263. **Grounding quality is the binding
constraint, not condition handling.** Everything below follows from that.

## Three arms

| arm | needs training | what it tests |
|---|---|---|
| **A. direct prompting** | no | the phase 1 baseline |
| **B. decompose-and-combine** | no | can perfect condition logic beat a model that has to hold the condition in its head? |
| **C. QLoRA fine-tune** | yes | does teaching grounding raise the ceiling that limits both A and B? |
| **D. hybrid** | no (selection only) | per-type choice between A and B, chosen on val |

Arm B is the one that makes arm C honest. If B matches C without any training,
the contribution is the decomposition and the paper should say so.

## The pipeline

```bash
# 1. benchmark and a leak-free split (clip-level, stable under regeneration)
python -m ctag.build_benchmark --source esc50 --n-clips 300 --p-overlap 0.45 \
       --out data/esc50 --esc50-root data/esc50_raw
python -m ctag.split --bench data/esc50/benchmark.jsonl --out data/esc50

# 2. training data, weighted towards plain grounding
python -m ctag.sft_data --bench data/esc50/benchmark_train.jsonl \
       --timelines data/esc50/timelines.jsonl --out data/esc50/sft_train.jsonl \
       --plain-ratio 0.6
python -m ctag.sft_data --bench data/esc50/benchmark_val.jsonl \
       --timelines data/esc50/timelines.jsonl --out data/esc50/sft_val.jsonl \
       --plain-ratio 0.6

# 3. arms A and B on the test split
python -m ctag.run_zeroshot --model qwen2.5-omni \
       --bench data/esc50/benchmark_test.jsonl --out runs/esc50/test_direct
python -m ctag.run_agent --grounder qwen2.5-omni \
       --bench data/esc50/benchmark_test.jsonl --out runs/esc50/test_agent

# 4. arm C
python -m ctag.train_lora --data data/esc50/sft_train.jsonl \
       --val data/esc50/sft_val.jsonl --out runs/lora_omni --epochs 2
python -m ctag.run_zeroshot --model qwen2.5-omni --adapter runs/lora_omni \
       --bench data/esc50/benchmark_test.jsonl --out runs/esc50/test_lora

# 5. arm D (no inference; selection on val, reported on test)
python -m ctag.hybrid --direct runs/esc50/qwen25_omni --agent runs/esc50/agent_omni \
       --out runs/esc50/hybrid
```

## Design choices worth defending in the write-up

**Plain ratio 0.6.** Grounding is the bottleneck, so most of the training signal
should be "where does X occur". `sft_data` synthesises extra plain examples for
every label in every training clip, which is free because the timeline already
holds the answer, plus a few absent-sound examples so an empty answer stays a
legitimate output.

**Audio encoder frozen, LoRA on the language side only.** There are roughly 210
training clips. That is not enough to retrain perception, and unfreezing the
encoder is the fastest route to overfitting the composed-audio distribution.
This is a limitation to state, not to hide: it caps how much arm C can fix the
grounding problem, which is precisely the problem that matters.

*Caveat recorded 2026-09-13:* the first adapter that trained to completion (Kaggle
v4) did **not** honour this. Its `target_modules` were bare names, which PEFT also
matched inside `audio_tower.layers.*` and `visual.blocks.*`, so 192 audio-encoder
and 192 vision-encoder LoRA tensors were trained beside the 392 language-model
ones. That adapter is scored as its own arm, **C-enc**, and `train_lora` now
targets the language model by full module path and refuses anything else. See
`results_kaggle_v4.md`.

**Clip-level split by stable hash.** Queries from one clip share audio and a
timeline, so a query-level split leaks. The hash is over (clip_id, seed) so
adding clips later never reshuffles existing assignments and a model trained
earlier is never silently evaluated on its own training clips.

**Loss on the answer only.** The collator masks prompt tokens. Training on them
teaches the model to reproduce our instruction text and no metric would reveal
it. Two tests cover the masking, including per-row masking within a batch.

**Recall over precision, if a threshold ever gets tuned.** Missing 25% of events
costs 35 f1 points; inventing 25% costs 6.

## What would make this negative, and that is fine

- If arm C does not beat arm A on plain grounding, the encoder freeze is the
  likely cause and the honest finding is that LoRA on the language side cannot
  fix an audio perception problem.
- If arm B matches arm C, the paper is about decomposition, not fine-tuning.
- If arm D beats everything, the contribution is a routing rule, which is a
  weaker but still publishable result.

Each of these is a real outcome. None requires the method to work.

## Open risks

1. Everything is composed ESC-50. Real recordings (DESED, TAG-Bench audio) are
   still untested, and the composed distribution is the one the model would
   overfit to.
2. Arm B costs two grounding calls per query. `run_agent` caches per
   (clip, sound), which is what makes it affordable, but the caching makes it a
   slightly optimistic cost estimate for a single-query deployment.
3. `train_lora` has never been executed; it is written against the documented
   API but the first GPU run should be a 20-step smoke test with `--max-steps 20`
   before committing to a full epoch.

## Memory, after the 2026-09-11 failure

The first unattended run completed the inference arms and lost both training
arms to CUDA out-of-memory on a 15 GB T4. Arm C burned 159 minutes before dying;
arm E died in one. Three causes, all now addressed:

1. **The logits, not the weights.** The failing allocation was 3.07 GiB, which is
   one `lm_head` output: 152,064 vocabulary entries by ~3,600 positions in fp16,
   plus its gradient and an fp32 softmax. `--max-seq-len` (default 3072) bounds
   it. Truncation keeps the tail, because the supervised answer is at the end.
2. **`modules_to_save` for arm E**, about 16 GB of weights, gradients and Adam
   state to move 301 rows. Replaced by a trainable delta on the new rows only —
   see `docs/timestamp_tokens.md`.
3. **fp32 Adam state.** `--optim` now defaults to `paged_adamw_8bit` when
   bitsandbytes is importable, a quarter of the optimiser memory.

And so the next failure is cheap rather than expensive, `--preflight` (on by
default) runs one forward and backward on the longest example before training
starts, prints peak memory against the card's capacity, and on OOM exits in about
a minute naming the flags to try. Two and a half hours of T4 time were spent
discovering this the other way.

**Not yet verified on a GPU.** All of the above is tested on CPU — the wrappers,
the truncation, the save/load round trip — but no fine-tuning arm has run on real
hardware yet. The preflight is what makes the first GPU minute informative.
