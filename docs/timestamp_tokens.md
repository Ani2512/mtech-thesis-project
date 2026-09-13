# Atomic timestamp tokens (arm E)

## The problem, measured

Qwen2.5-Omni's tokenizer splits a timestamp into one token per character:

```
'16.76'                          ->  5 tokens: '1' '6' '.' '7' '6'
'[[0.65, 3.15], [8.87, 11.37]]'  -> 25 tokens
```

Three consequences:

1. **Emitting a time is a five-step sequence.** Any single slip moves the answer
   by seconds. Qwen2-Audio's predicted intervals were about five times too short
   and clustered at the end of the clip, which looks more like a number-generation
   failure than a hearing one.
2. **Nearness is not representable.** 16.76 and 16.8 are adjacent in time and
   share almost no token structure, so the output space cannot express that a
   near miss is nearly right.
3. **Relational conditions become digit arithmetic.** "After the horn" requires
   comparing two digit strings the model itself produced mid-generation. This is
   consistent with our finding that decomposition wins precisely on the
   relational types, where the comparison is done in code instead.

## Prior work

**TEMPO** (arXiv:2608.29999) adds roughly 601 tokens covering `t ∈ {0.0, 0.1,
…, 60.0}` at 0.1 s resolution, making each timestamp "a single categorical
decision over approximately 600 candidates". New embeddings are initialised as
"the mean of the BPE decomposition of the corresponding numeric value". It adds
a distance-aware Gaussian auxiliary loss:

```
q_k ∝ exp(-(t_k - t*)² / (2σ²)),  σ = 0.3 s
L_time = -Σ_k q_k log p_k
L = L_CE + λ · L_time,            λ = 0.5
```

**TimeAudio** (arXiv:2511.11039) instead uses M=20 anchor/offset tokens written
`<a2><f5>`, anchors initialised from the numeral embedding and offsets from the
mean of the numeral and decimal-point embeddings. Its ablation credits the
markers alone with **+3.0 mIoU** on temporal grounding, +3.6 Eb-F1 on dense
captioning.

**SpotSound** (arXiv:2604.13023) and the Audio-Side Time Prompt work
(arXiv:2604.13715) attack the same representation problem from the encoder side.

## What we implement

TEMPO's flat scheme, in `ctag/timetokens.py`. With 20-second clips the
vocabulary is small, and the anchor/offset scheme's advantage only appears for
long-form audio where a flat vocabulary grows linearly with duration. That is
the right swap if this is extended to hour-long recordings.

- `TimeVocab(max_seconds=30.0, resolution=0.1)` → 301 time tokens plus a
  dedicated `<t=none>` so "nothing here" is also one categorical decision rather
  than a punctuation pattern.
- `encode` / `decode` between interval lists and token strings.
- `soft_labels` implements the Gaussian target above.
- `init_embeddings` implements TEMPO's mean-of-BPE-pieces initialisation.

Quantisation rounds half **up** with a 1e-6 tolerance rather than using
`round()`, because Python rounds halves to even (0.65 would go down to 0.6) and
`3.15 / 0.1` is `31.4999…` in binary, which makes plain rounding unpredictable
near midpoints. The residual error is at most 0.05 s, far below anything the
IoU metric resolves on ~2.5 s events.

The new embeddings are new rows, not low-rank updates to existing weights, so a
plain adapter would leave them at initialisation and the scheme would be inert.
They must genuinely train.

The obvious way to do that, LoRA's `modules_to_save=["embed_tokens", "lm_head"]`,
does not fit on the hardware. It unfreezes both matrices in full: for
Qwen2.5-Omni that is 152,064 x 3,584 twice, about 2.0 GB of fp32 weights, 2.0 GB
of gradients and 4.1 GB of Adam state each — roughly **16 GB before a single
activation**. On a 15 GB T4 the backward pass died in under a minute, which is
exactly how arm E failed on 2026-09-11.

Only the 301 timestamp rows need to move, about 4 MB. `timetokens.wrap_new_rows`
keeps both base matrices frozen and puts a small trainable delta on the tail: the
input embedding is `base(ids) + delta[ids - base_size]` for the new ids, so
TEMPO's mean-of-BPE initialisation is preserved and learned on top of, and the
output head supplies the new-token logits from the delta alone, with the base
rows zeroed so nothing has to fight a random initialisation. Same computation,
about 1/500th of the optimiser cost.

PEFT does not know about those deltas, so they are written beside the adapter as
`time_deltas.pt` and re-applied at load time. If that file is missing, the
backend says so loudly rather than silently scoring a model whose timestamp
tokens never moved.

## Running it

```bash
python -m ctag.sft_data --bench data/esc50/benchmark_train.jsonl \
       --timelines data/esc50/timelines.jsonl --out data/esc50/sft_train_tt.jsonl \
       --plain-ratio 0.6 --time-tokens

python -m ctag.train_lora --data data/esc50/sft_train_tt.jsonl \
       --val data/esc50/sft_val_tt.jsonl --out runs/lora_omni_tt \
       --time-tokens --time-sigma 0.3 --time-lambda 0.5 --epochs 2

python -m ctag.run_zeroshot --model qwen2.5-omni --adapter runs/lora_omni_tt \
       --bench data/esc50/benchmark_test.jsonl --out runs/esc50/test_lora_tt
```

`parse_intervals` reads timestamp tokens directly, so both output formats are
scored through the same path and the arms stay comparable. At inference the
processor is loaded from the adapter directory, because that is where the added
tokens live, and the base embeddings are resized to match before the adapter
attaches.

## Why this is a clean ablation

Only the output representation changes. Same data, same clips, same split, same
loss on everything except the timestamp positions. So a difference between arm C
and arm E is attributable to the representation rather than to anything else.

## What would falsify it

If arm E does not beat arm C, the likely reading is that grounding failure is
upstream of the output format: the model does not know when the sound happened,
so giving it a cleaner way to say so changes nothing. That would be consistent
with our decomposition result, where even perfect condition logic could not
exceed 0.263 at the model's real grounding quality. It is a reportable outcome,
not a failed experiment.
