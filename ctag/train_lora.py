"""QLoRA fine-tuning of Qwen2.5-Omni's thinker for temporal grounding.

Only the thinker is trained: it is the audio-plus-text-to-text path, and the
talker (speech synthesis) is irrelevant here and is not loaded.

Mix rationale lives in docs/phase2_decomposition.md. Briefly: with perfect
events the conditions are trivial (agent scores 1.000), while at the model's
real grounding quality flawless condition logic still reaches only 0.263. So
grounding is the binding constraint and `ctag.sft_data --plain-ratio` should
stay high rather than drilling conditional phrasing.

    python -m ctag.train_lora --data data/esc50/sft_train.jsonl \
           --val data/esc50/sft_val.jsonl --out runs/lora_omni --epochs 2

Then evaluate with the adapter:
    python -m ctag.run_zeroshot --model qwen2.5-omni --adapter runs/lora_omni \
           --bench data/esc50/benchmark_test.jsonl --out runs/esc50/omni_lora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bf16_ok() -> bool:
    """True only where bf16 runs on the hardware, not where it is emulated.

    torch.cuda.is_bf16_supported() takes including_emulation=True by default, so
    on a T4 (sm_75, no native bf16) it returns True and selecting bf16 gives
    emulated arithmetic. That cost roughly a 5x slowdown on the first run: 16-19
    seconds per step against 3.1-3.5 with fp16. Ampere is sm_80, so require
    capability 8.0 or above and ignore the emulation answer entirely.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def load_examples(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class GroundingCollator:
    """Builds a batch and masks everything except the answer.

    Masking is done by *answer length from the end*, not by prompt length from
    the start. The prompt carries one audio placeholder token which the
    processor expands into hundreds of audio positions, so a prompt-length mask
    computed from the text covers only the first few dozen tokens and leaves the
    whole audio region as a training target. That produced NaN gradients and a
    loss collapsing to zero on the first real run, while a stub processor that
    did no expansion let the unit test pass.

    Counting the answer tokens back from the last non-padding position is
    immune to however the processor expands the prompt.
    """

    def __init__(self, processor, sr: int = 16000, max_audio_s: float = 30.0,
                 max_seq_len: int | None = 3072):
        self.p = processor
        self.sr = sr
        self.max_audio_s = max_audio_s
        # The lm_head produces one logit per vocabulary entry per position:
        # 152,064 x 4,096 positions is 1.2 GiB in fp16, and the backward pass
        # needs the gradient and an fp32 softmax on top. That single tensor is
        # what raised "tried to allocate 3.07 GiB" on a T4, not the weights.
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[dict]):
        import librosa
        import torch

        tok = self.p.tokenizer
        texts, audios, answer_lens = [], [], []
        for ex in batch:
            conv = []
            for m in ex["messages"]:
                if m["role"] == "user":
                    content = [{"type": "audio", "audio": ex["audio"]},
                               {"type": "text", "text": m["content"]}]
                else:
                    content = [{"type": "text", "text": m["content"]}]
                conv.append({"role": m["role"], "content": content})
            prompt = self.p.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            answer = ex["target"] + (tok.eos_token or "")
            texts.append(prompt + answer)
            answer_lens.append(len(tok(answer, add_special_tokens=False).input_ids))
            a, _ = librosa.load(ex["audio"], sr=self.sr)
            audios.append(a[: int(self.max_audio_s * self.sr)])

        enc = self.p(text=texts, audio=audios, sampling_rate=self.sr,
                     return_tensors="pt", padding=True)
        full_len = enc["input_ids"].shape[1]
        if self.max_seq_len and full_len > self.max_seq_len:
            # Keep the tail: the answer is at the end, and truncating the front
            # drops leading audio rather than the supervision signal. Measure
            # against the original length -- comparing against input_ids while
            # rewriting it leaves every later tensor at full width, and the
            # answer then falls outside the mask entirely.
            keep = self.max_seq_len
            for k, v in list(enc.items()):
                if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2 and v.shape[1] == full_len:
                    enc[k] = v[:, -keep:]
        input_ids = enc["input_ids"]
        labels = torch.full_like(input_ids, -100)

        pad_id = tok.pad_token_id
        attn = enc.get("attention_mask")
        for i, n_ans in enumerate(answer_lens):
            if attn is not None:
                end = int(attn[i].sum())                       # last real token
            elif pad_id is not None:
                nonpad = (input_ids[i] != pad_id).nonzero()
                end = int(nonpad[-1]) + 1 if len(nonpad) else int(input_ids.shape[1])
            else:
                end = int(input_ids.shape[1])
            start = max(0, end - n_ans)
            labels[i, start:end] = input_ids[i, start:end]
        enc["labels"] = labels
        return enc


# Only the thinker's language model: Qwen2_5OmniThinkerForConditionalGeneration
# holds it at `model.layers.N`; the encoders are `audio_tower.layers.N` and
# `visual.blocks.N`. PEFT applies re.fullmatch to the module path when
# target_modules is a string.
LM_TARGET_MODULES = (r"model\.layers\.\d+\."
                     r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
# The audio encoder (Whisper-style: q/k/v/out_proj attention, fc1/fc2 MLP), for
# --train-encoder. The vision tower stays out in every configuration: nothing
# in this task is visual and its 192 tensors were pure waste in the v4 adapter.
AUDIO_TARGET_MODULES = (r"audio_tower\.layers\.\d+\."
                        r"(self_attn\.(q|k|v|out)_proj|fc1|fc2)")
LM_AND_AUDIO_TARGET_MODULES = f"(?:{LM_TARGET_MODULES})|(?:{AUDIO_TARGET_MODULES})"


def lora_targets_by_subtree(model) -> dict[str, int]:
    """Count LoRA tensors per top-level subtree, e.g. {'model': 392}. Used to
    refuse an adapter that has leaked into the encoders."""
    from collections import Counter

    counts: Counter = Counter()
    for name, _ in model.named_parameters():
        if "lora_" in name:
            parts = name.replace("base_model.model.", "", 1).split(".")
            counts[parts[0]] += 1
    return dict(counts)


def build_model(model_id: str, precision: str | None, lora_r: int, lora_alpha: int,
                lora_dropout: float, time_tokens: bool = False, max_seconds: float = 30.0,
                resolution: float = 0.1, train_encoder: bool = False):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    from .models import _fit_plan

    label, kw = _fit_plan(8.4, precision)
    print(f"[train] loading thinker in {label}")
    # Load the thinker alone: the talker is speech synthesis and is dead weight here.
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **kw)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    vocab = None
    if time_tokens:
        from .timetokens import TimeVocab

        vocab = TimeVocab(max_seconds, resolution)
        # The base size is the TOKENIZER's length before the new tokens, not the
        # embedding matrix's row count. Qwen pads the matrix (152,064 rows for a
        # 151,665-token vocabulary), so reading the matrix gave a base larger
        # than the resized matrix and the wrapper raised "no new rows".
        base_vocab = len(processor.tokenizer)
        added = processor.tokenizer.add_tokens(vocab.tokens, special_tokens=False)
        model.resize_token_embeddings(len(processor.tokenizer))
        assert model.get_input_embeddings().weight.shape[0] == base_vocab + added, \
            "resize did not land at tokenizer length; new-row bookkeeping would be wrong"
        # TEMPO (arXiv:2608.29999): initialise each new embedding as the mean of
        # the BPE pieces of the number it stands for, so the tokens start where
        # the model already represents those digits rather than at random.
        n = vocab.init_embeddings(processor.tokenizer, model.get_input_embeddings().weight)
        print(f"[train] added {added} timestamp tokens, initialised {n} embeddings")

    if "4bit" in label or "8bit" in label:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # prepare_model_for_kbit_training does this for quantised loads; with
        # full-precision weights and gradient checkpointing the LoRA inputs
        # would otherwise carry no grad and every step would be a no-op.
        model.enable_input_require_grads()
    model.config.use_cache = False

    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        # Language side only. The audio encoder is frozen: with ~200 training clips
        # there is not enough signal to retrain perception, and unfreezing it is the
        # fastest way to overfit the composed-audio distribution.
        #
        # A regex on the full module path, not a list of names. The bare names
        # ["q_proj", ...] also match the audio tower's and the vision tower's
        # attention projections (audio_tower.layers.N.self_attn.q_proj,
        # visual.blocks.N.attn.*), and the first adapter that trained to
        # completion had 192 audio-tower and 192 visual LoRA tensors alongside
        # the 392 language-model ones -- the "frozen encoder" was not frozen.
        # That adapter is kept as its own arm (lora_text_enc) rather than
        # passed off as this one.
        target_modules=LM_AND_AUDIO_TARGET_MODULES if train_encoder else LM_TARGET_MODULES,
        # Not modules_to_save=["embed_tokens", "lm_head"]: that makes both full
        # 152,064 x 3,584 matrices trainable, about 16 GiB of weights, gradients
        # and Adam state, which is why arm E died on a T4 inside a minute. The
        # new rows are still trained in full, but only the new rows -- see
        # timetokens.wrap_new_rows.
        modules_to_save=None,
    )
    if time_tokens:
        from .timetokens import wrap_new_rows

        wrap_new_rows(model, base_vocab)

    model = get_peft_model(model, cfg)

    where = lora_targets_by_subtree(model)
    print(f"[train] LoRA tensors per subtree: {where}")
    allowed = {"model", "audio_tower"} if train_encoder else {"model"}
    leaked = {k: v for k, v in where.items() if k not in allowed}
    if leaked or "model" not in where or (train_encoder and "audio_tower" not in where):
        raise RuntimeError(f"LoRA must attach to {sorted(allowed)} only, got {where}")

    if time_tokens:
        # get_peft_model freezes everything it does not own, the deltas included.
        n_delta = 0
        for name, param in model.named_parameters():
            if name.endswith(".delta"):
                param.requires_grad_(True)
                n_delta += param.numel()
        if not n_delta:
            raise RuntimeError("timestamp deltas were not registered; they would "
                               "stay at initialisation and the scheme would be inert")
        print(f"[train] {n_delta:,} trainable timestamp parameters "
              f"({n_delta * 4 / 1024 ** 2:.1f} MB), base embeddings frozen")

    # Keep every trainable tensor in fp32. With 4-bit weights and an fp16
    # compute dtype the adapter updates underflow and the optimiser state
    # overflows, which shows up as grad_norm = nan and a loss collapsing to
    # zero rather than as an exception. T4 and P100 are Turing and Pascal, so
    # bf16 is not available as an escape.
    n_cast = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.to(torch.float32)
            n_cast += 1
    if n_cast:
        print(f"[train] cast {n_cast} trainable tensors to fp32 for stability")

    model.print_trainable_parameters()
    return model, processor, vocab


def time_loss_terms(logits, labels, time_ids, Q, ignore_index: int = -100):
    """TEMPO's distance-aware auxiliary loss, restricted to timestamp positions.

    Plain cross-entropy treats a prediction 0.1 s off as exactly as wrong as one
    10 s off, which throws away the ordinal structure that makes a timestamp
    vocabulary worth having. Instead score those positions against a Gaussian
    over neighbouring times:

        q_k  proportional to  exp(-(t_k - t*)^2 / (2 sigma^2))
        L_time = - sum_k q_k log p_k

    `Q[i]` is the precomputed soft target for the i-th time token.
    Returns (loss, n_positions); loss is 0 when the batch has no timestamps.
    """
    import torch

    # causal shift: position j predicts token j+1
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    lut = torch.full((int(logits.shape[-1]),), -1, dtype=torch.long, device=labels.device)
    lut[time_ids] = torch.arange(len(time_ids), device=labels.device)
    safe = labels.clamp_min(0)
    row = torch.where(labels == ignore_index, torch.full_like(labels, -1), lut[safe])
    mask = row >= 0
    n = int(mask.sum())
    if n == 0:
        return logits.new_zeros(()), 0

    sel = logits[mask][:, time_ids]                      # [n, T] timestamp columns only
    logp = torch.log_softmax(sel.float(), dim=-1)
    q = Q[row[mask]]                                     # [n, T]
    return -(q * logp).sum(dim=-1).mean(), n


def _make_time_trainer():
    """Built lazily so importing this module does not require transformers."""
    from transformers import Trainer

    class TimeAwareTrainer(Trainer):
        def configure_time_loss(self, tokenizer, vocab, sigma: float, lam: float):
            import torch

            from .timetokens import EMPTY_TOKEN

            ids = [tokenizer.convert_tokens_to_ids(t) for t in vocab.tokens
                   if t != EMPTY_TOKEN]
            if any(i is None or i < 0 for i in ids):
                raise ValueError("timestamp tokens are missing from the tokenizer; "
                                 "add_tokens must run before the trainer is built")
            self._time_ids = torch.tensor(ids, dtype=torch.long)
            self._Q = torch.tensor([vocab.soft_labels(t, sigma)[:-1] for t in vocab.times],
                                   dtype=torch.float)
            self._lam = float(lam)
            self._time_seen = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss
            if labels is not None and getattr(self, "_lam", 0.0) > 0:
                dev = outputs.logits.device
                if self._time_ids.device != dev:
                    self._time_ids = self._time_ids.to(dev)
                    self._Q = self._Q.to(dev)
                l_time, n = time_loss_terms(outputs.logits, labels, self._time_ids, self._Q)
                self._time_seen += n
                loss = loss + self._lam * l_time
            return (loss, outputs) if return_outputs else loss

    return TimeAwareTrainer


def _preflight(model, collate, examples, a):
    """One forward+backward on the longest example, before the real run.

    The first full attempt spent 159 minutes training arm C and then died in the
    backward pass with CUDA out of memory. The longest example is the one that
    will fail, so try it first: a failure here costs about a minute and says what
    to change.
    """
    import torch

    longest = max(examples, key=lambda e: len(e.get("target", "")) +
                  sum(len(m.get("content", "")) for m in e["messages"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        batch = collate([longest] * max(1, a.batch_size))
        batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        model.train()
        out = model(**batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as e:
        free = total = 0
        if torch.cuda.is_available():
            free, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info())
        raise SystemExit(
            f"\n*** PREFLIGHT OOM -- stopping now rather than after hours of training.\n"
            f"{e}\n"
            f"GPU has {free:.1f} GiB free of {total:.1f} GiB. Sequence length was "
            f"{batch['input_ids'].shape[1] if 'batch' in dir() else 'unknown'} tokens.\n"
            f"Try, in order:  --max-seq-len {max(512, (a.max_seq_len or 3072) // 2)}  "
            f"|  --max-seconds {a.max_seconds / 2:.0f}  |  --grad-accum {a.grad_accum * 2} "
            f"with --batch-size 1  |  --lora-r {max(8, a.lora_r // 2)}\n") from None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        _, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info())
        print(f"[train] preflight OK: peak {peak:.1f} GiB of {total:.1f} GiB "
              f"on the longest example ({batch['input_ids'].shape[1]} tokens)")
    else:
        print(f"[train] preflight OK on CPU ({batch['input_ids'].shape[1]} tokens)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--precision", default=None, choices=["fp16", "bf16", "8bit", "4bit"],
                    help="bf16 = full-precision weights, for cards with native bf16 and >= 24 GB")
    ap.add_argument("--train-encoder", action="store_true",
                    help="also attach LoRA to the audio encoder (C-enc was worth ~+0.02 on 2,300 "
                         "examples; with tens of thousands it may matter more). Never the vision tower.")
    ap.add_argument("--amp", default="none", choices=["none", "fp16", "bf16", "auto"],
                    help="mixed precision. 'none' keeps gradients in fp32, which avoids the "
                         "fp16 overflow that produces nan grad_norm without paying for "
                         "emulated bf16 on pre-Ampere cards.")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--time-tokens", action="store_true",
                    help="atomic timestamp tokens plus the distance-aware Gaussian loss")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--max-seq-len", type=int, default=3072,
                    help="cap on tokens per example; bounds the lm_head logits, "
                         "which is the allocation that OOMs on a 15 GB card. 0 disables.")
    ap.add_argument("--optim", default=None,
                    help="optimiser; defaults to paged_adamw_8bit when bitsandbytes "
                         "is available (a quarter of the Adam state), else adamw_torch")
    ap.add_argument("--preflight", action="store_true", default=True,
                    help="run one forward+backward on the longest example first and "
                         "report peak memory, so an OOM costs seconds not hours")
    ap.add_argument("--no-preflight", dest="preflight", action="store_false")
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--time-sigma", type=float, default=0.3,
                    help="TEMPO uses 0.3 s")
    ap.add_argument("--time-lambda", type=float, default=0.5,
                    help="TEMPO uses 0.5")
    a = ap.parse_args(argv)

    from transformers import Trainer, TrainingArguments

    train = load_examples(Path(a.data))
    val = load_examples(Path(a.val)) if a.val else None
    print(f"[train] {len(train)} examples" + (f", {len(val)} val" if val else ""))

    # Measured on a T4 with this model: emulated bf16 trains correctly but runs
    # ~5x slower (18.3 s/step); fp16 runs at 3.5 s/step but overflows, giving
    # nan grad_norm and a loss that collapses to zero. 'none' keeps the
    # optimiser in fp32 and avoids both.
    amp = a.amp
    if amp == "auto":
        amp = "bf16" if _bf16_ok() else "none"
    bf16 = amp == "bf16"
    fp16 = amp == "fp16"
    print(f"[train] mixed precision: {amp}"
          + ("  (fp16 overflows on this model; expect nan grad_norm)" if fp16 else "")
          + ("  (emulated on pre-Ampere cards, ~5x slower)" if bf16 and not _bf16_ok() else ""))
    model, processor, vocab = build_model(a.model_id, a.precision, a.lora_r, a.lora_alpha,
                                          a.lora_dropout, a.time_tokens, a.max_seconds,
                                          a.resolution, a.train_encoder)
    collate = GroundingCollator(processor, max_seq_len=(a.max_seq_len or None))

    optim = a.optim
    if optim is None:
        try:
            import bitsandbytes  # noqa: F401
            optim = "paged_adamw_8bit"
        except Exception:
            optim = "adamw_torch"
    print(f"[train] optimiser: {optim}"
          + ("  (8-bit state: a quarter of fp32 Adam)" if "8bit" in optim else ""))

    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        # HF's per_device_eval_batch_size defaults to 8. The preflight covers a
        # training batch of 1; an eval batch of 8 builds eight times the logits
        # and is exactly where arm C died after 176 minutes of good training.
        per_device_eval_batch_size=a.batch_size,
        # Only the loss is needed from evaluation; gathering logits for the whole
        # val set on the GPU is a second way to run out of memory.
        prediction_loss_only=True,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val else "no",
        # bf16 where the card supports it (Ampere and later); fp16 otherwise.
        # bf16 has the same range as fp32 and removes the overflow that makes
        # QLoRA produce nan gradients.
        bf16=bf16, fp16=fp16,
        gradient_checkpointing=True,
        # PEFT wraps modules the reentrant checkpointer cannot see through, which
        # silently drops gradients for the wrapped embeddings.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    if a.time_tokens:
        trainer = _make_time_trainer()(model=model, args=args, train_dataset=train,
                                       eval_dataset=val, data_collator=collate)
        trainer.configure_time_loss(processor.tokenizer, vocab, a.time_sigma, a.time_lambda)
    else:
        trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                          data_collator=collate)
    if a.preflight:
        _preflight(model, collate, train, a)

    from transformers import TrainerCallback

    class SaveBeforeEval(TrainerCallback):
        """The Trainer evaluates and only then saves at the end of an epoch. If
        evaluation fails -- as it did with an out-of-memory in prediction_step
        after 176 minutes of training -- the adapter is never written and the
        whole epoch is lost. Save first."""
        def on_epoch_end(self, args, state, control, model=None, **kw):
            model.save_pretrained(a.out)
            print(f"[train] adapter saved at end of epoch {state.epoch:.0f} -> {a.out}", flush=True)
            return control

    trainer.add_callback(SaveBeforeEval())

    try:
        result = trainer.train()
    except Exception:
        if trainer.state.global_step > 0:
            model.save_pretrained(a.out)
            print(f"[train] crashed after {trainer.state.global_step} steps; adapter saved to {a.out} "
                  "so the training is not lost", flush=True)
        raise

    # The averaged training_loss can look healthy while every step was skipped,
    # which is how a run with nan grad_norm and a loss of 0 reported PASSED.
    # Inspect the logged history instead.
    if a.time_tokens:
        from .timetokens import save_deltas

        save_deltas(model, a.out)

    history = [h for h in trainer.state.log_history if "grad_norm" in h or "loss" in h]
    nan_grads = sum(1 for h in history
                    if isinstance(h.get("grad_norm"), float) and h["grad_norm"] != h["grad_norm"])
    zero_loss = sum(1 for h in history if h.get("loss") == 0.0)
    losses = [h["loss"] for h in history if isinstance(h.get("loss"), (int, float))]

    if nan_grads or zero_loss:
        raise SystemExit(
            f"\n*** TRAINING FAILED: {nan_grads} logged steps had nan grad_norm and "
            f"{zero_loss} had loss exactly 0. Every such step was skipped, so the adapter "
            f"at {a.out} has learned nothing. Try --amp none (or --amp bf16, which is "
            f"correct but ~5x slower on pre-Ampere), or a lower --lr. ***\n")
    if len(losses) >= 2 and losses[-1] >= losses[0]:
        print(f"\n*** WARNING: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}). ***\n")
    print(f"[train] loss {losses[0]:.4f} -> {losses[-1]:.4f}" if len(losses) >= 2
          else f"[train] final training loss {result.training_loss}")

    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    print(f"[train] adapter saved to {a.out}")


if __name__ == "__main__":
    main()
