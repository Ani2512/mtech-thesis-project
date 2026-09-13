"""Atomic timestamp tokens, so a time is one categorical decision.

Qwen2.5-Omni's tokenizer splits '16.76' into five tokens, one per character:
    '16.76' -> '1' '6' '.' '7' '6'
A two-interval answer costs 25 tokens. Three problems follow. Emitting a time
is a five-step sequence where any slip moves the answer by seconds; 16.76 and
16.8 share almost no token structure, so the representation cannot express that
a near miss is nearly right; and "after the horn" becomes a comparison between
digit strings the model wrote itself.

Prior work fixes this by giving each quantised time its own token:

  TEMPO (arXiv:2608.29999) adds ~601 tokens for t in {0.0, 0.1, ..., 60.0},
  initialising each embedding as "the mean of the BPE decomposition of the
  corresponding numeric value", making each timestamp "a single categorical
  decision over approximately 600 candidates". It adds a distance-aware
  Gaussian loss (see `soft_labels`) so near misses earn partial credit.

  TimeAudio (arXiv:2511.11039) instead uses M=20 anchor/offset tokens
  (<a2><f5> style), with anchors initialised from the numeral embedding and
  offsets from the mean of the numeral and decimal-point embeddings. Its
  ablation credits the markers alone with +3.0 mIoU on temporal grounding.

We follow TEMPO's flat scheme: it is simpler, and with 20-second clips the
vocabulary is small anyway. The anchor/offset scheme is the better choice if
this is ever extended to long-form audio, where a flat vocabulary would grow
linearly with duration.
"""
from __future__ import annotations

import math
import re

EMPTY_TOKEN = "<t=none>"
_TOK_RE = re.compile(r"<t=(\d+\.\d)>")


class TimeVocab:
    """Quantised time tokens covering [0, max_seconds] at `resolution`."""

    def __init__(self, max_seconds: float = 30.0, resolution: float = 0.1):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.max_seconds = float(max_seconds)
        self.resolution = float(resolution)
        self.n_steps = int(round(self.max_seconds / self.resolution)) + 1
        self.times = [round(i * self.resolution, 10) for i in range(self.n_steps)]
        self.tokens = [self._tok(t) for t in self.times] + [EMPTY_TOKEN]

    def _tok(self, t: float) -> str:
        return f"<t={t:.1f}>"

    # ---------------------------------------------------------------- mapping
    def index(self, t: float) -> int:
        """Nearest quantised index, clamped into range.

        Deliberately not round(): Python rounds halves to even, so 0.65 would
        quantise down to 0.6 while 0.75 goes up to 0.8. Worse, t/resolution is
        not exact in binary -- 3.15/0.1 is 31.4999999999999996 -- so a plain
        round is unpredictable near a midpoint. Round half up with a tolerance
        of 1e-6 of a step, which is 1e-7 s at 0.1 s resolution and far below
        anything the metric can see.
        """
        i = math.floor(float(t) / self.resolution + 0.5 + 1e-6)
        return max(0, min(self.n_steps - 1, i))

    def quantise(self, t: float) -> float:
        return self.times[self.index(t)]

    def token(self, t: float) -> str:
        return self.tokens[self.index(t)]

    # ---------------------------------------------------------------- codec
    def encode(self, intervals) -> str:
        """An interval list becomes a flat token string: two tokens per interval.
        An empty answer is its own single token, so 'nothing here' is also one
        categorical decision rather than a punctuation pattern."""
        if not intervals:
            return EMPTY_TOKEN
        out = []
        for a, b in intervals:
            a, b = float(a), float(b)
            if b < a:
                a, b = b, a
            out.append(self.token(a))
            out.append(self.token(b))
        return "".join(out)

    def decode(self, text: str):
        """Tokens back to intervals. Returns [] for the empty token, and None if
        nothing parseable is present, matching ctag.metrics.parse_intervals."""
        if text is None:
            return None
        if EMPTY_TOKEN in text:
            return []
        vals = [float(m) for m in _TOK_RE.findall(text)]
        if not vals:
            return None
        out = []
        for i in range(0, len(vals) - 1, 2):
            a, b = vals[i], vals[i + 1]
            if b < a:
                a, b = b, a
            if b > a:
                out.append((a, b))
        return out

    # ---------------------------------------------------------------- loss
    def soft_labels(self, t: float, sigma: float = 0.3) -> list[float]:
        """TEMPO's distance-aware target: q_k proportional to
        exp(-(t_k - t*)^2 / (2 sigma^2)), normalised over the time tokens.

        Cross-entropy against a one-hot target says a prediction 0.1 s away is
        exactly as wrong as one 10 s away. This says otherwise, which is the
        whole point of an ordinal vocabulary. The empty token gets zero mass:
        it is not near any time.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        w = [math.exp(-((tk - t) ** 2) / (2 * sigma * sigma)) for tk in self.times]
        z = sum(w)
        if z <= 0:                       # target far outside the range
            q = [0.0] * self.n_steps
            q[self.index(t)] = 1.0
            return q + [0.0]
        return [x / z for x in w] + [0.0]

    # ---------------------------------------------------------------- init
    def init_embeddings(self, tokenizer, embedding_matrix):
        """TEMPO: initialise each new embedding as the mean of the BPE pieces of
        the number it represents, so the tokens start where the model already
        represents those digits rather than at random.

        Returns the number of rows written.
        """
        import torch

        n = 0
        with torch.no_grad():
            for tok, t in zip(self.tokens, self.times + [None]):
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid < 0:
                    continue
                text = f"{t:.1f}" if t is not None else "none"
                pieces = tokenizer(text, add_special_tokens=False).input_ids
                if not pieces:
                    continue
                embedding_matrix[tid] = embedding_matrix[pieces].mean(dim=0)
                n += 1
        return n


# ---------------------------------------------------------------------------
# Training only the rows that are new
# ---------------------------------------------------------------------------
#
# PEFT's modules_to_save=["embed_tokens", "lm_head"] makes the *whole* of both
# matrices trainable. For Qwen2.5-Omni that is 152,064 x 3,584 twice: about
# 2.0 GiB of fp32 weights, 2.0 GiB of gradients and 4.1 GiB of Adam state per
# matrix, so roughly 16 GiB before a single activation. On a 15 GiB T4 the
# backward pass died in under a minute.
#
# Only the timestamp rows need to move -- 301 of them at a 0.1 s resolution,
# about 4 MB. These two wrappers keep the base matrices frozen and put a small
# trainable delta on the tail, which is the same computation at 1/4000th of the
# optimiser cost.


def _new_rows_modules():
    """Imported lazily: torch is a GPU-runtime dependency, not a package one."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class NewRowsEmbedding(nn.Module):
        """Frozen base embedding, plus a trainable delta on rows >= base_size.

        The base rows already hold TEMPO's mean-of-BPE initialisation, so the
        delta starts at zero and learns on top of it rather than replacing it.
        """

        def __init__(self, base: nn.Embedding, base_size: int):
            super().__init__()
            self.base = base
            self.base_size = int(base_size)
            n_new = base.num_embeddings - self.base_size
            if n_new <= 0:
                raise ValueError(f"no new rows: {base.num_embeddings} <= {base_size}")
            # On the device the base rows already occupy. A CPU-born parameter
            # next to a GPU-resident model fails at the first forward with
            # "indices should be either on cpu or on the same device as the
            # indexed tensor", which is how arm E died on its second attempt.
            self.delta = nn.Parameter(
                torch.zeros(n_new, base.embedding_dim, dtype=torch.float32,
                            device=base.weight.device))
            for p in self.base.parameters():
                p.requires_grad_(False)

        @property
        def weight(self):                     # some callers read .weight directly
            return self.base.weight

        def forward(self, ids):
            out = self.base(ids)
            # Under device_map the base carries an accelerate hook that moves
            # `ids` to its own device; the wrapper does not, so index on the
            # device the output (and the delta) actually live on.
            ids = ids.to(out.device)
            is_new = ids >= self.base_size
            idx = (ids - self.base_size).clamp_(min=0)
            add = self.delta.to(device=out.device, dtype=out.dtype)[idx]
            return out + add * is_new.unsqueeze(-1).to(out.dtype)

    class NewRowsLinear(nn.Module):
        """Frozen base projection, with the tail logits supplied by a trainable
        delta. The base rows for the new tokens are zeroed at construction, so
        the new-token logits are learned rather than fighting a random init."""

        def __init__(self, base: nn.Linear, base_size: int):
            super().__init__()
            self.base = base
            self.base_size = int(base_size)
            n_new = base.out_features - self.base_size
            if n_new <= 0:
                raise ValueError(f"no new rows: {base.out_features} <= {base_size}")
            with torch.no_grad():
                base.weight[self.base_size:].zero_()
                if base.bias is not None:
                    base.bias[self.base_size:].zero_()
            self.delta = nn.Parameter(
                torch.zeros(n_new, base.in_features, dtype=torch.float32,
                            device=base.weight.device))
            for p in self.base.parameters():
                p.requires_grad_(False)

        def forward(self, x):
            logits = self.base(x)
            tail = F.linear(x.to(self.delta.device),
                            self.delta.to(x.dtype)).to(logits.device)
            return torch.cat([logits[..., :self.base_size],
                              logits[..., self.base_size:] + tail], dim=-1)

    return NewRowsEmbedding, NewRowsLinear


DELTA_FILE = "time_deltas.pt"


def wrap_new_rows(thinker, base_size: int):
    """Swap in the trainable-tail wrappers. Returns the two new modules."""
    NewRowsEmbedding, NewRowsLinear = _new_rows_modules()
    emb = thinker.get_input_embeddings()
    head = thinker.get_output_embeddings()
    if head is None:
        raise RuntimeError("no output embedding to wrap; cannot train timestamp tokens")
    wrapped_emb = NewRowsEmbedding(emb, base_size)
    wrapped_head = NewRowsLinear(head, base_size)
    thinker.set_input_embeddings(wrapped_emb)
    thinker.set_output_embeddings(wrapped_head)
    return wrapped_emb, wrapped_head


def save_deltas(model, out_dir):
    """Write the timestamp deltas next to the adapter. PEFT does not know about
    them, so without this the tokens stay at their initialisation and the whole
    scheme is inert -- the same failure modules_to_save was there to prevent."""
    import os

    import torch

    from torch import nn

    found = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "delta") and hasattr(mod, "base_size"):
            # keyed on what is wrapped, not on the module's name
            key = "embedding" if isinstance(mod.base, nn.Embedding) else "lm_head"
            found[key] = {"delta": mod.delta.detach().cpu(), "base_size": mod.base_size}
            # The base rows under the delta are NOT reproducible at load time:
            # resize_token_embeddings fills new rows from a fitted normal
            # (mean_resizing=True), not from TEMPO's mean-of-BPE init the
            # training used. Save the rows the delta was trained against
            # (about 4 MB) so inference adds the delta to the same base.
            found[key]["base_rows"] = mod.base.weight[mod.base_size:].detach().float().cpu()
    if not found:
        return None
    path = os.path.join(out_dir, DELTA_FILE)
    torch.save(found, path)
    print(f"[train] saved timestamp deltas for {sorted(found)} -> {path}")
    return path


def vocab_from_tokenizer(tokenizer) -> "TimeVocab":
    """Recover the TimeVocab whose tokens were added to this tokenizer."""
    import re

    times = sorted(float(m.group(1)) for tok in tokenizer.get_added_vocab()
                   for m in [re.fullmatch(r"<t=(\d+(?:\.\d+)?)>", tok)] if m)
    if len(times) < 2:
        raise ValueError("tokenizer carries no timestamp tokens to rebuild a TimeVocab from")
    res = min(b - a for a, b in zip(times, times[1:]))
    v = TimeVocab(times[-1], res)
    if set(v.tokens) != set(tokenizer.get_added_vocab()) & set(v.tokens) or len(v.times) != len(times):
        raise ValueError(f"tokenizer's timestamp tokens do not form a TimeVocab({times[-1]}, {res})")
    return v


def load_deltas(thinker, adapter_dir, tokenizer=None):
    """Re-apply saved deltas at inference. Returns True when they were found.

    The new embedding rows must be the ones the delta was trained on top of.
    Newer delta files carry them; for an older file they are rebuilt with
    TEMPO's deterministic mean-of-BPE init, which needs the tokenizer.
    """
    import os

    import torch

    path = os.path.join(adapter_dir, DELTA_FILE)
    if not os.path.exists(path):
        return False
    blob = torch.load(path, map_location="cpu")
    base_size = next(iter(blob.values()))["base_size"]
    emb, head = wrap_new_rows(thinker, base_size)
    how = "no embedding delta in the file"
    with torch.no_grad():
        if "embedding" in blob:
            e = blob["embedding"]
            if "base_rows" in e:
                emb.base.weight[base_size:].copy_(e["base_rows"].to(emb.base.weight.dtype))
                how = "saved base rows"
            elif tokenizer is not None:
                n = vocab_from_tokenizer(tokenizer).init_embeddings(tokenizer, emb.base.weight)
                if n != emb.base.weight.shape[0] - base_size:
                    raise RuntimeError(f"rebuilt {n} timestamp rows, expected "
                                       f"{emb.base.weight.shape[0] - base_size}")
                how = f"base rows rebuilt from the tokenizer ({n} rows, old delta file)"
            else:
                raise RuntimeError(
                    f"{path} has no base rows and no tokenizer was given to rebuild them; "
                    "the delta would sit on random rows and the eval would be meaningless")
            emb.delta.copy_(e["delta"].to(emb.delta.dtype))
        if "lm_head" in blob:
            head.delta.copy_(blob["lm_head"]["delta"].to(head.delta.dtype))
    print(f"[qwen2.5-omni] loaded timestamp deltas from {path} ({how})")
    return True
