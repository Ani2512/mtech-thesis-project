"""Stop probabilities: how close the model came to closing the list before each event.

The timeline target is a JSON list. In Qwen's vocabulary the boundary after an
event is a single token: `},` to continue, `}]` to stop; before the first event
the choice is `{"` (start) against `]` (empty list). At each of those decisions
the probability mass the model put on stopping is a confidence for the event
that follows. The phase 3 error analysis found that 6 of 7 spurious events were
the last one emitted, so ctag.trailing tests a trailing event with this number.

    decision_steps(step_texts)         where each event was committed to
    stop_probs(tokenizer, ids, scores) P(stop) at each of those steps

Tokenisation is not assumed: a decision step is found in the decoded text and
the stop alternative is "the text of that token before the committing
character, then ']'", so `},`/`}]` and `,`/`]` both work.
"""
from __future__ import annotations

_TEXT_CACHE: dict[int, list[str]] = {}
_CLOSE_CACHE: dict[tuple[int, str], list[int]] = {}


def token_texts(tokenizer) -> list[str]:
    key = id(tokenizer)
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    return _TEXT_CACHE[key]


def close_ids(tokenizer, alt_prefix: str) -> list[int]:
    """Vocabulary ids whose text is `alt_prefix` followed by ']' (whitespace
    allowed in between). These are the tokens the model could have emitted
    instead of committing to another event."""
    key = (id(tokenizer), alt_prefix)
    if key not in _CLOSE_CACHE:
        out = []
        for i, t in enumerate(token_texts(tokenizer)):
            if t.startswith(alt_prefix) and t[len(alt_prefix):].lstrip().startswith("]"):
                out.append(i)
        _CLOSE_CACHE[key] = out
    return _CLOSE_CACHE[key]


def decision_steps(step_texts: list[str]) -> list[tuple[int, str]]:
    """One (step, alt_prefix) per event in generation order: the step whose
    token holds the first '{' (event 0) or the ',' that follows the k-th '}'
    (event k >= 1). alt_prefix is that token's text before the committing
    character; the stop alternative at that step is alt_prefix + ']'."""
    out: list[tuple[int, str]] = []
    want_open, after_close = True, False
    for i, t in enumerate(step_texts):
        for j, ch in enumerate(t):
            if want_open:
                if ch == "{":
                    out.append((i, t[:j]))
                    want_open = False
            elif ch == "}":
                after_close = True
            elif after_close and ch == ",":
                out.append((i, t[:j]))
                after_close = False
            elif after_close and ch == "]":
                after_close = False
    return out


def stop_probs(tokenizer, gen_ids, scores) -> list[float]:
    """P(stop) before each emitted event. `gen_ids` are the generated token ids
    (prompt removed), `scores` the per-step score tensors from
    generate(output_scores=True) with the same length. Greedy decoding leaves
    the scores as logits, so they are softmaxed here."""
    import torch

    ids = [int(x) for x in gen_ids]
    texts = token_texts(tokenizer)
    step_texts = [texts[i] if i < len(texts) else "" for i in ids]
    out = []
    for step, prefix in decision_steps(step_texts):
        if step >= len(scores):
            out.append(None)
            continue
        p = torch.softmax(scores[step][0].float(), dim=-1)
        cid = [c for c in close_ids(tokenizer, prefix) if c < p.shape[0]]
        out.append(float(p[cid].sum()) if cid else 0.0)
    return out
