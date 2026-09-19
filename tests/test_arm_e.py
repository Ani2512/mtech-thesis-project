"""Arm E retest: output-row init, the none weight, full rows, data balancing."""
import subprocess
import sys

import torch
from torch import nn


def _tiny(base_size, n_new, h):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(base_size + n_new, h)
            self.head = nn.Linear(h, base_size + n_new, bias=False)
        def get_input_embeddings(self): return self.emb
        def set_input_embeddings(self, m): self.emb = m
        def get_output_embeddings(self): return self.head
        def set_output_embeddings(self, m): self.head = m
    return Tiny()


def test_head_rows_can_start_from_their_bpe_mean_instead_of_zero(tmp_path):
    """With zeroed head rows every time symbol starts at the same logit; with the
    mean-of-BPE rows the symbols are ordered by their digits on the output side
    from step zero, and the exact rows survive a save-and-load."""
    try:
        from test_grounding import _FakeTok          # pytest puts tests/ on sys.path
    except ImportError:
        from tests.test_grounding import _FakeTok
    from ctag.timetokens import TimeVocab, load_deltas, save_deltas, wrap_new_rows

    v = TimeVocab(1.0, 0.5)
    tok = _FakeTok(v)
    base_size, n_new, h = len(tok.base), len(v.tokens), 6
    torch.manual_seed(0)
    m = _tiny(base_size, n_new, h)
    v.init_embeddings(tok, m.emb.weight)
    v.init_embeddings(tok, m.head.weight)              # what --head-init bpe does
    rows_before = m.head.weight[base_size:].detach().clone()
    assert rows_before.abs().sum() > 0
    _, head = wrap_new_rows(m, base_size, zero_head_rows=False)
    x = torch.randn(1, 2, h)
    logits = head(x)
    assert torch.allclose(logits[..., base_size:], x @ rows_before.T, atol=1e-5)   # kept, not zeroed
    with torch.no_grad():
        head.delta.normal_()
    want = head(x)
    save_deltas(m, tmp_path)
    fresh = _tiny(base_size, n_new, h)
    with torch.no_grad():
        fresh.head.weight[:base_size].copy_(head.base.weight[:base_size])
        fresh.head.weight[base_size:].normal_()          # as resize() would leave them
        fresh.emb.weight.copy_(m.emb.base.weight if hasattr(m.emb, "base") else m.emb.weight)
    assert load_deltas(fresh, tmp_path, tokenizer=tok)
    assert torch.allclose(fresh.head(x), want, atol=1e-5), "saved head rows were not restored"
    # the phase 2 default is unchanged: zero rows, zero logits at init
    z = _tiny(base_size, n_new, h)
    _, zh = wrap_new_rows(z, base_size)
    assert torch.allclose(zh(x)[..., base_size:], torch.zeros(1, 2, n_new), atol=1e-6)


def test_weighted_ce_downweights_only_the_none_token():
    from ctag.train_lora import weighted_ce
    import torch.nn.functional as F

    torch.manual_seed(1)
    V, none_id = 10, 7
    logits = torch.randn(1, 5, V)
    labels = torch.tensor([[-100, 2, none_id, 5, none_id]])
    plain = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(weighted_ce(logits, labels, none_id, 1.0), plain, atol=1e-6)
    # weight 0: the none positions vanish from the loss
    only_rest = F.cross_entropy(logits[:, :-1].reshape(-1, V)[[0, 2]], torch.tensor([2, 5]))
    assert torch.allclose(weighted_ce(logits, labels, none_id, 0.0), only_rest, atol=1e-6)
    # and in between it is a weighted mean, so it moves monotonically
    a, b, c = (weighted_ce(logits, labels, none_id, w) for w in (0.0, 0.3, 1.0))
    assert min(a, c) <= b <= max(a, c)
    # no supervised positions at all: zero, not nan
    assert weighted_ce(logits, torch.full_like(labels, -100), none_id, 0.3) == 0.0


def test_sft_caps_the_empty_share_and_the_size(tmp_path):
    from ctag.build_benchmark import build
    from ctag.sft_data import build as sft
    out = tmp_path / "proc"
    build("procedural", 12, out, seed=0)
    s0 = sft(out / "benchmark.jsonl", out / "timelines.jsonl", tmp_path / "a.jsonl", plain_ratio=0.5)
    s1 = sft(out / "benchmark.jsonl", out / "timelines.jsonl", tmp_path / "b.jsonl", plain_ratio=0.5,
             max_empty_share=0.1, max_examples=40)
    assert s0["empty_targets"] / s0["examples"] > 0.1
    assert s1["examples"] == 40 and s1["empty_targets"] / s1["examples"] <= 0.1 + 1 / 40


def test_train_lora_exposes_the_arm_e_flags():
    out = subprocess.run([sys.executable, "-m", "ctag.train_lora", "--help"], capture_output=True, text=True)
    for flag in ("--head-init", "--none-weight", "--time-rows"):
        assert flag in out.stdout


def test_runner_has_the_arm_e_block():
    src = open("nebius_phase3.py").read()
    for s in ("CTAG_ARM_E", "--head-init", "--none-weight", "--time-rows", "--max-empty-share", "sft_q_tt"):
        assert s in src


def test_full_rows_bridge_reconciles_fp32_copies_with_a_bf16_body():
    """--time-rows full: the trainable embedding and head copies are fp32, the
    frozen body is bf16 and there is no autocast. Without the bridge the first
    body matmul raises a dtype mismatch (the L40S dry run, 2026-09-19); with it
    the forward runs, the logits are fp32 and both copies receive gradients."""
    from ctag.train_lora import bridge_full_rows

    torch.manual_seed(0)
    m = _tiny(8, 2, 4)
    m.body = nn.Linear(4, 4, bias=False)
    m.to(torch.bfloat16)
    m.body.weight.requires_grad_(False)
    for p in (m.emb.weight, m.head.weight):        # what the fp32 cast does
        p.data = p.data.float()
    ids = torch.tensor([[1, 9, 3]])

    def forward():
        return m.head(m.body(m.emb(ids)))

    try:
        forward()
        assert False, "expected a dtype mismatch before the bridge"
    except RuntimeError as e:
        assert "dtype" in str(e)

    assert bridge_full_rows(m) == torch.bfloat16
    logits = forward()
    assert logits.dtype == torch.float32
    logits.float().sum().backward()
    assert m.emb.weight.grad is not None and m.emb.weight.grad.abs().sum() > 0
    assert m.head.weight.grad is not None and m.head.weight.grad.abs().sum() > 0
    assert m.body.weight.grad is None

    same = _tiny(8, 2, 4)                          # copies already match the body: no hooks
    same.body = nn.Linear(4, 4, bias=False)
    same.to(torch.bfloat16)
    same.body.weight.requires_grad_(False)
    assert bridge_full_rows(same) == torch.bfloat16
    assert not same.emb._forward_hooks and not same.head._forward_pre_hooks


def test_full_rows_adapter_forces_the_embedding_resize():
    """A full-rows adapter saved embed_tokens/lm_head at the tokenizer's length
    (151,967 rows) while the fresh thinker carries Qwen's padded 152,064; PEFT
    refuses the shape unless the matrix is resized first (L40S dry run,
    2026-09-19). A plain text adapter still leaves the padded matrix alone."""
    from ctag.models import adapter_needs_resize

    padded, with_time = 152064, 151967
    assert adapter_needs_resize(with_time, padded, has_deltas=False, full_rows=True)
    assert adapter_needs_resize(with_time, padded, has_deltas=True, full_rows=False)
    assert not adapter_needs_resize(151665, padded, has_deltas=False, full_rows=False)
    assert adapter_needs_resize(padded + 5, padded, has_deltas=False, full_rows=False)
