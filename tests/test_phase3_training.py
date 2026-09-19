"""Trainer options for the rented-GPU run: bf16 loading, encoder LoRA, the runner."""
import re
import os
import json
import subprocess
import sys

import pytest


def test_encoder_targets_cover_the_audio_tower_and_never_the_vision_tower():
    from ctag.train_lora import AUDIO_TARGET_MODULES, LM_AND_AUDIO_TARGET_MODULES, LM_TARGET_MODULES

    audio = ["audio_tower.layers.0.self_attn.q_proj", "audio_tower.layers.31.self_attn.out_proj",
             "audio_tower.layers.7.fc1", "audio_tower.layers.7.fc2"]
    lm = ["model.layers.0.self_attn.q_proj", "model.layers.27.mlp.down_proj"]
    never = ["visual.blocks.5.attn.qkv", "visual.blocks.5.mlp.gate_proj", "lm_head", "model.embed_tokens",
             "audio_tower.conv1", "audio_tower.layers.0.self_attn.o_proj"]
    assert all(re.fullmatch(AUDIO_TARGET_MODULES, n) for n in audio)
    assert not any(re.fullmatch(AUDIO_TARGET_MODULES, n) for n in lm + never)
    assert all(re.fullmatch(LM_AND_AUDIO_TARGET_MODULES, n) for n in audio + lm)
    assert not any(re.fullmatch(LM_AND_AUDIO_TARGET_MODULES, n) for n in never)
    # the phase 2 target is unchanged
    assert not any(re.fullmatch(LM_TARGET_MODULES, n) for n in audio)


def test_bf16_is_a_full_precision_load_plan():
    import torch

    from ctag.models import _fit_plan

    label, kw = _fit_plan(8.4, "bf16")
    assert label == "bf16" and kw["dtype"] is torch.bfloat16 and "quantization_config" not in kw
    pytest.importorskip("transformers")          # the 4-bit plan builds a BitsAndBytesConfig
    label, kw = _fit_plan(8.4, "4bit")
    assert "quantization_config" in kw


def test_train_lora_exposes_the_new_flags():
    out = subprocess.run([sys.executable, "-m", "ctag.train_lora", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--train-encoder" in out.stdout and "bf16" in out.stdout


def test_phase3_runner_compiles_and_documents_its_knobs():
    import py_compile
    py_compile.compile("nebius_phase3.py", doraise=True)
    src = open("nebius_phase3.py").read()
    for knob in ("CTAG_GEN_N", "CTAG_EPOCHS", "CTAG_PRECISION", "CTAG_TRAIN_ENC", "CTAG_SMOKE"):
        assert knob in src
    assert "event_f1_pooled" in src and "--grounder" in src and "timeline" in src


def test_every_runner_command_is_accepted_by_its_module():
    """Every flag and every choice value the Nebius runner passes must exist in
    the module it calls. An argparse error after an hour of generation is the
    kind of mistake that costs GPU money; this catches it on CPU."""
    import json
    import os
    import re

    env = {**os.environ, "CTAG_DRY_RUN": "1", "CTAG_ARM_E": "1"}
    out = subprocess.run([sys.executable, "nebius_phase3.py"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    cmds = [json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ")]
    assert len(cmds) >= 15, out.stdout
    helps = {}
    for cmd in cmds:
        module = cmd[2]
        if module not in helps:
            h = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
            assert h.returncode == 0, (module, h.stderr[-500:])
            helps[module] = h.stdout
        text = helps[module]
        flags = [t for t in cmd[3:] if t.startswith("--")]
        for f in flags:
            assert re.search(rf"(^|\s){re.escape(f)}(\s|,|$)", text), f"{module} does not accept {f}"
        # choice values: '--flag {a,b,c}' in the help text
        for f, v in zip(cmd[3:], cmd[4:]):
            if f.startswith("--") and not v.startswith("--"):
                m = re.search(rf"{re.escape(f)} \{{([^}}]+)\}}", text)
                if m:
                    assert v in m.group(1).split(","), f"{module} {f}: {v!r} not in {{{m.group(1)}}}"
    modules = {c[2] for c in cmds}
    assert {"ctag.build_benchmark", "ctag.split", "ctag.gen_train", "ctag.sft_data", "ctag.train_lora",
            "ctag.run_transcribe", "ctag.run_agent", "ctag.run_zeroshot"} <= modules


def test_warmup_is_a_step_count_not_a_ratio():
    """transformers 5.2 removed warmup_ratio; the Kaggle image only warned about
    it, a fresh install on the VM raises TypeError at TrainingArguments."""
    from ctag.train_lora import warmup_steps
    assert warmup_steps(20000, 2, 4, 3.0, -1) == round(0.03 * 2500 * 3)
    assert warmup_steps(4, 1, 1, 1.0, 2) == 1            # a 2-step smoke test still warms up
    src = open("ctag/train_lora.py").read()
    assert "warmup_ratio=" not in src


def test_runner_trains_at_rank_128_with_alpha_2r_by_default():
    import json
    import os

    env = {**os.environ, "CTAG_DRY_RUN": "1", "CTAG_ARM_E": "1"}
    env.pop("CTAG_LORA_R", None)
    out = subprocess.run([sys.executable, "nebius_phase3.py"], capture_output=True, text=True, env=env)
    cmds = [json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ")]
    trains = [c for c in cmds if c[2] == "ctag.train_lora"]
    assert len(trains) >= 4
    for c in trains:
        assert c[c.index("--lora-r") + 1] == "128" and c[c.index("--lora-alpha") + 1] == "256", c
    env["CTAG_LORA_R"] = "32"
    out = subprocess.run([sys.executable, "nebius_phase3.py"], capture_output=True, text=True, env=env)
    c = next(json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ") and "ctag.train_lora" in l)
    assert c[c.index("--lora-r") + 1] == "32" and c[c.index("--lora-alpha") + 1] == "64"


def test_latest_checkpoint_picks_the_newest_and_stops_at_the_done_marker(tmp_path):
    """Resume support for preemptible VMs: the newest checkpoint-N wins (numeric,
    not lexical: 900 < 2500), stray files and dirs are ignored, and a finished
    run (train_done.json) is never resumed."""
    from ctag.train_lora import DONE_FILE, latest_checkpoint

    assert latest_checkpoint(str(tmp_path / "missing")) is None
    out = tmp_path / "out"
    out.mkdir()
    assert latest_checkpoint(str(out)) is None
    for n in (900, 2500, 1800):
        (out / f"checkpoint-{n}").mkdir()
    (out / "checkpoint-notes.txt").write_text("")
    (out / "checkpoint-x").mkdir()
    assert latest_checkpoint(str(out)) == str(out / "checkpoint-2500")
    (out / DONE_FILE).write_text("{}")
    assert latest_checkpoint(str(out)) is None


def test_restore_deltas_puts_a_checkpoint_back_into_a_wrapped_model(tmp_path):
    """A Trainer checkpoint holds the adapter only; the timestamp deltas are
    saved beside it by a callback and must come back exactly on resume, base
    rows included. Without a file the resume must refuse, not continue inert."""
    import pytest
    import torch
    from torch import nn

    from ctag.timetokens import DELTA_FILE, restore_deltas, save_deltas, wrap_new_rows

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.head = nn.Linear(4, 10, bias=False)
        def get_input_embeddings(self): return self.emb
        def set_input_embeddings(self, m): self.emb = m
        def get_output_embeddings(self): return self.head
        def set_output_embeddings(self, m): self.head = m

    torch.manual_seed(0)
    m = Tiny()
    emb, head = wrap_new_rows(m, base_size=8)
    with torch.no_grad():
        emb.delta.fill_(0.5)
        head.delta.fill_(-0.25)
        emb.base.weight[8:].fill_(2.0)
    ck = tmp_path / "checkpoint-2"
    ck.mkdir()
    save_deltas(m, str(ck))
    assert (ck / DELTA_FILE).exists()
    with torch.no_grad():
        emb.delta.zero_()
        head.delta.zero_()
        emb.base.weight[8:].zero_()
    assert restore_deltas(m, str(ck))
    assert torch.all(emb.delta == 0.5) and torch.all(head.delta == -0.25)
    assert torch.all(emb.base.weight[8:] == 2.0)
    with pytest.raises(RuntimeError, match="missing"):
        restore_deltas(m, str(tmp_path / "checkpoint-9"))


def test_runner_checkpoints_every_train_and_skips_on_the_done_marker():
    """Every train_lora command the runner issues carries --save-steps, and a
    training counts as done on train_done.json (or the pre-2026-09-19 processor
    files), never on the adapter file the epoch-end save writes mid-run."""
    env = dict(os.environ, CTAG_DRY_RUN="1", CTAG_SAVE_STEPS="123")
    out = subprocess.run([sys.executable, "nebius_phase3.py"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    cmds = [json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ")]
    trains = [c for c in cmds if c[2] == "ctag.train_lora"]
    assert len(trains) >= 4
    for c in trains:
        assert c[c.index("--save-steps") + 1] == "123", c
    src = open("nebius_phase3.py").read()
    assert 'adapter_config.json")' not in src
    assert src.count("trained(") >= 5


def test_runner_can_run_arm_c_at_scale_alone_with_its_decomposition_eval():
    """CTAG_ARM_E_ARMS=text is the end-to-end control for the timeline result:
    the question target at the same scale, asked directly (arm C) and inside
    the decomposition (arm F). No time-symbol step may run, and the F eval must
    use the arm C adapter as the model grounder."""
    env = dict(os.environ, CTAG_DRY_RUN="1", CTAG_ARM_E="1", CTAG_ARM_E_ARMS="text")
    out = subprocess.run([sys.executable, "nebius_phase3.py"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    cmds = [json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ")]
    joined = [" ".join(c) for c in cmds]
    assert not any("--time-tokens" in j for j in joined)
    assert not any("lora_q_tt" in j for j in joined)
    trains = [c for c in cmds if c[2] == "ctag.train_lora" and "runs/lora_q_text" in c]
    assert len(trains) == 1 and "--save-steps" in trains[0]
    direct = [c for c in cmds if c[2] == "ctag.run_zeroshot" and "runs/lora_q_text" in c]
    assert len(direct) == 1
    f = [c for c in cmds if c[2] == "ctag.run_agent" and "runs/lora_q_text" in c]
    assert len(f) == 1 and f[0][f[0].index("--grounder") + 1] == "qwen2.5-omni" and "--adapter" in f[0]
    for bad in ("both", "tt", "nonsense"):
        env["CTAG_ARM_E_ARMS"] = bad
        r = subprocess.run([sys.executable, "nebius_phase3.py"], capture_output=True, text=True, env=env)
        if bad == "nonsense":
            assert r.returncode != 0 and "CTAG_ARM_E_ARMS" in r.stderr
        else:
            assert r.returncode == 0
            j = [" ".join(json.loads(l[4:])) for l in r.stdout.splitlines() if l.startswith("DRY ")]
            assert any("lora_q_tt" in x for x in j)
            assert any("test_f_text" in x for x in j) == (bad == "both")
