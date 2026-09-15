"""Trainer options for the rented-GPU run: bf16 loading, encoder LoRA, the runner."""
import re
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
