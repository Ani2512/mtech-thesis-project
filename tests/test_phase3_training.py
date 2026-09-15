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
