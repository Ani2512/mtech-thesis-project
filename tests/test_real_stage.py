"""The real-data training stage: the mixer and the nebius_real.py runner."""
import json
import os
import re
import subprocess
import sys

import pytest


def test_sft_mix_repeats_tags_and_shuffles(tmp_path):
    from ctag.sft_mix import mix, parse_part

    a, b = tmp_path / "real.jsonl", tmp_path / "synth.jsonl"
    a.write_text("".join(json.dumps({"audio": f"r{i}.wav", "target": "[]" if i == 0 else "[[1,2]]"}) + "\n" for i in range(3)))
    b.write_text("".join(json.dumps({"audio": f"s{i}.wav", "target": "[[0,1]]"}) + "\n" for i in range(5)))
    assert parse_part(f"{a}:4") == (a, 4) and parse_part(str(b)) == (b, 1)
    with pytest.raises(ValueError):
        parse_part(f"{a}:0")
    out = tmp_path / "mix.jsonl"
    s = mix([(a, 4), (b, 1)], out, seed=1)
    rows = [json.loads(l) for l in open(out)]
    assert s["examples"] == len(rows) == 3 * 4 + 5
    assert s["by_source"] == {"real": 12, "synth": 5} and s["empty_targets"] == 4
    assert rows != sorted(rows, key=lambda r: r["source"])        # shuffled, not blocked
    with pytest.raises(FileNotFoundError):
        mix([(tmp_path / "nope.jsonl", 1)], out)


def _dry(env_extra=None):
    env = {**os.environ, "CTAG_DRY_RUN": "1", **(env_extra or {})}
    out = subprocess.run([sys.executable, "nebius_real.py"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return [json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ")]


def test_every_real_runner_command_is_accepted_by_its_module():
    """Same guard as for nebius_phase3.py: every flag and choice value the
    runner passes must exist in the module it calls, checked on CPU."""
    cmds = _dry()
    assert len(cmds) >= 14, cmds
    helps = {}
    for cmd in cmds:
        module = cmd[2]
        if module not in helps:
            h = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
            assert h.returncode == 0, (module, h.stderr[-500:])
            helps[module] = h.stdout
        text = helps[module]
        for f in [t for t in cmd[3:] if t.startswith("--")]:
            assert re.search(rf"(^|\s){re.escape(f)}(\s|,|$)", text), f"{module} does not accept {f}"
        for f, v in zip(cmd[3:], cmd[4:]):
            if f.startswith("--") and not v.startswith("--"):
                m = re.search(rf"{re.escape(f)} \{{([^}}]+)\}}", text)
                if m:
                    assert v in m.group(1).split(","), f"{module} {f}: {v!r} not in {{{m.group(1)}}}"
    assert {"ctag.split", "ctag.gen_train", "ctag.sft_data", "ctag.sft_mix", "ctag.train_lora",
            "ctag.run_transcribe", "ctag.run_agent", "ctag.trailing"} <= {c[2] for c in cmds}


def test_real_runner_trains_open_vocab_on_the_mix_with_checkpoints_and_rank_128():
    cmds = _dry({"CTAG_SAVE_STEPS": "77", "CTAG_REAL_REPEAT": "5"})
    trains = [c for c in cmds if c[2] == "ctag.train_lora"]
    assert len(trains) == 2                                   # smoke + the run
    for c in trains:
        assert c[c.index("--save-steps") + 1] == "77"
        assert c[c.index("--lora-r") + 1] == "128" and c[c.index("--lora-alpha") + 1] == "256"
        assert "--train-encoder" in c and c[c.index("--data") + 1].endswith("sft_mix.jsonl")
    mixes = [c for c in cmds if c[2] == "ctag.sft_mix"]
    assert len(mixes) == 1 and any(p.endswith(":5") for p in mixes[0])
    # the open prompt everywhere: every transcription SFT file and every transcription run
    for c in cmds:
        if c[2] == "ctag.sft_data" and "transcribe" in c:
            assert "--no-vocab-in-prompt" in c, c
        if c[2] == "ctag.run_transcribe":
            assert "--no-vocab-in-prompt" in c and "--stop-probs" in c, c
    # no real clip may be used both for training and for the public test: the
    # split happens on the validation benchmark, the test is the public one
    split = next(c for c in cmds if c[2] == "ctag.split")
    assert "validation" in split[split.index("--bench") + 1]
    tests_ = [c[c.index("--bench") + 1] for c in cmds if c[2] == "ctag.run_transcribe"]
    assert any("public" in b for b in tests_) and any("benchmark_val" in b for b in tests_)
    assert any("esc50" in b for b in tests_)                   # the forgetting check


def test_real_runner_real_only_mode_has_no_generator_step():
    cmds = _dry({"CTAG_SYNTH": "0"})
    assert not any(c[2] == "ctag.gen_train" for c in cmds)
    mixes = [c for c in cmds if c[2] == "ctag.sft_mix"]
    assert len(mixes) == 1 and sum(1 for p in mixes[0] if p == "--part") == 1
    src = open("nebius_real.py").read()
    assert 'adapter_config.json")' not in src and "train_done.json" in src
    for knob in ("CTAG_REAL_DIR", "CTAG_REAL_REPEAT", "CTAG_SYNTH", "CTAG_GEN_DIR", "CTAG_ADAPTER", "CTAG_SMOKE",
                 "CTAG_SAVE_STEPS", "CTAG_DRY_RUN"):
        assert knob in src
