"""Trailing-event check: stop probabilities from generation scores, the three
rules, and the CLI on mock predictions."""
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from ctag.stopprob import decision_steps
from ctag.trailing import align_stop_probs, apply_rule, last_emitted_index, matched_flags, should_drop

TWO = '[{"sound": "dog", "start": 0.98, "end": 3.48}, {"sound": "car horn", "start": 4.83, "end": 7.33}]'


def _qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B", local_files_only=True)
    except Exception:
        return None


def test_decision_steps_find_the_start_and_every_comma_after_a_close():
    # one character per step: the decision for event 0 is at '{', for event 1 at the ',' after '}'
    steps = list('[{"a":1},{"b":2}]')
    assert decision_steps(steps) == [(1, ""), (8, "")]
    # merged tokens: '},' carries the comma, so the alt prefix is '}'
    assert decision_steps(["[", '{"', "x", "},", ' {"', "y", "}]"]) == [(1, ""), (3, "}")]
    # commas inside an event never count; an empty list has no decisions
    assert decision_steps(['[{"s": "a", "start": 1, "end": 2}]']) == [(0, "[")]
    assert decision_steps(["[]"]) == []


def test_stop_probs_read_the_close_token_mass_at_each_decision():
    tok = _qwen_tokenizer()
    if tok is None:
        pytest.skip("Qwen tokenizer not cached")
    import torch

    from ctag.stopprob import close_ids, stop_probs

    ids = tok(TWO)["input_ids"]
    texts = [tok.decode([i]) for i in ids]
    steps = decision_steps(texts)
    assert len(steps) == 2 and texts[steps[0][0]] == '{"' and texts[steps[1][0]] == "},"
    assert steps[1][1] == "}" and tok.convert_tokens_to_ids("}]") in close_ids(tok, "}")
    V = len(tok)
    scores = [torch.full((1, V), -30.0) for _ in ids]
    for i, t in enumerate(ids):
        scores[i][0, t] = 0.0                                  # the chosen token
    # before event 1 the model put 30% on '}]'
    s1 = steps[1][0]
    scores[s1][0, tok.convert_tokens_to_ids("}]")] = float(np.log(0.3 / 0.7))
    p = stop_probs(tok, ids, scores)
    assert len(p) == 2 and p[0] < 1e-6 and abs(p[1] - 0.3) < 1e-3


def _rows():
    ev = [{"label": "dog", "onset": 1.0, "offset": 3.0}, {"label": "cat", "onset": 5.0, "offset": 7.5},
          {"label": "siren", "onset": 5.2, "offset": 7.7}]
    raw = json.dumps([{"sound": e["label"], "start": e["onset"], "end": e["offset"]} for e in ev])
    return {"clip_id": "c1", "audio": "c1.wav", "events": ev, "raw": raw, "stop_probs": [0.01, 0.02, 0.6]}


def test_last_emitted_follows_the_raw_answer_not_the_sorted_list():
    row = _rows()
    assert last_emitted_index(row["events"], row["raw"]) == 2
    # the model wrote the siren before the cat: the last emitted is the cat
    raw = json.dumps([{"sound": "dog", "start": 1.0, "end": 3.0}, {"sound": "siren", "start": 5.2, "end": 7.7},
                      {"sound": "cat", "start": 5.0, "end": 7.5}])
    assert last_emitted_index(row["events"], raw) == 1
    assert align_stop_probs(row["events"], raw, [0.01, 0.6, 0.02]) == [0.01, 0.02, 0.6]
    assert last_emitted_index(row["events"], "not json") == 2
    assert align_stop_probs(row["events"], "not json", [0.1, 0.2, 0.3]) == [None] * 3
    assert last_emitted_index([], "[]") is None


def test_rules(tmp_path):
    row = _rows()
    assert should_drop("iou", 0.5, row, 2) and not should_drop("iou", 0.95, row, 2)
    assert not should_drop("iou", 0.5, row, 0)
    assert should_drop("stop", 0.5, row, 2) and not should_drop("stop", 0.7, row, 2)
    assert not should_drop("stop", 0.5, row, 1)
    sr = 16000
    y = np.zeros(10 * sr, dtype=np.float32)
    y[sr: 3 * sr] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)
    sf.write(tmp_path / "c1.wav", y, sr)
    assert should_drop("silence", 0.3, row, 2, audio_root=str(tmp_path))      # 5.2-7.7 s is silent
    assert not should_drop("silence", 0.3, row, 0, audio_root=str(tmp_path))  # 1-3 s holds the tone
    with pytest.raises(ValueError):
        should_drop("nope", 0.5, row, 0)


def test_apply_rule_counts_what_it_removes():
    row = _rows()
    gold = {"c1": {"events": [{"label": "dog", "onset": 1.0, "offset": 3.0}, {"label": "cat", "onset": 5.0, "offset": 7.5}]}}
    pred3 = [(e["label"], e["onset"], e["offset"]) for e in row["events"]]
    assert matched_flags(pred3, [("dog", 1.0, 3.0), ("cat", 5.0, 7.5)], 0.5) == [True, True, False]
    new, conf = apply_rule([row], "stop", 0.5, gold)
    assert conf == {"fired": 1, "removed_spurious": 1, "removed_correct": 0, "spurious_last_events": 1}
    assert len(new[0]["events"]) == 2 and new[0]["f1"] == 1.0 and new[0]["trailing_dropped"]
    # a correct last event removed is counted as such
    gold2 = {"c1": {"events": [{"label": e["label"], "onset": e["onset"], "offset": e["offset"]} for e in row["events"]]}}
    new, conf = apply_rule([row], "stop", 0.5, gold2)
    assert conf["removed_correct"] == 1 and conf["removed_spurious"] == 0 and new[0]["recall"] < 1.0


def test_cli_mock_stop_probs_end_to_end(tmp_path):
    from ctag.build_benchmark import main as build
    from ctag.run_transcribe import main as transcribe

    build(["--source", "procedural", "--n-clips", "12", "--out", str(tmp_path / "b")])
    out = tmp_path / "t"
    transcribe(["--model", "mock:oracle", "--bench", str(tmp_path / "b" / "benchmark.jsonl"),
                "--timelines", str(tmp_path / "b" / "timelines.jsonl"), "--out", str(out),
                "--spurious", "0.5", "--stop-probs", "--seed", "1"])
    rows = [json.loads(l) for l in open(out / "pred_timelines.jsonl")]
    assert all(len(r["stop_probs"]) == len(r["events"]) for r in rows)
    assert any(r["n_pred"] > r["n_gold"] for r in rows)
    r = subprocess.run([sys.executable, "-m", "ctag.trailing", "--pred-timelines", str(out / "pred_timelines.jsonl"),
                        "--timelines", str(tmp_path / "b" / "timelines.jsonl"), "--rule", "stop", "--thr", "0.5",
                        "--out", str(tmp_path / "f")], capture_output=True, text=True, check=True)
    s = json.loads((tmp_path / "f" / "summary.json").read_text())
    # the invented event lands anywhere in the sorted list, so only the clips
    # where it is last are fixed; nothing correct may be removed
    assert s["confusion"]["removed_correct"] == 0 and s["confusion"]["removed_spurious"] > 0
    assert s["after"]["event_f1_pooled"] > s["before"]["event_f1_pooled"], r.stdout
    r = subprocess.run([sys.executable, "-m", "ctag.trailing", "--pred-timelines", str(out / "pred_timelines.jsonl"),
                        "--timelines", str(tmp_path / "b" / "timelines.jsonl"), "--rule", "iou", "--sweep", "0.3", "0.9"],
                       capture_output=True, text=True, check=True)
    assert "rm_spur" in r.stdout and r.stdout.strip().count("\n") >= 3


def test_tune_on_picks_the_threshold_on_val_and_applies_it(tmp_path):
    from ctag.build_benchmark import main as build
    from ctag.run_transcribe import main as transcribe

    build(["--source", "procedural", "--n-clips", "12", "--out", str(tmp_path / "b")])
    for tag, seed in (("val", 1), ("test", 2)):
        transcribe(["--model", "mock:oracle", "--bench", str(tmp_path / "b" / "benchmark.jsonl"),
                    "--timelines", str(tmp_path / "b" / "timelines.jsonl"), "--out", str(tmp_path / tag),
                    "--spurious", "0.5", "--stop-probs", "--seed", str(seed)])
    r = subprocess.run([sys.executable, "-m", "ctag.trailing", "--rule", "stop",
                        "--pred-timelines", str(tmp_path / "test" / "pred_timelines.jsonl"),
                        "--tune-on", str(tmp_path / "val" / "pred_timelines.jsonl"),
                        "--timelines", str(tmp_path / "b" / "timelines.jsonl"), "--sweep", "0.1", "0.5", "0.9",
                        "--out", str(tmp_path / "f")], capture_output=True, text=True, check=True)
    assert "chosen thr=0.10" in r.stdout or "chosen thr=0.50" in r.stdout, r.stdout   # 0.9 never fires on the mock's 0.7
    s = json.loads((tmp_path / "f" / "summary.json").read_text())
    assert s["confusion"]["removed_correct"] == 0 and s["thr"] < 0.9
