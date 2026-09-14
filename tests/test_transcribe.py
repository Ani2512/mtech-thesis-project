"""Whole-timeline transcription: target, parser, scorer, grounder, generator."""
import json
import random
import subprocess
import sys

import pytest

from ctag.timeline import Event, Timeline
from ctag.transcribe import (parse_timeline, score_timeline, summarize_timelines, target_timeline,
                             timeline_grounder, transcribe_query)

GOLD = [("dog", 1.0, 2.0), ("car horn", 3.0, 4.0), ("dog", 5.0, 6.0)]


def test_target_roundtrips_through_the_parser():
    tl = Timeline(20.0, [Event("dog", 1.0, 2.0), Event("car_horn", 3.0, 4.0), Event("dog", 5.0, 6.0)])
    s = target_timeline(tl.events)
    assert s.startswith('[{"sound": "dog"')
    assert parse_timeline(s) == GOLD
    assert score_timeline(parse_timeline(s), GOLD)["f1"] == 1.0
    assert target_timeline([]) == "[]" and parse_timeline("[]") == []


def test_parser_reads_the_shapes_models_emit():
    assert parse_timeline("Here you go: [{'label': 'dog', 'onset': '1.0', 'offset': '2.0'}]") == [("dog", 1.0, 2.0)]
    assert parse_timeline('[["dog", 1, 2], ["cat", 2.5, 3]]') == [("dog", 1.0, 2.0), ("cat", 2.5, 3.0)]
    assert parse_timeline("[[1.0, 2.0]]") == [("", 1.0, 2.0)]        # no label: never matches
    assert parse_timeline("There are no sound events.") == []
    assert parse_timeline("I cannot help with that") is None
    assert parse_timeline('[{"sound": "Door_Wood_Knock", "start": 2, "end": 1}]') == [("door wood knock", 1.0, 2.0)]


def test_scoring_is_label_aware_but_reports_the_label_agnostic_view_too():
    wrong_name = [("cat", 1.0, 2.0), ("car horn", 3.0, 4.0), ("dog", 5.0, 6.0)]
    s = score_timeline(wrong_name, GOLD)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["f1_any_label"] == 1.0
    assert s["label_hits"] == {"dog": [1, 2], "car horn": [1, 1]}
    missed = [("dog", 1.0, 2.0)]
    s = score_timeline(missed, GOLD)
    assert s["recall"] == pytest.approx(1 / 3) and s["precision"] == 1.0 and s["n_pred"] < s["n_gold"]
    assert score_timeline(None, GOLD)["parse_fail"] and score_timeline(None, GOLD)["f1"] == 0.0
    assert score_timeline([], [])["f1"] == 1.0
    summ = summarize_timelines([score_timeline(missed, GOLD), score_timeline(wrong_name, GOLD)])
    assert summ["recall_by_label"]["dog"] == pytest.approx((1 + 1) / (2 + 2))
    assert summ["under_report_rate"] == 0.5


def test_timeline_grounder_answers_every_query_type_exactly(tmp_path):
    """Gold timelines fed back through the grounder must reproduce every answer:
    this is the 'conditions are free once the timeline is right' claim."""
    from ctag.agent import ground_query
    from ctag.queries import generate
    tl = Timeline(20.0, [Event("dog", 1.0, 2.0), Event("car_horn", 3.0, 4.0), Event("dog", 5.0, 6.0),
                         Event("siren", 7.0, 12.0), Event("dog", 8.0, 9.0), Event("footsteps", 9.5, 10.5)])
    preds = {"c0": {"events": [{"label": e.label, "onset": e.onset, "offset": e.offset} for e in tl.events]}}
    g = timeline_grounder(preds)
    qs = generate(tl, "c0", ["dog", "car_horn", "siren", "footsteps", "cat"], random.Random(3), max_per_type=3)
    assert {q.qtype for q in qs} >= {"PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ABSENT"}
    for q in qs:
        assert ground_query(q, "/x/c0.wav", g) == [tuple(a) for a in q.answer], q.text
    # normalised names: the model writes 'car horn', the query carries 'car_horn'
    assert g("/x/c0.wav", "car_horn") == [(3.0, 4.0)]


def _procedural(tmp_path, n=6):
    from ctag.build_benchmark import build
    from ctag.split import split_benchmark
    out = tmp_path / "proc"
    build("procedural", n, out, seed=0)
    split_benchmark(out / "benchmark.jsonl", out, seed=0, frac_train=0.5, frac_val=0.25)
    return out


def test_sft_transcribe_emits_one_parseable_target_per_clip(tmp_path):
    from ctag.sft_data import build_transcribe
    out = _procedural(tmp_path)
    s = build_transcribe(out / "timelines.jsonl", tmp_path / "sft.jsonl", out / "benchmark_train.jsonl")
    rows = [json.loads(l) for l in open(tmp_path / "sft.jsonl")]
    train_clips = {json.loads(l)["clip_id"] for l in open(out / "benchmark_train.jsonl")}
    assert s["examples"] == len(rows) == len(train_clips)
    for r in rows:
        assert r["kind"] == "transcribe" and r["messages"][1]["content"].startswith("Locate: every sound event")
        # the full closed set, not only the sounds that happen to be in the training clips
        assert "using only these sound names: beep, buzz, chirp, click, hiss" in r["messages"][1]["content"]
        assert parse_timeline(r["target"]) is not None
    # the prompt is the one inference sends
    assert transcribe_query(s["vocab"]) in rows[0]["messages"][1]["content"]


def test_run_transcribe_mock_scores_one_and_degrades_with_noise(tmp_path):
    out = _procedural(tmp_path)
    def run(extra, name):
        cmd = [sys.executable, "-m", "ctag.run_transcribe", "--model", "mock:oracle",
               "--bench", str(out / "benchmark.jsonl"), "--timelines", str(out / "timelines.jsonl"),
               "--out", str(tmp_path / name)] + extra
        subprocess.run(cmd, check=True, capture_output=True)
        return json.load(open(tmp_path / name / "summary.json"))
    clean = run([], "clean")
    assert clean["f1"] == 1.0 and clean["n_clips"] == 6 and clean["parse_fail_rate"] == 0.0
    noisy = run(["--drop", "0.5", "--relabel", "0.3", "--seed", "1"], "noisy")
    assert noisy["recall"] < 1.0 and noisy["f1_any_label"] >= noisy["f1"]
    # and the predictions answer the benchmark through run_agent with no model
    cmd = [sys.executable, "-m", "ctag.run_agent", "--grounder", "timeline",
           "--pred-timelines", str(tmp_path / "clean" / "pred_timelines.jsonl"),
           "--bench", str(out / "benchmark.jsonl"), "--out", str(tmp_path / "agent_tl")]
    subprocess.run(cmd, check=True, capture_output=True)
    summ = json.load(open(tmp_path / "agent_tl" / "summary.json"))
    assert summ["model"] == "agent:timeline"
    assert summ["by_type"]["ALL"]["f1@0.5"] == 1.0 and summ["by_type"]["ALL"]["rejection_f1"] == 1.0


def test_gen_train_is_deterministic_and_hard_mode_makes_hard_cases(tmp_path):
    from ctag.gen_train import build
    a = build("procedural", 12, tmp_path / "a", seed=7, hard=True, write_audio=False, queries=True)
    b = build("procedural", 12, tmp_path / "b", seed=7, hard=True, write_audio=False, workers=2)
    ta = [json.loads(l) for l in open(tmp_path / "a" / "timelines.jsonl")]
    tb = [json.loads(l) for l in open(tmp_path / "b" / "timelines.jsonl")]
    assert [t["events"] for t in ta] == [t["events"] for t in tb]        # same seed -> same clips, any worker count
    assert ta[0]["clip_id"] == "gen7_000000" and ta[0]["recipe"]["n_events"] >= 4
    assert a["clips_with_overlap"] > 0 and a["repeated_labels"] > 0 and a["queries"] > 0
    easy = build("procedural", 12, tmp_path / "e", seed=7, hard=False, write_audio=False)
    assert easy["events_per_clip_mean"] <= 6.0
    rows = [json.loads(l) for l in open(tmp_path / "a" / "benchmark.jsonl")]
    assert all(r["clip_id"].startswith("gen7_") for r in rows)
    # different seed -> different clips
    c = build("procedural", 12, tmp_path / "c", seed=8, hard=True, write_audio=False)
    tc = [json.loads(l) for l in open(tmp_path / "c" / "timelines.jsonl")]
    assert [t["events"] for t in tc] != [t["events"] for t in ta]


def test_gen_train_writes_audio_that_matches_the_timeline(tmp_path):
    import soundfile as sf
    from ctag.gen_train import build
    build("procedural", 2, tmp_path / "g", seed=3, hard=True, write_audio=True)
    rows = [json.loads(l) for l in open(tmp_path / "g" / "timelines.jsonl")]
    x, sr = sf.read(rows[0]["audio"])
    assert sr == 16000 and abs(len(x) / sr - rows[0]["duration"]) < 0.01
    assert all(0 <= e["onset"] < e["offset"] <= rows[0]["duration"] for e in rows[0]["events"])


def test_compose_gap_bounds_are_respected():
    from ctag.compose import ProceduralBank, compose_clip
    rng = random.Random(0)
    _, tl = compose_clip(ProceduralBank(rng), rng, n_events=8, n_labels=3, p_overlap=0.0, gap=(0.1, 0.2))
    evs = tl.events
    gaps = [b.onset - a.offset for a, b in zip(evs, evs[1:])]
    assert gaps and all(0.1 - 1e-6 <= g <= 0.2 + 1e-6 for g in gaps)
