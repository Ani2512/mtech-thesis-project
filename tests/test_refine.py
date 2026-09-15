"""Boundary refinement: energy snapping recovers jittered edges, never invents them."""
import json
import subprocess
import sys

import numpy as np
import soundfile as sf

from ctag.refine import refine_intervals, snap_interval


def _tone_clip(tmp_path, events, sr=16000, dur=10.0, snr_db=20.0):
    """Sine bursts with 20 ms fades on white noise; exact edges known."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    for a, b in events:
        n = int((b - a) * sr); t = np.arange(n) / sr
        y = 0.3 * np.sin(2 * np.pi * 660 * t)
        f = int(0.02 * sr); env = np.ones(n); env[:f] = np.linspace(0, 1, f); env[-f:] = np.linspace(1, 0, f)
        x[int(a * sr): int(a * sr) + n] += (y * env).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise *= (np.sqrt(np.mean(x ** 2)) + 1e-9) / (10 ** (snr_db / 20)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
    p = tmp_path / "tone.wav"; sf.write(p, np.clip(x + noise, -1, 1), sr)
    return str(p)


def test_snap_moves_jittered_edges_back_onto_the_bursts(tmp_path):
    events = [(1.0, 2.5), (4.0, 4.8), (7.0, 9.0)]
    wav = _tone_clip(tmp_path, events)
    jittered = [(1.3, 2.2), (3.7, 5.1), (7.25, 8.6)]          # off by up to 0.3 s each edge
    new, recs = refine_intervals(wav, jittered, window=0.5)
    for (a, b), (ga, gb), r in zip(new, events, recs):
        assert r["moved"] and abs(a - ga) < 0.05 and abs(b - gb) < 0.05, (a, b, ga, gb, r)


def test_snap_leaves_exact_edges_and_silence_alone(tmp_path):
    events = [(1.0, 2.5), (6.0, 7.0)]
    wav = _tone_clip(tmp_path, events)
    new, _ = refine_intervals(wav, events, window=0.5)
    for (a, b), (ga, gb) in zip(new, events):
        assert abs(a - ga) < 0.04 and abs(b - gb) < 0.04
    # a window on pure noise has nothing to snap to and is returned unchanged
    new, recs = refine_intervals(wav, [(3.5, 4.5)], window=0.3)
    assert new == [(3.5, 4.5)] and recs[0]["reason"] == "no_energy"
    # empty input, empty output
    assert refine_intervals(wav, []) == ([], [])


def test_snap_refuses_to_collapse_a_window():
    db = np.full(1000, -60.0); db[300:320] = -10.0             # one 200 ms burst
    iv, rec = snap_interval(db, 0.01, onset=2.0, offset=4.0, window=1.5)   # a window far wider than the burst
    assert iv == (2.0, 4.0) and rec["reason"] == "collapsed" or (iv[1] - iv[0] >= 0.05)


def test_refine_cli_rescores_timelines_and_query_predictions(tmp_path):
    from ctag.build_benchmark import build
    out = tmp_path / "proc"
    build("procedural", 6, out, seed=0)
    py = [sys.executable, "-m"]
    subprocess.run(py + ["ctag.run_transcribe", "--model", "mock:oracle", "--jitter", "0.3", "--seed", "1",
                         "--bench", str(out / "benchmark.jsonl"), "--timelines", str(out / "timelines.jsonl"),
                         "--out", str(tmp_path / "tr")], check=True, capture_output=True)
    subprocess.run(py + ["ctag.refine", "--pred-timelines", str(tmp_path / "tr" / "pred_timelines.jsonl"),
                         "--timelines", str(out / "timelines.jsonl"), "--out", str(tmp_path / "ref")], check=True, capture_output=True)
    s = json.load(open(tmp_path / "ref" / "summary.json"))
    assert s["edges_moved"] > 0
    assert s["after"]["event_f1_pooled"] > s["before"]["event_f1_pooled"]
    assert s["after"]["centre_error_median"] < s["before"]["centre_error_median"]
    rows = [json.loads(l) for l in open(tmp_path / "ref" / "pred_timelines.jsonl")]
    assert all(r["refined"] and "events_before" in r for r in rows)
    # the refined timelines still feed the query grounder
    subprocess.run(py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", str(tmp_path / "ref" / "pred_timelines.jsonl"),
                         "--bench", str(out / "benchmark.jsonl"), "--out", str(tmp_path / "agent")], check=True, capture_output=True)
    # per-question predictions from the (jittered) oracle agent, refined with the benchmark's audio
    subprocess.run(py + ["ctag.run_agent", "--grounder", "oracle", "--jitter", "0.3", "--timelines", str(out / "timelines.jsonl"),
                         "--bench", str(out / "benchmark.jsonl"), "--out", str(tmp_path / "agent_j")], check=True, capture_output=True)
    subprocess.run(py + ["ctag.refine", "--predictions", str(tmp_path / "agent_j" / "predictions.jsonl"),
                         "--bench", str(out / "benchmark.jsonl"), "--out", str(tmp_path / "agent_j_ref")], check=True, capture_output=True)
    before = json.load(open(tmp_path / "agent_j" / "summary.json"))["by_type"]["ALL"]["f1@0.5"]
    after = json.load(open(tmp_path / "agent_j_ref" / "summary.json"))["by_type"]["ALL"]["f1@0.5"]
    assert after > before, (before, after)
    # exactly one input kind
    r = subprocess.run(py + ["ctag.refine", "--out", str(tmp_path / "x")], capture_output=True)
    assert r.returncode != 0
