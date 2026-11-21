"""Exact labels from DCASE 2020's synthetic-evaluation recipes (scripts/desed_synth.py)."""
import csv
import json
import sys

sys.path.insert(0, "scripts")
import desed_synth as ds  # noqa: E402


def test_post_process_merges_strips_and_drops_like_desed():
    ev = [("Speech", 1.0, 1.5), ("Speech", 1.6, 2.4),          # gap 0.1 < 0.15 -> merged
          ("Speech", 3.0, 3.1),                               # 0.1 s < 0.25 -> dropped
          ("Dog", 1.0, 1.5), ("Dog", 1.7, 2.0)]               # gap 0.2 -> kept apart
    assert ds.post_process(ev) == [("Dog", 1.0, 1.5), ("Speech", 1.0, 2.4), ("Dog", 1.7, 2.0)]


def test_recipe_to_tsv_round_trip(tmp_path):
    rec = {"file_metadata": {"duration": 10.0},
           "annotations": [{"namespace": "scaper", "data": [
               {"time": 0.0, "duration": 10.0, "value": {"role": "background", "label": "noise"}},
               {"time": 9.2, "duration": 3.0, "value": {"role": "foreground", "label": "Vacuum_cleaner_nOff"}},
               {"time": 2.0, "duration": 0.9, "value": {"role": "foreground", "label": "Dishes"}},
               {"time": 2.95, "duration": 0.6, "value": {"role": "foreground", "label": "Dishes"}}]}]}
    (tmp_path / "07.jams").write_text(json.dumps(rec))
    (tmp_path / "08.jams").write_text(json.dumps({"file_metadata": {"duration": 10.0}, "annotations": [{"namespace": "scaper", "data": []}]}))
    out = tmp_path / "labels.tsv"
    s = ds.write_labels(tmp_path, out)
    rows = list(csv.DictReader(open(out), delimiter="\t"))
    assert s["clips"] == 2 and s["events"] == 2 and s["labels"] == {"Dishes": 1, "Vacuum_cleaner": 1}
    assert [(r["filename"], r["event_label"], r["onset"], r["offset"]) for r in rows] == [
        ("07.wav", "Dishes", "2.000", "3.550"),                # merged across the 0.05 s gap
        ("07.wav", "Vacuum_cleaner", "9.200", "10.000"),       # clipped to the clip, suffix stripped
        ("08.wav", "", "", "")]                                # the empty soundscape keeps one blank row


def test_desed_runner_can_skip_the_question_adapter_steps():
    import os
    import subprocess

    env = {**os.environ, "CTAG_DRY_RUN": "1", "CTAG_DESED_ARMS": "0", "CTAG_DESED_ZEROSHOT_N": "0",
           "CTAG_DESED_BENCH": "data/desed/synth20_eval/benchmark.jsonl",
           "CTAG_DESED_TIMELINES": "data/desed/synth20_eval/timelines.jsonl", "CTAG_T_ADAPTER": "runs/lora_real",
           "CTAG_OPEN_VOCAB": "1"}
    out = subprocess.run([sys.executable, "nebius_desed.py"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-1500:]
    cmds = [json.loads(l[4:]) for l in out.stdout.splitlines() if l.startswith("DRY ")]
    assert cmds and not any(c[2] == "ctag.run_zeroshot" for c in cmds)
    assert not any("--grounder" in c and c[c.index("--grounder") + 1] == "qwen2.5-omni" for c in cmds)
    t = next(c for c in cmds if c[2] == "ctag.run_transcribe")
    assert "--no-vocab-in-prompt" in t and t[t.index("--adapter") + 1] == "runs/lora_real" and "synth20_eval" in t[t.index("--bench") + 1]
    env["CTAG_DESED_ARMS"] = "1"
    out = subprocess.run([sys.executable, "nebius_desed.py"], capture_output=True, text=True, env=env)
    assert any(json.loads(l[4:])[2] == "ctag.run_zeroshot" for l in out.stdout.splitlines() if l.startswith("DRY "))
