"""Real recordings: strong-label import, query generation, hand-review round trip."""
import csv
import json

import numpy as np
import pytest
import soundfile as sf

from ctag.real_data import apply_review, build_queries, export_review, import_strong, read_mid_map

DESED_TSV = """filename\tonset\toffset\tevent_label
a.wav\t0.000\t2.500\tDog
a.wav\t1.000\t4.000\tSpeech
a.wav\t6.000\t7.000\tDog
a.wav\t8.000\t9.500\tDishes
b.wav\t0.500\t3.000\tCat
b.wav\t3.000\t3.000\tCat
b.wav\t5.000\t9.000\tRunning_water
b.wav\t5.100\t5.200\tDog
"""


def _desed(tmp_path, with_audio=True):
    tsv = tmp_path / "val.tsv"; tsv.write_text(DESED_TSV)
    root = tmp_path / "audio"; root.mkdir()
    if with_audio:
        sf.write(root / "a.wav", np.zeros(16000 * 10, dtype="float32"), 16000)   # 10 s
        sf.write(root / "b.wav", np.zeros(16000 * 8, dtype="float32"), 16000)    # 8 s
    return tsv, root


def test_import_desed_builds_timelines_with_durations_from_the_audio(tmp_path):
    tsv, root = _desed(tmp_path)
    s = import_strong(tsv, root, tmp_path / "out", "desed")
    rows = {json.loads(l)["clip_id"]: json.loads(l) for l in open(tmp_path / "out" / "timelines.jsonl")}
    assert s["clips"] == 2 and s["clips_without_audio"] == 0
    assert rows["a"]["duration"] == 10.0 and rows["b"]["duration"] == 8.0
    assert [e["label"] for e in rows["a"]["events"]] == ["Dog", "Speech", "Dog", "Dishes"]   # sorted by onset
    assert len(rows["b"]["events"]) == 3            # the zero-length Cat row was dropped
    assert rows["a"]["audio"].endswith("audio/a.wav")


def test_import_without_audio_uses_the_given_duration(tmp_path):
    tsv, root = _desed(tmp_path, with_audio=False)
    s = import_strong(tsv, root, tmp_path / "out", "desed", duration=10.0)
    rows = [json.loads(l) for l in open(tmp_path / "out" / "timelines.jsonl")]
    assert s["clips_without_audio"] == 2 and all(r["duration"] == 10.0 for r in rows)
    s = import_strong(tsv, root, tmp_path / "out2", "desed", duration=10.0, require_audio=True)
    assert s["clips"] == 0


def test_import_audioset_strong_maps_mids_and_drops_unknown_ones(tmp_path):
    tsv = tmp_path / "as.tsv"
    tsv.write_text("segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n"
                   "Yx_000\t0.0\t1.5\t/m/0bt9lr\nYx_000\t2.0\t3.0\t/m/unknown\nYx_001\t1.0\t2.0\t/m/01yrx\n")
    mm = tmp_path / "mid.tsv"; mm.write_text("/m/0bt9lr\tDog\n/m/01yrx\tCat\n")
    assert read_mid_map(mm) == {"/m/0bt9lr": "Dog", "/m/01yrx": "Cat"}
    s = import_strong(tsv, None, tmp_path / "out", "audioset", mid_map=mm, duration=10.0)
    rows = {json.loads(l)["clip_id"]: json.loads(l) for l in open(tmp_path / "out" / "timelines.jsonl")}
    assert s["clips"] == 2 and rows["Yx_000"]["events"] == [{"label": "Dog", "onset": 0.0, "offset": 1.5}]


def test_import_refuses_a_file_with_the_wrong_columns(tmp_path):
    tsv = tmp_path / "x.tsv"; tsv.write_text("file\tstart\tend\tlabel\na\t0\t1\tDog\n")
    with pytest.raises(ValueError):
        import_strong(tsv, None, tmp_path / "out", "desed", duration=10.0)


def test_queries_use_the_composed_generator_and_filter_sliver_overlaps(tmp_path):
    tsv, root = _desed(tmp_path)
    import_strong(tsv, root, tmp_path / "out", "desed")
    s = build_queries(tmp_path / "out" / "timelines.jsonl", tmp_path / "q", seed=0, max_per_type=3, min_overlap=0.3)
    rows = [json.loads(l) for l in open(tmp_path / "q" / "benchmark.jsonl")]
    assert s["queries"] == len(rows) > 0
    assert set(s["vocab"]) == {"Cat", "Dishes", "Dog", "Running_water", "Speech"}
    # question text carries phrases, predicates carry raw labels, answers are exact
    assert all("_" not in r["text"] for r in rows)
    assert all(r["expects_empty"] == (len(r["answer"]) == 0) for r in rows)
    # in clip b the Dog overlaps Running_water by 0.1 s: a WHILE question about
    # dog/water is ambiguous and must have been dropped, not asked
    for r in rows:
        if r["qtype"] == "WHILE" and r["clip_id"] == "b":
            assert {r["x"], r["y"]} != {"Dog", "Running_water"}
    s2 = build_queries(tmp_path / "out" / "timelines.jsonl", tmp_path / "q2", seed=0, max_per_type=3, min_overlap=0.0)
    assert s2["dropped"].get("while_ambiguous", 0) == 0
    assert (s["dropped"].get("while_ambiguous", 0) > 0) or s2["queries"] >= s["queries"]


def test_review_round_trip_keeps_confirmed_applies_corrections_and_drops_the_rest(tmp_path):
    tsv, root = _desed(tmp_path)
    import_strong(tsv, root, tmp_path / "out", "desed")
    build_queries(tmp_path / "out" / "timelines.jsonl", tmp_path / "q", seed=1, max_per_type=1, min_overlap=0.0)
    bench = tmp_path / "q" / "benchmark.jsonl"
    n = export_review(bench, tmp_path / "review.csv")
    rows = list(csv.DictReader(open(tmp_path / "review.csv")))
    assert n == len(rows) >= 4 and rows[0]["ok"] == "" and "text" in rows[0]
    # reviewer: confirm the first, correct the second, drop the third, leave the rest blank
    rows[0]["ok"] = "y"
    rows[1]["ok"] = "yes"; rows[1]["corrected_answer"] = "[[1.0, 2.0]]"
    rows[2]["ok"] = "n"
    with open(tmp_path / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    s = apply_review(bench, tmp_path / "review.csv", tmp_path / "verified.jsonl")
    assert s == {**s, "kept": 2, "confirmed": 1, "corrected": 1, "dropped": 1, "unreviewed": n - 3}
    kept = [json.loads(l) for l in open(tmp_path / "verified.jsonl")]
    assert kept[0]["verified"] == "confirmed"
    assert kept[1]["verified"] == "corrected" and kept[1]["answer"] == [[1.0, 2.0]] and kept[1]["expects_empty"] is False
    # a malformed correction is refused loudly
    rows[1]["corrected_answer"] = "1 to 2"
    with open(tmp_path / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError):
        apply_review(bench, tmp_path / "review.csv", tmp_path / "v2.jsonl")


def test_verified_real_benchmark_scores_through_the_oracle_grounder(tmp_path):
    """The imported timelines and the generated questions agree: the oracle
    grounder fed the real timelines reproduces every answer."""
    import subprocess, sys
    tsv, root = _desed(tmp_path)
    import_strong(tsv, root, tmp_path / "out", "desed")
    build_queries(tmp_path / "out" / "timelines.jsonl", tmp_path / "q", seed=2, max_per_type=2, min_overlap=0.0)
    cmd = [sys.executable, "-m", "ctag.run_agent", "--grounder", "oracle", "--timelines", str(tmp_path / "out" / "timelines.jsonl"),
           "--bench", str(tmp_path / "q" / "benchmark.jsonl"), "--out", str(tmp_path / "agent")]
    subprocess.run(cmd, check=True, capture_output=True)
    summ = json.load(open(tmp_path / "agent" / "summary.json"))
    assert summ["by_type"]["ALL"]["f1@0.5"] == 1.0
