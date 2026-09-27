"""The DESED runner's command list and the review sampler, CPU only."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from desed_review_sample import sample  # noqa: E402


def test_runner_dry_run_lists_every_step_in_order():
    env = dict(os.environ, CTAG_DRY_RUN="1", CTAG_N="3", CTAG_DESED_ZEROSHOT_N="50")
    out = subprocess.run([sys.executable, "nebius_desed.py"], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    cmds = [json.loads(l[4:]) for l in out.splitlines() if l.startswith("DRY ")]
    modules = [c[2] for c in cmds]
    assert modules == ["ctag.run_zeroshot", "ctag.run_agent", "ctag.run_transcribe", "ctag.run_agent",
                       "ctag.trailing", "ctag.run_agent", "ctag.run_zeroshot"]
    # the question-trained adapter first, the timeline adapter for the transcription
    assert cmds[0][cmds[0].index("--adapter") + 1] == "runs/lora_q_text"
    assert cmds[2][cmds[2].index("--adapter") + 1] == "runs/lora_transcribe" and "--stop-probs" in cmds[2]
    assert cmds[2][cmds[2].index("--timelines") + 1] == "data/desed/public/timelines.jsonl"
    # the trailing check uses a fixed threshold (no val split on real audio)
    assert "--thr" in cmds[4] and "--tune-on" not in cmds[4]
    # the untrained reference is capped by CTAG_N when that is smaller
    assert cmds[6][cmds[6].index("--n") + 1] == "3" and "--adapter" not in cmds[6]


def test_runner_skips_decomposition_with_a_local_model_id():
    env = dict(os.environ, CTAG_DRY_RUN="1", CTAG_MODEL_ID="/tmp/tiny", CTAG_DESED_ZEROSHOT_N="0")
    out = subprocess.run([sys.executable, "nebius_desed.py"], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    cmds = [json.loads(l[4:]) for l in out.splitlines() if l.startswith("DRY ")]
    assert [c[2] for c in cmds] == ["ctag.run_zeroshot", "ctag.run_transcribe", "ctag.run_agent", "ctag.trailing", "ctag.run_agent"]
    assert all("--model-id" in c for c in cmds if c[2] in ("ctag.run_zeroshot", "ctag.run_transcribe"))


def _rows():
    rows = []
    for clip in range(100):
        for t in ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ABSENT"]:
            for k in range(2):
                target = "speech" if (clip + k) % 3 == 0 else "dog"
                rows.append({"qid": f"c{clip}_{t}_{k}", "clip_id": f"c{clip}", "qtype": t, "text": f"every {target}"})
    return rows


def test_review_sample_is_stratified_and_spread_over_clips():
    chosen = sample(_rows(), 150, seed=0, per_clip=2)
    assert len(chosen) == 150
    by_type = {}
    for r in chosen:
        by_type[r["qtype"]] = by_type.get(r["qtype"], 0) + 1
    assert by_type["ABSENT"] == 10 and all(by_type[t] >= 20 for t in by_type if t != "ABSENT")
    per_clip = {}
    for r in chosen:
        per_clip[r["clip_id"]] = per_clip.get(r["clip_id"], 0) + 1
    assert max(per_clip.values()) <= 2
    # non-speech questions are preferred while they last
    assert sum(1 for r in chosen if "speech" in r["text"]) < len(chosen) // 3
    assert sample(_rows(), 150, seed=0, per_clip=2) == chosen   # deterministic


def test_review_sample_on_the_real_csv_if_present(tmp_path):
    src = ROOT / "data/desed/public/review.csv"
    if not src.exists():
        return
    out = tmp_path / "s.csv"
    subprocess.run([sys.executable, "scripts/desed_review_sample.py", "--review", str(src), "--out", str(out), "--n", "300"],
                   cwd=ROOT, check=True, capture_output=True)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 300 and set(rows[0]) >= {"qid", "clip_id", "qtype", "text", "answer", "ok", "corrected_answer"}
