"""Run all of phase 2 unattended, then print the comparison table.

Ordered so the essential comparison finishes first. Measured on a T4 at
4.53 s/example with --amp none:

    arm C train   2.9 h      arm A (direct)      0.8 h
    arm C eval    0.8 h      arm B (agent)       0.2 h  (grounding cached per clip+sound)
    arm E train   2.9 h      union k=3           2.3 h
    arm E eval    0.8 h      hybrid              instant

That totals about 10.7 h. Two Kaggle limits cut across it: the enforced weekly
GPU quota is *floating* (the API reported 6 h on 2026-09-12 while the editor
showed "30 hrs"), and a single session is capped. So a full pass may need more
than one session. Every step is skipped when its output exists and results are
copied to the notebook output after every step, so a stopped session hands back
whatever finished and re-running this file continues from there.

    %run /kaggle/working/bootstrap.py      # once, to set up and smoke test
    %run /kaggle/working/phase2.py         # this

Every step is skipped if its output already exists, so re-running after a crash
resumes rather than restarting. Roughly 8 hours on a T4, inside Kaggle's
12-hour session.

One epoch, not two: there are only 200 training clips, so a second pass buys
little and doubles the most expensive item.
"""
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/temp/mtech-project/research_project"   # clone lives outside the 500-file output cap
TEST = "data/esc50/benchmark_test.jsonl"
# Adapters go under /kaggle/working, which Kaggle keeps as the version output
# even when a session is stopped by the quota or the session limit.
# /kaggle/temp is discarded, which is where the first run put them.
ADAPTERS = "/kaggle/working/adapters"
VAL = "data/esc50/benchmark_val.jsonl"
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")
os.chdir(WORK)

EPOCHS = os.environ.get("CTAG_EPOCHS", "1")
# Which mixed-precision setting the bootstrap found to actually train on this card.
AMP = os.environ.get("CTAG_AMP", "none")
# The hybrid selects per type on validation clips and reports on test clips.
# Scoring the whole val split would cost another ~50 min; 400 queries is ~55 per
# type, enough for six binary choices, and ctag.hybrid warns below 20.
VAL_N = os.environ.get("CTAG_VAL_N", "400")
print(f"[phase2] epochs={EPOCHS}  amp={AMP}")


RESULTS = "/kaggle/working/results"


def persist():
    """Copy every finished run to the notebook output now, not at the end.

    The first full run copied results only after the last arm. A session that
    is stopped by the weekly GPU quota or the session limit never reaches that
    line, and everything computed until then is lost. Copying after each step
    means a stopped session still hands back every arm that completed.
    """
    import glob
    import shutil

    os.makedirs(RESULTS, exist_ok=True)
    for src in glob.glob("runs/esc50/*"):
        dst = os.path.join(RESULTS, os.path.basename(src))
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)


def run(label, cmd, produces):
    if produces and os.path.exists(produces):
        print(f"\n=== {label}: already done ===", flush=True)
        return True
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m"] + cmd).returncode
    print(f"--- {label}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {(time.time()-t0)/60:.0f} min",
          flush=True)
    persist()
    return rc == 0


ok = True

# --- seed from a previous run's output, attached as a Kaggle dataset ---------
# v4 (2026-09-12) produced arms A, B, both val runs and the union run, plus a
# trained adapter. Copying them in means those steps are skipped (their
# `produces` file exists) and only the missing arms cost GPU time.
SEED = os.environ.get("CTAG_SEED_DIR", "/kaggle/input/ctag-phase2-v4")
if os.path.isdir(SEED):
    import glob as _glob
    import shutil as _shutil

    # Found by marker file rather than by path: Kaggle extracts uploaded
    # archives itself and the nesting depth is not worth depending on.
    os.makedirs("runs/esc50", exist_ok=True)
    for marker in sorted(_glob.glob(f"{SEED}/**/summary.json", recursive=True)):
        src = os.path.dirname(marker)
        dst = os.path.join("runs/esc50", os.path.basename(src))
        if not os.path.exists(dst):
            _shutil.copytree(src, dst)
            print(f"[seed] {os.path.basename(src)} <- {src}")
    for marker in sorted(_glob.glob(f"{SEED}/**/adapter_model.safetensors", recursive=True)):
        src = os.path.dirname(marker)
        dst = os.path.join(ADAPTERS, os.path.basename(src))
        if not os.path.exists(dst):
            _shutil.copytree(src, dst)
            print(f"[seed] adapter {os.path.basename(src)} <- {src}")
    persist()
else:
    print(f"[seed] no seed dir at {SEED}; every arm runs from scratch")

# --- the v4 adapter: LoRA that leaked into the audio and vision encoders -----
# Its target_modules were bare names, which also matched audio_tower.*.q_proj
# and visual.blocks.*.attn.*. It is a legitimate "LoRA on encoder + LM" arm,
# so it is scored under its own name rather than discarded; arm C below is the
# language-model-only adapter the plan describes. Cheap (one eval) and first,
# so a stopped session still hands back a fine-tuned number.
if os.path.exists(f"{ADAPTERS}/lora_text_enc/adapter_model.safetensors"):
    ok &= run("eval arm C-enc (v4 adapter, LoRA on encoders + LM)",
              ["ctag.run_zeroshot", "--model", "qwen2.5-omni",
               "--adapter", f"{ADAPTERS}/lora_text_enc",
               "--bench", TEST, "--out", "runs/esc50/test_lora_text_enc"],
              "runs/esc50/test_lora_text_enc/summary.json")

# --- arm C: it is the main comparison ---------------------------------------
ok &= run("train arm C (text timestamps)",
          ["ctag.train_lora", "--data", "data/esc50/sft_train.jsonl",
           "--val", "data/esc50/sft_val.jsonl", "--out", f"{ADAPTERS}/lora_text",
           "--epochs", EPOCHS, "--amp", AMP],
          f"{ADAPTERS}/lora_text/adapter_model.safetensors")
ok &= run("eval arm C",
          ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", f"{ADAPTERS}/lora_text",
           "--bench", TEST, "--out", "runs/esc50/test_lora_text"],
          "runs/esc50/test_lora_text/summary.json")

# --- cheap untrained baselines, so arm C has something to be compared against
ok &= run("arm A (direct prompting)",
          ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--bench", TEST,
           "--out", "runs/esc50/test_direct"],
          "runs/esc50/test_direct/summary.json")
ok &= run("arm B (decompose and combine)",
          ["ctag.run_agent", "--grounder", "qwen2.5-omni", "--bench", TEST,
           "--out", "runs/esc50/test_agent"],
          "runs/esc50/test_agent/summary.json")
# The hybrid needs val-split scores to choose from. Without these the selection
# set is empty and the hybrid silently becomes a copy of arm A, which is exactly
# what happened on the first full run.
ok &= run("arm A on val (for hybrid selection)",
          ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--bench", VAL,
           "--n", VAL_N, "--out", "runs/esc50/val_direct"],
          "runs/esc50/val_direct/summary.json")
ok &= run("arm B on val (for hybrid selection)",
          ["ctag.run_agent", "--grounder", "qwen2.5-omni", "--bench", VAL,
           "--n", VAL_N, "--out", "runs/esc50/val_agent"],
          "runs/esc50/val_agent/summary.json")
run("arm D (hybrid, selection on val)",
    ["ctag.hybrid", "--direct", "runs/esc50/test_direct",
     "--agent", "runs/esc50/test_agent",
     "--direct-val", "runs/esc50/val_direct",
     "--agent-val", "runs/esc50/val_agent",
     "--out", "runs/esc50/test_hybrid"],
    "runs/esc50/test_hybrid/summary.json")

# --- arm E: a reproduction of published work, so it yields if time runs short
ok &= run("train arm E (timestamp tokens)",
          ["ctag.train_lora", "--data", "data/esc50/sft_train_tt.jsonl",
           "--val", "data/esc50/sft_val_tt.jsonl", "--out", f"{ADAPTERS}/lora_tt",
           "--epochs", EPOCHS, "--amp", AMP,
           "--time-tokens", "--time-sigma", "0.3", "--time-lambda", "0.5"],
          f"{ADAPTERS}/lora_tt/adapter_model.safetensors")
ok &= run("eval arm E",
          ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", f"{ADAPTERS}/lora_tt",
           "--bench", TEST, "--out", "runs/esc50/test_lora_tt"],
          "runs/esc50/test_lora_tt/summary.json")

# --- recall-biased decoding last: k forward passes per query is the priciest item.
# k=3 rather than 5 keeps the run inside one session; simulations put the optimum
# at k=5/2 votes but k=3 captures most of the gain (docs/recall_bias.md).
K = os.environ.get("CTAG_UNION_K", "3")
ok &= run(f"recall-biased decoding (k={K}, 2 votes)",
          ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--bench", TEST,
           "--out", "runs/esc50/test_union", "--samples", K, "--min-votes", "2",
           "--temperature", "0.7"],
          "runs/esc50/test_union/summary.json")

# --- optional extra backbones: arm A and arm B on the test split -------------
# CTAG_EXTRA_MODELS="audio-flamingo-3" (comma-separated). Off by default so the
# main pipeline's budget is unchanged; ~45 min per model for arm A, ~15 for B.
for extra_model in [m for m in os.environ.get("CTAG_EXTRA_MODELS", "").split(",") if m.strip()]:
    tag = extra_model.replace(".", "").replace("-", "")
    ok &= run(f"arm A ({extra_model})",
              ["ctag.run_zeroshot", "--model", extra_model, "--bench", TEST,
               "--out", f"runs/esc50/test_direct_{tag}"],
              f"runs/esc50/test_direct_{tag}/summary.json")
    ok &= run(f"arm B ({extra_model})",
              ["ctag.run_agent", "--grounder", extra_model, "--bench", TEST,
               "--out", f"runs/esc50/test_agent_{tag}"],
              f"runs/esc50/test_agent_{tag}/summary.json")

# --- results ----------------------------------------------------------------
import glob

TYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]
runs = {}
for p in sorted(glob.glob("runs/esc50/test_*/summary.json")):  # val_* are selection only, not arms
    name = os.path.basename(os.path.dirname(p)).replace("test_", "")
    runs[name] = json.load(open(p))["by_type"]

for metric in ("f1@0.5", "f_beta", "count_acc"):
    print(f"\n{metric}" + ("   (recall weighted 5.6x, the measured asymmetry)"
                           if metric == "f_beta" else ""))
    print("-" * (14 + 13 * len(TYPES)))
    print(f"{'arm':<14}" + "".join(f"{t:>13}" for t in TYPES))
    for m in sorted(runs):
        print(f"{m:<14}" + "".join(
            (f"{runs[m][t][metric]:>13.3f}"
             if isinstance(runs[m].get(t, {}).get(metric), (int, float)) else f"{'-':>13}")
            for t in TYPES))

# final persist, plus the docs for context
import shutil

persist()
for doc in glob.glob("docs/*.md"):
    shutil.copy(doc, RESULTS)
print(f"\nresults copied to {RESULTS} (download from the Output tab)")
print("ALL STEPS OK" if ok else "SOME STEPS FAILED - check the log above")
