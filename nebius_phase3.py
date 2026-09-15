"""Phase 3 on a rented GPU (Nebius L40S or similar): generate, train, evaluate.

Same discipline as kaggle_phase2.py: every step is skipped when its output
exists, results are copied to RESULTS after every step, and a crash resumes
rather than restarts. Unlike Kaggle there is no output-file cap and no session
limit, so this runs to completion once.

    # on the VM, after docs/nebius.md's setup
    cd ~/mtech-thesis-project && nohup python nebius_phase3.py > logs/phase3.log 2>&1 &
    tail -f logs/phase3.log

Knobs (environment variables, all optional):
    CTAG_GEN_N       generated training clips            (20000)
    CTAG_EPOCHS      passes over the generated set       (3)
    CTAG_BS          per-device batch size               (2)
    CTAG_ACCUM       gradient accumulation               (4)
    CTAG_LR          learning rate                        (1e-4)
    CTAG_PRECISION   bf16 | fp16 | 4bit                   (bf16)
    CTAG_TRAIN_ENC   1 to also train the audio encoder    (1)
    CTAG_SMOKE       1 to stop after the 20-step smoke test
    CTAG_WORKERS     generator processes                  (cpu count)
    CTAG_RESULTS     where finished runs are copied       (~/phase3_results)

Order: benchmark (if missing) -> generated set -> SFT files -> 20-step smoke
train -> full train -> transcribe val (the 0.95 gate) and test -> every query
type from the test timelines -> a direct-prompting run with the same adapter
for reference. Roughly 3-4 h on an L40S at 20k clips x 3 epochs.
"""
import json
import os
import shutil
import subprocess
import sys
import time

WORK = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK)
os.makedirs("logs", exist_ok=True)

GEN_N = os.environ.get("CTAG_GEN_N", "20000")
EPOCHS = os.environ.get("CTAG_EPOCHS", "3")
BS = os.environ.get("CTAG_BS", "2")
ACCUM = os.environ.get("CTAG_ACCUM", "4")
LR = os.environ.get("CTAG_LR", "1e-4")
PRECISION = os.environ.get("CTAG_PRECISION", "bf16")
TRAIN_ENC = os.environ.get("CTAG_TRAIN_ENC", "1") == "1"
SMOKE_ONLY = os.environ.get("CTAG_SMOKE", "0") == "1"
WORKERS = os.environ.get("CTAG_WORKERS", str(max(1, (os.cpu_count() or 2) - 1)))
RESULTS = os.path.expanduser(os.environ.get("CTAG_RESULTS", "~/phase3_results"))

BENCH = "data/esc50"
GEN = "data/gen_esc50"
ADAPTER = "runs/lora_transcribe"
TEST, VAL = f"{BENCH}/benchmark_test.jsonl", f"{BENCH}/benchmark_val.jsonl"
AMP = "bf16" if PRECISION == "bf16" else "none"
print(f"[phase3] gen={GEN_N} epochs={EPOCHS} bs={BS} accum={ACCUM} lr={LR} precision={PRECISION} "
      f"train_encoder={TRAIN_ENC} workers={WORKERS}", flush=True)


def persist():
    os.makedirs(RESULTS, exist_ok=True)
    for src in ("runs/esc50", ADAPTER, "logs"):
        if os.path.isdir(src):
            dst = os.path.join(RESULTS, os.path.basename(src))
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt", "checkpoint-*"))


def run(label, cmd, produces):
    if produces and os.path.exists(produces):
        print(f"\n=== {label}: already done ===", flush=True)
        return True
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    mins = (time.time() - t0) / 60
    print(f"--- {label}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {mins:.0f} min", flush=True)
    persist()
    return rc == 0


py = [sys.executable, "-m"]
ok = True

# ---------------------------------------------------------------- 1. benchmark
ok &= run("benchmark (300 ESC-50 clips)",
          py + ["ctag.build_benchmark", "--source", "esc50", "--n-clips", "300", "--p-overlap", "0.45",
                "--out", BENCH, "--esc50-root", "data/esc50_raw"], f"{BENCH}/benchmark.jsonl")
ok &= run("split", py + ["ctag.split", "--bench", f"{BENCH}/benchmark.jsonl", "--out", BENCH], TEST)

# ---------------------------------------------------------------- 2. generated set
ok &= run(f"generate {GEN_N} hard clips",
          py + ["ctag.gen_train", "--source", "esc50", "--n-clips", GEN_N, "--hard", "--workers", WORKERS,
                "--out", GEN, "--esc50-root", "data/esc50_raw"], f"{GEN}/gen_stats.json")

# ---------------------------------------------------------------- 3. SFT files
ok &= run("SFT: transcription targets for the generated set",
          py + ["ctag.sft_data", "--task", "transcribe", "--timelines", f"{GEN}/timelines.jsonl",
                "--out", f"{GEN}/sft_transcribe.jsonl"], f"{GEN}/sft_transcribe.jsonl")
ok &= run("SFT: transcription targets for the benchmark val clips",
          py + ["ctag.sft_data", "--task", "transcribe", "--timelines", f"{BENCH}/timelines.jsonl",
                "--bench", VAL, "--out", f"{BENCH}/sft_transcribe_val.jsonl"], f"{BENCH}/sft_transcribe_val.jsonl")
if not ok:
    raise SystemExit("data preparation failed; see above")

train_common = ["ctag.train_lora", "--data", f"{GEN}/sft_transcribe.jsonl", "--val", f"{BENCH}/sft_transcribe_val.jsonl",
                "--precision", PRECISION, "--amp", AMP, "--batch-size", BS, "--grad-accum", ACCUM, "--lr", LR,
                "--max-seq-len", "4096"] + (["--train-encoder"] if TRAIN_ENC else [])

# ---------------------------------------------------------------- 4. smoke test, then the real run
ok &= run("smoke train (20 steps)", py + train_common + ["--out", "runs/lora_smoke", "--max-steps", "20"],
          "runs/lora_smoke/adapter_config.json")
if not ok:
    raise SystemExit("the smoke test failed; fix before spending hours")
if SMOKE_ONLY:
    raise SystemExit("CTAG_SMOKE=1: stopping after the smoke test")
ok &= run(f"train transcription ({EPOCHS} epochs)", py + train_common + ["--out", ADAPTER, "--epochs", EPOCHS],
          f"{ADAPTER}/adapter_config.json")
if not ok:
    raise SystemExit("training failed")

# ---------------------------------------------------------------- 5. evaluate
for split, tag in ((VAL, "val"), (TEST, "test")):
    run(f"transcribe {tag} clips",
        py + ["ctag.run_transcribe", "--model", "qwen2.5-omni", "--adapter", ADAPTER, "--precision", PRECISION,
              "--bench", split, "--timelines", f"{BENCH}/timelines.jsonl", "--out", f"runs/esc50/transcribe_{tag}"],
        f"runs/esc50/transcribe_{tag}/summary.json")
    run(f"every query type from the {tag} timelines",
        py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", f"runs/esc50/transcribe_{tag}/pred_timelines.jsonl",
              "--bench", split, "--out", f"runs/esc50/{tag}_from_timeline"],
        f"runs/esc50/{tag}_from_timeline/summary.json")

# the same adapter asked the phase 2 way, for a like-for-like reference row
run("direct prompting with the transcription adapter (reference)",
    py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", ADAPTER, "--precision", PRECISION,
          "--bench", TEST, "--out", "runs/esc50/test_direct_transcribe_adapter"],
    "runs/esc50/test_direct_transcribe_adapter/summary.json")

# ---------------------------------------------------------------- 6. the gate and the table
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

v = load("runs/esc50/transcribe_val/summary.json")
t = load("runs/esc50/transcribe_test/summary.json")
q = load("runs/esc50/test_from_timeline/summary.json")
print("\n================ phase 3 ================")
if v:
    f1 = v.get("event_f1_pooled")
    print(f"val   event F1 (pooled) {f1:.3f}  recall {v['event_recall_pooled']:.3f}  "
          f"precision {v['event_precision_pooled']:.3f}   GATE 0.95: {'PASS' if f1 and f1 >= 0.95 else 'not yet'}")
if t:
    print(f"test  event F1 (pooled) {t['event_f1_pooled']:.3f}  under-report {t['under_report_rate']:.3f}  "
          f"duration ratio {t['duration_ratio_median']}")
    print("      lowest recall by sound:", ", ".join(f"{k} {x:.2f}" for k, x in sorted(t["recall_by_label"].items(), key=lambda kv: kv[1])[:4]))
if q:
    types = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]
    print("test  f1@0.5 from the timeline: " + "  ".join(f"{ty} {q['by_type'][ty]['f1@0.5']:.3f}" for ty in types if ty in q["by_type"]))
persist()
print(f"\nresults copied to {RESULTS}; pull them with:  rsync -av <vm>:{RESULTS}/ runs_nebius_phase3/")
