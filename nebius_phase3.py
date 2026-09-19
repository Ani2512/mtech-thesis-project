"""Phase 3 on a rented GPU (Nebius L40S or similar): generate, train, evaluate.

Same discipline as kaggle_phase2.py: every step is skipped when its output
exists, results are copied to RESULTS after every step, and a crash resumes
rather than restarts: finished stages are skipped and a training that was cut
off (preempted VM, OOM in evaluation) continues from its newest checkpoint-N,
which train_lora writes every CTAG_SAVE_STEPS steps. On a preemptible VM just
start the VM again and rerun the same command.

    # on the VM, after docs/nebius.md's setup
    cd ~/mtech-thesis-project && nohup python nebius_phase3.py > logs/phase3.log 2>&1 &
    tail -f logs/phase3.log

Knobs (environment variables, all optional):
    CTAG_GEN_N       generated training clips            (20000)
    CTAG_EPOCHS      passes over the generated set       (3)
    CTAG_BS          per-device batch size               (2)
    CTAG_ACCUM       gradient accumulation               (4)
    CTAG_LR          learning rate                        (1e-4)
    CTAG_LORA_R      LoRA rank; alpha is always 2r        (128, TEMPO's setting; phase 2 used 32)
    CTAG_PRECISION   bf16 | fp16 | 4bit                   (bf16)
    CTAG_TRAIN_ENC   1 to also train the audio encoder    (1)
    CTAG_SMOKE       1 to stop after the 20-step smoke test
    CTAG_ARM_E       1 to also run the arm E retest: text digits vs time symbols
                     on the same generated questions, same epochs, same card    (1)
    CTAG_ARM_E_N     question examples for that comparison                        (20000)
    CTAG_ARM_E_ROWS  delta | full   how the new symbol rows are trained          (full)
    CTAG_SAVE_STEPS  resumable checkpoint every N steps    (200, ~15 min on the L40S; 0 = epoch ends)
    CTAG_WORKERS     generator processes                  (cpu count)
    CTAG_RESULTS     where finished runs are copied       (~/phase3_results)
    CTAG_DRY_RUN     1 to print every command and run nothing (tests validate the flags)

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
LORA_R = os.environ.get("CTAG_LORA_R", "128")
LORA_ALPHA = str(2 * int(LORA_R))
PRECISION = os.environ.get("CTAG_PRECISION", "bf16")
TRAIN_ENC = os.environ.get("CTAG_TRAIN_ENC", "1") == "1"
SMOKE_ONLY = os.environ.get("CTAG_SMOKE", "0") == "1"
ARM_E = os.environ.get("CTAG_ARM_E", "1") == "1"
ARM_E_N = os.environ.get("CTAG_ARM_E_N", "20000")
ARM_E_ROWS = os.environ.get("CTAG_ARM_E_ROWS", "full")
SAVE_STEPS = os.environ.get("CTAG_SAVE_STEPS", "200")
WORKERS = os.environ.get("CTAG_WORKERS", str(max(1, (os.cpu_count() or 2) - 1)))
RESULTS = os.path.expanduser(os.environ.get("CTAG_RESULTS", "~/phase3_results"))
DRY_RUN = os.environ.get("CTAG_DRY_RUN", "0") == "1"

BENCH = "data/esc50"
GEN = "data/gen_esc50"
ADAPTER = "runs/lora_transcribe"
TEST, VAL = f"{BENCH}/benchmark_test.jsonl", f"{BENCH}/benchmark_val.jsonl"
AMP = "bf16" if PRECISION == "bf16" else "none"
print(f"[phase3] gen={GEN_N} epochs={EPOCHS} bs={BS} accum={ACCUM} lr={LR} lora_r={LORA_R} precision={PRECISION} "
      f"train_encoder={TRAIN_ENC} workers={WORKERS}", flush=True)


def persist():
    if DRY_RUN:
        return
    os.makedirs(RESULTS, exist_ok=True)
    for src in ("runs/esc50", ADAPTER, "runs/lora_q_text", "runs/lora_q_tt", "logs"):
        if os.path.isdir(src):
            dst = os.path.join(RESULTS, os.path.basename(src))
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt", "checkpoint-*"))


def trained(out):
    """A training is complete when train_lora wrote its marker. The processor
    files are the marker of runs from before 2026-09-19 (written only at the
    end, unlike the adapter, which the epoch-end save writes mid-run)."""
    return [f"{out}/train_done.json", f"{out}/tokenizer_config.json"]


def run(label, cmd, produces):
    if DRY_RUN:
        print("DRY " + json.dumps(cmd), flush=True)
        return True
    if isinstance(produces, str):
        produces = [produces]
    if produces and any(os.path.exists(p) for p in produces):
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
                "--queries", "--out", GEN, "--esc50-root", "data/esc50_raw"], f"{GEN}/gen_stats.json")

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
                "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
                "--max-seq-len", "4096", "--save-steps", SAVE_STEPS] + (["--train-encoder"] if TRAIN_ENC else [])

# ---------------------------------------------------------------- 4. smoke test, then the real run
ok &= run("smoke train (20 steps)", py + train_common + ["--out", "runs/lora_smoke", "--max-steps", "20"],
          trained("runs/lora_smoke"))
if not ok:
    raise SystemExit("the smoke test failed; fix before spending hours")
if SMOKE_ONLY and not DRY_RUN:
    raise SystemExit("CTAG_SMOKE=1: stopping after the smoke test")
ok &= run(f"train transcription ({EPOCHS} epochs)", py + train_common + ["--out", ADAPTER, "--epochs", EPOCHS],
          trained(ADAPTER))
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
    # boundary refinement: the signal decides the exact edges (ctag.refine)
    run(f"refine the {tag} timelines against the audio",
        py + ["ctag.refine", "--pred-timelines", f"runs/esc50/transcribe_{tag}/pred_timelines.jsonl",
              "--timelines", f"{BENCH}/timelines.jsonl", "--out", f"runs/esc50/transcribe_{tag}_refined"],
        f"runs/esc50/transcribe_{tag}_refined/summary.json")
    run(f"every query type from the refined {tag} timelines",
        py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", f"runs/esc50/transcribe_{tag}_refined/pred_timelines.jsonl",
              "--bench", split, "--out", f"runs/esc50/{tag}_from_timeline_refined"],
        f"runs/esc50/{tag}_from_timeline_refined/summary.json")

# the same adapter asked the phase 2 way, for a like-for-like reference row
run("direct prompting with the transcription adapter (reference)",
    py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", ADAPTER, "--precision", PRECISION,
          "--bench", TEST, "--out", "runs/esc50/test_direct_transcribe_adapter"],
    "runs/esc50/test_direct_transcribe_adapter/summary.json")

# ---------------------------------------------------------------- 5b. arm E retest
# Phase 2's arm E (time symbols) lost to text digits, 0.194 vs 0.530, but it was
# trained on a T4: one 4-bit epoch, zeroed output rows, a delta on 302 rows,
# <t=none> unweighted. Here both representations get the same generated
# questions, the same epochs and the same card, and the symbols get TEMPO's
# configuration: mean-of-BPE rows on the head too, the full tables trainable,
# the empty answer capped in the data and down-weighted in the loss.
if ARM_E:
    qcommon = ["ctag.sft_data", "--task", "queries", "--timelines", f"{GEN}/timelines.jsonl",
               "--bench", f"{GEN}/benchmark.jsonl", "--plain-ratio", "0.6", "--max-empty-share", "0.15",
               "--max-examples", ARM_E_N]
    vcommon = ["ctag.sft_data", "--task", "queries", "--timelines", f"{BENCH}/timelines.jsonl",
               "--bench", VAL, "--plain-ratio", "0.6"]
    for tag, extra in (("text", []), ("tt", ["--time-tokens"])):
        run(f"SFT: question targets ({tag})", py + qcommon + extra + ["--out", f"{GEN}/sft_q_{tag}.jsonl"],
            f"{GEN}/sft_q_{tag}.jsonl")
        run(f"SFT: question targets for val ({tag})", py + vcommon + extra + ["--out", f"{BENCH}/sft_q_{tag}_val.jsonl"],
            f"{BENCH}/sft_q_{tag}_val.jsonl")
    tcommon = ["ctag.train_lora", "--precision", PRECISION, "--amp", AMP, "--batch-size", BS, "--grad-accum", ACCUM,
               "--lr", LR, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA, "--epochs", EPOCHS,
               "--max-seq-len", "3072", "--save-steps", SAVE_STEPS] + (["--train-encoder"] if TRAIN_ENC else [])
    run("train arm C at scale (text digits)",
        py + tcommon + ["--data", f"{GEN}/sft_q_text.jsonl", "--val", f"{BENCH}/sft_q_text_val.jsonl",
                        "--out", "runs/lora_q_text"], trained("runs/lora_q_text"))
    run("train arm E retest (time symbols)",
        py + tcommon + ["--data", f"{GEN}/sft_q_tt.jsonl", "--val", f"{BENCH}/sft_q_tt_val.jsonl",
                        "--time-tokens", "--head-init", "bpe", "--none-weight", "0.3", "--time-rows", ARM_E_ROWS,
                        "--out", "runs/lora_q_tt"], trained("runs/lora_q_tt"))
    for tag in ("text", "tt"):
        run(f"eval arm {'C' if tag == 'text' else 'E'} at scale on test",
            py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", f"runs/lora_q_{tag}",
                  "--precision", PRECISION, "--bench", TEST, "--out", f"runs/esc50/test_q_{tag}"],
            f"runs/esc50/test_q_{tag}/summary.json")

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
def num(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

if v:
    f1 = v.get("event_f1_pooled")
    print(f"val   event F1 (pooled) {num(f1)}  recall {num(v.get('event_recall_pooled'))}  "
          f"precision {num(v.get('event_precision_pooled'))}   GATE 0.95: {'PASS' if f1 and f1 >= 0.95 else 'not yet'}")
if t:
    print(f"test  event F1 (pooled) {num(t.get('event_f1_pooled'))}  under-report {num(t.get('under_report_rate'))}  "
          f"duration ratio {num(t.get('duration_ratio_median'))}")
    print("      lowest recall by sound:", ", ".join(f"{k} {x:.2f}" for k, x in sorted(t["recall_by_label"].items(), key=lambda kv: kv[1])[:4]))
r = load("runs/esc50/transcribe_test_refined/summary.json")
if r and "after" in r:
    print(f"test  refined: event F1 {num(r['before']['event_f1_pooled'])} -> {num(r['after']['event_f1_pooled'])}  "
          f"duration ratio {num(r['before']['duration_ratio_median'])} -> {num(r['after']['duration_ratio_median'])}  "
          f"edges moved {r['edges_moved']}")
types = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]
if q:
    print("test  f1@0.5 from the timeline: " + "  ".join(f"{ty} {q['by_type'][ty]['f1@0.5']:.3f}" for ty in types if ty in q["by_type"]))
for tag, name in (("text", "arm C at scale (text digits)"), ("tt", "arm E retest (time symbols)")):
    s = load(f"runs/esc50/test_q_{tag}/summary.json")
    if s:
        print(f"test  {name}: " + "  ".join(f"{ty} {s['by_type'][ty]['f1@0.5']:.3f}" for ty in types if ty in s["by_type"])
              + f"   empty-answer rate {s['by_type']['ALL']['false_rejection_rate']:.2f}")
persist()
print(f"\nresults copied to {RESULTS}; pull them with:  rsync -av <vm>:{RESULTS}/ runs_nebius_phase3/")
