"""The real-data stage (TEMPO's second stage) on a rented GPU: train the
timeline adapter on real DESED recordings mixed with varied generated clips,
then test on the 692-clip DESED public set and on the composed ESC-50 test.

Why: composed training reached 0.983 on composed audio and 0.02 on real audio
(docs/results_desed.md). The pilot retrain fixed the window-length and density
habits but could not teach sounds the ESC-50 bank lacks (speech, dishes,
frying, blender). Only real labelled audio can; DESED's validation set (911
clips with audio, data/desed/validation, no video shared with the public test
set) is the training half.

    # on the VM: scripts/nebius_sync.sh push-desed <ip> && scripts/nebius_sync.sh push-real <ip>
    cd ~/mtech-thesis-project && source .venv/bin/activate && export HF_HOME=$PWD/hf_cache
    CTAG_SMOKE=1 python nebius_real.py            # data files + a 20-step train, ~15 min
    nohup python nebius_real.py > logs/real.log 2>&1 &

Same discipline as nebius_phase3.py: a step is skipped when its output exists,
a training that was cut off resumes from its newest checkpoint, results are
copied to RESULTS/real after every step; rerun the same command after a
preemption.

Knobs (environment variables, all optional):
    CTAG_REAL_DIR     imported real set (timelines.jsonl + benchmark.jsonl)   (data/desed/validation)
    CTAG_REAL_VAL     share of real clips held out for validation             (0.1)
    CTAG_REAL_REPEAT  how many times the real part appears in the mix        (4)
    CTAG_SYNTH        1 to mix in varied generated clips, 0 = real only       (1)
    CTAG_GEN_DIR      the generated set (made here if missing)                (data/gen_varied)
    CTAG_GEN_N        generated clips if it has to be made                    (5000)
    CTAG_GEN_DUR      their length in seconds                                 (10)
    CTAG_EPOCHS / CTAG_BS / CTAG_ACCUM / CTAG_LR / CTAG_LORA_R / CTAG_PRECISION / CTAG_TRAIN_ENC
                      as in nebius_phase3.py                                  (3 / 2 / 4 / 1e-4 / 128 / bf16 / 1)
    CTAG_SAVE_STEPS   resumable checkpoint every N steps                     (100)
    CTAG_ADAPTER      where the adapter goes                                  (runs/lora_real)
    CTAG_REAL_OUT     evaluation outputs                                      (runs/real)
    CTAG_DESED_BENCH / CTAG_DESED_TIMELINES   the public test set   (data/desed/public/benchmark.jsonl, .../timelines.jsonl)
    CTAG_ESC50        composed test split for the forgetting check, "" skips (data/esc50/benchmark_test.jsonl)
    CTAG_STOP_SWEEP   P(stop) thresholds tried on the real val split          (0.02 0.05 0.1 0.2 0.3 0.5)
    CTAG_SMOKE        1 to stop after the 20-step smoke train
    CTAG_N            first N clips only (dry runs)
    CTAG_MODEL_ID     a local checkpoint instead of Qwen2.5-Omni-7B (CPU dry runs)
    CTAG_WORKERS, CTAG_RESULTS, CTAG_DRY_RUN   as in nebius_phase3.py

Budget on an L40S: ~8,300 examples x 3 epochs at batch 8 is ~3,100 steps, about
3.5 h at the pilot's 3.8 s/step, plus ~1.5 h of evaluation; a 5,000-clip
generated set takes ~20 min to make if it is not there already.
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

REAL = os.environ.get("CTAG_REAL_DIR", "data/desed/validation")
REAL_VAL = os.environ.get("CTAG_REAL_VAL", "0.1")
REAL_REPEAT = os.environ.get("CTAG_REAL_REPEAT", "4")
SYNTH = os.environ.get("CTAG_SYNTH", "1") == "1"
GEN = os.environ.get("CTAG_GEN_DIR", "data/gen_varied")
GEN_N = os.environ.get("CTAG_GEN_N", "5000")
GEN_DUR = os.environ.get("CTAG_GEN_DUR", "10")
EPOCHS = os.environ.get("CTAG_EPOCHS", "3")
BS = os.environ.get("CTAG_BS", "2")
ACCUM = os.environ.get("CTAG_ACCUM", "4")
LR = os.environ.get("CTAG_LR", "1e-4")
LORA_R = os.environ.get("CTAG_LORA_R", "128")
LORA_ALPHA = str(2 * int(LORA_R))
PRECISION = os.environ.get("CTAG_PRECISION", "bf16")
TRAIN_ENC = os.environ.get("CTAG_TRAIN_ENC", "1") == "1"
SAVE_STEPS = os.environ.get("CTAG_SAVE_STEPS", "100")
ADAPTER = os.environ.get("CTAG_ADAPTER", "runs/lora_real")
OUT = os.environ.get("CTAG_REAL_OUT", "runs/real")
PUB_BENCH = os.environ.get("CTAG_DESED_BENCH", "data/desed/public/benchmark.jsonl")
PUB_TL = os.environ.get("CTAG_DESED_TIMELINES", "data/desed/public/timelines.jsonl")
ESC50_TEST = os.environ.get("CTAG_ESC50", "data/esc50/benchmark_test.jsonl")
ESC50_TL = os.path.join(os.path.dirname(ESC50_TEST), "timelines.jsonl") if ESC50_TEST else ""
STOP_SWEEP = os.environ.get("CTAG_STOP_SWEEP", "0.02 0.05 0.1 0.2 0.3 0.5").split()
SMOKE_ONLY = os.environ.get("CTAG_SMOKE", "0") == "1"
N = os.environ.get("CTAG_N")
MODEL_ID = os.environ.get("CTAG_MODEL_ID")
WORKERS = os.environ.get("CTAG_WORKERS", str(max(1, (os.cpu_count() or 2) - 1)))
RESULTS = os.path.expanduser(os.environ.get("CTAG_RESULTS", "~/phase3_results"))
DRY_RUN = os.environ.get("CTAG_DRY_RUN", "0") == "1"
AMP = "bf16" if PRECISION == "bf16" else "none"
TYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]

REAL_TL = f"{REAL}/timelines.jsonl"
REAL_TRAIN, REAL_VALB = f"{REAL}/benchmark_train.jsonl", f"{REAL}/benchmark_val.jsonl"
SFT_REAL_TRAIN, SFT_REAL_VAL = f"{OUT}/sft_real_train.jsonl", f"{OUT}/sft_real_val.jsonl"
SFT_SYNTH, SFT_MIX = f"{GEN}/sft_transcribe.jsonl", f"{OUT}/sft_mix.jsonl"
GOLD_BOTH = f"{OUT}/gold_val_plus_public.jsonl"     # one gold file for the trailing check's two splits
open_vocab = ["--no-vocab-in-prompt"]                # the open prompt, training and evaluation alike
print(f"[real] real={REAL} val_share={REAL_VAL} repeat={REAL_REPEAT} synth={SYNTH} gen={GEN} epochs={EPOCHS} "
      f"bs={BS} accum={ACCUM} lr={LR} lora_r={LORA_R} precision={PRECISION} train_encoder={TRAIN_ENC} "
      f"adapter={ADAPTER} out={OUT}", flush=True)


def persist():
    if DRY_RUN:
        return
    os.makedirs(RESULTS, exist_ok=True)
    for src, name in ((OUT, "real"), (ADAPTER, os.path.basename(ADAPTER)), ("logs", "logs")):
        if os.path.isdir(src):
            dst = os.path.join(RESULTS, name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt", "checkpoint-*", "sft_*.jsonl"))


def trained(out):
    """Complete when train_lora wrote its marker (never the adapter file, which
    the epoch-end save writes mid-run)."""
    return [f"{out}/train_done.json"]


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
    print(f"--- {label}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {(time.time() - t0) / 60:.0f} min", flush=True)
    persist()
    return rc == 0


def check_inputs():
    if DRY_RUN:
        return
    missing = [p for p in (REAL_TL, f"{REAL}/benchmark.jsonl", PUB_BENCH, PUB_TL) if not os.path.exists(p)]
    if ESC50_TEST:
        missing += [p for p in (ESC50_TEST, ESC50_TL) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {missing}; on the Mac: scripts/nebius_sync.sh push-desed <ip> and push-real <ip>")
    for tl in (REAL_TL, PUB_TL):
        first = json.loads(open(tl, encoding="utf-8").readline())
        if not os.path.exists(first["audio"]):
            raise SystemExit(f"audio not found at {first['audio']} (from {tl}); run the sync first")
    real_ids = {json.loads(l)["clip_id"] for l in open(REAL_TL, encoding="utf-8")}
    pub_ids = {json.loads(l)["clip_id"] for l in open(PUB_TL, encoding="utf-8")}
    if real_ids & pub_ids:
        raise SystemExit(f"{len(real_ids & pub_ids)} clips are in both the real training set and the public test set")


py = [sys.executable, "-m"]
n_arg = ["--n", N] if N else []
mid = ["--model-id", MODEL_ID] if MODEL_ID else []
check_inputs()
ok = True

# ---------------------------------------------------------------- 1. data
# real clips: a held-out tenth for validation (by clip, stable hash); the test
# set is the public one, so frac-train + frac-val = 1 leaves this split's
# "test" file empty on purpose
ok &= run("split the real clips (train / val)",
          py + ["ctag.split", "--bench", f"{REAL}/benchmark.jsonl", "--out", REAL,
                "--frac-train", str(round(1 - float(REAL_VAL), 4)), "--frac-val", REAL_VAL], REAL_VALB)
ok &= run("SFT: real train clips, whole timeline, open prompt",
          py + ["ctag.sft_data", "--task", "transcribe"] + open_vocab + ["--timelines", REAL_TL, "--bench", REAL_TRAIN,
                "--out", SFT_REAL_TRAIN], SFT_REAL_TRAIN)
ok &= run("SFT: real val clips",
          py + ["ctag.sft_data", "--task", "transcribe"] + open_vocab + ["--timelines", REAL_TL, "--bench", REAL_VALB,
                "--out", SFT_REAL_VAL], SFT_REAL_VAL)
parts = [f"{SFT_REAL_TRAIN}:{REAL_REPEAT}"]
if SYNTH:
    ok &= run(f"generate {GEN_N} varied clips, all 50 classes",
              py + ["ctag.gen_train", "--source", "esc50", "--n-clips", GEN_N, "--varied", "--classes", "all",
                    "--duration", GEN_DUR, "--workers", WORKERS, "--out", GEN, "--esc50-root", "data/esc50_raw"],
              f"{GEN}/gen_stats.json")
    ok &= run("SFT: generated clips, open prompt",
              py + ["ctag.sft_data", "--task", "transcribe"] + open_vocab + ["--timelines", f"{GEN}/timelines.jsonl",
                    "--out", SFT_SYNTH], SFT_SYNTH)
    parts.append(f"{SFT_SYNTH}:1")
ok &= run("mix real x repeat + generated",
          py + ["ctag.sft_mix"] + sum((["--part", p] for p in parts), []) + ["--out", SFT_MIX], SFT_MIX)
if not ok:
    raise SystemExit("data preparation failed; see above")

# ---------------------------------------------------------------- 2. train
train_common = ["ctag.train_lora", "--data", SFT_MIX, "--val", SFT_REAL_VAL, "--precision", PRECISION, "--amp", AMP,
                "--batch-size", BS, "--grad-accum", ACCUM, "--lr", LR, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
                "--max-seq-len", "4096", "--save-steps", SAVE_STEPS] + (["--train-encoder"] if TRAIN_ENC else []) + mid
ok &= run("smoke train (20 steps)", py + train_common + ["--out", f"{ADAPTER}_smoke", "--max-steps", "20"],
          trained(f"{ADAPTER}_smoke"))
if not ok:
    raise SystemExit("the smoke test failed; fix before spending hours")
if SMOKE_ONLY and not DRY_RUN:
    raise SystemExit("CTAG_SMOKE=1: stopping after the smoke test")
ok &= run(f"train on the mix ({EPOCHS} epochs)", py + train_common + ["--out", ADAPTER, "--epochs", EPOCHS], trained(ADAPTER))
if not ok:
    raise SystemExit("training failed")

# ---------------------------------------------------------------- 3. evaluate
# (a) the held-out real clips: the number to watch, and where P(stop) is tuned
# (b) the public test set, every query type, then trailing-checked
# (c) the composed test split: how much composed accuracy the real stage costs
def transcribe(label, bench, timelines, out):
    return run(label, py + ["ctag.run_transcribe", "--model", "qwen2.5-omni"] + open_vocab +
               ["--adapter", ADAPTER, "--precision", PRECISION, "--stop-probs", "--bench", bench, "--timelines", timelines,
                "--out", out] + n_arg + mid, f"{out}/summary.json")


def query_types(label, pred, bench, out):
    return run(label, py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", f"{pred}/pred_timelines.jsonl",
                            "--bench", bench, "--out", out] + n_arg, f"{out}/summary.json")


transcribe("transcribe the held-out real clips", REAL_VALB, REAL_TL, f"{OUT}/transcribe_val")
query_types("every query type, held-out real clips", f"{OUT}/transcribe_val", REAL_VALB, f"{OUT}/val_from_timeline")
transcribe("transcribe DESED public (the test)", PUB_BENCH, PUB_TL, f"{OUT}/transcribe_public")
query_types("every query type, DESED public", f"{OUT}/transcribe_public", PUB_BENCH, f"{OUT}/public_from_timeline")
if not DRY_RUN and not os.path.exists(GOLD_BOTH) and os.path.exists(f"{OUT}/transcribe_public/summary.json"):
    with open(GOLD_BOTH, "w", encoding="utf-8") as f:
        for p in (REAL_TL, PUB_TL):
            f.write(open(p, encoding="utf-8").read().rstrip("\n") + "\n")
run("trailing check: P(stop) threshold from the real val clips, applied to the public test",
    py + ["ctag.trailing", "--rule", "stop", "--pred-timelines", f"{OUT}/transcribe_public/pred_timelines.jsonl",
          "--tune-on", f"{OUT}/transcribe_val/pred_timelines.jsonl", "--timelines", GOLD_BOTH, "--sweep"] + STOP_SWEEP +
    ["--out", f"{OUT}/transcribe_public_trailing"], f"{OUT}/transcribe_public_trailing/summary.json")
query_types("every query type, DESED public after the trailing check", f"{OUT}/transcribe_public_trailing", PUB_BENCH,
            f"{OUT}/public_from_timeline_trailing")
if ESC50_TEST:
    transcribe("transcribe the composed ESC-50 test (forgetting check)", ESC50_TEST, ESC50_TL, f"{OUT}/transcribe_esc50_test")
    query_types("every query type, composed ESC-50 test", f"{OUT}/transcribe_esc50_test", ESC50_TEST, f"{OUT}/esc50_test_from_timeline")


# ---------------------------------------------------------------- 4. the table
def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def num(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


if not DRY_RUN:
    print("\n================ real-data stage ================")
    print("reference (docs/results_desed.md): DESED public label-aware event F1 0.023 (old adapter) / 0.039 (pilot v2); "
          "label-agnostic 0.148 / 0.282; every-type ALL 0.020 / 0.028; composed test ALL 0.983 / 0.718")
    for tag, name in (("transcribe_val", "held-out real clips"), ("transcribe_public", "DESED public"),
                      ("transcribe_esc50_test", "composed ESC-50 test")):
        t = load(f"{OUT}/{tag}/summary.json")
        if t:
            print(f"{name}: event F1 label-aware {num(t.get('event_f1_pooled'))}  label-agnostic {num(t.get('f1_any_label'))}  "
                  f"recall {num(t.get('event_recall_pooled'))}  precision {num(t.get('event_precision_pooled'))}  "
                  f"duration ratio {num(t.get('duration_ratio_median'))}  parse fail {num(t.get('parse_fail_rate'))}")
            if t.get("recall_by_label"):
                print("      recall by sound: " + ", ".join(f"{k} {x:.2f}" for k, x in sorted(t["recall_by_label"].items(),
                                                                                              key=lambda kv: kv[1])))
    tr = load(f"{OUT}/transcribe_public_trailing/summary.json")
    if tr and tr.get("after"):
        c = tr["confusion"]
        print(f"trailing check (thr {tr.get('thr')}): DESED event F1 {num(tr['before']['event_f1_pooled'])} -> "
              f"{num(tr['after']['event_f1_pooled'])}  removed spurious {c['removed_spurious']} correct {c['removed_correct']}")
    for tag, name in (("val_from_timeline", "held-out real clips, every type"), ("public_from_timeline", "DESED public, every type"),
                      ("public_from_timeline_trailing", "DESED public, every type, trailing-checked"),
                      ("esc50_test_from_timeline", "composed ESC-50 test, every type")):
        s = load(f"{OUT}/{tag}/summary.json")
        if s:
            bt = s["by_type"]
            print(f"{name}: " + "  ".join(f"{ty} {num(bt[ty].get('f1@0.5'))}" for ty in TYPES if ty in bt)
                  + (f"   ABSENT rejection f1 {num(bt['ABSENT'].get('rejection_f1'))}" if "ABSENT" in bt else ""))
    persist()
    print(f"\nresults copied to {RESULTS}/real and {RESULTS}/{os.path.basename(ADAPTER)}; pull with scripts/nebius_sync.sh pull <ip>")
