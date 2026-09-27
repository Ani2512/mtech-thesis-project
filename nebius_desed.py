"""Real recordings: the phase 3 adapters on the DESED public evaluation set.

Everything measured before this is composed ESC-50 audio. This runner asks the
two trained adapters the same questions on 692 real 10 s domestic recordings
with strong labels (docs/real_recordings.md), in the order the report needs:

    1. arm C at scale (runs/lora_q_text) asked directly            ~65 min on an L40S
    2. arm F at scale: that adapter inside the decomposition       ~25 min
    3. the timeline adapter (runs/lora_transcribe) transcribes every
       clip once, with stop probabilities; every query type from the
       timelines, then again after the trailing check at the ESC-50
       threshold                                                    ~90 min
    4. the untrained model asked directly on the first N questions ~50 min
       (the "before training" row on real audio; CTAG_DESED_ZEROSHOT_N=0 skips)

Same discipline as nebius_phase3.py: a step is skipped when its summary.json
exists, results are copied to RESULTS/desed after every step, rerun the same
command after an interruption.

    # on the VM, after scripts/nebius_sync.sh push <ip> and push-desed <ip>
    cd ~/mtech-thesis-project && source .venv/bin/activate && export HF_HOME=$PWD/hf_cache
    nohup python nebius_desed.py > logs/desed.log 2>&1 &

Knobs (environment variables, all optional):
    CTAG_DESED_BENCH        benchmark.jsonl to answer     (data/desed/public/benchmark.jsonl;
                            benchmark_verified.jsonl after the hand review)
    CTAG_DESED_TIMELINES    gold timelines for scoring    (data/desed/public/timelines.jsonl)
    CTAG_DESED_OUT          output directory              (runs/desed)
    CTAG_Q_ADAPTER          question-trained adapter      (runs/lora_q_text)
    CTAG_T_ADAPTER          timeline adapter              (runs/lora_transcribe)
    CTAG_DESED_ZEROSHOT_N   untrained reference, questions (700; 0 skips)
    CTAG_DESED_TRAIL_THR    trailing-check threshold      (0.02, tuned on ESC-50 val)
    CTAG_PRECISION          bf16 | fp16 | 8bit | 4bit     (bf16)
    CTAG_N                  first N clips / questions only (dry runs)
    CTAG_MODEL_ID           a local checkpoint instead of Qwen2.5-Omni-7B (the CPU
                            dry run's tiny model; the decomposition step is skipped,
                            ctag.run_agent has no override)
    CTAG_RESULTS            where results are copied      (~/phase3_results)
    CTAG_DRY_RUN=1          print the commands, run nothing
"""
import json
import os
import shutil
import subprocess
import sys
import time

BENCH = os.environ.get("CTAG_DESED_BENCH", "data/desed/public/benchmark.jsonl")
TIMELINES = os.environ.get("CTAG_DESED_TIMELINES", "data/desed/public/timelines.jsonl")
OUT = os.environ.get("CTAG_DESED_OUT", "runs/desed")
Q_ADAPTER = os.environ.get("CTAG_Q_ADAPTER", "runs/lora_q_text")
T_ADAPTER = os.environ.get("CTAG_T_ADAPTER", "runs/lora_transcribe")
ZEROSHOT_N = os.environ.get("CTAG_DESED_ZEROSHOT_N", "700")
TRAIL_THR = os.environ.get("CTAG_DESED_TRAIL_THR", "0.02")
PRECISION = os.environ.get("CTAG_PRECISION", "bf16")
N = os.environ.get("CTAG_N")
MODEL_ID = os.environ.get("CTAG_MODEL_ID")
RESULTS = os.path.expanduser(os.environ.get("CTAG_RESULTS", "~/phase3_results"))
DRY_RUN = os.environ.get("CTAG_DRY_RUN", "0") == "1"
TYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]

print(f"[desed] bench={BENCH} out={OUT} q_adapter={Q_ADAPTER} t_adapter={T_ADAPTER} precision={PRECISION} "
      f"zeroshot_n={ZEROSHOT_N} trail_thr={TRAIL_THR} n={N} model_id={MODEL_ID}", flush=True)


def persist():
    if DRY_RUN:
        return
    os.makedirs(RESULTS, exist_ok=True)
    for src, name in ((OUT, "desed"), ("logs", "logs")):
        if os.path.isdir(src):
            dst = os.path.join(RESULTS, name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)


def run(label, cmd, produces):
    if DRY_RUN:
        print("DRY " + json.dumps(cmd), flush=True)
        return True
    if os.path.exists(produces):
        print(f"\n=== {label}: already done ===", flush=True)
        return True
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    mins = (time.time() - t0) / 60
    print(f"--- {label}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {mins:.0f} min", flush=True)
    persist()
    return rc == 0


def check_inputs():
    if DRY_RUN:
        return
    missing = [p for p in (BENCH, TIMELINES) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {missing}; on the Mac: scripts/nebius_sync.sh push-desed <ip>")
    first = json.loads(open(BENCH, encoding="utf-8").readline())
    if not os.path.exists(first["audio"]):
        raise SystemExit(f"audio not found at {first['audio']}; scripts/nebius_sync.sh push-desed <ip> copies the wav files")
    for name, adapter in (("question", Q_ADAPTER), ("timeline", T_ADAPTER)):
        if not os.path.exists(os.path.join(adapter, "adapter_config.json")):
            print(f"[desed] WARNING: the {name} adapter is not at {adapter}; its steps will fail "
                  f"(scripts/nebius_sync.sh push <ip>, or train it first)", flush=True)


py = [sys.executable, "-m"]
n_arg = ["--n", N] if N else []
mid = ["--model-id", MODEL_ID] if MODEL_ID else []
check_inputs()
os.makedirs("logs", exist_ok=True)

# 1. the question-trained adapter, asked directly (the report's Method C at scale)
run("arm C at scale asked directly on DESED public",
    py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", Q_ADAPTER, "--precision", PRECISION,
          "--bench", BENCH, "--out", f"{OUT}/public_q_text"] + n_arg + mid,
    f"{OUT}/public_q_text/summary.json")

# 2. the same adapter as the sound locator inside the decomposition (Method F at scale)
if MODEL_ID:
    print("\n=== arm F at scale on DESED public: skipped, ctag.run_agent has no --model-id ===", flush=True)
else:
    run("arm F at scale on DESED public (arm C adapter inside the decomposition)",
        py + ["ctag.run_agent", "--grounder", "qwen2.5-omni", "--adapter", Q_ADAPTER, "--precision", PRECISION,
              "--bench", BENCH, "--out", f"{OUT}/public_f_text"] + n_arg,
        f"{OUT}/public_f_text/summary.json")

# 3. the timeline adapter: one transcription per clip, every type from it, trailing check
run("transcribe DESED public with the timeline adapter (stop probabilities on)",
    py + ["ctag.run_transcribe", "--model", "qwen2.5-omni", "--adapter", T_ADAPTER, "--precision", PRECISION, "--stop-probs",
          "--bench", BENCH, "--timelines", TIMELINES, "--out", f"{OUT}/transcribe_public"] + n_arg + mid,
    f"{OUT}/transcribe_public/summary.json")
run("every query type from the DESED timelines",
    py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", f"{OUT}/transcribe_public/pred_timelines.jsonl",
          "--bench", BENCH, "--out", f"{OUT}/public_from_timeline"] + n_arg,
    f"{OUT}/public_from_timeline/summary.json")
run(f"trailing check at the ESC-50 threshold ({TRAIL_THR})",
    py + ["ctag.trailing", "--rule", "stop", "--thr", TRAIL_THR, "--pred-timelines", f"{OUT}/transcribe_public/pred_timelines.jsonl",
          "--timelines", TIMELINES, "--out", f"{OUT}/transcribe_public_trailing"],
    f"{OUT}/transcribe_public_trailing/summary.json")
run("every query type from the trailing-checked DESED timelines",
    py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", f"{OUT}/transcribe_public_trailing/pred_timelines.jsonl",
          "--bench", BENCH, "--out", f"{OUT}/public_from_timeline_trailing"] + n_arg,
    f"{OUT}/public_from_timeline_trailing/summary.json")

# 4. the untrained model, asked directly, on the first N questions (phase 1's row on real audio)
if ZEROSHOT_N != "0":
    zn = min(int(ZEROSHOT_N), int(N)) if N else int(ZEROSHOT_N)
    run(f"untrained model asked directly on DESED public (first {zn} questions)",
        py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--precision", PRECISION,
              "--bench", BENCH, "--n", str(zn), "--out", f"{OUT}/public_zeroshot"] + mid,
        f"{OUT}/public_zeroshot/summary.json")


# ---------------------------------------------------------------- the table
def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def num(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def empty_answer_counts(pred_path):
    """The per-type f1 in summary.json is over answerable questions (the phase 2
    convention). Real recordings have many questions whose answer is empty, so
    count them too: gold-empty questions answered empty, and the mean f1@0.5
    over every typed question (ABSENT excluded, as in the composed tables)."""
    try:
        rows = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    except Exception:
        return None
    typed = [r for r in rows if r.get("qtype") != "ABSENT" and "f1@0.5" in r]
    absent = [r for r in rows if r.get("qtype") == "ABSENT"]
    empty = [r for r in typed if not r.get("answer")]
    return {"typed": len(typed), "mean_f1_all_typed": sum(r["f1@0.5"] for r in typed) / max(1, len(typed)),
            "gold_empty": len(empty), "gold_empty_answered_empty": sum(1 for r in empty if not r.get("pred")),
            "absent": len(absent), "absent_answered_empty": sum(1 for r in absent if not r.get("pred"))}


if not DRY_RUN:
    print("\n================ DESED public evaluation ================")
    t = load(f"{OUT}/transcribe_public/summary.json")
    if t:
        print(f"timeline adapter: event F1 (pooled) {num(t.get('event_f1_pooled'))}  recall {num(t.get('event_recall_pooled'))}  "
              f"precision {num(t.get('event_precision_pooled'))}  under-report {num(t.get('under_report_rate'))}  "
              f"duration ratio {num(t.get('duration_ratio_median'))}  parse fail {num(t.get('parse_fail_rate'))}")
        if t.get("recall_by_label"):
            print("      recall by sound: " + ", ".join(f"{k} {x:.2f}" for k, x in sorted(t["recall_by_label"].items(), key=lambda kv: kv[1])))
    tr = load(f"{OUT}/transcribe_public_trailing/summary.json")
    if tr and tr.get("after"):
        c = tr["confusion"]
        print(f"trailing check (thr {tr['thr']}): event F1 {num(tr['before']['event_f1_pooled'])} -> {num(tr['after']['event_f1_pooled'])}  "
              f"removed spurious {c['removed_spurious']} correct {c['removed_correct']}")
    for tag, name in (("public_zeroshot", f"untrained, asked directly (first {ZEROSHOT_N} q)"),
                      ("public_q_text", "arm C at scale, asked directly"),
                      ("public_f_text", "arm F at scale, decomposition"),
                      ("public_from_timeline", "timeline route"),
                      ("public_from_timeline_trailing", "timeline route, trailing-checked")):
        s = load(f"{OUT}/{tag}/summary.json")
        if not s:
            continue
        bt = s["by_type"]
        line = f"{name}: " + "  ".join(f"{ty} {num(bt[ty].get('f1@0.5'))}" for ty in TYPES if ty in bt)
        if "ABSENT" in bt:
            line += f"   ABSENT rejection f1 {num(bt['ABSENT'].get('rejection_f1'))}"
        print(line)
        e = empty_answer_counts(f"{OUT}/{tag}/predictions.jsonl")
        if e:
            print(f"      all {e['typed']} typed questions: mean f1@0.5 {num(e['mean_f1_all_typed'])};  gold-empty answered empty "
                  f"{e['gold_empty_answered_empty']}/{e['gold_empty']};  absent-sound answered empty {e['absent_answered_empty']}/{e['absent']}")
    persist()
    print(f"\nresults copied to {RESULTS}/desed; pull them with:  scripts/nebius_sync.sh pull <ip>")
