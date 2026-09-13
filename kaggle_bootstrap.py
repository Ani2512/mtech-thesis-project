"""One self-contained cell. Paste into Kaggle and run; safe to re-run any time.

Updates the code without deleting anything, rebuilds only what is missing, and
finishes with the trainer smoke test. It has no ordering dependency on other
cells, which is what kept going wrong.

    !curl -sL https://raw.githubusercontent.com/Ani2512/mtech-project/compositional-temporal-grounding/research_project/kaggle_bootstrap.py -o /kaggle/working/bootstrap.py
    %run /kaggle/working/bootstrap.py
"""
import os
import subprocess
import sys
import time

# Clone OUTSIDE /kaggle/working. Kaggle publishes at most 500 output files, and
# the clone (.git alone is hundreds) consumed the cap before results/ was
# reached -- v2 completed and its summaries never made it into the output.
REPO = "/kaggle/temp/mtech-project"
WORK = f"{REPO}/research_project"
BRANCH = "compositional-temporal-grounding"
CACHE = "/kaggle/temp/hf"

os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.makedirs(CACHE, exist_ok=True)


def run(cmd, **kw):
    print(f"$ {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}", flush=True)
    return subprocess.run(cmd, **kw)


def step(label):
    print(f"\n=== {label} ===", flush=True)


# ---------------------------------------------------------------- code
step("code")
os.chdir("/kaggle/working")
if os.path.isdir(f"{REPO}/.git"):
    # data/ and runs/ are gitignored, so a hard reset refreshes tracked code and
    # leaves the dataset, the benchmark and finished runs untouched.
    run(["git", "-C", REPO, "fetch", "-q", "origin"], check=True)
    run(["git", "-C", REPO, "reset", "-q", "--hard", f"origin/{BRANCH}"], check=True)
    print("updated in place; data preserved")
else:
    run(["git", "clone", "-q", "-b", BRANCH,
         "https://github.com/Ani2512/mtech-project.git", REPO], check=True)
    print("cloned fresh")
os.chdir(WORK)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())

# ---------------------------------------------------------------- deps
step("dependencies")
run([sys.executable, "-m", "pip", "install", "-q", "scipy", "soundfile", "librosa",
     "pyyaml", "pytest", "transformers>=4.52", "qwen-omni-utils", "accelerate",
     "bitsandbytes", "peft"], check=False)
# The Kaggle image ships torchao 0.10; peft's LoRA dispatcher checks the
# version whenever a target module is not a bitsandbytes layer and raises
# "Found an incompatible version of torchao ... only versions above 0.16.0 are
# supported". That killed the arm C eval in v4 after a 3-hour train. Nothing
# here uses torchao, and with it absent the dispatcher simply moves on.
run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)

step("self-test")
if run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:warnings"]).returncode:
    raise SystemExit("tests failed; stopping before anything expensive")

# ---------------------------------------------------------------- data
step("benchmark")
if os.path.exists("data/esc50/benchmark.jsonl"):
    print("already built")
else:
    run([sys.executable, "-m", "ctag.build_benchmark", "--source", "esc50",
         "--n-clips", "300", "--p-overlap", "0.45", "--out", "data/esc50",
         "--esc50-root", "data/esc50_raw"], check=True)

step("splits")
if os.path.exists("data/esc50/benchmark_train.jsonl"):
    print("already split")
else:
    run([sys.executable, "-m", "ctag.split", "--bench", "data/esc50/benchmark.jsonl",
         "--out", "data/esc50"], check=True)

step("training data")
for split in ("train", "val"):
    for tt, suffix in ((False, ""), (True, "_tt")):
        out = f"data/esc50/sft_{split}{suffix}.jsonl"
        if os.path.exists(out):
            continue
        cmd = [sys.executable, "-m", "ctag.sft_data",
               "--bench", f"data/esc50/benchmark_{split}.jsonl",
               "--timelines", "data/esc50/timelines.jsonl",
               "--out", out, "--plain-ratio", "0.6"]
        if tt:
            cmd.append("--time-tokens")
        run(cmd, check=True)
for f in sorted(os.listdir("data/esc50")):
    if f.startswith("sft_"):
        print(" ", f, sum(1 for _ in open(f"data/esc50/{f}")), "examples")

# ---------------------------------------------------------------- smoke test
step("trainer smoke test: find a precision that trains AND is fast")

# Measured on a T4 with this model:
#   emulated bf16  18.3 s/step, trains correctly
#   fp16            3.5 s/step, overflows -> nan grad_norm, loss collapses to 0
# So neither is automatically right. Try the cheap-and-stable option first and
# fall back. Weights are cached, so each attempt costs about 90 seconds.
attempts = [
    ("none", "fp32 gradients, no autocast"),
    ("bf16", "correct but emulated on pre-Ampere, ~5x slower"),
]
chosen = None
for amp, why in attempts:
    print(f"\n--- trying --amp {amp}  ({why})", flush=True)
    t0 = time.time()
    rc = run([sys.executable, "-m", "ctag.train_lora",
              "--data", "data/esc50/sft_train.jsonl", "--out", f"/kaggle/temp/smoke_{amp}",
              "--max-steps", "20", "--grad-accum", "1", "--amp", amp]).returncode
    secs = (time.time() - t0)
    if rc == 0:
        chosen = amp
        print(f"--- --amp {amp} TRAINED, {secs:.0f}s for 20 steps "
              f"({secs / 20:.1f} s/step)", flush=True)
        break
    print(f"--- --amp {amp} failed (exit {rc})", flush=True)

print("\n" + "=" * 70)
if chosen:
    per_step = secs / 20
    epoch_h = 2300 * per_step / 3600
    print(f"USE --amp {chosen}.  {per_step:.1f} s/example, so one epoch over 2300 "
          f"examples is about {epoch_h:.1f} h.")
    print(f"Run phase 2 with:  CTAG_AMP={chosen} %run /kaggle/working/phase2.py")
    if epoch_h > 5:
        print("WARNING: that is slow enough that two arms will not fit one session.")
else:
    print("No precision setting trained. Send me the output above.")
print("=" * 70)
