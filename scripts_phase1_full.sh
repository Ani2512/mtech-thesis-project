#!/usr/bin/env bash
# One self-contained phase 1 run. Safe to launch with nohup and forget:
# it survives the browser disconnecting, logs everything, and skips work
# that is already done so it can be re-launched after a crash.
#
#   nohup bash scripts_phase1_full.sh > logs/phase1_full.log 2>&1 &
set -u
cd /content/mtech-thesis-project 2>/dev/null || {
  cd /content && rm -rf mtech-thesis-project
  git clone -q https://github.com/Ani2512/mtech-thesis-project.git
  cd mtech-thesis-project
}
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "refreshing code"
git fetch -q origin && git reset -q --hard origin/main

say "installing deps"
pip install -q scipy soundfile librosa pyyaml pytest \
    "transformers>=4.52" qwen-omni-utils accelerate bitsandbytes 2>&1 | tail -2

say "self-test"
python -m pytest tests -q -p no:warnings 2>&1 | tail -2

if [ ! -f data/esc50/benchmark.jsonl ]; then
  say "building ESC-50 benchmark"
  python -m ctag.build_benchmark --source esc50 --n-clips 300 --p-overlap 0.45 \
         --out data/esc50 --esc50-root data/esc50_raw
else
  say "benchmark already present"
fi

for m in oracle ignore_condition first_only; do
  if [ ! -f "runs/esc50/mock_$m/summary.json" ]; then
    say "mock:$m"
    python -m ctag.run_zeroshot --model "mock:$m" --bench data/esc50/benchmark.jsonl \
           --out "runs/esc50/mock_$m" > /dev/null
  fi
done
say "mocks ready"

# Qwen2.5-Omni first: it is the viable backbone, so its numbers matter most.
for spec in "qwen2.5-omni:qwen25_omni" "qwen2-audio:qwen2_audio"; do
  model="${spec%%:*}"; out="${spec##*:}"
  if [ -f "runs/esc50/${out}_n1200/summary.json" ]; then
    say "$model already done at n=1200, skipping"; continue
  fi
  say "starting $model at n=1200 (about 80 minutes)"
  python -m ctag.run_zeroshot --model "$model" --bench data/esc50/benchmark.jsonl \
         --n 1200 --out "runs/esc50/${out}_n1200" 2>&1 | grep -v "it/s\]" | tail -40
  say "$model finished"
done

say "ALL DONE"
