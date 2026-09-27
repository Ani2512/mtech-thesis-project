#!/usr/bin/env bash
# Move phase 3 results between the Mac and a Nebius VM.
#
#   scripts/nebius_sync.sh push <ip>   # Mac -> VM: the transcription adapter and the finished
#                                      # evaluations, so the runner skips the 10 h transcription
#                                      # training and its evaluation on a fresh (preemptible) VM
#   scripts/nebius_sync.sh pull <ip>   # VM -> Mac: everything in ~/phase3_results
#   scripts/nebius_sync.sh push-desed <ip>  # Mac -> VM: the DESED public-eval questions,
#                                      # timelines and 692 wav files (1.1 GB) for nebius_desed.py
#
# Local side: runs_nebius_phase3/ in the repo (gitignored). Checkpoints and optimiser
# state are never copied.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?push|pull}; ip=${2:?vm ip}
SSH="ssh -i $HOME/.ssh/id_ed25519_nebius"
EXCL=(--exclude 'checkpoint-*' --exclude 'optimizer.pt')
case "$mode" in
  push)
    for d in lora_transcribe esc50; do
      [ -d "runs_nebius_phase3/$d" ] || { echo "missing runs_nebius_phase3/$d (pull first)" >&2; exit 1; }
    done
    $SSH "ubuntu@$ip" 'mkdir -p ~/mtech-thesis-project/runs'
    rsync -az "${EXCL[@]}" -e "$SSH" runs_nebius_phase3/lora_transcribe runs_nebius_phase3/esc50 \
      "ubuntu@$ip:mtech-thesis-project/runs/"
    $SSH "ubuntu@$ip" 'ls ~/mtech-thesis-project/runs/lora_transcribe/train_done.json ~/mtech-thesis-project/runs/lora_transcribe/tokenizer_config.json 2>/dev/null | head -1; ls ~/mtech-thesis-project/runs/esc50'
    ;;
  push-desed)
    for f in data/desed/public/benchmark.jsonl data/desed/public/timelines.jsonl data/desed/audio/dataset/audio/eval/public; do
      [ -e "$f" ] || { echo "missing $f (docs/real_recordings.md: import + queries first)" >&2; exit 1; }
    done
    $SSH "ubuntu@$ip" 'mkdir -p ~/mtech-thesis-project/data/desed/public ~/mtech-thesis-project/data/desed/audio/dataset/audio/eval'
    rsync -az -e "$SSH" data/desed/public/benchmark.jsonl data/desed/public/timelines.jsonl \
      "ubuntu@$ip:mtech-thesis-project/data/desed/public/"
    [ -f data/desed/public/benchmark_verified.jsonl ] && rsync -az -e "$SSH" data/desed/public/benchmark_verified.jsonl \
      "ubuntu@$ip:mtech-thesis-project/data/desed/public/"
    rsync -az -e "$SSH" data/desed/audio/dataset/audio/eval/public \
      "ubuntu@$ip:mtech-thesis-project/data/desed/audio/dataset/audio/eval/"
    $SSH "ubuntu@$ip" 'ls ~/mtech-thesis-project/data/desed/audio/dataset/audio/eval/public | wc -l; wc -l ~/mtech-thesis-project/data/desed/public/*.jsonl'
    ;;
  pull)
    mkdir -p runs_nebius_phase3
    rsync -az "${EXCL[@]}" -e "$SSH" "ubuntu@$ip:phase3_results/" runs_nebius_phase3/
    du -sh runs_nebius_phase3; ls runs_nebius_phase3/esc50
    ;;
  *) echo "usage: $0 push|pull|push-desed <ip>" >&2; exit 2;;
esac
