#!/usr/bin/env bash
# Move phase 3 results between the Mac and a Nebius VM.
#
#   scripts/nebius_sync.sh push <ip>   # Mac -> VM: the transcription adapter and the finished
#                                      # evaluations, so the runner skips the 10 h transcription
#                                      # training and its evaluation on a fresh (preemptible) VM
#   scripts/nebius_sync.sh pull <ip>   # VM -> Mac: everything in ~/phase3_results
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
  pull)
    mkdir -p runs_nebius_phase3
    rsync -az "${EXCL[@]}" -e "$SSH" "ubuntu@$ip:phase3_results/" runs_nebius_phase3/
    du -sh runs_nebius_phase3; ls runs_nebius_phase3/esc50
    ;;
  *) echo "usage: $0 push|pull <ip>" >&2; exit 2;;
esac
