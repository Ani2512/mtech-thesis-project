#!/usr/bin/env bash
# One-shot setup for a fresh Nebius L40S VM (Ubuntu + NVIDIA drivers image).
# Idempotent: rerunning skips what is already done. Mirrors docs/nebius.md.
#
#   ssh -i ~/.ssh/id_ed25519_nebius ubuntu@<ip> \
#     'curl -fsSL https://raw.githubusercontent.com/Ani2512/mtech-thesis-project/main/scripts/nebius_bootstrap.sh | bash'
#
# Then:  cd ~/mtech-thesis-project && source .venv/bin/activate && export HF_HOME=$PWD/hf_cache
#        CTAG_SMOKE=1 python nebius_phase3.py                       # ~30 min
#        nohup python nebius_phase3.py > logs/phase3.log 2>&1 &     # full run
set -euo pipefail

REPO=https://github.com/Ani2512/mtech-thesis-project.git
DIR=$HOME/mtech-thesis-project

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "no GPU visible; pick the Ubuntu image with NVIDIA drivers" >&2; exit 1; }

sudo apt-get update -qq
sudo apt-get install -y -qq git ffmpeg libsndfile1 python3-venv rsync tmux

if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q --ff-only; else git clone -q "$REPO" "$DIR"; fi
cd "$DIR"

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
python -c "import torch, torchvision" 2>/dev/null || \
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt "transformers>=4.52" accelerate peft librosa audioread qwen-omni-utils
pip uninstall -y -q torchao 2>/dev/null || true   # peft's dispatcher trips on old torchao builds

export HF_HOME=$PWD/hf_cache
mkdir -p logs "$HF_HOME"
grep -q HF_HOME ~/.bashrc || echo "export HF_HOME=$PWD/hf_cache" >> ~/.bashrc

python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest tests -q
python scripts/dry_run_gpu_paths.py

echo
echo "bootstrap done. Next:"
echo "  cd $DIR && source .venv/bin/activate && export HF_HOME=\$PWD/hf_cache"
echo "  CTAG_SMOKE=1 python nebius_phase3.py"
echo "  nohup python nebius_phase3.py > logs/phase3.log 2>&1 &"
