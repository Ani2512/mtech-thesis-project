# Phase 3 on a rented GPU

Decided 2026-09-12 (GPU) and 2026-09-14 (what to run). Kaggle's 30 h/week of
T4 time cannot hold a 20,000-clip, 3-epoch run in full precision; a single
on-demand L40S does it in an afternoon.

## Machine

| | |
|---|---|
| GPU | **L40S 48 GB** (Ada, sm_89): fp16/bf16 weights fit, native bf16, no 4-bit. On-demand, about $1.55/h |
| Faster | H100 80 GB, about $3.85/h, roughly 2x the step rate |
| Avoid | Blackwell cards (sm_120) until the CUDA/torch stack is boring; preemptible VMs unless the disk is persistent |
| Disk | 100 GB: model cache ~20 GB, 20k wav clips ~13 GB (half as FLAC), adapters and runs < 5 GB |
| CPU | 8+ cores helps the generator (`--workers`) |

## Setup (once)

```bash
sudo apt-get install -y git ffmpeg libsndfile1
git clone https://github.com/Ani2512/mtech-thesis-project.git && cd mtech-thesis-project
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # torchvision: the Omni processor imports it
pip install -r requirements.txt "transformers>=4.52" accelerate peft librosa audioread qwen-omni-utils
# audioread: qwen-omni-utils imports it for audio loading but does not declare it (found by the dry run)
pip uninstall -y torchao 2>/dev/null   # peft's dispatcher trips on old torchao builds
export HF_HOME=$PWD/hf_cache            # keep the 20 GB model out of $HOME quotas
python -m pytest tests -q               # 80+ tests, CPU only, ~10 s
python scripts/dry_run_gpu_paths.py     # every GPU code path on a tiny random model, CPU, a few minutes
```

ESC-50 downloads once (~600 MB) on the first `build_benchmark`/`gen_train`.

## Run

```bash
mkdir -p logs
CTAG_SMOKE=1 python nebius_phase3.py          # data + a 20-step smoke train, ~30 min incl. generation
nohup python nebius_phase3.py > logs/phase3.log 2>&1 &   # the full run, resumes if restarted
tail -f logs/phase3.log
```

Every step is skipped once its output exists and results are copied to
`~/phase3_results` after each step, so a stopped VM hands back what finished.
Pull them: `rsync -av <vm>:~/phase3_results/ runs_nebius_phase3/` (gitignored).

What the run produces and where the gate is read:

| output | meaning |
|---|---|
| `transcribe_val/summary.json` `event_f1_pooled` | **the 0.95 gate**; also recall per sound |
| `transcribe_test/summary.json` | same on the test clips; `duration_ratio_median` shows whether the fixed-duration habit survived |
| `test_from_timeline/summary.json` | every query type computed from the predicted timelines, the number to compare with phase 2's 0.645 |
| `test_direct_transcribe_adapter/summary.json` | the same adapter asked per question, a like-for-like reference |

## Cost

At 20,000 clips x 3 epochs with batch 2 x accumulation 4, expect roughly 2.5-3.5 h
of training on an L40S plus ~30 min of generation and ~40 min of evaluation:
about $6-8. The arm E retest and a second epoch count are extra runs of the same
size. Stop the VM when `phase3.log` ends with the results table; the disk can be
kept for the next run.

## Arm E retest (in the runner, `CTAG_ARM_E=1`, default on)

Phase 2's time-symbol arm lost to text digits (0.194 vs 0.530) under T4
constraints. The runner retrains both on the same generated questions for the
same epochs, and gives the symbols TEMPO's own configuration:
`--head-init bpe` (output rows start from the mean of their digit pieces, not
zero), `--time-rows full` (both tables trainable; ~16 GB extra, fine on 48 GB),
`--none-weight 0.3` and `--max-empty-share 0.15` (the empty answer can no longer
be won by default). The table at the end prints both arms side by side with
their empty-answer rate. Roughly the cost of the transcription run again.

## Not in this runner yet

- Boundary refinement against signal energy: a post-processor on
  `pred_timelines.jsonl`, to be scored with `run_transcribe`'s metrics.
- The real-recording set (`ctag.real_data`): evaluate the same adapter on it
  with `run_transcribe --timelines data/desed/timelines.jsonl` once verified.
