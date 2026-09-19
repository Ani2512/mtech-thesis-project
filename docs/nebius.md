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

One command from the Mac does all of the below (idempotent, safe to rerun):

```bash
ssh -i ~/.ssh/id_ed25519_nebius ubuntu@<ip> 'curl -fsSL https://raw.githubusercontent.com/Ani2512/mtech-thesis-project/main/scripts/nebius_bootstrap.sh | bash'
```

By hand, the same steps:

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

## LoRA rank

The runner trains at rank 128 (alpha 256), TEMPO's setting for the same
7B-class language model; phase 2 used rank 32 on the T4. Rank 128 is roughly
four times the adapter parameters (about 320M) and about 4 GB of weights,
gradients and optimiser state in fp32, well within 48 GB. `CTAG_LORA_R=32`
restores the phase 2 size for a like-for-like comparison.

## Cost

At 20,000 clips x 3 epochs with batch 2 x accumulation 4, expect roughly 2.5-3.5 h
of training on an L40S plus ~30 min of generation and ~40 min of evaluation:
about $6-8. The arm E retest and a second epoch count are extra runs of the same
size. Stop the VM when `phase3.log` ends with the results table; the disk can be
kept for the next run.

## Preemptible VMs (about half price) and resuming

Nebius sells the same L40S as *Preemptible* for roughly half the on-demand
rate ($0.90/h Intel, $0.74/h AMD in September 2026), with the catch that the
VM can be stopped at any moment. The runner and `train_lora` are built to
survive that:

- `train_lora --save-steps N` writes a resumable Trainer checkpoint (adapter,
  optimiser, scheduler, RNG, step counter) every N steps into
  `<out>/checkpoint-N`, keeping the two newest (about 5 GB each at rank 128).
  The runner passes `CTAG_SAVE_STEPS` (default 200, about 15 minutes on the
  L40S). Timestamp deltas (arm E, delta rows) are saved into each checkpoint
  too, since PEFT does not know about them.
- On the next start, `train_lora` resumes from the newest checkpoint on its
  own (`--resume auto`, the default) unless the run already finished, which
  it records in `<out>/train_done.json`. The runner skips a training only on
  that marker, never on the adapter file the epoch-end save writes mid-run.
- So after a preemption: start the VM again in the console, ssh in, and rerun
  the exact same command. Everything finished is skipped, the training
  continues from at most N steps back, and the results table prints at the
  end as usual. Nothing else to do.

The disk persists across a preemption (the VM is stopped, not deleted), so
the model cache, the generated set and the checkpoints are all still there.
The public IP may change on restart. Measured on 2026-09-19: a full
transcription run is about 10 h on-demand ($19); preemptible with resume
costs about $9 plus whatever partial steps the kills throw away.

## Arm C at scale on a preemptible VM (the end-to-end control)

The phase 3 result (every query type 0.961–1.000 from the written timeline,
`results_nebius_phase3.md`) changed the training target *and* the data scale,
precision, rank and epochs at once relative to phase 2 (0.645). The control that
separates the two is arm C at the same scale: the same 20,000 generated clips as
question-and-answer examples, the same recipe, asked directly at test time. The
runner has it (`CTAG_ARM_E=1 CTAG_ARM_E_ARMS=text`), together with arm F at scale
(that adapter as the grounder inside the decomposition), so the final table shows
the three-way comparison at one scale. About 6–8 h; the time-symbol half is
skipped.

Procedure, on a fresh preemptible VM (the stopped Regular VM cannot be converted):

1. Console: Create VM, **Preemptible**, NVIDIA L40S Intel 1 GPU / 16 vCPU / 64 GiB,
   Ubuntu 24.04 for NVIDIA GPUs (CUDA 13), 150 GiB SSD, public IP Auto (dynamic),
   user `ubuntu`, key `~/.ssh/id_ed25519_nebius.pub`. Note the public IP.
2. Bootstrap (idempotent, ~10 min):
   `ssh -i ~/.ssh/id_ed25519_nebius ubuntu@<ip> 'curl -fsSL https://raw.githubusercontent.com/Ani2512/mtech-thesis-project/main/scripts/nebius_bootstrap.sh | bash'`
3. Push the transcription adapter and its evaluations from the Mac so the runner
   skips them (1.7 GB): `scripts/nebius_sync.sh push <ip>`
4. Run:
   `ssh -i ~/.ssh/id_ed25519_nebius ubuntu@<ip> 'cd ~/mtech-thesis-project && source .venv/bin/activate && export HF_HOME=$PWD/hf_cache && CTAG_ARM_E=1 CTAG_ARM_E_ARMS=text nohup python nebius_phase3.py > logs/armc.log 2>&1 &'`
   The data regenerates in 3 min, the smoke train reruns (3 min, a check of the
   new machine), transcription and its evaluation are skipped, then: SFT question
   targets → train arm C at scale (checkpoint every 200 steps) → eval C directly →
   eval F (decomposition) → table.
5. After a preemption: start the VM again in the console (the IP may change), ssh in,
   rerun the command of step 4. Training resumes from the newest checkpoint.
6. Pull: `scripts/nebius_sync.sh pull <ip>`; then delete the VM.

Reading the result: near 0.55 (phase 2's arm C) means the timeline target is the
result; near the timeline row means scale did most of it and the timeline route's
contribution is the wrong-"nothing" elimination and one model call per clip.

## Arm E retest (in the runner, `CTAG_ARM_E=1`, default on)

Phase 2's time-symbol arm lost to text digits (0.194 vs 0.530) under T4
constraints. The runner retrains both on the same generated questions for the
same epochs, and gives the symbols TEMPO's own configuration:
`--head-init bpe` (output rows start from the mean of their digit pieces, not
zero), `--time-rows full` (both tables trainable; ~16 GB extra, fine on 48 GB),
`--none-weight 0.3` and `--max-empty-share 0.15` (the empty answer can no longer
be won by default). The table at the end prints both arms side by side with
their empty-answer rate. Roughly the cost of the transcription run again.

## Boundary refinement (in the runner)

`ctag.refine` snaps each predicted edge to the nearest energy rise or fall
within 0.5 s, keeping the model's *what* and *where* and letting the signal
decide *exactly when*. The runner scores the test timelines before and after
(`transcribe_test_refined/summary.json`) and answers every query type from
both, so the table shows whether it helps on real predictions. On procedural
clips it recovers most of a 0.3-0.5 s edge error and costs about a point on
exact edges where an impulsive sound overlaps a continuous one.

## Not in this runner yet

- The real-recording set (`ctag.real_data`): evaluate the same adapter on it
  with `run_transcribe --timelines data/desed/timelines.jsonl` once verified.
