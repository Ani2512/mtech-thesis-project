# Nebius phase 3 run — whole-timeline transcription, results as printed by the runner

Nebius VM `cyan-cardinal-instance-5` (eu-north1, NVIDIA L40S 48 GB, 16 vCPU, 64 GiB, $1.78/h
incl. disk), `nebius_phase3.py` launched 2026-09-19 05:07 UTC, finished ~16:00 UTC.
Code: `main` at fea8ccd minus the resume support of PR #13 (the run started from
`fix/preflight-gradient-checkpointing` = ed03f43, the same trainer code). Knobs:
`CTAG_ARM_E=0 CTAG_BS=4 CTAG_ACCUM=2`, everything else default (20,000 generated clips,
3 epochs, lr 1e-4 cosine, LoRA rank 128 / alpha 256, bf16, audio encoder trained,
max 4,096 tokens). Output in `runs_nebius_phase3/` (gitignored; the adapter is 1.7 GB).
Cost of the whole day incl. bootstrap, two smoke attempts and the dry runs: about $20.

## Step outcomes

```
--- benchmark (300 ESC-50 clips): ok in 1 min
--- split: ok in 0 min
--- generate 20000 hard clips: ok in 3 min          (15 workers; the doc estimated 20-30 min)
--- SFT: transcription targets for the generated set: ok in 0 min
--- SFT: transcription targets for the benchmark val clips: ok in 0 min
--- smoke train (20 steps): FAILED rc=1 in 5 min     (attempt 1: preflight OOM, see below)
--- smoke train (20 steps): ok in 3 min              (attempt 2)
--- train transcription (3 epochs): ok in 583 min    (7,500 steps, 4.6 s/step, 8 examples/step)
--- transcribe val clips: ok in 6 min
--- every query type from the val timelines: ok in 0 min
--- refine the val timelines against the audio: ok in 0 min
--- every query type from the refined val timelines: ok in 0 min
--- transcribe test clips: ok in 6 min
--- every query type from the test timelines: ok in 0 min
--- refine the test timelines against the audio: ok in 0 min
--- every query type from the refined test timelines: ok in 0 min
--- direct prompting with the transcription adapter (reference): ok in 53 min   (699 q at 4.5 s)
```

Preflight (attempt 2, bs 2): `peak 22.6 GiB of 44.4 GiB on the longest example (958 tokens),
gradient checkpointing on`. Full run (bs 4): `peak 27.0 GiB`, 35-42 GB in use during
training, GPU at 100%.

## Training curve

| point | train loss (logged) | val loss |
|---|---|---|
| smoke, 20 steps | 0.642 -> 0.249 | 0.211 |
| end of epoch 1 (step 2,500) | ~0.13-0.22 | 0.051 |
| end of epoch 2 (step 5,000) | ~0.06-0.12 | 0.030 |
| end of epoch 3 (step 7,500) | 0.024 (last logged; 1.358 at step 10) | 0.025 |

Grad norm finite throughout (0.77 at step 10 -> 0.15-0.28 late), no nan steps, no zero-loss
steps. Learning rate 1e-4 peak, cosine to 2e-6 at the end.

## Transcription quality (the gate)

Label-aware one-to-one matching at IoU 0.5, pooled over every event of every clip.

```
split  n_clips  P_pooled  R_pooled  F1_pooled  f1_any_label  count_acc  under_report  parse_fail  centre_err_med  centre<1s  dur_ratio_med
val       52     0.990     0.993     0.992       0.992         0.904      0.038         0.000       0.002 s         0.997      1.000
test      48     0.985     0.993     0.989       0.990         0.917      0.021         0.000       0.002 s         0.996      1.000
```

**GATE 0.95 on val: PASS (0.992).**

Recall per sound, test: car horn 0.95, cat 0.96, every other label 1.00
(church bells, clock alarm, dog, door wood knock, footsteps, glass breaking, keyboard typing,
laughing, rooster, siren, sneezing, train). Val: keyboard typing 0.944, everything else 1.00.

`f1_any_label` equals `f1` on both splits: no event was heard but misnamed. The residual is
a small number of missed or invented events (recall 0.993, precision 0.985-0.990), and a
clip-level count accuracy of about 0.91, i.e. one clip in eleven has one event too many or
too few even though the pooled event scores barely register it.

## Every query type computed from the predicted timelines

`ctag.run_agent --grounder timeline`: the same decompose-and-combine agent as arms B/D/F, with
the per-sound grounder replaced by a lookup in the predicted timeline. No model call per
question. Numbers are over questions that have an answer (ABSENT shows rejection F1).

```
f1@0.5 (val, 768 q)
PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL   | ABSENT rej_f1
0.998 0.988   0.983 1.000  0.986      0.987 0.995        0.991 | 1.000
f_beta (val)                                              0.992
rejection: precision 0.986  recall 1.000  false_rejection_rate 0.005  under_report 0.007

f1@0.5 (test, 699 q)
PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL   | ABSENT rej_f1
0.992 0.963   0.997 1.000  1.000      0.961 0.974        0.983 | 1.000
f_beta (test)
0.993 0.963   0.999 1.000  1.000      0.958 0.977        0.984
count_acc (test)
0.969 0.979   0.978 1.000  1.000      0.948 0.958        0.977
rejection (test): f1 0.982  false_rejection_rate 0.008  under_report 0.016  centre<1s 0.993
```

Against phase 2 (`results_kaggle_v6.md`, same 699 test questions):

```
                        PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL
F-combined (phase 2)    0.763 0.575   0.780 0.639  0.587      0.464 0.666        0.645
timeline (phase 3)      0.992 0.963   0.997 1.000  1.000      0.961 0.974        0.983
```

Every type clears the >= 0.95 per-type target on test as well as val. The two weakest,
WHILE (0.961) and ORDINAL (0.963), are the two whose answer depends on the exact count of
occurrences of the target sound (WHILE also on the overlap), so the count errors above are
where they lose. The phase 2 error budget's largest item, a wrong "nothing" (19% of
answerable questions for arm F), is gone by construction: false rejection is 0.8%.

## Boundary refinement: harmful at this accuracy

`ctag.refine` (snap each edge to the nearest energy rise/fall within 0.5 s, alpha 0.5) was
run on the predicted test timelines: **190 edges moved, event F1 0.989 -> 0.978**, centre
error median 0.002 -> 0.008 s, recall by sound dropped for dog (1.00 -> 0.95), door wood
knock (1.00 -> 0.95) and glass breaking (1.00 -> 0.92). Query types from the refined
timelines: ALL 0.983 -> 0.973, PLAIN 0.992 -> 0.971, ORDINAL 0.963 -> 0.938. Val: 0.992 ->
(refined) lower as well. The step was designed for a transcriber with 0.3-0.5 s edge errors;
this one places centres within 2 ms, so the only thing the snap can do is move a correct edge
to a neighbouring sound's onset. Report as a negative result; leave it off by default.

## Direct prompting with the transcription adapter (reference)

`ctag.run_zeroshot` with the same adapter, asked each of the 699 test questions directly:
**0.000 on every type, parse_fail_rate 1.000**. The adapter answers every prompt with a
whole-clip timeline (`[{"sound": "dog", "start": 0.98, "end": 3.48}, ...]`), which the
question parser rejects. This is the expected control: the adapter was trained on one target
only, and the gain comes from the timeline route, not from the extra training on its own.
A fair "direct" arm at this scale is arm C at scale below (run 2): 0.951 on answerable questions,
0.000 on the 197 empty-answer ones.

## Run 2 (2026-09-26): arm C and arm F at scale, the trailing check

Preemptible L40S (`computeinstance-e00s9ekmwbsya6907n`, 195.242.29.105, $0.90/h), fresh
disk, bootstrapped from `main` and run on `fix/trailing-none-format` = e176ca3 (merged as
PR #18, 64742fb). Launched 2026-09-26 ~05:10 UTC, finished ~20:25 UTC the same day, no
preemption (the VM was up 23 h by the time the results were pulled). Knobs: `CTAG_ARM_E=1 CTAG_ARM_E_ARMS=text CTAG_SAVE_STEPS=100 CTAG_BS=4
CTAG_ACCUM=2`. The transcription adapter and its evaluations were pushed from the Mac
(`scripts/nebius_sync.sh push`) so the runner skipped them. Two bugs surfaced on the way in,
both fixed in PR #18: the trailing step crashed when every clip was unparseable (the dry run's
random model) and the sync script's `--info=progress2` is unknown to macOS's rsync.

```
--- smoke train (20 steps): ok in 8 min
--- transcribe val clips with stop probabilities: ok in 5 min
--- transcribe test clips with stop probabilities: ok in 4 min
--- trailing check: threshold from val, applied to test: ok in 0 min
--- SFT: question targets (text): ok in 0 min        (20,000 ex: 11,965 plain, 8,035 conditional, 3,000 empty targets = 15%)
--- SFT: question targets for val (text): ok in 0 min (600 ex)
--- train arm C at scale (text digits): ok in 441 min (7,500 steps, 3.45 s/step, checkpoint every 100)
--- eval arm C at scale on test: ok in 9 min          (699 q, one call each)
--- eval arm F at scale on test: ok in 3 min          (699 q, decomposition over the arm C adapter)
```

Arm C at scale is the control the phase 3 result needed: the same 20,000 generated clips,
the same recipe (rank 128, bf16, 3 epochs, encoder trained), the same prompt byte for byte,
with the *question* as the training target instead of the timeline. Arm F at scale is that
adapter as the per-sound grounder inside the decomposition.

### Training curve, arm C at scale

| point | train loss (logged) | val loss (600 val questions) |
|---|---|---|
| step 10 | 0.76 | |
| end of epoch 1 (step 2,500) | 0.7-1.2 | 0.525 |
| end of epoch 2 (step 5,000) | 0.5-1.4 | 0.429 |
| end of epoch 3 (step 7,500) | 1.03 (last logged) | 0.422 |

The question target does not converge the way the timeline target did (val loss 0.025 at
the end): each example is one short answer whose tokens are mostly digits, so the loss stays
near 1 per token even when the intervals are right. Grad norm 0.4-1.5 throughout, no nan.

### The three-way comparison (test, f1@0.5, answerable questions)

Same 699 test questions, same metric as every table above: per-type scores over the
questions that have an answer (502), as the runner prints them.

```
                                           PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL    empty-answer rate
timeline route (run 1)                     0.992 0.963   0.997 1.000  1.000      0.961 0.974        0.983  0.00 (rejection f1 1.000)
arm C at scale, asked directly             0.979 0.887   0.981 0.970  0.984      0.928 0.935        0.951  0.00 (rejection f1 0.000)
arm F at scale, arm C inside decomposition 0.982 0.950   0.983 0.986  0.968      0.907 0.959        0.963  0.02

f_beta                                     PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED
timeline                                   0.993 0.963   0.999 1.000  1.000      0.958 0.977
arm C at scale                             0.976 0.887   0.981 0.978  0.984      0.928 0.945
arm F at scale                             0.977 0.950   0.982 0.984  0.968      0.898 0.957

count_acc                                  PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED
timeline                                   0.969 0.979   0.978 1.000  1.000      0.948 0.958
arm C at scale                             0.948 0.833   0.708 0.618  0.708      0.594 0.615
arm F at scale                             0.958 0.969   0.978 0.966  1.000      0.906 0.927
```

Against phase 2 (`results_kaggle_v6.md`, same questions, T4, rank 16, 4-bit, ~2,000 clips):

```
                        PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL
C-enc (phase 2)         0.764 0.450   0.821 0.696  0.667      0.257 0.208        0.554
C at scale (phase 3)    0.979 0.887   0.981 0.970  0.984      0.928 0.935        0.951
F-hybrid (phase 2)      0.763 0.575   0.780 0.639  0.587      0.464 0.666        0.645
F at scale (phase 3)    0.982 0.950   0.983 0.986  0.968      0.907 0.959        0.963
timeline (phase 3)      0.992 0.963   0.997 1.000  1.000      0.961 0.974        0.983
```

**Reading.** Scale did most of the lift: the direct arm went from 0.554 to 0.951 on the same
recipe as the timeline adapter, so the phase 2 -> phase 3 jump is mostly data, rank and
precision. The timeline target still wins on all seven types against both arms, by 0.032 overall
over arm C and by 0.076 on ORDINAL, 0.039 on NOT_FOLLOWED, 0.033 on WHILE; the gaps are on the compositional
types, and the count accuracy shows why: asked directly, the adapter gets the *first*
occurrence and drops the rest (count_acc 0.59-0.71 on the conditional types against
0.95-1.00 from the timeline). Decomposition over the arm C adapter recovers most of the count
loss (ORDINAL 0.887 -> 0.950, count_acc back to 0.91-1.00) but loses on WHILE and NEXT_AFTER,
where the combine step inherits the grounder's boundary errors twice.

### Arm C never says "nothing": the empty answer, 197 of 699 questions

The per-type f1 above is over answerable questions, the phase 2 convention. The test set
also has 197 questions whose correct answer is empty: 48 ABSENT rejection queries (a sound
that is not in the clip) and 149 conditional questions whose condition is never met
("the next siren once the door wood knock has ended", when no knock precedes a siren).

```
                 gold-empty questions answered empty   f1@0.5 over all 651 typed questions (7 types)
timeline route   146 / 149 conditional, 48 / 48 ABSENT   0.982
arm C at scale     0 / 149,               0 / 48        0.733
arm F at scale   148 / 149,               0 / 48        0.970
```

The arm C adapter at scale emitted `[]` on **zero** of the 699 test questions. Its raw output
on "all occurrences of train", with no train in the clip, is `[[4.83, 7.33]]`: it invents an
interval for a sound that is not there. This is with 3,000 empty targets (15%) in its training
set and the prompt byte-identical to training, so it is the model's behaviour, not a format
mismatch. It is the *opposite* failure from phase 2, where the same arm at small scale
rejected too often (false rejection 29%, arm E answered empty on 73%); at scale with the
empty share capped at 15% the interval answer wins every time. Either way, the direct arm
does not know when to say nothing; the timeline route gets rejection right by construction
(the sound is either in the transcript or it is not). Decomposition over the arm C adapter
gets the 149 conditional empties right because the combine step, not the model, decides
emptiness, but still fails all 48 ABSENT queries: a plain query on a missing sound goes
straight to the grounder, which invents the interval.

So the honest summary of the control is two sentences. On questions that have an answer,
scale closes most of the gap and the timeline target adds 0.03. On the third of the questions
whose answer is "nothing", the direct arm scores zero and the timeline route scores 0.98,
and that difference (0.982 vs 0.733 over all typed questions) is structural, not a matter of
more data.

### Trailing-event check: positive, small

`ctag.trailing --rule stop`, P(stop) from the generation scores (PR #17), threshold tuned on
val, applied to test:

```
tune thr=0.02..0.30  val event_f1_pooled 0.9917 -> 0.9933  fired=3  rm_spur=2 rm_corr=1
tune thr=0.50, 0.70  no change (fired=0)
test  rule=stop thr=0.02  fired=4  removed_spurious=4  removed_correct=0
      event_f1_pooled 0.989 -> 0.996   count_acc 0.917 -> 0.958
```

On test it removed all 4 spurious trailing events and no correct one, and the query types
from the trailing-checked test timelines improve with it, the best row of the project so far:

```
f1@0.5 (test, from the trailing-checked timelines)
PLAIN ORDINAL AFTER BEFORE NEXT_AFTER WHILE NOT_FOLLOWED ALL
0.997 0.975   1.000 1.000  1.000      0.961 0.991        0.989   (was 0.992 0.963 0.997 1.000 1.000 0.961 0.974 0.983)
```

Cheap (one extra transcription
pass with scores), positive on both splits, worth keeping on. The val tuning is flat from 0.02
to 0.30, so the threshold is not sensitive.

## First GPU run of the phase 3 code: three bugs, all invisible on CPU

All found on 2026-09-19 by the bootstrap's dry run and the first smoke attempt, each fixed
with a CPU test the same day:

1. **`--time-rows full` dtype mismatch** (PR #11): PEFT's `modules_to_save` copies of
   `embed_tokens`/`lm_head` are cast to fp32 for stability, the thinker is bf16, `--amp none`
   has no autocast; the first LoRA linear raised *mat1 and mat2 must have the same dtype*.
   Fixed by casting at the two boundaries (`bridge_full_rows`), as the delta path already did.
2. **Full-rows adapter would not load** (PR #11): the loader resized the embedding matrix only
   for a delta file or a tokenizer larger than Qwen's padded 152,064 rows; the saved copies
   have 151,967. `adapter_needs_resize()` now resizes for full rows too.
3. **Preflight OOM at 42 GiB** (PR #12): the preflight forward+backward ran before
   `Trainer.train()` switched gradient checkpointing on; the kbit path had it from
   `prepare_model_for_kbit_training`, the bf16 path did not. Now enabled in `build_model`;
   the same example then peaks at 22.6 GiB.

Plus one operational fix (PR #13): the runner could only resume at epoch ends, so
`train_lora --save-steps N` + `--resume auto` + `train_done.json` were added for preemptible
VMs. Not used by this run (it started before the merge).

## Timing and cost, measured vs. estimated

| item | estimated (docs/nebius.md, 2026-09-15) | measured |
|---|---|---|
| 20k-clip generation | 20-30 min | 3 min (16 vCPU) |
| training, 20k x 3 epochs, r=128 | 3-4 h | 9 h 43 min (4.6 s/step, 8 ex/step, examples ~1,000 chars each) |
| evaluation | ~40 min | 65 min, of which 53 min is the direct-prompting reference |
| whole run | $6-8 | ~$19 |
| arm E retest at default size (not run) | "the cost of training again" | ~20 h / $36 -> deferred, to be run at ARM_E_N ~4,000 on a preemptible VM |

Batch 4 x accum 2 was 7% faster per example than batch 2 x accum 4 (4.6 vs 4.9 s/step).
GPU utilisation sat at 62-100%; audio decoding (2 loader workers) is the likely gap.

## Caveats to state with these numbers

- **Synthetic composition, shared sources.** Training clips and benchmark clips are composed
  from the same ESC-50 recordings (a composition split, not a source split). The real-recording
  track (DESED public eval, 692 clips, 4,948 questions, hand review pending) is what tests
  generalisation.
- **Fixed-duration prior is still there**: `duration_ratio_median` 1.000 on both splits, as in
  every phase 2 arm. The generated set caps durations at 2.5 s. Harmless here, wrong on real
  audio.
- **The decomposition is structural.** Questions carry their type and parts; no model parses
  the English. The number measures transcription + rule application, not question
  understanding. A text-only parser would be needed for free-text questions.
- **Small test split**: 48 clips / 699 questions, 89-96 answerable per type; per-type scores
  carry roughly +/-2 points at this level (the 1,500-clip rebuild remains planned).
- **Closed vocabulary**: the prompt lists the 14 sound names.

## Next

1. Error analysis: DONE, see `error_analysis_phase3.md` (10 of 100 val+test clips
   imperfect, all count errors: quiet sounds dropped inside overlaps, duplicate events
   appended at the end of the list; no timing errors).
2. Trailing-event check: DONE (run 2 above), positive: test event F1 0.989 -> 0.996, 4 of 4
   spurious last events removed, 0 correct ones; query types ALL 0.983 -> 0.989.
3. Arm C and arm F at scale: DONE (run 2 above). Scale explains most of the gain on
   answerable questions (0.554 -> 0.951); the timeline target adds 0.03 there and 0.25 over
   all typed questions, because the direct arm never answers "nothing".
4. Arm E retest (time symbols at scale): still deferred; about 7.5 h on a preemptible L40S
   with `CTAG_ARM_E_ARMS=tt`.
5. DESED public-eval: hand review, then `run_transcribe --timelines data/desed/timelines.jsonl`
   with this adapter, for the composed-vs-real number.
6. SED ceiling arm (a conventional sound-event-detection model on the same clips).
7. The report's phase 3 chapter and deck slides 13-14.
