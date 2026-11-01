# Results: the real-data stage (DESED validation → public), 2026-11-01

The timeline adapter trained on real recordings for the first time. Composed
training alone had given 0.983 on composed audio and 0.02 on real audio
(docs/results_desed.md); the pilot retrain fixed window length and density but
could not teach sounds the ESC-50 bank lacks. This run adds the real half.

## Setup

- **Runner**: `nebius_real.py` (PR #25), main at e78e163, L40S preemptible.
  Two preemptions (one after training had finished, one during evaluation);
  the boot service (`scripts/nebius_service.sh`) resumed both times, nothing
  was redone.
- **Training data**: 8,136 examples = 784 real DESED validation clips
  (docs/real_recordings.md, 16 kHz mono, open prompt) repeated 4× (3,136) +
  the pilot's 5,000 varied generated clips (2,500 × 10 s + 2,500 × 20 s, 50
  ESC-50 classes, open prompt). 115 real clips held out for validation (split
  by clip, stable hash). The test is the 692-clip DESED public set, no video
  shared with the training clips.
- **Training**: 3 epochs, 3,051 steps at batch 8, 179 min; LoRA rank 128,
  alpha 256, bf16, audio encoder trained; loss 4.79 → 1.93. Eval loss on the
  held-out real clips **0.772 / 0.803 / 0.880** by epoch: the model overfits
  the 784 real clips after the first pass. The saved adapter is the epoch-3
  one; the epoch-1 checkpoint was not kept (two most recent only).
- **Evaluation**: every clip transcribed once with the open prompt and P(stop)
  recorded; every query type computed from the timelines (`ctag.run_agent
  --grounder timeline`); the trailing check tuned on the held-out real clips.
  Output in `runs_nebius_phase3/real/` and the adapter in
  `runs_nebius_phase3/lora_real/` (gitignored).

## DESED public: real recordings, 692 clips, 2,765 events, 4,948 questions

### Events (label-aware unless stated)

| adapter | event F1 | any-label F1 | recall | precision | median length ratio | events / clip (gold 4.0) | unreadable |
|---|---|---|---|---|---|---|---|
| composed-only (20k, 14 classes) | 0.023 | 0.148 | 0.02 | | 2.50 s constant | 8.5 | 0 % |
| pilot v2 (5k varied, 50 classes) | 0.039 | 0.282 | | | 1.12 s vs 0.85 | 5.6 | 15.2 % |
| **real-data stage** | **0.671** | **0.791** | **0.632** | **0.715** | **1.006** | **3.53** | **0.3 %** |

Centre error median 0.046 s, 92.8 % of matched events within 1 s. Under-report
rate 0.308 (clips with fewer events written than present), count accuracy 0.532.

### Recall by sound

| sound (gold events) | composed-only | pilot v2 | **real-data stage** | held-out real (115 clips) |
|---|---|---|---|---|
| speech (913) | 0.00 | 0.00 | **0.80** | 0.80 |
| cat (240) | 0.15 | 0.18 | **0.78** | 0.62 |
| vacuum cleaner (96) | 0.00 | 0.51 | **0.76** | 1.00 |
| blender (84) | 0.02 | 0.01 | **0.74** | 0.67 |
| frying (90) | 0.00 | 0.00 | **0.70** | 0.75 |
| running water (109) | 0.00 | 0.00 | **0.63** | 0.95 |
| electric shaver / toothbrush (108) | 0.00 | 0.00 | **0.56** | 0.71 |
| dog (441) | 0.12 | 0.08 | **0.50** | 0.82 |
| alarm bell ringing (196) | 0.00 | 0.00 | **0.42** | 0.71 |
| dishes (488) | 0.01 | 0.00 | **0.41** | 0.42 |

The vocabulary followed the data exactly as before: of 2,444 events written on
the public set, 2,407 carry one of the ten DESED names (speech 831, dishes 394,
dog 372, cat 216, alarm 155, water 120, vacuum 92, blender 85, shaver 72, frying
71) and 37 an ESC-50 name (church bells 9, crackling fire 5, ...). Speech, which
the earlier adapters never wrote, is now the most-written name, in proportion.

### Every query type from the timelines, f1@0.5

| method on real audio | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | **ALL** |
|---|---|---|---|---|---|---|---|---|
| untrained model, asked directly (first 700 q) | 0.490 | 0.223 | 0.527 | 0.265 | 0.333 | 0.507 | 0.225 | 0.362 |
| arm C at scale, asked directly | 0.190 | 0.231 | 0.208 | 0.225 | 0.228 | 0.219 | 0.174 | 0.203 |
| arm F at scale, decomposition | 0.196 | 0.186 | 0.136 | 0.118 | 0.110 | 0.212 | 0.140 | 0.174 |
| timeline, composed-only adapter | 0.025 | 0.042 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | 0.020 |
| timeline, pilot v2 | | | | | | | | 0.028 |
| **timeline, real-data stage** | **0.687** | **0.551** | **0.433** | **0.473** | **0.404** | **0.502** | **0.577** | **0.582** |
| same, after the trailing check (thr 0.05) | 0.680 | 0.543 | 0.410 | 0.467 | 0.382 | 0.477 | 0.564 | 0.570 |

ABSENT rejection F1 0.993 (the model almost never invents a sound that is not
there). Question counts: PLAIN 1,047, ORDINAL 866, NOT_FOLLOWED 710, WHILE 502,
AFTER / BEFORE / NEXT_AFTER 377 each, ABSENT 692.

The type ordering is the composed benchmark's: the sequence questions (AFTER,
BEFORE, NEXT_AFTER) pay for every missed event twice, once as the anchor and
once as the answer, so a 0.63 event recall costs them most.

**The trailing check does nothing on real audio** (0.671 → 0.670; 56 removals,
34 right, 22 wrong). It was built for the composed clips' duplicated last
event; real recordings fail by omission (under-report 0.308), not by trailing
repeats. It stays off here.

## Held-out real clips (115, same source as the training clips)

Event F1 0.713, any-label 0.755, recall 0.719, precision 0.708; every type ALL
0.655 (PLAIN 0.755, ORDINAL 0.580, AFTER 0.609, BEFORE 0.568, NEXT_AFTER 0.500,
WHILE 0.580, NOT_FOLLOWED 0.697). A little above the public set, so the public
set is somewhat harder but there is no train/test gap to worry about.

## Composed ESC-50 test: the forgetting check (48 clips)

| adapter | event F1 | every type ALL |
|---|---|---|
| composed-only (20k clips, 14 classes, closed prompt) | 0.989 | 0.983 |
| pilot v2 (5k varied, 50 classes, open prompt) | 0.824 | 0.718 |
| **real-data stage** (same 5k + real) | **0.873** | **0.772** |

Adding the real clips *raised* the composed score over the pilot that used the
same generated data, so the real half costs nothing there. The remaining gap to
0.983 is the generated set (5k varied, 50 classes versus 20k, 14 classes) and
the open prompt, not the real data. Lowest composed recall: door wood knock
0.67, car horn 0.75, sneezing 0.75.

## What it means

1. **Real labelled audio was the missing ingredient, and 784 clips of it were
   enough to move the real-audio number from 0.02 to 0.58.** This is the single
   largest change in the project and the thesis's central result on real
   recordings: the timeline route now beats the untrained model asked directly
   (0.362) and every question-trained arm (0.203, 0.174) on real audio, which
   none of the composed-only adapters did.
2. **The three habits are gone**: window lengths match (ratio 1.006), density is
   below gold rather than double it (3.53 vs 4.0), and the vocabulary is the
   data's. The remaining error is omission: dishes and alarm at 0.41-0.42 recall,
   both short, quiet, overlapping sounds.
3. **The model overfits after one epoch** (eval loss 0.772 → 0.880). The
   reported numbers are from the epoch-3 adapter. A one-epoch rerun
   (`CTAG_EPOCHS=1`, own `CTAG_ADAPTER`) is the cheapest likely improvement,
   about one GPU hour.
4. **Caveats stand**: the DESED questions are automatic and the hand review
   (0/300) has not been done; the untrained row is on 700 of 4,948 questions;
   the test set is a single domain (domestic audio, 10 classes).

## Follow-ups, in order of expected value per GPU hour

1. One-epoch (or 1.5) rerun with the same data; evaluate exactly as here.
2. Real-data stage on top of the 20k generated set rather than the pilot's 5k,
   to recover composed accuracy without losing the real-audio gain.
3. The hand review of `data/desed/public/review_sample.csv`, which decides
   which per-type numbers on real audio can be quoted.

## One-epoch rerun (2026-11-01, evening): the overfitting read was wrong

Same data, same settings, one epoch instead of three (1,017 steps, 60 min, loss
4.57 → 1.82), own adapter `runs_nebius_phase3/lora_real_1ep`, outputs in
`runs_nebius_phase3/real_1ep`. The held-out eval loss was indeed lower
(0.767 vs 0.880), but every score that matters was lower too:

| | 3 epochs (reported above) | 1 epoch |
|---|---|---|
| DESED public: label-aware event F1 | **0.671** | 0.644 |
| DESED public: any-label F1 | **0.791** | 0.766 |
| DESED public: precision / recall | **0.715** / 0.632 | 0.653 / **0.636** |
| DESED public: every type ALL | **0.582** | 0.568 |
| held-out real: event F1 / ALL | **0.713 / 0.655** | 0.684 / 0.639 |
| composed ESC-50 test: event F1 / ALL | **0.873 / 0.772** | 0.793 / 0.656 |

The one-epoch adapter finds about as many real events (recall 0.636 vs 0.632)
but names and bounds them worse (precision 0.653 vs 0.715), and it is far
behind on composed audio, where the third pass still had much to learn. The
token-level eval loss on 115 clips tracked overconfidence, not accuracy, and
is not a selection criterion here. **The three-epoch adapter stands as the
reported result.** The trailing check again did nothing (0.644 → 0.644).

Follow-up 1 of the list above is closed; follow-ups 2 (real stage on the 20k
generated set) and 3 (the hand review) remain.
