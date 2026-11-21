# 6 REAL RECORDINGS

## 6.1 The test set: DESED

Everything in chapters 4 and 5 was measured on audio this project composed.
The six-month plan of September 2026 said that the gap between composed and
real audio would be the most quoted number in the thesis, and it is.

The real-recording track uses DESED (Domestic Environment Sound Event
Detection, DCASE Task 4), the public strongly labelled set of 10-second
domestic YouTube clips over ten classes: alarm or bell ringing, blender, cat,
dishes, dog, electric shaver or toothbrush, frying, running water, speech,
vacuum cleaner. Two subsets carry onset and offset labels. The public
evaluation set, 692 clips and 2,765 events, is the test set throughout this
chapter. The validation set, 1,168 clips, becomes the training half in
Section 6.4; 911 of its clips could still be fetched from YouTube, the rest
being private or removed, and no video is shared between the two subsets.

The questions are generated from the annotators' timelines by the same
rules as the composed benchmark (Section 3.3), with one addition: WHILE
questions that rest on an overlap shorter than 0.3 s are dropped, since the
condition must be audible. The public set yields 4,948 questions (PLAIN
1,047, ORDINAL 866, NOT_FOLLOWED 710, WHILE 502, AFTER, BEFORE and
NEXT_AFTER 377 each, ABSENT 692). The program reproduces every answer
exactly from the gold timelines, so, as before, the measured gap belongs to
the model.

Real recordings differ from the composed clips in four ways that matter here.
Events last from a fraction of a second to the full ten seconds, where the
composed set capped them at 2.5 s. Same-class events can be adjacent or
fragmented, speech especially, so ORDINAL questions depend on the annotators'
segmentation. Overlaps are real and frequent, speech over everything. And of
the ten classes only dog and cat are among the fourteen ESC-50 classes the
model was trained on.

## 6.2 The composed training does not transfer

Table 6.1. f1@0.5 over answerable questions on the DESED public set, every
method of chapter 5, with its composed test score for comparison.

| method | composed test ALL | DESED PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | DESED ALL | ABSENT rejection F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained Qwen2.5-Omni-7B, asked directly (first 700 q) | 0.207 | 0.490 | 0.223 | 0.527 | 0.265 | 0.333 | 0.507 | 0.225 | **0.362** | 0.595 |
| arm C at scale, asked directly | 0.951 | 0.190 | 0.231 | 0.208 | 0.225 | 0.228 | 0.219 | 0.174 | 0.203 | 0.000 |
| arm F at scale, decomposition over arm C | 0.963 | 0.196 | 0.186 | 0.136 | 0.118 | 0.110 | 0.212 | 0.140 | 0.174 | 0.000 |
| timeline transcript, composed-only adapter | 0.983 | 0.025 | 0.042 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | **0.020** | 0.965 |

The ordering is the reverse of the composed benchmark. The model that was
never trained on this project's data scores 0.362; every trained variant
scores below it, and the one that was best on composed audio is worst on
real audio. This is a result about synthetic training, not a fault in the
pipeline: the oracle scores 1.000 on all 4,948 questions, every audio file
is present, and the model's outputs are well formed.

The transcript shows why. On the 692 clips the composed-only adapter wrote
5,892 events, 8.5 per clip against a true 4.0, every one of them 2.50 s long
against a true median of 0.85 s, and named them from the fourteen ESC-50
classes: cat 1,340 times, keyboard typing 1,131, glass breaking 492,
footsteps 455, in a domestic recording that contains none of keyboard typing,
glass breaking, church bells, trains, sneezing or roosters. It never wrote
"speech", which is 913 of the 2,765 real events; over real speech it wrote
cat 369 times. The prompt had offered it the ten DESED names, since the name
list is built from the timelines it is given; it wrote the fourteen it was
trained on anyway (48 of 5,892 events use a DESED-only name). Three habits,
all in the weights:

1. **Vocabulary.** Eight of the ten classes were never in training; recall on
   six of them is exactly zero.
2. **Duration.** Every window is the ESC-50 trim length. For the two shared
   classes the placement is there and the size is not: recall of real dog
   events is 0.12 at IoU 0.5, 0.25 at 0.3 and 0.84 at 0.1.
3. **Density.** The generated clips carry 6.5 events per 20 s and the model
   reproduces that rate on 10 s of audio.

The untrained model wins because its windows are the right length (length
ratio 1.00) and it can say "nothing" (rejection F1 0.595); the question-
trained adapter answered "nothing" on none of the 1,938 questions that
required it.

## 6.3 Varying the generator: two habits fixed, one it cannot fix

The generator was changed to draw event length log-uniformly from 0.25 to
8 s by cropping or looping the source, three to nine events per clip, all 50
ESC-50 classes, and the vocabulary list was removed from the prompt. A pilot
adapter was trained on 5,000 such clips (half of 10 s, half of 20 s), same
recipe otherwise.

Table 6.2. The pilot against the composed-only adapter on the public set.

| | composed-only (20k clips, 14 classes, 2.5 s cap) | varied pilot (5k clips, 50 classes) | gold |
|---|---|---|---|
| event F1, label-aware | 0.023 | 0.039 | |
| event F1, label-agnostic | 0.148 | **0.282** | |
| predicted event length, median | 2.50 s | **1.12 s** | 0.85 s |
| events written per clip | 8.5 | **5.6** | 4.0 |
| unreadable answers (hit the token cap) | 0 % | 15.2 % | |
| every question type, ALL | 0.020 | 0.028 | |
| composed test, ALL | 0.983 | 0.718 | |

The duration and density habits were artefacts of the generator and are gone.
Twice as many windows now sit on a real sound with the right extent. The
vocabulary is as wide as the training set and no wider: the one DESED class
that gained an ESC-50 name, vacuum cleaner, went from 0.00 to 0.51 recall
with no real training data, and over the other real sounds the model writes
the nearest ESC-50 names (pouring water, brushing teeth, door wood creaks). It
still never writes "speech", "dishes", "frying" or "blender", which is why
the label-aware and question-level scores stay near zero. Synthetic variation
fixes what synthetic regularity broke, and it cannot supply classes the
source bank does not contain.

## 6.4 The real-data stage

TEMPO trains in two stages, synthetic then real. This project did the same.
The 911 DESED validation clips were split by clip, 784 for training and 115
held out, and turned into one transcription example each with the open
prompt; the 784 were repeated four times and mixed with the pilot's 5,000
generated clips, 8,136 examples in all. Three epochs, 3,051 steps at eight
examples per step, 179 minutes on the L40S, rank 128, bf16, audio encoder
trained. The held-out loss rose after the first epoch (0.772, 0.803, 0.880),
which looked like overfitting; a one-epoch rerun with the same data settled
the question the other way, scoring lower on every metric that matters
(public set 0.644 against 0.671 event F1, 0.568 against 0.582 across types,
composed test 0.656 against 0.772) despite its lower loss. The token-level
loss on 115 clips tracked overconfidence, not accuracy, and is not a
selection criterion here; the three-epoch adapter is the one reported.

### 6.4.1 Events on the public set

Table 6.3. Transcription quality on the 692 public clips, label-aware unless
stated.

| adapter | event F1 | any-label F1 | recall | precision | length ratio (median) | events per clip (gold 4.0) | unreadable |
|---|---|---|---|---|---|---|---|
| composed-only | 0.023 | 0.148 | 0.02 | | 2.50 s constant | 8.5 | 0 % |
| varied pilot | 0.039 | 0.282 | | | 1.12 s vs 0.85 | 5.6 | 15.2 % |
| **real-data stage** | **0.671** | **0.791** | **0.632** | **0.715** | **1.006** | **3.53** | **0.3 %** |

Median centre error 46 ms, 92.8 percent of matched events within one second.
The vocabulary followed the data exactly as before: of 2,444 events written,
2,407 carry one of the ten DESED names and speech, never written before, is
now the most-written name in proportion.

Table 6.4. Recall by sound on the public set.

| sound (gold events) | composed-only | varied pilot | real-data stage | held-out real clips |
|---|---|---|---|---|
| speech (913) | 0.00 | 0.00 | **0.80** | 0.80 |
| cat (240) | 0.15 | 0.18 | **0.78** | 0.62 |
| vacuum cleaner (96) | 0.00 | 0.51 | **0.76** | 1.00 |
| blender (84) | 0.02 | 0.01 | **0.74** | 0.67 |
| frying (90) | 0.00 | 0.00 | **0.70** | 0.75 |
| running water (109) | 0.00 | 0.00 | **0.63** | 0.95 |
| electric shaver or toothbrush (108) | 0.00 | 0.00 | **0.56** | 0.71 |
| dog (441) | 0.12 | 0.08 | **0.50** | 0.82 |
| alarm or bell ringing (196) | 0.00 | 0.00 | **0.42** | 0.71 |
| dishes (488) | 0.01 | 0.00 | **0.41** | 0.42 |

The remaining error is omission (under-report rate 0.308): dishes and alarm,
both short, quiet and overlapping, are found in fewer than half of cases.

### 6.4.2 Every question type on real audio

Table 6.5. f1@0.5 over answerable questions, DESED public set.

| method | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| untrained model, asked directly (first 700 q) | 0.490 | 0.223 | 0.527 | 0.265 | 0.333 | 0.507 | 0.225 | 0.362 |
| arm C at scale, asked directly | 0.190 | 0.231 | 0.208 | 0.225 | 0.228 | 0.219 | 0.174 | 0.203 |
| timeline, composed-only adapter | 0.025 | 0.042 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | 0.020 |
| **timeline, real-data stage** | **0.687** | **0.551** | **0.433** | **0.473** | **0.404** | **0.502** | **0.577** | **0.582** |

ABSENT rejection F1 0.993: the model almost never invents a sound that is not
there. The type ordering is the composed benchmark's: the sequence questions
(AFTER, BEFORE, NEXT_AFTER) pay for every missed event twice, once as the
anchor and once as the answer, so a recall of 0.63 costs them most. The
trailing check of Section 5.5 does nothing on real audio (0.671 to 0.670; 56
removals, 34 right, 22 wrong): it was built for the composed clips'
duplicated last event, and real recordings fail by omission. It is left off.

On the 115 held-out real clips the adapter scores 0.713 event F1 and 0.655
across types, a little above the public set, so there is no train-test gap
to worry about. On the composed ESC-50 test it scores 0.873 event F1 and
0.772 across types, above the varied pilot's 0.718 that used the same
generated data, so the real half cost nothing there; the remaining gap to
0.983 is the smaller, wider generated set and the open prompt.

## 6.5 The exact-label cross-check

The DESED questions are automatic, derived from human annotation, and human
annotators disagree on boundaries by a few hundred milliseconds. A hand
review was planned and then dropped, because a reviewer cannot resolve that
disagreement either. What can is a test set whose labels are exact: the
synthetic evaluation soundscapes of DCASE 2020 Task 4, 828 clips built with
Scaper from isolated real recordings of the same ten DESED classes, each
shipped as a recipe saying which recording plays when, at what level and
pitch shift. The recipe is the ground truth. DESED's own post-processing was
applied (merge same-label events closer than 150 ms, drop events under
250 ms), giving 1,957 events on 806 clips with events, median length 0.97 s,
2.36 events per clip, and 6,098 questions. The same adapter, untouched, was
scored on it.

Table 6.6. The real-data adapter on human-labelled and exact-labelled
real audio.

| test set | event F1 | any-label F1 | recall | precision | centre error (median) | events per clip | under-report | ALL across types | ABSENT rejection |
|---|---|---|---|---|---|---|---|---|---|
| DESED public, human labels (692 clips) | 0.671 | 0.791 | 0.632 | 0.715 | 46 ms | 3.53 (gold 4.0) | 0.308 | 0.582 | 0.993 |
| synthetic soundscapes, exact labels (806 clips) | **0.701** | **0.848** | 0.700 | 0.702 | **26 ms** | 2.42 (gold 2.36) | 0.195 | 0.573 | 0.984 |

Table 6.7. Per type on the exact-label set, with the public set for
comparison.

| set | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| DESED public | 0.687 | 0.551 | 0.433 | 0.473 | 0.404 | 0.502 | 0.577 | 0.582 |
| exact-label synthetic | 0.662 | **0.700** | 0.451 | 0.398 | 0.442 | **0.339** | 0.608 | 0.573 |

Three things follow. Event detection is not an annotation artefact: with
exact labels the event F1 rises, the centre error halves and the length ratio
stays at 1.0, so the public-set score was held down by its human labels, not
propped up by them. The per-sound picture is what changes: alarm (0.42 to
0.78) and shaver (0.56 to 0.74) were under-credited by the public annotation,
while blender (0.74 to 0.20), frying (0.70 to 0.28) and vacuum (0.76 to
0.52) collapse on the synthetic clips, where these long stationary textures
appear as short partial events at controlled levels rather than filling the
whole clip as they do in the public recordings; this is a real weakness of
the adapter on textures heard only in part. And the question-type total is
unchanged while its composition moves: ORDINAL gains because fewer events per
clip make counting easier, WHILE loses because Scaper's overlaps are short
and a 26 ms centre error is enough to miss an IoU of 0.5 on a window under a
second. WHILE at 0.34 on exact labels is the clearest open weakness in the
question stack. The trailing check again did nothing.

## 6.6 Summary

Training on composed clips with a closed vocabulary and a fixed event length
gives 0.983 on composed audio and negative transfer to real recordings
(0.020, below the untrained 0.362). Varying the generator removes the length
and density habits and cannot supply the vocabulary. Adding 784 real strongly
labelled clips supplies it: 0.671 event F1 and 0.582 across every question
type on the public set, 0.772 kept on composed audio, and a cross-check on
exact labels that agrees (0.701 and 0.573). The number the thesis quotes for
real recordings is the public-set one, with the exact-label result beside it.
