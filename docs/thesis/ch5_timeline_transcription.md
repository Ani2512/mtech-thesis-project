# 5 WHOLE-TIMELINE TRANSCRIPTION AT SCALE

## 5.1 From seven targets to one

Chapter 4 ended with an error budget. Of the 502 answerable test questions,
the best Phase II method answered 55 percent perfectly, answered "nothing"
wrongly on 19 percent, listed too few occurrences on 8 percent and misplaced
the right number on 14 percent. The wrong "nothing" was a decision the model
made, and the decomposition of Section 4.3 had already removed it for the
types it handled, by computing the condition in a program rather than asking
the model to apply it.

This chapter takes that idea to its end. The model is asked one thing per
clip: write out every sound event in the recording, as a list of objects with
a name, a start and an end, sorted by start. Every question type is then
computed from that list by the same predicates that generated the benchmark
(Section 3.3). The model never decides whether an answer is empty; a sound is
either in its transcript or it is not. The seven question-type scores
collapse into one quantity, the accuracy of the transcript, which is measured
directly: event-level precision, recall and F1 with one-to-one matching at an
IoU of 0.5, label-aware (the name must match) and label-agnostic (a sound
heard but misnamed is not a miss), plus the count accuracy per clip, the
under-report rate, the median centre error and the recall per sound.

The prompt is the same system prompt and layout as every other method, so
training and inference are byte-identical:

> Locate: every sound event in the recording, as a JSON list of objects
> {"sound": name, "start": seconds, "end": seconds} sorted by start, using
> only these sound names: car horn, cat, ... . The recording is 20.0 seconds
> long. Reply with only the JSON list.

With the gold timelines in place of the model's, the program reproduces every
benchmark answer exactly (a unit test guards this), so the measured gap is the
transcript's and nothing else's.

## 5.2 Training data at scale

The Phase II study trained on 2,300 examples drawn from the 300 benchmark
clips. Here the generator of Section 3.1 is used as a training-set factory:
20,000 clips of 20 seconds, composed from the same fourteen ESC-50 classes as
a pure function of a seed and an index, so the timelines file is the durable
artefact and the audio can be regenerated anywhere with one command. A
"hard" recipe draws each clip's layout from three modes: crowded (seven to
ten events, two or three labels, gaps of 0.1 to 0.8 s), overlap-heavy
(overlap probability 0.55 to 0.85) and benchmark-like with mild variation;
signal-to-noise ratio 6 to 30 dB; minimum overlap 0.2 to 0.6 s. Generation
took three minutes on sixteen CPU cores. One training example per clip, the
whole transcript as the target.

The split is by composition, not by source: the training clips and the
benchmark clips are built from the same ESC-50 recordings, as the benchmark's
own splits already were. This is stated here because it is the reason the
next chapter exists.

## 5.3 Training

Qwen2.5-Omni-7B, the thinker part only, with a LoRA adapter of rank 128 and
alpha 256 (the setting TEMPO uses), bf16 precision, the audio encoder trained
rather than frozen, learning rate 1e-4 with a cosine schedule, three epochs
over the 20,000 examples at eight examples per step (7,500 steps). The
hardware moved from the free Kaggle T4 of Phase II to a rented NVIDIA L40S
(48 GB), because a 20,000-clip run in full precision does not fit the T4's
weekly allowance. Training took 9 h 43 min at 4.6 s per step, with the
validation loss falling from 0.211 after twenty steps to 0.051, 0.030 and
0.025 at the end of each epoch; the gradient norm stayed finite throughout.

Three bugs appeared on the first GPU run that no CPU test had caught: a
precision mismatch between the fp32 copies of the embedding rows and the bf16
model, an adapter that would not reload because of a vocabulary-size
assumption, and an out-of-memory in the preflight pass because gradient
checkpointing was switched on only after it. Each was fixed with a CPU test
the same day; they are recorded because the cost of a wasted GPU day is part
of the method.

## 5.4 Results on composed audio

### 5.4.1 The transcript

Table 5.1. Transcription quality on the composed benchmark, label-aware,
IoU 0.5, pooled over every event of every clip.

| split | clips | precision | recall | event F1 | any-label F1 | count accuracy | under-report | centre error (median) | length ratio (median) |
|---|---|---|---|---|---|---|---|---|---|
| validation | 52 | 0.990 | 0.993 | 0.992 | 0.992 | 0.904 | 0.038 | 0.002 s | 1.000 |
| test | 48 | 0.985 | 0.993 | 0.989 | 0.990 | 0.917 | 0.021 | 0.002 s | 1.000 |

The gate set for this phase, event F1 of at least 0.95 on validation, is
passed with 0.992. The label-agnostic F1 equals the label-aware one: no event
was heard but misnamed. Recall per sound on test is 1.00 for twelve of the
fourteen classes, 0.96 for cat and 0.95 for car horn. What remains is a small
number of missed or invented events: one clip in eleven has one event too many
or too few, which the pooled scores barely register and the count accuracy of
0.91 does.

### 5.4.2 Every question type from the transcript

Table 5.2. f1@0.5 over answerable questions, same 699 test questions as
chapter 4.

| method | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| Phase II best (choose per type, chapter 4) | 0.763 | 0.575 | 0.780 | 0.639 | 0.587 | 0.464 | 0.666 | 0.645 |
| timeline transcript, this chapter | 0.992 | 0.963 | 0.997 | 1.000 | 1.000 | 0.961 | 0.974 | **0.983** |
| the same after the trailing check (5.5) | 0.997 | 0.975 | 1.000 | 1.000 | 1.000 | 0.961 | 0.991 | **0.989** |

Every type clears 0.95 on test as well as validation. ABSENT questions are
rejected with an F1 of 1.000; the false-rejection rate on answerable
questions is 0.8 percent. The largest item of the Phase II error budget, the
wrong "nothing", is gone by construction. The two weakest types, WHILE and
ORDINAL, are the two whose answer flips on the exact count of the target
sound's occurrences, which is where the count errors of Table 5.1 land.

### 5.4.3 The control: same recipe, question as the target

Two questions had to be answered before the gain could be attributed. Is it
the training target, or just the scale? And what does the direct method do
with the empty answers? A second run trained the same model, with the same
20,000 clips, rank, precision, epochs and prompt layout, on the *question*
as the target (the Phase II "arm C"), then used that adapter both directly
and as the sound-locator inside the Phase II decomposition ("arm F").

Table 5.3. The three-way comparison on the 699 test questions.

| method | ALL, answerable questions | count accuracy, conditional types | gold-empty conditional answered empty (of 149) | ABSENT rejected (of 48) | f1@0.5 over all 651 typed questions |
|---|---|---|---|---|---|
| Phase II arm C (2,000 clips, rank 16, 4-bit, T4) | 0.554 | | | | |
| arm C at scale, asked directly | 0.951 | 0.59 to 0.71 | 0 | 0 | 0.733 |
| arm F at scale, arm C inside the decomposition | 0.963 | 0.91 to 1.00 | 148 | 0 | 0.970 |
| timeline transcript | **0.983** | 0.95 to 1.00 | 146 | 48 | **0.982** |

Two sentences summarise it. On questions that have an answer, scale did most
of the lift (0.554 to 0.951 on the same recipe) and the timeline target adds
0.03, concentrated on the compositional types: 0.076 on ORDINAL, 0.039 on
NOT_FOLLOWED, 0.033 on WHILE, because asked directly the model finds the first
occurrence and stops. On the third of the questions whose answer is "nothing",
the direct method scores zero: the arm C adapter answered empty on none of
the 699 questions, inventing an interval for a sound that is not in the clip,
with 15 percent empty targets in its training set and the prompt identical to
training. The decomposition recovers the conditional empties because its
program decides emptiness, but still fails every ABSENT question, since a
plain query on a missing sound goes straight to the locator. The difference
of 0.982 against 0.733 over all typed questions is structural, not a matter
of more data.

## 5.5 Where the transcript still loses, and the one fix that worked

All 100 validation and test clips were examined (Table 5.4 summarises; the
full list is in the repository's error analysis). Ten clips are imperfect,
and all ten are count errors: four dropped events and seven spurious ones.
No clip has a timing error that costs a match; the median centre error is
2 ms and the worst 0.40 s.

Table 5.4. The eleven errors on 100 clips.

| kind | count | where |
|---|---|---|
| dropped event | 4 | always inside an overlap; three of four are the quieter sound of the pair (keyboard typing under a train or a siren, a cat under a car horn and a dog, a car horn inside a four-deep pile) |
| spurious event | 7 | six of seven sit on top of a sound already transcribed and six of seven are the last event the model emitted; one lands in silence |

Crowding drives it: clips whose densest moment has four sounds at once fail
two times in three, three at once one in nine, one or two at once about one in
fourteen. Keyboard typing, a quiet broadband texture, is the weakest sound at
0.962 recall and precision. The cost to the questions is concentrated: the 18
imperfect test questions come from four clips, and one dropped car horn alone
breaks seven of them.

Two repairs were tried. **Boundary refinement**, snapping each edge to the
nearest energy rise or fall within 0.5 s, made things worse (test event F1
0.989 to 0.978; ALL 0.983 to 0.973): the transcriber places centres within
2 ms, so the only thing a snap can do is move a correct edge onto a
neighbouring sound. It is reported as a negative result and left off.
**The trailing-event check** uses the model's own stopping confidence. At
every event boundary the model chooses between continuing the list and
closing it, and the probability it put on closing before the last event is
recorded from the generation scores. If that probability exceeds a threshold
tuned on validation (0.02; the tuning is flat from 0.02 to 0.30), the last
event is dropped. On test it removed all four spurious trailing events and no
correct one: event F1 0.989 to 0.996, count accuracy 0.917 to 0.958, and the
question types 0.983 to 0.989 (last row of Table 5.2). Two geometric versions
of the same check, dropping the last event when it overlaps another prediction
or when the audio under it is quiet, both failed, because 45 percent of the
benchmark's events overlap and its quiet sounds sit under loud ones.

## 5.6 What this chapter established, and what it did not

A model trained to transcribe the whole timeline answers every compositional
question type at 0.96 or better on composed audio, and it does so for a
structural reason: the program, not the model, decides when the answer is
empty. The question-trained control at the same scale shows that the
remaining difference is exactly on the compositional types and on the
empty answers.

Four caveats stand. The training and test clips share their source
recordings. Every trained model predicts the training set's event length
(length ratio 1.000), which is harmless here and wrong on real audio. The
decomposition is structural: questions carry their type and parts, so the
number measures transcription plus rule application, not understanding of
English. And the test split is small, 48 clips and 699 questions, so per-type
scores carry about two points of noise. The first two are the subject of the
next chapter.
