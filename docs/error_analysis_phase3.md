# Phase 3 error analysis: where the timeline transcriber still loses (2026-09-19)

Companion to `results_nebius_phase3.md`. Every number here comes from
`scripts/error_analysis_phase3.py` run on the Mac against
`runs_nebius_phase3/esc50/transcribe_{val,test}/pred_timelines.jsonl` and the rebuilt
`data/esc50/timelines.jsonl` (the build is deterministic; the gold event counts match the
ones recorded in the run). CPU only, no model calls.

## The short version

- **100 clips scored (52 val + 48 test), 10 imperfect.** All 10 are count errors: 4 dropped
  events and 7 spurious events (one clip has both). No clip has a timing error that costs a
  match: 568 matched events, centre error median 0.002 s, worst 0.40 s, only 5 above 0.2 s.
- **Dropped events (4) are always inside an overlap**, and 3 of the 4 are the quieter of the
  pair: keyboard typing under a train, keyboard typing under a siren, a cat under a car horn
  and a dog, a car horn inside a four-deep pile of bells, horn and dog.
- **Spurious events (7) are duplicates, not silence hallucinations.** 6 of 7 sit on top of
  sounds that are already transcribed (energy in the window 0.75-2.35x the clip average);
  only 1 lands in silence (keyboard typing at 14.85-17.35 s in a clip whose four earlier
  keyboard-typing events it continues; energy 0.10x). **6 of 7 are the last event the model
  emitted**: the model finishes the real list and appends one more.
- **Crowding is the driver.** Clips whose densest moment has 4 sounds at once fail 2 times
  in 3; 3 at once, 1 in 9; 1-2 at once, about 1 in 14. Six-event clips fail 11%, five-event
  clips 7%.
- **Keyboard typing is the weakest sound** (recall 0.962, precision 0.962; 2 of the 4 drops
  and 2 of the 7 spurious events). It is a quiet, broadband texture that is masked by
  anything loud and easy to imagine where there is nothing.
- **Query cost is concentrated.** The 18 imperfect test questions (of 699) come from 4 clips.
  One dropped car horn in `esc50_00075` alone breaks 7 questions (PLAIN, ORDINAL, WHILE x2,
  NOT_FOLLOWED x2 and the AFTER that uses it as the anchor). This is why WHILE (0.961) and
  ORDINAL (0.963) are the lowest types: they are the ones whose answer flips on a single
  count, while BEFORE/NEXT_AFTER survive an extra trailing event.
- **Refinement cannot help.** None of the 11 errors is an edge error, which is consistent
  with the refinement step making test scores worse (0.989 -> 0.978).

## Every error

| split | clip | kind | label | window (s) | gold events | max concurrency | note |
|---|---|---|---|---|---|---|---|
| val | esc50_00045 | spurious | church bells | 8.21-10.71 | 6 | 3 | on top of car horn, church bells, car horn; last emitted=yes; rms ratio 2.35 |
| val | esc50_00135 | spurious | sneezing | 6.61-9.11 | 6 | 2 | on top of glass breaking, laughing; last emitted=no; rms ratio 1.41 |
| val | esc50_00189 | dropped | keyboard typing | 5.51-8.01 | 5 | 2 | overlaps 1 other event: train |
| val | esc50_00251 | dropped | keyboard typing | 0.66-3.16 | 6 | 2 | overlaps 1 other event: siren |
| val | esc50_00298 | spurious | keyboard typing | 9.51-12.01 | 6 | 4 | on top of dog; last emitted=yes; rms ratio 1.12 |
| test | esc50_00047 | dropped | cat | 10.13-12.63 | 6 | 3 | overlaps 2 other events: car horn, dog |
| test | esc50_00075 | dropped | car horn | 3.32-5.82 | 6 | 4 | overlaps 4 other events: car horn, church bells, dog, church bells |
| test | esc50_00075 | spurious | dog | 4.42-6.92 | 6 | 4 | on top of church bells, car horn, dog, church bells; last emitted=yes; rms ratio 2.20 |
| test | esc50_00174 | spurious | glass breaking | 6.23-8.16 | 6 | 3 | on top of car horn, rooster, train; last emitted=yes; rms ratio 1.62 |
| test | esc50_00193 | spurious | keyboard typing | 14.85-17.35 | 6 | 2 | tail, after the last gold event; last emitted=yes; rms ratio 0.10 |
| test | esc50_00233 | spurious | clock alarm | 13.97-16.47 | 5 | 1 | on top of laughing; last emitted=yes; rms ratio 0.75 |

"rms ratio" is the RMS energy of the audio inside the spurious window divided by the RMS of
the whole clip. In `esc50_00075` the dropped car horn (3.32-5.82) and the spurious dog
(4.42-6.92) overlap by 1.4 s: the model heard something there and gave it the wrong name and
the neighbouring dog's timing (IoU with the real horn 0.39, below the 0.5 match threshold,
so it counts as one drop and one spurious).

## Failure rate by clip difficulty

| max concurrency in the clip | clips | imperfect | rate |
|---|---|---|---|
| 1 | 18 | 1 | 6% |
| 2 | 52 | 4 | 8% |
| 3 | 27 | 3 | 11% |
| 4 | 3 | 2 | 67% |

| gold events in the clip | clips | imperfect | rate |
|---|---|---|---|
| 5 | 28 | 2 | 7% |
| 6 | 72 | 8 | 11% |

The training set (`data/gen_esc50`, 20,000 clips, `--hard`) has 4-10 events per clip
(mode 7), so the benchmark's 5-6 events are not out of distribution in count. What it is
short of is four-deep overlap with a quiet sound underneath.

## Per-sound recall and precision (val + test pooled, label-aware, IoU 0.5)

| sound | gold | predicted | hits | recall | precision |
|---|---|---|---|---|---|
| keyboard typing | 53 | 53 | 51 | 0.962 | 0.962 |
| car horn | 39 | 38 | 38 | 0.974 | 1.000 |
| cat | 48 | 47 | 47 | 0.979 | 1.000 |
| sneezing | 22 | 23 | 22 | 1.000 | 0.957 |
| glass breaking | 23 | 24 | 23 | 1.000 | 0.958 |
| dog | 44 | 45 | 44 | 1.000 | 0.978 |
| clock alarm | 45 | 46 | 45 | 1.000 | 0.978 |
| church bells | 47 | 48 | 47 | 1.000 | 0.979 |
| laughing | 47 | 47 | 47 | 1.000 | 1.000 |
| door wood knock | 41 | 41 | 41 | 1.000 | 1.000 |
| footsteps | 42 | 42 | 42 | 1.000 | 1.000 |
| siren | 36 | 36 | 36 | 1.000 | 1.000 |
| rooster | 39 | 39 | 39 | 1.000 | 1.000 |
| train | 46 | 46 | 46 | 1.000 | 1.000 |

## The 18 imperfect test questions

| clip | root cause | questions hit | types |
|---|---|---|---|
| esc50_00075 | car horn dropped in a 4-deep overlap, spurious dog | 7 | PLAIN x2, ORDINAL, WHILE x2, NOT_FOLLOWED x2 |
| esc50_00233 | trailing spurious clock alarm | 5 | PLAIN, ORDINAL, AFTER, WHILE, NOT_FOLLOWED |
| esc50_00047 | cat dropped under car horn + dog | 4 | ORDINAL, WHILE x2, NOT_FOLLOWED |
| esc50_00193 | trailing spurious keyboard typing (in silence) | 2 | ORDINAL, AFTER |

`esc50_00174` (spurious glass breaking) costs no question because no test question asks
about glass breaking in that clip. 681 of 699 test questions are answered perfectly.

## What this suggests (not done)

1. **Harder training mix, not more of the same.** The 20k `--hard` set could be biased
   toward clips with 3-4 concurrent events and a quiet sound (keyboard typing, cat,
   footsteps) under a loud one; the failures are concentrated there and the model sees few
   such cases now.
2. **A trailing-event check, but not a geometric one.** 6 of 7 spurious events are the last
   one emitted, so a check on the last event is the right place. Measured with
   `ctag.trailing` (proper one-to-one matching, 100 val+test clips), the two rules that need
   no model rerun both fail:

   | rule on the last emitted event | thr | fired | removed spurious | removed correct | event F1 |
   |---|---|---|---|---|---|
   | overlaps another predicted event, val | IoU 0.5 | 4 | 1 | 3 | 0.992 -> 0.988 |
   | overlaps another predicted event, val | IoU 0.9 | 1 | 1 | 0 | 0.992 -> 0.993 |
   | overlaps another predicted event, test | IoU 0.5 | 5 | 3 | 2 | 0.989 -> 0.991 |
   | overlaps another predicted event, test | IoU 0.9 | 0 | 0 | 0 | unchanged |
   | audio under it is quiet, val | RMS ratio 0.3 | 9 | 0 | 9 | 0.992 -> 0.976 |
   | audio under it is quiet, test | RMS ratio 0.3 | 7 | 1 | 6 | 0.989 -> 0.980 |

   The benchmark overlaps 45% of its events (30 of 300 gold clips end with an event that
   overlaps another at IoU >= 0.5), so a correct last event often looks like a duplicate;
   and its quiet sounds sit under loud ones, so a silence gate removes real keyboard typing
   and cats. The threshold that helps on val (IoU 0.9) does nothing on test. What remains
   is the model's own stopping confidence: at every event boundary it chooses `},`
   (continue) or `}]` (stop), and the probability it put on stopping before the last event
   is recorded by `run_transcribe --stop-probs` (`ctag.stopprob`) and tested by
   `ctag.trailing --rule stop`, with the threshold picked on val. The runner does this in
   the next VM session (`CTAG_STOP_PROBS=1`, default); it needs one more pass over the 100
   clips. An earlier version of this note claimed the geometric rule would remove 6 of 7
   spurious events without touching a correct one; that was wrong.
3. **Do not rely on energy alone.** Only 1 of 7 spurious events is in silence, so an energy
   gate on its own removes one error and would have made the refinement mistake again.
4. **Report per-type numbers with the count-error mechanism attached.** WHILE and ORDINAL
   are not harder to reason about here; they are the types that a single miscount breaks.
   The 1,500-clip rebuild will shrink the +/-2-point noise but not this mechanism.
