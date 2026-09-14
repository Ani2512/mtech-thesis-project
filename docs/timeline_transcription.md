# Whole-timeline transcription (phase 3 target)

Decided 2026-09-14 from the phase 2 error budget (`results_kaggle_v6.md`): of
the 502 non-rejection test queries, the best arm gets 55% perfect, answers
"nothing" wrongly on 19%, lists too few occurrences on 8%, and misplaces the
right number on 14%. The false empties are a *decision* the model makes; the
decomposition already removed them for the types it handles by computing the
condition in code. This takes that to its end: the model is asked one thing per
clip, **write out every event with its start and end**, and every query type is
computed from that list with the predicates in `ctag.timeline`. The "nothing"
decision is never the model's, and the benchmark collapses to a single
event-transcription accuracy.

## Target

```
Locate: every sound event in the recording, as a JSON list of objects
{"sound": name, "start": seconds, "end": seconds} sorted by start, using only
these sound names: car horn, cat, ... . The recording is 20.0 seconds long.
Reply with only the JSON list.

[{"sound": "dog", "start": 0.98, "end": 3.48}, {"sound": "siren", "start": 2.31, "end": 4.81}, ...]
```

Same `SYSTEM` and `prompt_for` as every other arm, so training and inference
are byte-identical. Names are the human phrases (`door wood knock`); the
parser and scorer normalise both sides (`ctag.agent._norm`).

## Code

| what | where |
|---|---|
| target string, parser, event-level scorer, grounder from predictions | `ctag/transcribe.py` |
| training set: one example per clip | `python -m ctag.sft_data --task transcribe --timelines ... --bench benchmark_train.jsonl --out sft_transcribe.jsonl` |
| transcribe + score clips (CPU mock or a model) | `python -m ctag.run_transcribe --model mock:oracle|qwen2.5-omni [--adapter ...] --bench ... --timelines ... --out ...` |
| every query type from predicted timelines, no model calls | `python -m ctag.run_agent --grounder timeline --pred-timelines <out>/pred_timelines.jsonl --bench ... --out ...` |
| large training set with hard cases | `python -m ctag.gen_train --source esc50 --n-clips 20000 --hard --workers 8 --out data/gen_esc50 --esc50-root data/esc50_raw` |

The scorer reports event-level precision, recall and F1 with **label-aware**
one-to-one matching at IoU 0.5, plus `f1_any_label` (a heard-but-misnamed
event is not a miss), count accuracy, under-report rate, centre error,
duration ratio and recall per sound. The gate for phase 3 is event F1 ≥ 0.95
on the validation clips; with the gold timelines the grounder reproduces every
benchmark answer exactly (`test_timeline_grounder_answers_every_query_type_exactly`).

## Generator

`ctag.gen_train` composes clips as a pure function of `(seed, index)`, so the
timelines file is the durable artefact and audio can be regenerated anywhere
with the same command. `--hard` draws the recipe per clip from three modes:
crowded (7–10 events, 2–3 labels, gaps 0.1–0.8 s), overlap-heavy (p_overlap
0.55–0.85), and benchmark-like with mild variation; SNR 6–30 dB; minimum
overlap 0.2–0.6 s. `gen_stats.json` records events per clip, clips with an
overlap, repeated labels and the label histogram. `--queries` also emits
per-question rows in the benchmark format, for mixed training. Clip ids are
`gen<seed>_<i>` and never collide with `esc50_*`.

Sizes: a 20 s clip is 0.64 MB as PCM-16 wav; 20,000 clips ≈ 13 GB (about half
with `--format flac`). Generation is CPU-bound and parallel (`--workers`).

## Caveats to state

- ESC-50 source recordings are shared between the generated training clips and
  the benchmark test clips (as they already are between the benchmark's own
  splits). This is a *composition* split, not a *source* split; the real-recording
  track is what tests generalisation to unseen sources.
- Every fine-tuned arm so far predicts the training-set duration
  (`duration_ratio_median` 1.000). The hard recipes vary event length only through
  the source clip; `min_overlap` and gaps vary, durations still cap at 2.5 s.
- The vocabulary list in the prompt closes the label set. Leave it out
  (`--no-vocab-in-prompt`) for the open-vocabulary variant.
