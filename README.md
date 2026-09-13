# Compositional Temporal Audio Grounding

**Task.** Given an audio recording and a query with a temporal condition,
return *every* time interval that satisfies it, or an empty list if none does.

| Condition type | Example query | Answer |
|---|---|---|
| `PLAIN` | "every dog bark" | all bark intervals |
| `ORDINAL` | "the second dog bark", "the last dog bark" | one interval, or none |
| `AFTER` / `BEFORE` | "every dog bark after the car horn" | bark intervals starting after the horn ends |
| `NEXT_AFTER` | "the first dog bark after the car horn" | one interval |
| `WHILE` | "every dog bark while music is playing" | bark intervals that overlap music |
| `NOT_FOLLOWED` | "every dog bark not followed by footsteps within 3 s" | subset of barks |
| `ABSENT` | "every cat meow" (no meow in the audio) | `[]` |

Every query has ground truth computed by a deterministic predicate over an
exact event timeline (see `docs/query_semantics.md`), so the benchmark has no
annotation noise. Audio comes from two tracks:

- **composed**: isolated event clips (ESC-50 / FSD50K) mixed onto backgrounds
  with controlled overlap, so `WHILE` conditions are manufactured exactly;
- **procedural**: synthetic tones/noises with the same pipeline, no downloads,
  used for tests and dry runs.

Real-recording evaluation (DESED, TAG-Bench audio) is phase 1b.

## Phase 1: zero-shot failure curve

Run open audio LLMs on the benchmark and plot metric-by-condition-type. That
figure is the paper's Figure 1 regardless of what the fine-tuning later does.

```bash
pip install -r requirements.txt
python -m pytest tests -q

# procedural benchmark + mock backends, no GPU, no downloads
python -m ctag.build_benchmark --source procedural --n-clips 60 --out data/proc
python -m ctag.run_zeroshot --model mock:ignore_condition --bench data/proc/benchmark.jsonl --out runs/mock_ignore
python -m ctag.run_zeroshot --model mock:first_only      --bench data/proc/benchmark.jsonl --out runs/mock_first

# composed benchmark from ESC-50 (downloads ~600 MB once)
python -m ctag.build_benchmark --source esc50 --n-clips 300 --out data/esc50_bench

# real models (GPU runtime)
python -m ctag.run_zeroshot --model qwen2.5-omni --bench data/esc50_bench/benchmark.jsonl --out runs/q25o
python -m ctag.run_zeroshot --model qwen2-audio  --bench data/esc50_bench/benchmark.jsonl --out runs/q2a
python -m ctag.run_zeroshot --model gemini       --bench data/esc50_bench/benchmark.jsonl --out runs/gemini   # needs GEMINI_API_KEY
python -m ctag.run_zeroshot --model audio-flamingo-3 --bench data/esc50_bench/benchmark.jsonl --out runs/af3  # NVIDIA AF3, noncommercial licence
```

Outputs per run: `predictions.jsonl` (query, raw model text, parsed intervals,
per-query metrics) and `summary.json` (per condition type: union-IoU,
F1@0.5 with Hungarian matching, count accuracy, rejection precision/recall,
parse-failure rate).

## Layout

```
ctag/
  timeline.py        Event / Timeline and the predicates behind every condition type
  queries.py         query templates + ground-truth generation from a Timeline
  compose.py         build timelines + mixed audio (procedural or ESC-50 event bank)
  metrics.py         interval parsing, union-IoU, matched F1, count accuracy, rejection metrics
  models.py          backends: mock:*, qwen2.5-omni, qwen2-audio, gemini
  build_benchmark.py CLI: compose clips -> benchmark.jsonl + wavs
  run_zeroshot.py    CLI: model x benchmark -> predictions + summary
docs/
  query_semantics.md              exact definitions (cite in the paper)
  compositional_grounding_check.md prior-art check, 2026-09-11
  data_licensing.md               sources and licences
configs/phase1.yaml               run matrix
tests/                            no GPU, no downloads
```

## Baselines to beat (phase 2)
CoMET-style training-free agent (decompose the condition, ground each part,
combine) versus LoRA fine-tuning on composed queries. Both evaluated on the
same benchmark plus the real-recording track.

## Running it

`phase1_kaggle.ipynb` is the recommended path. Kaggle gives 30 GPU hours a week
and 12-hour sessions, and "Save & Run All (Commit)" executes the whole notebook
server-side with the browser closed. Set **Accelerator → GPU** and **Internet →
On** in the settings panel first, or cells 2 and 5 fail.

`phase1.ipynb` is the Colab equivalent. It works, but free-tier Colab reclaims
runtimes mid-job, which cost us an 80-minute run. If you use it, launch through
`scripts_phase1_full.sh`, which is idempotent and resumes rather than restarting.

Neither notebook is required: `scripts_phase1_full.sh` runs the whole pipeline
unattended on any machine with a GPU.
