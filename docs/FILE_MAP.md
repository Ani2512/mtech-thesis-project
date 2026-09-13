# What each file does

Repository `github.com/Ani2512/mtech-project`, branch `compositional-temporal-grounding`,
directory `research_project/`. 45 tracked files. Generated data, run outputs and
PDF exports are gitignored and rebuilt on demand.

---

## Top level

| File | What it does |
|---|---|
| `README.md` | The quick-start: the task in one table, how to build the benchmark, how to run each arm, what the output files contain. |
| `requirements.txt` | CPU dependencies (numpy, scipy, soundfile, pyyaml, pytest). GPU-only packages (torch, transformers, peft, bitsandbytes, librosa) are listed but commented out — they are installed on the GPU runtime, not locally. |
| `.gitignore` | Excludes `data/`, `runs/`, `logs/`, `__pycache__/` and `presentation/*.pdf`. Everything excluded is regenerable. |
| `kaggle_bootstrap.py` | **First cell on Kaggle.** Clones the branch, installs GPU deps, downloads ESC-50, builds the benchmark and splits, generates SFT data, then sweeps `--amp` settings with a 20-step smoke train to find one that both trains and is fast on the card. Safe to re-run; skips anything already done. |
| `kaggle_phase2.py` | **Second cell on Kaggle.** Runs all five arms unattended in an order that lands the essential arm C comparison first, scores arms A and B on the val split for the hybrid, prints the comparison tables, and copies `runs/` to the notebook output. Every step is skipped if its output exists, so a timeout resumes. |
| `scripts_phase1_full.sh` | Phase 1 as one resumable shell script: build benchmark, run the three mocks, run both real models. For Colab / any box with a GPU. |
| `phase1.ipynb` | Phase 1 as a Colab notebook (CPU half runs anywhere; the model cells need a GPU). |
| `phase1_kaggle.ipynb` | Same, laid out for Kaggle's runtime and 20 GB `/kaggle/working` limit. |
| `configs/phase1.yaml` | Benchmark parameters for phase 1: clip length, events per clip, overlap probability, seed. |

---

## `ctag/` — the package (16 modules)

### Building the benchmark

| File | What it does |
|---|---|
| `timeline.py` | **The task definition.** `Event(label, onset, offset)` and `Timeline`, with the eight predicates that define every condition type: `plain`, `ordinal`, `after`, `before`, `next_after`, `while_`, `not_followed`, `absent`. Pure and deterministic — ground truth is a function of the timeline alone. |
| `compose.py` | Builds the audio. Two sound banks (`ProceduralBank` for tests, `ESC50Bank` for real clips), silence-trimming so labelled onsets match audible ones, and `compose_clip()` which places events on a timeline with a controlled overlap probability — the thing that makes `WHILE` askable. Returns the mixed audio *and* the exact timeline. |
| `queries.py` | Turns a timeline into queries. Two or three phrasings per type, exact answers from the predicates, and `_pick()` which keeps about a quarter of relational queries as rejection cases (answer `[]`). Every query also stores its unconditioned answer for the diagnostic mocks. |
| `build_benchmark.py` | The CLI that ties the above together: `python -m ctag.build_benchmark --source esc50 --n-clips 300 --out data/esc50`. Writes `benchmark.jsonl`, `timelines.jsonl` and `wav/`. Deterministic seed, so everyone's benchmark is byte-identical. |
| `split.py` | Clip-level train / val / test split by a stable hash of `(seed, clip_id)`, so adding clips never reshuffles existing ones and no clip leaks across splits. Asserts the leak check. |

### Scoring

| File | What it does |
|---|---|
| `metrics.py` | `parse_intervals()` — a defensive parser for the shapes models actually emit (single-quoted dicts, `'x_start'/'x_end'` keys, `"1.2-3.4"` strings, time tokens). Then the metrics: union-IoU, Hungarian-matched F1 at a threshold, count accuracy, under-report rate, rejection precision/recall/F1, and `localisation()` which separates *where* from *how long*. `summarize()` aggregates per condition type. |
| `recall_bias.py` | The method. `MEASURED_BETA = √5.60 = 2.37`, pinned by a test. `f_beta()`, `union_decode()` (sample-and-vote merging), and `make_preference_pairs()` whose rejected side always under-detects. |
| `rescore.py` | Re-parses and re-scores a finished run offline, so a parser improvement never requires re-running the model. |

### The arms

| File | What it does |
|---|---|
| `models.py` | The backends behind one `ground()` interface: `MockBackend` (`oracle`, `ignore_condition`, `first_only`, `jitter` — executable hypotheses about how models fail), `Qwen25OmniBackend`, `Qwen2AudioBackend`, `GeminiBackend`. Holds the system prompt used identically at train and inference, the fp16-vs-4-bit memory planner, and adapter loading. |
| `run_zeroshot.py` | **Arm A** (and C / E at inference). Runs one backend over a benchmark file → `predictions.jsonl` + `summary.json`. `--adapter` attaches a trained LoRA; `--samples k --min-votes m` enables vote-and-merge decoding. |
| `agent.py` | **Arm B.** Decompose-and-combine: two plain grounding calls (target X, reference Y), then `combine()` applies the condition locally, mirroring `timeline.py` exactly. Also `oracle_grounder` and `noisy_grounder`, which turn the agent into the ceiling diagnostic that runs on CPU. |
| `run_agent.py` | CLI for arm B: `--grounder qwen2.5-omni` for the real model, `--grounder oracle` with `--jitter/--drop/--spurious` for the diagnostic. Caches groundings per `(clip, sound)`. |
| `hybrid.py` | **Arm D.** Chooses direct-or-agent per condition type on the *val* split, reports on the *test* split. Refuses to run with an empty selection set and rejects val runs that overlap test runs. |
| `sft_data.py` | Builds supervised fine-tuning examples in the exact inference prompt format. `--plain-ratio` weights the mix toward plain grounding (the diagnostic's conclusion) and synthesises extra PLAIN and absent-sound examples for free from the timeline. `--time-tokens` emits atomic timestamp targets for arm E. |
| `train_lora.py` | **Arms C and E.** QLoRA on the Qwen2.5-Omni thinker, audio encoder frozen. The collator masks loss to the answer *counted from the end*, caps sequence length to bound the logits allocation, and `--preflight` runs one forward+backward on the longest example before training so an OOM costs a minute. Refuses to report success on a run with NaN gradients or zero loss. |
| `timetokens.py` | **Arm E's representation.** `TimeVocab`: one token per 0.1 s, embeddings initialised as the mean of the number's BPE pieces (TEMPO), a distance-aware soft-label loss, and round-half-up quantisation. `wrap_new_rows()` trains only the 301 new vocabulary rows (~4 MB) instead of both full matrices (~16 GB); `save_deltas()`/`load_deltas()` persist them beside the adapter. |
| `__init__.py` | Package marker. |

---

## `tests/`

| File | What it does |
|---|---|
| `test_grounding.py` | 46 tests. Timeline predicates against hand-checked cases; query generation invariants; parser shapes; metric edge cases (empty vs unparseable, rejection metrics returning `None`); the agent's `combine()` matching `Timeline`; the split leak check; collator masking under placeholder expansion and sequence capping; the new-rows embedding/head wrappers and their save/load round trip; `MEASURED_BETA` pinned; hybrid selection and its refusal cases. |

---

## `docs/`

| File | What it does |
|---|---|
| `PROJECT_EXPLAINER.md` | **Start here.** The whole project from scratch for someone who knows nothing: problem, literature, core insight, architecture, every module with code snippets explained, all results, failures, limitations, plan, and anticipated questions. |
| `query_semantics.md` | The precise definition of each condition type — what "after" means at the boundary, why the reference must be unique, what the `NOT_FOLLOWED` window is. |
| `compositional_grounding_check.md` | The prior-art check from 11 September: what neighbours exist, what each lacks, what is genuinely new, and the risks. |
| `references.md` | All 24 cited works with verified titles, authors, dates and arXiv IDs, plus a log of the citation errors found and corrected. |
| `data_licensing.md` | Licences of every data source used and avoided. |
| `phase1_findings.md` | Zero-shot results for both backbones with the artefact checks, the difficulty ordering, and the sample-size caveats. |
| `phase2_decomposition.md` | The ceiling diagnostic: perfect grounding → 1.000, degraded → 0.263, and what that implies for training. |
| `phase2_plan.md` | The five arms, what each isolates, recommended settings, and the memory post-mortem after the first run. |
| `recall_bias.md` | The asymmetry measurement (slopes 1.411 vs 0.252), the derivation of beta, the two mechanisms, and the caveats. |
| `timestamp_tokens.md` | Why atomic time tokens, the TEMPO vs TimeAudio designs, the quantisation trap, and the new-rows memory fix. |
| `six_month_plan.md` | Month-by-month plan with the decision rule for a failed arm C. |
| `FILE_MAP.md` | This file. |

---

## `presentation/`

| File | What it does |
|---|---|
| `Compositional_Temporal_Audio_Grounding_2026-09-12.pptx` | The 18-slide review deck. |
| `SPEAKER_NOTES.md` | Read-aloud notes per slide with timings, plus answers to likely questions. |
| `build_deck.py` | Regenerates the deck from the numbers. Change a value here, run it, and the slide updates. |
| `make_figures.py` | Regenerates the four charts in `figures/` with matplotlib. |
| `figures/fig1_backbone.png` | Phase 1: Qwen2.5-Omni vs Qwen2-Audio per condition type. |
| `figures/fig2_direct_vs_decompose.png` | Arm A vs arm B on the test split, grouped by prediction, marked held/flipped. |
| `figures/fig3_asymmetry.png` | The two degradation curves and their slopes. |
| `figures/fig4_decoding.png` | Vote-and-merge vs single sample, with the simulated 0.951 line. |
| `*.pdf` | Exports of the deck and notes — gitignored, regenerated from the sources above. |

---

## Generated, not tracked

| Path | What appears there |
|---|---|
| `data/esc50/` | `benchmark.jsonl`, `timelines.jsonl`, `wav/`, the three split files, `sft_*.jsonl`. Built by `build_benchmark`, `split`, `sft_data`. |
| `data/esc50_raw/` | The ESC-50 download (~600 MB, once). |
| `runs/esc50/<arm>/` | `predictions.jsonl` (one row per query with raw text, parsed intervals and every metric) and `summary.json` (per-type aggregates). |
| `/kaggle/temp/lora_text`, `/kaggle/temp/lora_tt` | Trained adapters for arms C and E, with `time_deltas.pt` beside the arm E one. |
