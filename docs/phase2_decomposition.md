# Decomposition ceiling: where the conditional failure actually comes from

A conditional query names a target X and usually a reference Y. The
decompose-and-combine agent (`ctag/agent.py`) never asks a model to honour
the condition. It asks only for plain groundings of X and of Y, then applies
the condition itself using the same predicates that define the ground truth.

Substituting a **perfect grounder** and then degrading it in controlled ways
separates two questions that the zero-shot numbers confound:

- Are these conditions intrinsically hard?
- Or is the model simply bad at locating sounds, so anything built on top fails?

All runs below are the full ESC-50 benchmark, 4404 queries, CPU only.

## 1. The conditions are not hard

| grounding quality | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| perfect | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| boundaries ±0.25 s | 0.983 | 0.979 | 0.970 | 0.977 | 0.969 | 0.978 | 0.960 | 0.974 |
| boundaries ±0.5 s | 0.935 | 0.940 | 0.928 | 0.922 | 0.923 | 0.902 | 0.921 | 0.925 |
| boundaries ±1.0 s | 0.797 | 0.768 | 0.751 | 0.736 | 0.762 | 0.718 | 0.740 | 0.756 |
| misses 10% of events | 0.902 | 0.810 | 0.836 | 0.820 | 0.793 | 0.808 | 0.907 | 0.844 |
| misses 25% of events | 0.781 | 0.606 | 0.620 | 0.571 | 0.553 | 0.579 | 0.767 | 0.650 |
| 25% false detections | 0.941 | 0.904 | 0.954 | 0.962 | 0.935 | 0.957 | 0.911 | 0.937 |
| ±0.5 s, 20% miss, 20% false | 0.735 | 0.631 | 0.580 | 0.596 | 0.625 | 0.553 | 0.710 | 0.640 |

Perfect grounding gives 1.000 on every condition type, which also confirms the
agent implements exactly the benchmark's semantics rather than approximating
them. Conditions are trivial *given* the events.

**Recall matters far more than precision.** Missing 25% of events costs 35
points (0.650); inventing 25% spurious ones costs 6 (0.937). A grounder for
this task should be tuned to over-detect, not under-detect.

## 2. Calibrating to the real model

Qwen2.5-Omni's measured PLAIN f1 is 0.375. Degrading the oracle until its
PLAIN score matches gives a like-for-like comparison:

| | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| agent at j1.5/d.35/s.35 | 0.334 | 0.245 | 0.205 | 0.252 | 0.190 | 0.240 | 0.334 | **0.263** |
| Qwen2.5-Omni direct | 0.375 | 0.333 | 0.146 | 0.379 | 0.091 | 0.152 | 0.095 | **0.243** |

Overall they are nearly identical, 0.263 against 0.243. But the per-type split
is the interesting part:

| decomposition wins | direct prompting wins |
|---|---|
| NOT_FOLLOWED 0.334 vs 0.095 | ORDINAL 0.245 vs 0.333 |
| NEXT_AFTER 0.190 vs 0.091 | BEFORE 0.252 vs 0.379 |
| WHILE 0.240 vs 0.152 | |
| AFTER 0.205 vs 0.146 | |

Decomposition rescues precisely the four types the model collapses on — the
ones requiring a relation to a second event plus a selection among matches —
and loses on the two the model already handles, where it pays for grounding Y
without needing to.

## 3. What this means for phase 2

**Grounding quality is the binding constraint, not condition handling.** With
perfect events, composition scores 1.000. At the model's actual grounding
quality, perfect composition still only reaches 0.263. So essentially the whole
gap from 1.000 to 0.26 is attributable to locating sounds, not to reasoning
about conditions.

Consequences, in priority order:

1. **Fine-tune for grounding first.** Training the model to respect conditions
   has a low ceiling while PLAIN sits at 0.375. Training data should therefore
   be weighted towards plain grounding, with conditional queries layered on.
2. **Decomposition is a free win on the relational types** and requires no
   training at all. It belongs in the paper as a baseline, and a hybrid that
   decomposes only AFTER/NEXT_AFTER/WHILE/NOT_FOLLOWED while answering ORDINAL
   and BEFORE directly should beat both arms.
3. **Tune the grounder for recall.** Section 1 shows misses cost six times what
   false alarms do.

## 4. Caveats

- The degraded oracle uses idealised noise: uniform boundary jitter, random
  drops, random false detections. Real model errors are structured — Qwen2-Audio
  showed a strong end-of-clip bias and intervals roughly five times too short.
  The calibration in section 2 matches on PLAIN f1 alone and is indicative, not
  exact.
- Comparison uses Qwen2.5-Omni at n=150, whose own confidence interval is wide.
  Rerun section 2 against the n=1200 numbers when available.
- Composed ESC-50 audio only; nothing here is validated on real recordings.
- The agent issues two grounding calls per query, so it costs roughly twice the
  inference of direct prompting, before caching. `run_agent` caches per
  (clip, sound), which is what makes it affordable on a full benchmark.

## 5. Reproduce

```bash
python -m ctag.run_agent --grounder oracle --bench data/esc50/benchmark.jsonl \
       --timelines data/esc50/timelines.jsonl --out runs/esc50/agent_oracle
python -m ctag.run_agent --grounder oracle --jitter 1.5 --drop 0.35 --spurious 0.35 \
       --bench data/esc50/benchmark.jsonl --timelines data/esc50/timelines.jsonl \
       --out runs/esc50/agent_calibrated
# with a real model as the grounder (GPU):
python -m ctag.run_agent --grounder qwen2.5-omni --bench data/esc50/benchmark.jsonl \
       --out runs/esc50/agent_omni
```
