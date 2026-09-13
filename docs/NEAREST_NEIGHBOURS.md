# The nearest neighbours, and on which axis

The four works closest to this project are each close on a *different* axis.
Name them in this order; the intersection of all four is empty, and that empty
intersection is the gap.

| Neighbour | Closest on | What it shares | What it lacks |
|---|---|---|---|
| **CoMET-Bench** (video, arXiv:2606.15320) | task shape | Conditional multi-event grounding, rejection queries, interval-set answers, Rejection-F1, a decompose-style agent | Audio; superposition-made "while" |
| **TREA** (Interspeech 2025, arXiv:2505.13115) | data construction | Combines ESC-50 clips and knows the answer by construction | Concatenates, so nothing overlaps; multiple choice over 600 items |
| **TAG-Bench** (arXiv:2609.01542) | problem statement | "Return *every* interval matching the query"; documents that models find one occurrence and stop | Any conditional query; human-annotated |
| **TEMPO** (arXiv:2608.29999) | method | Atomic timestamp tokens; GRPO with verifiable rewards on temporal metrics | A task where its residual "segment merging" propagates into anything |

---

## Nearest by task shape — CoMET-Bench

*Conditional Multi-Event Temporal Grounding in Long-Form Video*, Zou et al., June 2026.

It is this task, in video — and video *only*: the full text has no mention of
audio, speech or transcripts. Four temporal condition types (Causal, Sequential,
Synchronous, Bounded) and three spatial (Static, Dynamic, Identity), rejection
queries, a Rejection-F1 metric, and a training-free agent (CoMET-Agent). The
mapping onto this project is nearly one-to-one: Sequential ≈ AFTER / BEFORE /
NEXT_AFTER, Synchronous ≈ WHILE, Bounded ≈ NOT_FOLLOWED. Causal and the three
spatial types have no audio analogue. The deck says the framing is a port of it and
cites it as such. Anyone who knows CoMET-Bench will see the resemblance
immediately, so name it first rather than let a reviewer find it.

The defences: mixing-made concurrency (video cannot superimpose two events at
the same pixels the way two sounds sum), rejection queries with exact
programmatic ground truth, the ceiling diagnostic, and the measured
miss/false-alarm asymmetry.

## Nearest by data construction — TREA

*Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning*,
Bhattacharya, Kulkarni, Ganapathy, Interspeech 2025.

Same source, same idea: combine ESC-50 recordings and know the answer because
you built the clip. Its ordering task literally asks *"which sound occurred
after X?"* — the AFTER query, with an event name as the answer instead of an
interval.

Three differences, all structural:

- TREA **concatenates**; this project concatenates *and mixes*. Sounds never
  overlap in TREA, so a "while" question cannot be asked there at all.
- TREA answers are **multiple choice** over 600 items; here they are **interval
  sets** over 4,404 queries.
- TREA's construction cannot produce rejection queries with exact ground truth
  for relational conditions; here about a quarter of relational queries have an
  empty answer by design.

This is the neighbour a committee member is most likely to raise as "hasn't this
been done?"

## Nearest by problem statement — TAG-Bench

*Benchmarking Temporal Audio Grounding in Large Audio Language Models*, Dai et
al., 1 September 2026.

Same base task — return every interval matching a natural-language query — and
the paper that documents the failure this project builds on. Across 21 systems:
*"every model under-reports the number of occurrences on one-to-many queries"*;
*"Models usually find one occurrence and stop"*; under-report rates 79.3–100 %;
best count accuracy 13.2 %.

It has no conditional queries and is human-annotated (1,750 pairs). It is the
strongest external motivation the project has, and the natural source of real
recordings for months 2–3 (CC BY 4.0).

## Nearest by method — TEMPO

*Temporally-grounded Multi-task Post-training for Large Audio-Language Models*,
Kulkarni et al., 30 August 2026.

Where the atomic timestamp tokens come from (one token per 0.1 s, embeddings
initialised as the mean of the number's BPE pieces, a distance-aware loss) and
where the training recipe for month 4 comes from (GRPO with verifiable rewards
that *"directly optimize the temporal metrics used at evaluation"*). Arm E
reproduces its representation and cites it.

Its own residual-error finding — *"dominated by segment merging and boundary
misalignment, even when the underlying semantic content is correct"* — is why
enumeration matters: merge two barks into one and "the second bark" has no
answer.

---

## Two more to be able to name

- **Auto-AEG / AEGBench** (arXiv:2607.04383) — the "list all intervals of X"
  benchmark with RL; the other half of the one-to-many story beside TAG-Bench.
- **DAQA** (Fayek & Johnson, TASLP 2020) and **Sridhar et al.** (NAACL 2025
  Industry, Qualcomm) — the query-generation precedents. They generate
  before/after/ordinal questions programmatically, with text answers instead of
  intervals. Cite them for the question templates; do not claim those as new.

---

## The one-sentence version

**CoMET-Bench is the same task in video; TREA is the same data built the simpler
way; TAG-Bench is the failure being explained; TEMPO is the method being
borrowed.** Four neighbours, four axes, empty intersection.
