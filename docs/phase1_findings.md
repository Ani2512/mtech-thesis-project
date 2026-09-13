# Phase 1 findings — compositional temporal audio grounding

All numbers from the ESC-50 composed benchmark (300 clips, 4404 queries, fixed
seed). Models run zero-shot on a Colab T4 in NF4 4-bit, 150 queries each,
`--max-new-tokens 96`. Mock baselines scored on the identical benchmark.

**Status: provisional.** The headline comparison rests on 150 queries per
model, 11–22 per condition type. Section 5 says exactly which claims survive
that sample size and which do not. A 1200-query run is the confirmation step.

## 1. Headline

| f1@0.5 | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| mock:oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mock:ignore_condition | 1.000 | 0.497 | 0.789 | 0.722 | 0.590 | 0.749 | 0.910 | 0.764 |
| mock:first_only | 0.758 | 0.269 | 0.260 | 0.885 | 0.370 | 0.627 | 0.590 | 0.542 |
| **Qwen2.5-Omni-7B** | **0.375** | 0.333 | 0.146 | 0.379 | 0.091 | 0.152 | 0.095 | 0.243 |
| **Qwen2-Audio-7B-Instruct** | **0.094** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 |

Count accuracy (predicted interval count equals gold count):

| count_acc | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED | ABSENT | ALL |
|---|---|---|---|---|---|---|---|---|---|
| mock:ignore_condition | 1.000 | 0.000 | 0.254 | 0.211 | 0.125 | 0.303 | 0.482 | 1.000 | 0.387 |
| Qwen2.5-Omni | 0.500 | 0.762 | 0.368 | 0.368 | 0.579 | 0.150 | 0.250 | 0.500 | 0.433 |
| Qwen2-Audio | 0.409 | 0.619 | 0.211 | 0.474 | 0.474 | 0.300 | 0.150 | 0.800 | 0.407 |

## 2. Qwen2-Audio cannot ground at all

0.094 on PLAIN and exactly 0.000 on every conditional type. Before accepting
this we ruled out three artefacts:

- **Is the audio reaching the model?** Yes. It produces 7.9 distinct answers
  per 10.6 queries on a clip and never a single canned response.
- **Is the parser at fault?** Partly, and it was fixed. It emits shapes like
  `[{'sneeze_start': '0.63', 'sneeze_end': '1.09'}]`, which `json.loads`
  rejects. Recovering those cut parse failures from 16% to 4% and raised
  f1@0.5 from 0.004 to 0.021. The conclusion did not change.
- **Is it pointing anywhere sensible?** Barely. 38% of PLAIN predictions land
  within 1 s of a gold event centre against 18% for a uniform random guess,
  but the predicted centre falls inside *any* labelled event 62% of the time
  when events cover 63% of the timeline — chance.

Its intervals are also ~5× too short: median 0.53 s against a 2.50 s gold.
That alone caps IoU near 0.21 and guarantees 0 at the 0.5 threshold whatever
the placement, so the two failures are confounded in the headline metric.
This is why `centre_error_median` and `duration_ratio_median` were added.

This matches **SpotSound** (arXiv:2604.13023, §4.1 Qualitative Results),
which reports on SpotSound-Bench that Qwen2-Audio "generates syntactically
complete time windows that suffer from severe semantic misalignment,
resulting in zero overlap with the ground truth."

*(Corrected 2026-09-12: this quote was previously attributed to TAG-Bench.
Qwen2-Audio is not evaluated in TAG-Bench at all — its 21 systems are Audio
Flamingo 2/3, FireRedAudio, GLM-4-Voice, Kimi-Audio, MiDashengLM, MiMo-Audio,
four MOSS-Audio variants, four Step-Audio-2 variants, Step-Audio-R1,
LLaMA-Omni2-14B, LLaMA-3.1-8B-Omni, Gemini-3.1-Pro, LAT-Audio and TimeAudio.)*

**Counter-evidence to weigh.** *From Semantics to Readout* (arXiv:2607.25355)
measures Qwen2-Audio at **0.3653 mIoU zero-shot** on its grounding task, rising
to 0.6199 after fine-tuning. That is not "cannot ground at all". The results are
not directly comparable — different data, mIoU rather than f1@0.5 (which is
all-or-nothing at the threshold), a different prompt, and 4-bit quantisation
here — but the claim should be stated as *cannot ground on this benchmark under
this prompt*, not as a property of the model. The independent support for the
strong reading is local: predicted centres land within 1 s of a gold centre only
38% of the time against 18% for random, and durations are ~5x too short.

**Consequence: Qwen2-Audio is a negative reference, not a subject.** Phase 2
cannot teach it to respect a condition when it cannot locate the sound.

## 3. Qwen2.5-Omni grounds, then drops the condition

PLAIN 0.375, conditional mean 0.199, parse failures 0.000. This is the
pattern phase 2 targets: the model finds the sound and then ignores the
qualifier attached to it.

The difficulty ordering is not uniform, and it is the more interesting result:

| Handled | Collapses |
|---|---|
| BEFORE 0.379, ORDINAL 0.333 | WHILE 0.152, AFTER 0.146, NOT_FOLLOWED 0.095, NEXT_AFTER 0.091 |

The three worst all require relating the target to a *second* event and then
selecting among the matches. ORDINAL needs only counting within one event
type, and BEFORE only a comparison against a single reference. So the
bottleneck looks like cross-event relational selection rather than "conditions"
in general.

Supporting detail: on NOT_FOLLOWED the model under-reports on 69% of queries,
the highest of any type, consistent with it collapsing a set of matches to one.

## 4. Diagnostic mock gaps

Mean absolute gap across conditional types, per model:

| | to ignore_condition | to first_only |
|---|---|---|
| Qwen2.5-Omni | 0.510 | 0.322 |
| Qwen2-Audio | 0.709 | 0.500 |

Neither model matches either failure mode cleanly. Both are worse than both
mocks, because the mocks are handed perfect grounding and only the *condition*
logic is ablated. The mocks therefore bound the achievable range rather than
describing real behaviour, which is how they should be read in the paper.

## 5. What does not yet survive the sample size

The gap between PLAIN and the conditional mean for Qwen2.5-Omni:

| quantity | value |
|---|---|
| n (PLAIN / conditional, non-rejection) | 22 / 77 |
| observed gap | 0.170 |
| permutation test p | 0.0298 |
| 95% bootstrap CI on the gap | [-0.010, 0.353] |

The permutation test is nominally significant; the bootstrap interval crosses
zero. Two reasonable tests disagreeing is what thin data looks like. **Do not
state the PLAIN-versus-conditional gap as established** until the 1200-query
run is in. The per-type ordering in section 3 rests on 11–17 queries per cell
and is weaker still — treat it as a hypothesis to confirm, not a result.

What *is* solid at this n: Qwen2-Audio scoring 0.000 across six conditional
types is not a sampling artefact, and the Qwen2.5-Omni versus Qwen2-Audio gap
on PLAIN (0.375 vs 0.094) is far too large to be noise.

## 6. Reproduction

```bash
python -m ctag.build_benchmark --source esc50 --n-clips 300 --p-overlap 0.45 \
       --out data/esc50 --esc50-root data/esc50_raw
for m in oracle ignore_condition first_only; do
  python -m ctag.run_zeroshot --model mock:$m --bench data/esc50/benchmark.jsonl \
         --out runs/esc50/mock_$m
done
python -m ctag.run_zeroshot --model qwen2.5-omni --bench data/esc50/benchmark.jsonl \
       --n 1200 --out runs/esc50/qwen25_omni
python -m ctag.run_zeroshot --model qwen2-audio --bench data/esc50/benchmark.jsonl \
       --n 1200 --out runs/esc50/qwen2_audio
```

Rate on a T4 in 4-bit: 4.0–4.1 s per query for both models, so 1200 queries is
roughly 80 minutes each. `ctag.rescore <run_dir>` re-parses a finished run
offline if the parser improves, so no run needs repeating for that reason.

## 7. Known gaps in this evidence

1. Composed ESC-50 audio only. Real recordings (DESED, TAG-Bench audio) are
   untested, so nothing here generalises beyond synthetic mixtures yet.
2. One prompt. No prompt sensitivity study, and TAG-Bench notes that an
   explicit "enumerate all intervals" instruction is an untested intervention.
3. Two models. Audio Flamingo 3 and the stronger systems in TAG-Bench are
   unmeasured here.
4. `f1@0.5` is the headline but conflates placement with duration. Report
   `centre_error_median` and `duration_ratio_median` alongside it.
