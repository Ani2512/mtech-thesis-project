# Compositional temporal audio grounding — prior-art check, 2026-09-11

Candidate: queries with temporal conditions ("the horn after the siren", "the
second dog bark", "barking while music plays", "a slam not followed by
footsteps") whose answer is the set of matching time intervals; build data
programmatically from AudioSet-strong (+ mixing for "while"), LoRA-tune an
open LALM, evaluate with IoU.

## Verdict: not taken, but every half of it has a neighbour

| Half | Closest prior work | What it has | What it lacks |
|---|---|---|---|
| Query side (relational/ordinal questions from AudioSet-strong metadata) | **DAQA** (Fayek & Johnson, 2019, TASLP; facebookresearch/daqa, CC BY 4.0) | Programmatic before/after/ordinal/count questions over concatenated AudioSet events | Answers are event names / yes-no / counts, never intervals |
| Query side + LALM fine-tuning | **Sridhar et al., NAACL 2025 Industry** (Qualcomm) | GPT-4-generated temporal QA from AudioSet-SL timestamps (before/after, ordering, duration, counting); curriculum fine-tune of LTU | Free-text answers scored as QA; no interval output, no IoU, no ordinal grounding, no mixing |
| Text-side temporal cues | Oncescu et al., ACM MM 2024 (before/after/then/while in captions) | Synthetic captions + ordering loss | Retrieval, not localisation |
| Grounding benchmarks | TAG-Bench (2609.01542), AEGBench/Auto-AEG (2607.04383), DCASE 2026 AMR | Interval outputs, multi-occurrence, IoU metrics | **No relational, ordinal or conditional queries at all** (verified: TAG-Bench six query types are word/event/description/semantic/rough/emotion; Auto-AEG is "list all intervals of [label]"; DCASE AMR is free-text captions, one moment) |
| Concurrency reasoning | PolyBench (Interspeech 2026, 2603.05128) | "While X, is there another sound?" as 4-way MCQ over DESED/MAESTRO | MCQ only, no intervals, no method |
| Reasoning benchmarks | CompA-order (ICLR 2024), TREA (Interspeech 2025), STAR-Bench (2510.24693), MMAU | Order/count/duration as MCQ | No grounding |
| Video precedent | **CoMET-Bench** (2606.15320): conditional multi-event grounding, 4 temporal + 3 spatial condition types, Rejection-F1, training-free agent. **CompSTVG** (2608.30584): compositional queries, synthetic difficulty-graded curriculum + RL | Exactly the task shape, in video | No audio; no superposition-based "while" conditions |

## What is genuinely new in the audio version
1. Interval-set output for relational/ordinal/conditional queries (no audio work does this).
2. "While"/"during" conditions manufactured exactly by mixing — impossible in video.
3. Negative/rejection queries ("a slam not followed by footsteps") with a Rejection-F1-style metric, extending SpotSound's presence/absence idea to conditions.
4. A fine-tuning result (LoRA on Qwen2.5-Omni-7B or AF3) versus the CoMET-style training-free agent.

## Risks
- Framing is a port of CoMET/CompSTVG to audio; reviewers will say so. Answer: items 2–4.
- TAG-Bench (Alibaba) and Auto-AEG authors could add relational queries in a v2 within months. Mitigation: publish the benchmark + zero-shot failure curve early (workshop/arXiv), then the method.
- DAQA and Sridhar et al. must be cited as query-generation precedents; do not claim the question templates are new.

## First-week plan
Generate ~500 queries (5 condition types × ~100) from AudioSet-strong eval + mixed pairs; run Qwen2.5-Omni-7B, Audio Flamingo 3 and Gemini zero-shot; report mIoU / count accuracy / rejection-F1 per condition type. That curve is the paper's Figure 1 regardless of what the method later achieves.
