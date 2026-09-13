# References used in the review deck

Two related-work slides cite 24 works.

Sourcing:

- **[verified]** — fetched from arXiv on 2026-09-12; title and authors below are
  the ones on the abs page.
- **[doc]** — the identifier appears in a repo doc written during the
  2026-09-11 prior-art check, so it was read at the time, but it has **not**
  been re-fetched. Check before it goes into a paper.

Three identifiers were wrong when first written into the deck and are corrected
below; see the bottom of this file.

## Temporal grounding in audio LLMs (deck slide 3)

All twelve verified against arXiv on 2026-09-12.

| Work | Verified title / venue | ID |
|---|---|---|
| TimeAudio | *Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models* — Wang, Li, Ma, Liu, Wang; 14 Nov 2025 (v2 12 Dec 2025). **TimeAudio is the method name, not the title.** | arXiv:2511.11039 |
| SpotSound | *SpotSound: Enhancing Large Audio-Language Models with Fine-Grained Temporal Grounding* — Sun, Zhou, Li, Zhang, Wang, Xie; 14 Apr 2026 (v2 10 Aug 2026). Introduces SpotSound-Bench. | arXiv:2604.13023 |
| Audio-Side Time Prompt | *Towards Fine-grained Temporal Perception: Post-Training Large Audio-Language Models with Audio-Side Time Prompt* — Shi, Cai, Liu, Gu, Jiang, Dai, McLoughlin, Song; 15 Apr 2026. Method is TimePro-RL. | arXiv:2604.13715 |
| TEMPO | *TEMPO: Temporally-grounded Multi-task Post-training for Large Audio-Language Models* — Kulkarni, Jayakumar, Ghosh, Aich, Duraiswami, Manocha; 30 Aug 2026 | arXiv:2608.29999 |
| Auto-AEG | *Auto-AEG: Scalable Data Construction for Open-Vocabulary Audio Event Grounding* — Zhang, Cheng, Yan, Zhang, Fu, Zhang, He, Jin; 5 Jul 2026. Introduces AEGBench. | arXiv:2607.04383 |
| AudioMap | *AudioMap: Cloze-and-Choice Reinforcement Learning for Time-Aware Dense Audio Captioning* — Rong, Ma, Li, Wang, Zhang, Liu; 10 Aug 2026 | arXiv:2608.09559 |
| Listening with Time | *Listening with Time: Precise Temporal Awareness for Long-Form Audio Understanding* — Shao, Su, Tian, Mu, Lin, Fan, Luo, Luan, Xie; 24 Apr 2026 | arXiv:2604.22245 |
| Encode Once, Decode Never | *Encode Once, Decode Never: Reusing Audio LM Internals for Efficient Temporal Localization* — An, Keung, Wang, Ahia, Smith; 10 Feb 2026 (v2 22 Jul 2026) | arXiv:2602.10230 |
| LA-RAG | *Event-Grounded Question Answering over Long Audio via Structured Retrieval* — Hegde, Sridhar, Vakada, Guo, Visser; 16 Feb 2026 (v6 14 Aug 2026) | arXiv:2602.14612 |
| SpectCount | *SpectCount: Spectrotemporal Counting via Synthetic Signals Improves Large Audio Language Models* — Kim, Jun, Kang, Hong, Lee, Kim; 5 Jun 2026 | arXiv:2606.06907 |
| TAG-Bench | *TAG-Bench: Benchmarking Temporal Audio Grounding in Large Audio Language Models* — Dai, Shu, Li, Xie, Li, Yu; 1 Sep 2026 (v2 2 Sep). 1,750 query–recording pairs, 149.5 h, eight subsets, 7 s–20 min. | arXiv:2609.01542 |
| DCASE 2026 | **Task 6, Audio Moment Retrieval from Long Audio.** (Task 5 is Audio-Dependent Question Answering, multiple choice.) | dcase.community |

## Compositional and relational queries (deck slide 4)

All verified against arXiv, the ACL Anthology or ISCA on 2026-09-12.

| Work | Verified title / venue | ID |
|---|---|---|
| DAQA | *Temporal Reasoning via Audio Question Answering* — Fayek & Johnson. **IEEE/ACM TASLP vol. 28, pp. 2283–2294, published 2020**; arXiv Nov 2019. Code: facebookresearch/daqa. Proposes the MALiMo model. | arXiv:1911.09655 |
| Sridhar et al. | *Enhancing Temporal Understanding in Audio Question Answering for Large Audio Language Models* — Sridhar, Guo, Visser (Qualcomm). NAACL 2025 Industry Track, `2025.naacl-industry.78`. | arXiv:2409.06223 |
| **TREA** | *Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning* — Bhattacharya, Kulkarni, Ganapathy. Interspeech 2025. **Built by concatenating ESC-50 recordings — the same source and construction as this project.** 600 MCQ items over TREA-D (duration), TREA-O (ordering), TREA-C (counting). | arXiv:2505.13115 |
| Oncescu et al. | *Dissecting Temporal Understanding in Text-to-Audio Retrieval* — Oncescu, Henriques, Koepke. ACM MM 2024. TempTest_rev / TempTest_rep sets, AudioCaps_uni. | arXiv:2409.00851 |
| PolyBench | *PolyBench: A Benchmark for Compositional Reasoning in Polyphonic Audio* — Chen, Xiao, Yin, Liu, Huang, Dang; 5 Mar 2026. Five subsets: counting, classification, detection, concurrency, duration. | arXiv:2603.05128 |
| CompA-order | *CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models* — Ghosh et al. ICLR 2024. CompA-order = 400 audio–caption matching instances probing event order. | arXiv:2310.08753 |
| STAR-Bench | *STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence* — Liu et al.; 28 Oct 2025 (v2 28 Nov 2025). Segment reordering, spatial localisation, 19 models. | arXiv:2510.24693 |
| MMAU | *MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark*. 10k clips, 27 tasks, 18 models. | arXiv:2410.19168 |
| CoMET-Bench | *Conditional Multi-Event Temporal Grounding in Long-Form Video* — Zou et al.; 13 Jun 2026. **Video only — full text checked 2026-09-12: no audio, speech or transcript anywhere.** Temporal conditions: Causal, Sequential, Synchronous, Bounded. Spatial: Static, Dynamic, Identity. Rejection-F1, CoMET-Agent. Sequential ≈ AFTER/BEFORE/NEXT_AFTER, Synchronous ≈ WHILE, Bounded ≈ NOT_FOLLOWED here; Causal and the spatial types have no audio analogue. | arXiv:2606.15320 |
| CompSTVG | *Learning Compositional Spatio-Temporal Video Grounding with Synthetic Curriculum* — Wang et al.; 31 Aug 2026. **Video.** Scene-graph synthetic data engine, curriculum RL. | arXiv:2608.30584 |
| MUSEG | *MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding* — Luo et al.; 27 May 2025 (v2 18 Apr 2026). **Video.** NeurIPS 2025 venue not stated on the abs page — claim dropped. | arXiv:2505.20715 |
| One-to-Many TG | *Towards One-to-Many Temporal Grounding* — Xu, Tan, Chen, Meng, Wang, Ji, Fei, Li; 4 Jun 2026 (v2 21 Jun). **Video.** | arXiv:2606.06294 |

## Backbones and infrastructure

Qwen2.5-Omni-7B (subject), Qwen2-Audio-7B-Instruct (negative reference),
ESC-50 (source clips).

*Reinforcement Learning for Data-Efficient Code-Switched ASR* — Ye & Vickers,
2 Jul 2026, arXiv:2607.02757. Cited in `six_month_plan.md` as a verifiable-reward
precedent, which is accurate. It is **not** a general RLVR reference — do not cite
it as one.

## Corrections made on 2026-09-12

Descriptions that did not match the paper, now fixed in the deck:

1. **2608.09559** was "GRPO with IoU reward". AudioMap is cloze-and-choice RL for
   dense audio captioning.
2. **2602.10230** was "frame-level Poisson head". It is *Encode Once, Decode
   Never*. **If a Poisson-head paper exists its identifier was never recorded and
   it is currently uncited.**
3. **2604.22245** was "LAT-Audio / LAT-Bench", sliding window plus crop tool. The
   title is *Listening with Time*; the mechanism is global-to-local iterative
   reasoning. **"LAT-Bench" appears nowhere and should not be cited.**
4. **2511.11039** — TimeAudio is the method; the paper is *Listening Between the
   Frames*. The "+3.0 mIoU from markers alone" figure is a body claim, not in the
   abstract; it was removed from the deck.
5. **2609.01542** — the deck claimed "21 models" and six named query types.
   Neither is in the abstract; replaced with the verified 1,750 pairs / 149.5 h /
   eight subsets.
6. **DAQA** is TASLP **2020**, not 2019 (arXiv 2019).
7. **PolyBench** — the "DESED / MAESTRO, 4-way MCQ" description came from the
   2026-09-11 reading and is not in the abstract; replaced with the five subsets
   the abstract names, one of which is **detection**.

## The finding that matters

**TREA (Interspeech 2025) builds its audio exactly the way this project does — by
combining ESC-50 recordings.** It was in the notes as a weak "order/count MCQ"
neighbour; it is in fact the closest work by construction. The distinctions still
hold and must be stated rather than assumed:

- TREA **concatenates**; this project concatenates *and mixes*, which is the only
  reason WHILE conditions can exist.
- TREA answers are **multiple choice** over 600 items; this project answers with
  **interval sets** over 4404 queries.
- TREA's ordering task asks "which sound occurred after X?" and the answer is an
  event name. Here the same question returns a time interval.

Expect a panel member who knows TREA to ask. The answer is concurrency and
interval-set output.

## Body claims checked against the PDFs (2026-09-12)

Both load-bearing claims were verified in the full text, and a third turned out
to be misattributed.

**TAG-Bench (arXiv:2609.01542) — confirmed, and stronger than recorded.**

- 21 systems evaluated, across general-purpose LALMs, omni-modal systems and
  temporal-grounding experts.
- "387 queries (22.1%) carry ≥2 ground-truth intervals (up to 18 per query)."
- "The highest one-to-many count accuracy is 13.2% (Step-Audio-R1), followed by
  Gemini-3.1-Pro at 7.0%, FireRedAudio at 4.7%." No model exceeds 13.2%.
- "every model under-reports the number of occurrences on one-to-many queries";
  "Models usually find one occurrence and stop." Under-report rates run from
  **79.3% to 100%** — this was not in the notes and is worth quoting.
- Best system reaches 31.2 mIoU; 9 of 21 fall below 5 mIoU.
- The eight subsets are Word, Acoustic event, Acoustic description, Semantic,
  Rough semantic, Emotion, Long–semantic, Long–acoustic description. The notes
  recorded **six**; there are eight.

**TEMPO (arXiv:2608.29999) — confirmed verbatim.**

- Uses "GRPO with verifiable rewards designed to directly optimize the temporal
  metrics used at evaluation"; the grounding reward "averages soft F1@IoU0.5,
  symmetric mIoU, and a boundary score".
- §6: "TEMPO's residual errors are dominated by **segment merging** and boundary
  misalignment, even when the underlying semantic content is correct."

**Misattribution found and fixed.** The line "Qwen2-Audio generates syntactically
complete time windows that suffer from severe semantic misalignment, resulting in
zero overlap with the ground truth" was credited to TAG-Bench. It is **SpotSound**,
arXiv:2604.13023, §4.1 Qualitative Results, on SpotSound-Bench. **Qwen2-Audio is
not evaluated in TAG-Bench at all** — the 21 systems are Audio Flamingo 2 / AF2
Sound-CoT / Audio Flamingo 3, FireRedAudio, GLM-4-Voice-9B, Kimi-Audio-7B,
MiDashengLM-7B, MiMo-Audio-7B-Instruct, four MOSS-Audio variants, four
Step-Audio-2-mini variants, Step-Audio-R1, LLaMA-Omni2-14B, LLaMA-3.1-8B-Omni,
Gemini-3.1-Pro, LAT-Audio and TimeAudio. Corrected in `phase1_findings.md` and on
deck slide 7.

**Counter-evidence on the "cannot ground at all" claim.** *From Semantics to
Readout* (arXiv:2607.25355) measures Qwen2-Audio at **0.3653 mIoU zero-shot**,
improving to 0.6199 after grounding fine-tuning. Not comparable to the 0.094
f1@0.5 measured here — different data, mIoU versus an all-or-nothing threshold
metric, a different prompt, 4-bit quantisation — but the claim must be phrased as
*cannot ground on this benchmark under this prompt*. Added to the limitations
slide.

**Earlier correction walked back:** "LAT-Audio" **is** a real system name —
TAG-Bench lists it among its temporal-awareness expert baselines alongside
TimeAudio. Only "LAT-Bench" remains unconfirmed.
