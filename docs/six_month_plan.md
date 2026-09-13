# Six-month plan

Decided 2026-09-11. The governing constraint is not compute or ideas: it is how
fast real recordings enter the pipeline. Everything measured so far lives on
composed ESC-50 audio that we generated, and that is the single sentence a
reviewer will press on.

## Already done (day 1)

- Benchmark: 8 condition types, 300 composed clips, 4404 queries, ground truth
  exact by construction. `WHILE` conditions manufactured by mixing, which video
  cannot do.
- Prior-art check: no audio benchmark contains relational, ordinal or conditional
  queries (`docs/compositional_grounding_check.md`).
- Phase 1: Qwen2-Audio cannot ground at all (PLAIN 0.094, all conditional 0.000);
  Qwen2.5-Omni grounds and drops the condition (0.375 vs 0.199).
- **Decomposition ceiling**: perfect grounding scores 1.000, but at the model's
  real grounding quality perfect condition logic reaches only 0.263. Grounding,
  not reasoning, is the bottleneck. Reproduced independently on Kaggle.
- **Recall asymmetry**: a unit of miss rate costs 1.411, a unit of false-alarm
  rate 0.252, a ratio of 5.60. Measurable only on a benchmark where conditions
  depend on a second event.
- Phase 2 code for five arms, 38 tests.

## Month 1 — close composed-audio, publish early

| | |
|---|---|
| Goal | Every arm measured on composed audio; benchmark and diagnosis on arXiv |
| Deliverable | Workshop-length paper; the benchmark released |
| Why now | TAG-Bench (Alibaba) appeared 2026-09-02 and is one step from this. Publishing timestamps the contribution. |

1. Get `train_lora` training properly (in progress; the smoke test currently
   shows nan grad_norm, fixes pushed).
2. Arms A–E on the test split, plus the n=1200 phase 1 confirmation.
3. Write up benchmark + decomposition ceiling + recall asymmetry. The method can
   be described as ongoing; the diagnosis is the contribution.

**Decision point:** if arm C does not beat arm A on PLAIN, the frozen audio
encoder is the likely cause. Report it as a limitation and move on. Do not spend
month 2 unfreezing the encoder on 210 clips.

## Months 2–3 — real recordings (critical path)

| | |
|---|---|
| Goal | A compositional test set over real audio, conditions verified by hand |
| Deliverable | ~300 real clips with verified conditional queries; every arm rerun |
| Risk | The composed numbers may look optimistic. That is a finding. |

1. Sources: DESED (CC BY 4.0, strong labels, 10 s domestic), AudioSet-strong
   labels, TAG-Bench audio (CC BY 4.0, 149.5 h).
2. Derive conditional queries from existing strong labels using the same
   predicates, then **verify by hand**. The predicates give candidate answers;
   a human confirms the event boundaries are right and the condition reads
   naturally.
3. Rerun everything. Expect the gap between composed and real to be the most
   quoted number in the thesis.

**Start this in month 2, not month 4.** It is the part that cannot be rushed and
the part that makes the work hard to scoop.

## Month 4 — the method, properly

1. F-beta as a verifiable reward under GRPO, following Auto-AEG (2607.04383) and
   the code-switching RLVR work (2607.02757), rather than the current offline
   preference pairs.
2. Rerun recall-biased decoding with a **real** model. Current numbers come from
   simulated grounders with independent noise; real errors are correlated, which
   should weaken both the union gain and the voting filter.
3. The hybrid routing rule, selected on val and reported on test.

## Months 5–6 — write and defend

Full paper, the ablations reviewers will demand, advisor revisions, thesis
document, defence.

## Explicitly out of scope

Dropped relative to a twelve-month version, and worth saying so in the thesis as
future work:

- Large-scale human annotation beyond the ~300 verified clips.
- Long-form audio, where a flat timestamp vocabulary grows linearly with duration
  and TimeAudio's anchor/offset scheme becomes necessary.
- Whether the 5.60x asymmetry holds in video. This is the most interesting
  extension: if it does, the claim generalises beyond audio.
- More backbones for their own sake. Adding models without new insight is padding.

## Standing risks

1. **Scooped on the benchmark.** Mitigation is the month-1 arXiv posting, not
   secrecy.
2. **Real recordings invalidate a composed-audio finding.** Mitigation is to run
   them early enough to reshape the story rather than contradict it.
3. **The frozen encoder caps arm C.** Already expected; state it rather than
   fight it with 210 clips.
4. **GPU quota.** Kaggle gives 30 h/week; a full arm sweep is roughly 22 h. Union
   decoding at k=5 is the expensive item and can drop to k=3.
