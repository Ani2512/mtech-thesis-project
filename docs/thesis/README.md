# Thesis chapters for the final report (drafts, 2026-11-21)

The Phase II Word report (`~/Downloads/MTech_Project_Report_24MDA067_...docx`,
28 pages) stays as it is: chapters 1 to 5 covering the benchmark, the diagnosis
and the composed-audio training study up to 0.645. The final thesis keeps its
chapters 1 to 3 and chapter 4 nearly unchanged, and adds the material below.
These files are the drafts of the new chapters, written from the result
documents in `docs/` (every number here has a source there; nothing is new).

| file | becomes | source documents |
|---|---|---|
| `abstract.md` | the abstract, rewritten for the whole project | all of the below |
| `ch5_timeline_transcription.md` | **5 Whole-timeline transcription at scale** | results_nebius_phase3.md, error_analysis_phase3.md, timeline_transcription.md |
| `ch6_real_recordings.md` | **6 Real recordings** | results_desed.md, real_recordings.md, results_real_stage.md |
| `ch7_conclusion.md` | **7 Conclusion** (replaces the Phase II chapter 5) | the two above and six_month_plan.md |
| `references_addendum.md` | entries to add to REFERENCES | data_licensing.md, references.md |

Chapter numbering: the Phase II report's "4 RESULTS AND DISCUSSION" becomes
"4 Composed audio: diagnosis and the first training study" and its "5
CONCLUSION" is replaced by chapter 7 here. The Phase II abstract's last
sentence ("points the next phase at ...") is superseded by `abstract.md`.

Conventions kept from the Phase II report: plain English, no symbols that
need defining, scores given to three decimals as f1@0.5 over the questions
that have an answer unless stated, ABSENT questions reported as rejection F1,
and each table named by what it measures. Where a number is on a subset
(first 700 questions, 48 test clips) the subset is stated next to it.

What is still open, and said so in chapter 7: the real-data stage trained on
the pilot's 5,000 generated clips rather than the 20,000-clip set (the run is
queued); the DESED questions are automatic, with the exact-label synthetic set
as the cross-check instead of a hand review.
