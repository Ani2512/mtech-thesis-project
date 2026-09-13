# Recall-biased grounding

The method that falls out of the phase 2 diagnosis. `docs/phase2_decomposition.md`
showed grounding quality, not condition handling, is the binding constraint.
This asks a narrower question: *which kind* of grounding error actually costs us?

## 1. The asymmetry, measured

Holding everything else perfect and degrading one failure mode at a time, over
the full 4404-query benchmark:

| rate | miss → f1 | false alarm → f1 |
|---|---|---|
| 0.05 | 0.913 | 0.987 |
| 0.10 | 0.844 | 0.975 |
| 0.15 | 0.780 | 0.961 |
| 0.20 | 0.712 | 0.953 |
| 0.25 | 0.650 | 0.937 |
| 0.30 | 0.590 | 0.923 |

Both responses are close to linear. Least-squares slopes through the origin:

```
cost per unit miss rate        1.411
cost per unit false-alarm rate 0.252
asymmetry                      5.60x
```

The cause is structural. A missed occurrence removes a correct interval *and*
can silently change a relational answer: lose the car horn and "every bark after
the horn" becomes unanswerable, so one miss can corrupt every query that
references it. A spurious detection usually adds one wrong interval and leaves
the rest intact.

**Nobody has measured this, because nobody had a benchmark where conditions
depend on a second event.** It is the first thing this task can say that plain
grounding benchmarks cannot.

## 2. The objective follows from the measurement

F1 weights precision and recall equally, which the data says is wrong. Use
F-beta with `beta² = 5.60`, so `beta = 2.37`. **Beta is read off the degradation
slopes, not tuned.** `ctag/recall_bias.py` exposes it as `MEASURED_BETA`, and
a test pins the value so it cannot quietly become a hyperparameter.

The effect on a concrete case, three gold intervals:

| prediction | f1 | F2.37 |
|---|---|---|
| misses one | 0.800 | 0.702 |
| invents one | 0.857 | 0.952 |

Under f1 the two are nearly equivalent. Under the measured beta they are not.

## 3. Two mechanisms

**Training: under-detection as the negative.** `make_preference_pairs` builds
preference data whose rejected side always has *fewer* intervals than the gold.
Over-detections are deliberately never used as negatives, because penalising a
behaviour that costs a fifth as much optimises the wrong direction. An example
with a single gold interval yields a pair against the empty answer, the most
severe miss available.

**Decoding: sample and merge.** `union_decode` clusters intervals across k
samples by IoU and keeps clusters appearing in at least `min_votes` samples.

## 4. Decoding results

Simulated grounder with a 35% miss rate **and** a 35% false-alarm rate per
sample, 1500 queries:

| k | min_votes | f1 | F2.37 |
|---|---|---|---|
| 1 | 1 | 0.633 | 0.651 |
| 2 | 1 | 0.744 | 0.826 |
| 3 | 1 | 0.736 | 0.863 |
| 5 | 1 | 0.645 | 0.837 |
| 3 | 2 | 0.734 | 0.718 |
| **5** | **2** | **0.951** | **0.952** |
| 5 | 3 | 0.795 | 0.780 |

Three things worth stating plainly:

1. **Pure union does not scale.** At `min_votes=1`, f1 peaks around k=2–3 then
   falls, because false alarms accumulate across samples while recall has
   already saturated. Reporting only k=3 would have hidden this.
2. **Voting recovers it.** k=5 with 2 votes reaches 0.951, far above the 0.633
   single-sample baseline, because a spurious detection rarely recurs in the
   same place while a real event does.
3. **Too strict is worse again.** k=5 with 3 votes drops to 0.795: the
   threshold starts discarding real events. The optimum is interior.

Note also that F-beta and f1 disagree at `min_votes=1` — F-beta stays high as f1
falls — which is exactly the point of choosing the metric to match the
downstream cost.

## 5. Caveats

- These are **simulated** grounders with idealised independent noise. Real
  sampling errors are correlated: a model that mislocates an event will often
  mislocate it the same way on every sample, which would weaken both the union
  gain and the voting filter. This must be rerun with a real model before any
  of section 4 is claimed.
- The 5.60x asymmetry is measured on composed ESC-50 with this query mix. It
  will differ on other distributions, and the right move is to re-derive beta
  rather than to import this number.
- k samples cost k forward passes. At k=5 that is five times the inference, which
  is the honest price of the decoding-side gain.

## 6. Why this is a contribution rather than a trick

The parameter is measured, not chosen. The mechanism follows from the
measurement rather than preceding it. And the measurement is only possible on a
benchmark where conditions depend on a second event, which is what this project
built. The chain is: new task → new measurable quantity → objective derived from
it → method that follows.
