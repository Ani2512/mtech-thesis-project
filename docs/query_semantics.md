# Query semantics (v0.1)

A *timeline* is a list of events `(label, onset, offset)` with `onset < offset`,
sorted by onset. `occ(X)` is the list of events with label `X` in onset order.
Intervals in answers are the full event intervals, never clipped.

| Type | Parameters | Answer set | Generated only when |
|---|---|---|---|
| `PLAIN` | X | `occ(X)` | `occ(X)` non-empty |
| `ORDINAL` | X, k (1-based) or `last` | `[occ(X)[k-1]]`, or `[]` if `len(occ(X)) < k` | `len(occ(X)) >= 2` (so ordinal matters); k may exceed count to create rejection queries |
| `AFTER` | X, Y | `{e in occ(X) : e.onset >= Y.offset}` | Y occurs exactly once |
| `BEFORE` | X, Y | `{e in occ(X) : e.offset <= Y.onset}` | Y occurs exactly once |
| `NEXT_AFTER` | X, Y | first element of `AFTER(X, Y)` or `[]` | Y occurs exactly once |
| `WHILE` | X, Y | `{e in occ(X) : exists y in occ(Y) with overlap(e, y) > 0}` | `occ(Y)` non-empty; X != Y |
| `NOT_FOLLOWED` | X, Y, w | `{e in occ(X) : no y in occ(Y) with e.offset <= y.onset <= e.offset + w}` | `occ(X)` non-empty |
| `ABSENT` | X | `[]` | X not in timeline |

Design rules:
- **Y is unique for AFTER/BEFORE/NEXT_AFTER.** Relational conditions on a
  repeated reference event are ambiguous in natural language; we avoid them
  rather than legislate a reading.
- **Answers are X's intervals, not intersections.** For `WHILE`, the answer is
  the whole overlapping X event. A model returning the intersection is wrong
  under this spec; the paper states this explicitly.
- **Rejection.** Any query whose answer is `[]` is a *rejection query*
  (`expects_empty = true`). Correct behaviour is to return an empty list.
- **Boundary tolerance.** Metrics use IoU; `F1@0.5` is the headline, `F1@0.7`
  reported. Tolerance for onset/offset is implicit in IoU, not a separate fudge.
- **Overlap threshold for WHILE** is strictly positive. Compose with a minimum
  overlap of 0.3 s so the condition is audible.
