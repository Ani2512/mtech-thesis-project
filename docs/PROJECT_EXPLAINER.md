# Compositional Temporal Audio Grounding — a complete explanation

*Anirudh Rangavajhala · M.Tech, IIIT Dharwad · 12 September 2026*

This document assumes no prior knowledge of the project. It explains the
problem, the literature it sits in, every technique used, what each piece of
code does and why, what has been measured, what has failed, and where it goes
next.

---

## Part 0 — The project in one page

**The task.** Give a model an audio recording and a question about *when*
something happens, where the question depends on something else in the
recording. "The car horn **after** the siren." "The **second** bark." "The dog
barking **while** music plays." The answer is a **set of time intervals**, and
sometimes the correct answer is *nothing* — no event satisfies the condition.

**Why it is not solved.** Audio language models are now asked to localise
sounds, but always one sound named on its own: "find the dog barking." Nobody
asks them a question whose answer depends on a *second* event. The video
community has started doing exactly this; audio has not.

**Why it is tractable.** If you *build* the audio by placing known sounds at
known times, the correct answer to any such question is a **pure function of
the timeline you constructed**. No annotation, no label noise, exact ground
truth, at any scale.

**What was found.** Three results, in order of importance:

1. **Grounding quality, not condition handling, is the binding constraint.**
   Give a system perfect knowledge of when each sound occurs, and every
   condition type is answered perfectly (1.000). Degrade that knowledge to the
   level a real model achieves, and even flawless condition logic reaches only
   0.263. Almost the entire failure is perception, not reasoning.
2. **A missed sound costs 5.60× a spurious one.** Measured, not assumed. This
   quantity is only measurable on a benchmark where conditions depend on a
   second event, which is what this project built. It yields an objective —
   F-beta with beta = 2.37 — that is *read off the measurement* rather than tuned.
3. **Decomposition beats direct prompting on exactly the query types that
   require relating two events**, and loses on the ones the model already
   handles. Predicted from simulation, then confirmed on a real model.

---

## Part 1 — The problem, precisely

### 1.1 Temporal audio grounding

The base task: given audio and a text query, return every time interval where
the query is satisfied. Evaluated with Intersection-over-Union (IoU) against
ground-truth intervals.

```
audio:  [-----dog----]      [--siren--]   [--dog--]        [-horn-]
time:   0    2.1   4.3      7.0    9.5    11.2  13.0       15.5  16.8

query:  "every dog bark"
answer: [(2.1, 4.3), (11.2, 13.0)]
```

### 1.2 What makes a query *compositional*

A compositional query adds a condition that refers to **another event**, to
**order**, or to **overlap**:

| Type | Example | Answer for the timeline above |
|---|---|---|
| `PLAIN` | every dog bark | `[(2.1,4.3), (11.2,13.0)]` |
| `ORDINAL` | the second bark | `[(11.2,13.0)]` |
| `AFTER` | every bark after the siren | `[(11.2,13.0)]` |
| `BEFORE` | every bark before the siren | `[(2.1,4.3)]` |
| `NEXT_AFTER` | the first bark after the siren | `[(11.2,13.0)]` |
| `WHILE` | every bark while the siren sounds | `[]` (none overlap) |
| `NOT_FOLLOWED` | every bark not followed by a horn within 3 s | `[(2.1,4.3)]` |
| `ABSENT` | every cat meow | `[]` (rejection query) |

Three properties make this harder than plain grounding:

- **The output is a set**, of unknown size. The model must decide *how many*
  intervals, not just where one is.
- **The answer can legitimately be empty.** A model that never says "nothing
  satisfies this" is wrong on a quarter of the benchmark by construction.
- **The condition couples two events.** Getting the reference event wrong
  corrupts the answer even when the target event was located perfectly.

---

## Part 2 — Literature: where this sits

Twenty-four works, all verified against arXiv / ACL Anthology / ISCA on
2026-09-12 (full titles, authors and dates in `docs/references.md`).

### 2.1 Temporal grounding in audio LLMs — they all localise *one* sound

| Work | Contribution |
|---|---|
| **TimeAudio** (arXiv:2511.11039) | Temporal markers + time-aware encoding + segment-level token merging. Paper title is *"Listening Between the Frames"*; TimeAudio is the method name. |
| **SpotSound** (arXiv:2604.13023) | Suppresses hallucinated timestamps for absent events; introduces SpotSound-Bench |
| **Audio-Side Time Prompt** (arXiv:2604.13715) | TimePro-RL: timestamps embedded *in the audio feature sequence*, SFT then RL |
| **TEMPO** (arXiv:2608.29999) | GRPO with verifiable rewards on temporal metrics; atomic timestamp tokens |
| **Auto-AEG** (arXiv:2607.04383) | Scalable synthetic data + RL; introduces AEGBench |
| **AudioMap** (arXiv:2608.09559) | Cloze-and-choice RL for time-aware dense audio captioning |
| **Listening with Time** (arXiv:2604.22245) | Global-to-local iterative reasoning for long-form audio |
| **Encode Once, Decode Never** (arXiv:2602.10230) | Reuses audio-LM internals to localise; 50× faster than token decoding |
| **LA-RAG** (arXiv:2602.14612) | Event-grounded QA over long audio via structured retrieval |
| **SpectCount** (arXiv:2606.06907) | Counting via synthetic spectrotemporal signals |
| **TAG-Bench** (arXiv:2609.01542) | 1,750 query–recording pairs, 149.5 h, 8 subsets, 21 systems |
| **DCASE 2026 Task 6** | Audio Moment Retrieval from Long Audio |

**None of them takes a query whose answer depends on a second event.**

TAG-Bench is the closest and the most important to know. Its own findings are
the strongest external motivation this project has:

- *"387 queries (22.1%) carry ≥2 ground-truth intervals (up to 18 per query)"*
- *"The highest one-to-many count accuracy is 13.2% (Step-Audio-R1), followed by
  Gemini-3.1-Pro at 7.0%, FireRedAudio at 4.7%"*
- *"every model under-reports the number of occurrences on one-to-many queries"*;
  *"Models usually find one occurrence and stop."* Under-report rates run
  **79.3% to 100%**.

TEMPO adds the other half: after GRPO training, *"TEMPO's residual errors are
dominated by **segment merging** and boundary misalignment, even when the
underlying semantic content is correct."*

So the field knows models cannot enumerate multiple occurrences. Nobody has
asked what happens when a *condition* is layered on top.

### 2.2 Compositional and relational queries — nobody outputs intervals

| Work | What it has | What it lacks |
|---|---|---|
| **DAQA** (Fayek & Johnson, TASLP 2020, arXiv:1911.09655) | Programmatic before/after/ordinal/count questions over concatenated AudioSet events; MALiMo model | Answers are event names, yes/no or counts — **never intervals** |
| **Sridhar et al.** (NAACL 2025 Industry, Qualcomm, arXiv:2409.06223) | GPT-4 temporal QA from AudioSet-SL; curriculum fine-tune of LTU | Free-text QA; no interval output, no IoU |
| **TREA** (Interspeech 2025, arXiv:2505.13115) | **Concatenates ESC-50 recordings — the same construction used here.** Duration / ordering / counting | 600 multiple-choice items; concatenation only so **concurrency cannot arise**; no intervals |
| **Oncescu et al.** (ACM MM 2024, arXiv:2409.00851) | Temporal ordering in text-to-audio retrieval; TempTest sets | Retrieval, not localisation |
| **PolyBench** (Interspeech 2026, arXiv:2603.05128) | Counting, classification, detection, concurrency, duration over polyphonic audio | Multiple choice; no conditional interval sets |
| **CompA-order** (ICLR 2024, arXiv:2310.08753) | 400 audio–caption matching instances probing event order | Matching, not grounding |
| **STAR-Bench** (arXiv:2510.24693), **MMAU** (arXiv:2410.19168) | Segment reordering, spatial localisation; 27 tasks, 10k clips | Multiple choice; no grounding |
| **CoMET-Bench** (arXiv:2606.15320) — **video** | Conditional multi-event grounding, 4 temporal + 3 spatial conditions, Rejection-F1, CoMET-Agent | Video only; no superposition-based "while" |
| **CompSTVG** (arXiv:2608.30584) — **video** | Compositional queries, scene-graph synthetic curriculum + RL | Video only |
| **MUSEG** (arXiv:2505.20715), **One-to-Many TG** (arXiv:2606.06294) — **video** | Multi-segment grounding; RL with completeness rewards | Video only |

### 2.3 The gap, stated as a matrix

| | Interval-set output | Condition on a second event | Audio |
|---|---|---|---|
| Audio grounding (TAG-Bench, TEMPO, Auto-AEG) | yes | **no** | yes |
| Audio relational QA (DAQA, Sridhar, TREA, PolyBench) | **no** | yes | yes |
| Video compositional grounding (CoMET-Bench, CompSTVG) | yes | yes | **no** |
| **This project** | **yes** | **yes** | **yes** |

**Stated honestly:** the task framing is a port of CoMET-Bench from video to
audio, and a reviewer will say so. The defences are four things video cannot
do or did not do: mixing-made concurrency, rejection queries with exact
ground truth, the ceiling diagnostic, and the miss/false-alarm asymmetry.

### 2.4 The closest neighbour, and how to answer about it

**TREA builds its audio exactly the way this project does — by combining
ESC-50 recordings.** Expect this question. The answer:

- TREA **concatenates**; this project concatenates *and mixes*. Concurrency
  cannot exist in TREA's data, so `WHILE` is impossible to ask there.
- TREA's answers are **multiple choice over 600 items**; here they are
  **interval sets over 4,404 queries**.
- TREA's ordering task asks *"which sound occurred after X?"* and the answer is
  an event name. Here the same question returns a **time interval**.

---

## Part 3 — The core insight

> **Ground truth is a pure function of the timeline.**

Because the audio is *constructed* by placing known sounds at known times, the
answer to any compositional query is *derived*, not annotated. Consequences:

- **Zero annotation cost.** 4,404 queries with exact answers, built in 79
  seconds.
- **No label noise.** The usual weak link in temporal grounding datasets is
  boundary annotation disagreement. Here there is none.
- **Rejection queries are free and exactly correct.** Asking about a sound that
  is genuinely absent is a matter of checking the vocabulary.
- **The diagnostic becomes possible.** Because a *perfect grounder* can be
  written in ten lines (just read the timeline), the ceiling of any method can
  be measured on CPU, before spending a single GPU hour.

The cost is that the audio is synthetic mixtures, not real recordings. That is
the single biggest limitation of the project and the critical path of the next
phase.

---

## Part 4 — System architecture

```
                    ctag/compose.py
                 (places sounds, mixes audio)
                            |
                            v
                    ctag/timeline.py
        Timeline(duration, [Event(label, onset, offset), ...])
                            |
              +-------------+-------------+
              |                           |
              v                           v
      ctag/queries.py                 wav files
  (8 condition types, exact
   answers from predicates)
              |
              v
      benchmark.jsonl  ---> ctag/split.py ---> train / val / test
              |
              v
   +----------+-----------+-----------+-----------+
   |          |           |           |           |
 arm A      arm B       arm C       arm D       arm E
 direct   decompose    QLoRA      hybrid     QLoRA + time
 prompt   & combine    (text ts)  (per type)   tokens
   |          |           |           |           |
   +----------+-----------+-----------+-----------+
                            |
                            v
                     ctag/metrics.py
       union-IoU · Hungarian F1 · count · rejection · F-beta
```

Sixteen modules, 2,718 lines, 46 tests.

---

## Part 5 — The code, module by module

### 5.1 `timeline.py` — the specification (95 lines)

This file *is* the task definition. Everything else serves it.

```python
@dataclass(frozen=True)
class Event:
    label: str
    onset: float
    offset: float

    def __post_init__(self):
        if self.offset <= self.onset:
            raise ValueError(f"empty event {self}")
```

`frozen=True` makes events immutable — a timeline cannot be accidentally
mutated between generating a query and scoring it. The validation rejects
zero-length events, which would make IoU undefined.

```python
@dataclass
class Timeline:
    duration: float
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self.events = sorted(self.events, key=lambda e: (e.onset, e.offset))
```

**Sorting on construction is load-bearing.** Because events are always in
temporal order, `ordinal` reduces to list indexing, and every predicate returns
intervals in order without re-sorting.

The eight predicates:

```python
def plain(self, x):           return self.occ(x)

def ordinal(self, x, k):
    xs = self.occ(x)
    if k == "last":  return xs[-1:] if xs else []
    return [xs[k - 1]] if len(xs) >= k else []
```

Note `ordinal` returns `[]` when `k` exceeds the count — that is how "the
fourth bark" becomes a rejection query when there are only two barks.

```python
def _unique(self, y) -> Event:
    ys = self.occ(y)
    if len(ys) != 1:
        raise ValueError(f"reference event {y!r} must occur exactly once, found {len(ys)}")
    return ys[0]

def after(self, x, y):   ref = self._unique(y); return [e for e in self.occ(x) if e.onset >= ref.offset]
def before(self, x, y):  ref = self._unique(y); return [e for e in self.occ(x) if e.offset <= ref.onset]
def next_after(self, x, y):  return self.after(x, y)[:1]
```

**Why `_unique` raises.** "The bark after the siren" is meaningless if there
are three sirens — *after which one?* Rather than pick a convention silently,
the code refuses. Query generation enforces the precondition by drawing the
reference only from labels occurring exactly once.

`after` uses `onset >= ref.offset` (strictly after the reference *ends*) and
`before` uses `offset <= ref.onset` (strictly before it *starts*). These are
choices; they are documented in `docs/query_semantics.md` and mirrored exactly
in the agent.

```python
def while_(self, x, y):
    if x == y: raise ValueError("WHILE requires distinct labels")
    ys = self.occ(y)
    return [e for e in self.occ(x) if any(e.overlap(yy) > 0 for yy in ys)]

def not_followed(self, x, y, window):
    ys = self.occ(y)
    return [e for e in self.occ(x) if not any(e.offset <= yy.onset <= e.offset + window for yy in ys)]

def absent(self, x):
    if self.occ(x): raise ValueError(f"{x!r} is present")
    return []
```

`while_` requires **strictly positive** overlap — touching endpoints do not
count. `absent` *raises* rather than returning `[]` when the sound is actually
present, so a malformed rejection query crashes the builder instead of becoming
a plausible-looking query with a false premise.

### 5.2 `compose.py` — building the audio (205 lines)

Two interchangeable banks:

- `ProceduralBank` — five synthesised classes (beep, buzz, hiss, click, chirp)
  with distinct spectra and 50 ms fades. Used by tests and dry runs; needs no
  download.
- `ESC50Bank` — 14 real classes from ESC-50 (dog, siren, car_horn,
  glass_breaking, footsteps, …).

ESC-50 clips are cleaned before placement:

```python
# trim leading/trailing near-silence so onset/offset labels are tight
thr = 0.02 * (np.abs(x).max() + 1e-9)
nz = np.where(np.abs(x) > thr)[0]
if len(nz):
    x = x[max(0, nz[0] - int(0.02 * SR)): nz[-1] + int(0.02 * SR)]
return x[: int(2.5 * SR)]          # cap at 2.5 s so several events fit
```

**Why trimming matters:** ESC-50 clips are 5 s with the sound somewhere inside.
If you place the raw clip, the *labelled* onset is the file start but the
*audible* onset is later, and every ground-truth interval is systematically
wrong. Trimming to the energy envelope makes the label match what is heard.

The label sequence is constructed, not drawn at random:

```python
def _label_sequence(labels, n_events, rng):
    """...guaranteeing at least one label occurring >=2 times (so ORDINAL
    queries are meaningful) and at least one occurring exactly once (so
    AFTER/BEFORE/NEXT_AFTER have an unambiguous reference)..."""
    repeated, unique = labels[0], labels[1]
    seq = [repeated, repeated, unique]
```

**This is a design decision with teeth.** Without it, a random clip might
contain no repeated label (no valid ORDINAL query) or no unique label (no valid
AFTER/BEFORE reference), and the benchmark would silently become unbalanced.

Placement, and the single most important parameter in the project:

```python
overlap_ok = (prev is not None and prev.label != lab
              and (prev.offset - prev.onset) > min_overlap
              and rng.random() < p_overlap)
if overlap_ok:
    latest = prev.offset - min_overlap
    t = rng.uniform(prev.onset + 0.05, latest) if latest > prev.onset + 0.05 else prev.onset
elif prev is not None:
    t = cursor + rng.uniform(0.4, 2.5)
```

With probability `p_overlap` (0.35) an event starts *inside* the previous one,
overlapping by at least `min_overlap` (0.3 s). **This is what makes `WHILE`
satisfiable by construction — and it is the thing TREA structurally cannot
do, because concatenation never produces simultaneity.**

Audio is summed, then noise is added at a controlled SNR:

```python
sig = np.sqrt(np.mean(audio ** 2)) + 1e-9
noise *= sig / (10 ** (snr_db / 20)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
audio = np.clip(audio + noise, -1, 1)
```

*A bug already fixed here:* the original placer drew all repeats from one
label, so cross-label overlap was rare and most `WHILE` answers came out empty.

### 5.3 `queries.py` — generating questions with exact answers (130 lines)

```python
TEMPLATES = {
    "AFTER": ["every {X} after the {Y}",
              "all {X} that happen once the {Y} has ended",
              "each {X} following the {Y}"],
    ...
}
```

**Two to three surface forms per type**, chosen so results are not an artefact
of one phrasing, with `template_idx` recorded on every query so the effect can
be checked after the fact.

The selection routine controls the rejection-query rate:

```python
def _pick(cands, answer_fn, rng, k, empty_frac):
    """Choose up to k candidates, preferring non-empty answers so that about
    empty_frac of the chosen ones are rejection queries."""
    rng.shuffle(cands)
    scored = [(c, answer_fn(*c)) for c in cands]
    nonempty = [sc for sc in scored if sc[1]]
    empty = [sc for sc in scored if not sc[1]]
    n_empty = min(len(empty), sum(rng.random() < empty_frac for _ in range(k)))
    chosen = nonempty[: k - n_empty] + empty[:n_empty]
```

It evaluates *every* candidate's answer first, partitions into empty and
non-empty, then draws per-slot with probability `empty_frac` (0.25). Result:
about a quarter of relational queries have `[]` as the correct answer. Without
this, a benchmark built from "interesting" queries would contain almost no
rejection cases and would reward a model that never says "nothing."

Each query also stores the **unconditioned** answer:

```python
out.append(Query(..., answer=ans, expects_empty=len(ans) == 0,
                 plain_intervals=[e.interval for e in tl.occ(x)], ...))
```

`plain_intervals` = `occ(X)` = the answer if the condition were ignored. This
single field is what makes the `ignore_condition` diagnostic mock possible.

### 5.4 `metrics.py` — scoring (287 lines, half of it defensive)

**Parsing.** Models do not reliably emit the requested format. The parser tries
four strategies in order:

```python
def parse_intervals(text):
    if _TIME_TOK.search(text):                       # 1) atomic time tokens
        return TimeVocab().decode(text)
    for m in re.finditer(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text):
        for loader in (json.loads, ast.literal_eval):    # 2) JSON, then Python literal
            ...
    if _EMPTY.search(text):                          # 3) explicit empty answer
        return []
    pairs = [(float(a), float(b)) for a, b in _PAIR.findall(text)]   # 4) loose "12.3-15.6"
```

`ast.literal_eval` is there because models emit **single-quoted Python dicts**
that `json.loads` rejects. `_pair_from_dict` accepts keys ending in
`start`/`end` with any prefix, so `{'sneeze_start': '0.63', 'sneeze_end': '1.09'}`
parses. **This took Qwen2-Audio's parse-failure rate from 16% to 4%** — the
difference between measuring a model and measuring a regex.

The distinction `None` vs `[]` is deliberate: `None` means *unparseable*
(counted as `parse_fail`), `[]` means *the model correctly said nothing*.
Conflating them would score a confused model as a good rejector.

**Four scoring axes.**

```python
def union_iou(pred, gt):            # TAG-Bench style: seconds of overlap / seconds of union
    p, g = _merge(pred), _merge(gt)
    inter = sum(_inter(a, b) for a in p for b in g)
    union = _length(_merge(p + g))
    return inter / union if union > 0 else 0.0
```

```python
def matched_f1(pred, gt, thr=0.5):  # AEGBench style: one-to-one Hungarian assignment
    M = np.array([[iou(p, g) for g in gt] for p in pred])
    r, c = linear_sum_assignment(-M)
    tp = int(sum(M[i, j] >= thr for i, j in zip(r, c)))
    prec, rec = tp / len(pred), tp / len(gt)
```

The Hungarian assignment matters for set-valued answers: it prevents one good
prediction from being credited against several ground-truth intervals.

**The honest metric.** `f1@0.5` cannot distinguish *wrong place* from *wrong
duration* — both score zero:

```python
def localisation(pred, gt):
    """A model that centres an interval perfectly but makes it a fifth as long
    scores 0 at IoU>=0.5, and so does a model pointing somewhere else entirely."""
    gc = [_centre(g) for g in gt]
    dists = [min(abs(_centre(p) - c) for c in gc) for p in pred]
    pdur = sorted(b - a for a, b in pred)[len(pred) // 2]
    gdur = sorted(b - a for a, b in gt)[len(gt) // 2]
    return dists, (pdur / gdur if gdur > 0 else None)
```

This is how Qwen2-Audio's failure was actually diagnosed: median predicted
duration **0.53 s against a 2.50 s gold**, which caps IoU near 0.21 *regardless
of placement* and guarantees 0 at threshold 0.5.

**Rejection metrics return `None`, not `0.0`,** when a condition type has no
rejection queries — reporting 0.0 would read as failure at a task never posed.

### 5.5 `models.py` — backends and the diagnostic mocks (228 lines)

The prompt, used **identically** at training and inference:

```python
SYSTEM = ("You are an audio analysis assistant. ... Reply with ONLY a JSON list "
          "of [start, end] pairs in seconds, e.g. [[1.2, 2.0], [7.5, 8.1]]. "
          "If nothing in the recording satisfies the query, reply [].")

def prompt_for(query_text, duration=None):
    d = f" The recording is {duration:.1f} seconds long." if duration else ""
    return f"Locate: {query_text}.{d} Reply with only the JSON list."
```

Passing the clip duration is a deliberate affordance — without it a model must
guess the time range it is allowed to answer in.

**The mocks are a measurement instrument, not test scaffolding:**

```python
if self.mode == "oracle":            iv = query.answer            # upper bound
elif self.mode == "ignore_condition": iv = query.plain_intervals  # hears perfectly, ignores the condition
elif self.mode == "first_only":       iv = query.plain_intervals[:1]  # "find one and stop"
```

Each mock is a *hypothesis about how a model fails*, made executable. Scoring
them on the same benchmark produces reference curves; a real model's per-type
profile can then be compared to them by mean absolute gap. `first_only` encodes
precisely the behaviour TAG-Bench reports: *"Models usually find one occurrence
and stop."*

**Memory planning:**

```python
def choose_precision(total_gib, param_billions=8.4):
    """fp16 when the weights plus working headroom genuinely fit, else 4-bit."""
    return "fp16" if total_gib >= param_billions * 2 * 1.25 else "4bit"
```

8.4B parameters in fp16 is ~17 GB; a T4 has ~15 GB. Without this check,
`device_map="auto"` silently offloads to CPU and inference becomes unusably
slow — a failure that looks like bad luck rather than a configuration error.

One hard-won workaround:

```python
# Qwen2_5OmniForConditionalGeneration.generate builds the talker's kwargs dict
# unconditionally -- it reads self.talker.codec_pad_token before checking whether
# audio output was requested -- so with enable_audio_output=False it raises
#   AttributeError: ... object has no attribute 'talker'
gen = self.model.thinker.generate(...)
```

### 5.6 `run_zeroshot.py` — the evaluation loop (87 lines)

Straightforward except for the sampling path:

```python
if a.samples > 1:
    raws = [backend.ground(audio, q.text, query=q, duration=duration,
                           temperature=a.temperature) for _ in range(a.samples)]
    parsed = [parse_intervals(r) for r in raws]
    # all unparseable means a genuine failure; otherwise ignore the duds
    pred = None if all(p is None for p in parsed) else union_decode(parsed, a.min_votes)
```

The comment marks a real decision: if *some* samples parse, the unparseable
ones are dropped rather than poisoning the answer; only total failure counts as
a parse failure.

### 5.7 `agent.py` — decompose-and-combine (134 lines)

The conceptual centrepiece. **Never ask the model to honour the condition.**
Ask only for plain groundings, then apply the condition in Python:

```python
def ground_query(query, audio_path, grounder):
    """Two plain grounding calls at most, then local composition."""
    xs = grounder(audio_path, query.x)
    ys = grounder(audio_path, query.y) if query.y else []
    return combine(query.qtype, xs, ys, query.k, query.window)
```

`combine()` mirrors `timeline.py` predicate for predicate:

```python
if qtype in ("AFTER", "BEFORE", "NEXT_AFTER"):
    ref = _unique_reference(ys)
    if ref is None: return []
    if qtype == "BEFORE":  return [x for x in xs if x[1] <= ref[0]]
    after = [x for x in xs if x[0] >= ref[1]]
    return after[:1] if qtype == "NEXT_AFTER" else after
```

**The proof the mirror is exact:** oracle grounder + agent scores **1.000 on
all 4,404 queries and every condition type**. Two independent implementations
of the same semantics agreeing perfectly. If `combine` deviated even slightly
from `Timeline`, that number would not be 1.000.

Three subtleties:

```python
def _norm(label):
    """Queries carry raw labels ('glass_breaking'); prompts carry phrases
    ('glass breaking'). Normalise both sides identically or multi-word sounds
    silently never match and the oracle stops being an oracle."""
    return label.strip().lower().replace("_", " ")
```

This was a **real bug**: normalising one side only meant `glass_breaking` never
matched anything, and "perfect" grounding scored 0.603 instead of 1.000.
Single-word procedural labels hid it entirely.

```python
def _unique_reference(ys):
    """AFTER/BEFORE/NEXT_AFTER need one unambiguous reference. If the grounder
    returns several, take the longest: on a correct grounding of a
    single-occurrence event the extras are spurious fragments."""
    return max(ys, key=lambda p: p[1] - p[0])
```

And the two grounders that turn the agent into a **measuring instrument**:

```python
def oracle_grounder(timelines):   # perfect: reads the timeline
def noisy_grounder(base, jitter, drop, spurious, seed):   # degrade one mode at a time
```

`noisy_grounder` is why the entire phase-2 diagnosis ran **on CPU, before any
GPU time was spent**.

### 5.8 `split.py` — leak-proof splitting (73 lines)

```python
def assign(clip_id, seed, frac_train, frac_val):
    """Stable hash rather than a shuffle: adding clips later does not reshuffle
    the ones already assigned, so a model trained earlier is never silently
    evaluated on its own training clips."""
    h = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / 2 ** 64
    if x < frac_train: return "train"
    if x < frac_train + frac_val: return "val"
    return "test"
```

Two properties, both deliberate:

- **Split by clip, not by query.** Queries from one clip share the same audio
  and timeline; splitting by query would leak the audio across the boundary.
- **Stable hash, not a shuffle.** Regenerating the benchmark with more clips
  leaves existing assignments untouched. A shuffle would silently reshuffle and
  invalidate every earlier trained model.

And it is checked, not assumed:

```python
overlap = (clips["train"] & clips["val"]) | (clips["train"] & clips["test"]) | ...
assert not overlap, f"clip leak across splits: {sorted(overlap)[:5]}"
```

### 5.9 `sft_data.py` — training examples (172 lines)

```python
"""The prompt format here is byte-identical to inference (`ctag.models.SYSTEM`
and `prompt_for`), so the model is never trained on one phrasing and evaluated
on another."""
```

The mix is chosen from the diagnostic, not by intuition:

```python
"""Training is therefore weighted towards plain grounding. Extra PLAIN examples
are synthesised for every label in every training clip, which costs nothing
because the timeline already has the answer."""
```

Because the ceiling experiment showed **grounding is the binding constraint**,
`--plain-ratio 0.6` weights training toward plain localisation rather than
condition handling. Synthesising extra PLAIN examples is free — the timeline
already knows every answer. Absent-sound examples are added too, so the model
learns to emit `[]`.

### 5.10 `train_lora.py` — QLoRA fine-tuning (473 lines)

The audio encoder is **frozen**:

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
# Language side only. The audio encoder is frozen: with ~200 training clips
# there is not enough signal to retrain perception, and unfreezing it is the
# fastest way to overfit the composed-audio distribution.
```

**The masking is the subtle part.** Labels are masked by answer length counted
back from the *end*:

```python
"""Masking is done by *answer length from the end*, not by prompt length from
the start. The prompt carries one audio placeholder token which the processor
expands into hundreds of audio positions, so a prompt-length mask computed from
the text covers only the first few dozen tokens and leaves the whole audio
region as a training target. That produced NaN gradients and a loss collapsing
to zero on the first real run, while a stub processor that did no expansion let
the unit test pass."""
```

That last clause is the lesson: **a test with a simplified fixture passed while
the real system was broken.**

Stability, on hardware without bf16:

```python
# With 4-bit weights and an fp16 compute dtype the adapter updates underflow and
# the optimiser state overflows, which shows up as grad_norm = nan and a loss
# collapsing to zero rather than as an exception.
for _, param in model.named_parameters():
    if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
        param.data = param.data.to(torch.float32)
```

And the completion check refuses to report success on a run that silently did
nothing:

```python
# The averaged training_loss can look healthy while every step was skipped,
# which is how a run with nan grad_norm and a loss of 0 reported PASSED.
nan_grads = sum(1 for h in history if h.get("grad_norm") != h.get("grad_norm"))
zero_loss = sum(1 for h in history if h.get("loss") == 0.0)
if nan_grads or zero_loss:
    raise SystemExit("*** TRAINING FAILED: ...")
```

### 5.11 `timetokens.py` — atomic timestamp tokens (291 lines)

The motivation, measured on the real tokenizer:

```python
"""Qwen2.5-Omni's tokenizer splits '16.76' into five tokens, one per character:
    '16.76' -> '1' '6' '.' '7' '6'
A two-interval answer costs 25 tokens. Three problems follow. Emitting a time is
a five-step sequence where any slip moves the answer by seconds; 16.76 and 16.8
share almost no token structure, so the representation cannot express that a
near miss is nearly right; and "after the horn" becomes a comparison between
digit strings the model wrote itself."""
```

The fix, following TEMPO: one token per quantised time, 0–30 s at 0.1 s = 301
tokens, each embedding initialised as the mean of the BPE pieces of the number.

Quantisation has a trap:

```python
"""Deliberately not round(): Python rounds halves to even, so 0.65 would
quantise down to 0.6 while 0.75 goes up to 0.8. Worse, t/resolution is not
exact in binary -- 3.15/0.1 is 31.4999999999999996 -- so a plain round is
unpredictable near a midpoint. Round half up with a tolerance."""
```

A **distance-aware loss** gives partial credit for near misses:

```
q_k ∝ exp(-(t_k - t*)² / 2σ²)      σ = 0.3 s
L = L_CE + λ · L_time               λ = 0.5
```

Without it, predicting 16.8 when the truth is 16.7 is penalised exactly as hard
as predicting 3.0, and the ordinal structure of the new vocabulary is wasted.

**The memory problem and its fix.** The obvious way to train new vocabulary
rows is PEFT's `modules_to_save=["embed_tokens", "lm_head"]`. That unfreezes
both full 152,064 × 3,584 matrices — ~2.0 GB of fp32 weights, 2.0 GB of
gradients and 4.1 GB of Adam state *each*, about **16 GB to move 301 rows**. On
a 15 GB T4 the backward pass died in under a minute.

```python
class NewRowsEmbedding(nn.Module):
    """Frozen base embedding, plus a trainable delta on rows >= base_size."""
    def forward(self, ids):
        out = self.base(ids)
        is_new = ids >= self.base_size
        idx = (ids - self.base_size).clamp_(min=0)
        add = self.delta.to(out.dtype)[idx]
        return out + add * is_new.unsqueeze(-1).to(out.dtype)
```

Same computation, ~4 MB instead of 16 GB, and TEMPO's initialisation is
preserved because the delta starts at zero and learns *on top of* the base row.
PEFT does not know about these, so they are saved as `time_deltas.pt` beside
the adapter and reloaded at inference — with a loud warning if missing, since
otherwise arm E would silently score a model whose timestamp tokens never moved.

### 5.12 `recall_bias.py` — the method (150 lines)

```python
MEASURED_ASYMMETRY = 5.60
MEASURED_BETA = MEASURED_ASYMMETRY ** 0.5      # 2.37
```

Pinned by a unit test so it cannot quietly become a tuned hyperparameter.

```python
def f_beta(pred, gt, beta=MEASURED_BETA, thr=0.5):
    """beta = 1 reproduces the f1@0.5 used elsewhere, so the two are comparable."""
```

Two mechanisms follow from the measurement.

**Training:**

```python
def make_preference_pairs(examples, rng, ...):
    """Preference data whose rejected side always under-detects. Over-detection
    is never used as a negative: the measured cost of a false alarm is a fifth
    that of a miss, so teaching the model to avoid over-calling would optimise
    the wrong direction."""
```

**Decoding:**

```python
def union_decode(samples, min_votes=1, merge_iou=0.5):
    """Intervals are clustered by IoU across samples; a cluster is kept if it
    appears in at least `min_votes` distinct samples."""
```

---

## Part 6 — Results

### 6.1 Phase 1 — which backbone can do this at all

Composed ESC-50, 150 queries per model, zero-shot, 4-bit on a T4.

| f1@0.5 | PLAIN | ORDINAL | BEFORE | AFTER | WHILE | NEXT_AFTER | NOT_FOLLOWED | ALL |
|---|---|---|---|---|---|---|---|---|
| mock: oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mock: ignore condition | 1.000 | 0.497 | 0.722 | 0.789 | 0.749 | 0.590 | 0.910 | 0.764 |
| mock: first match only | 0.758 | 0.269 | 0.885 | 0.260 | 0.627 | 0.370 | 0.590 | 0.542 |
| **Qwen2.5-Omni-7B** | **0.375** | 0.333 | 0.379 | 0.146 | 0.152 | 0.091 | 0.095 | 0.243 |
| **Qwen2-Audio-7B-Instruct** | **0.094** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 |

**Qwen2-Audio cannot ground on this benchmark at all.** Before accepting that,
three artefacts were ruled out: audio *is* reaching the model (7.9 distinct
answers per 10.6 queries per clip, never a canned response); the parser was
partly at fault and was fixed (16% → 4% parse failures, which did not change
the conclusion); and placement is barely above chance (38% of predictions
within 1 s of a gold centre against 18% for random). Its intervals are ~5× too
short. SpotSound (arXiv:2604.13023, §4.1) reports the same failure mode:
Qwen2-Audio *"generates syntactically complete time windows that suffer from
severe semantic misalignment, resulting in zero overlap with the ground truth."*

**Caveat to state:** *From Semantics to Readout* (arXiv:2607.25355) measures
Qwen2-Audio at 0.3653 mIoU zero-shot on its own task. Different data, different
metric, 4-bit here — so the honest claim is "cannot ground **on this benchmark
under this prompt**", not a property of the model.

**Qwen2.5-Omni is the backbone.** It finds the sound, then drops the qualifier.
And the difficulty ordering is itself the interesting result:

| Handled | Collapses |
|---|---|
| BEFORE 0.379, ORDINAL 0.333 | WHILE 0.152, AFTER 0.146, NOT_FOLLOWED 0.095, NEXT_AFTER 0.091 |

ORDINAL needs only counting within one event type; BEFORE needs one comparison
against a single reference. **The four that collapse all require relating the
target to a second event and then selecting among the matches.** That is the
specific missing capability.

### 6.2 The ceiling diagnostic — is it conditions, or is it hearing?

Full benchmark, 4,404 queries, CPU only.

| Grounding quality given to the composer | f1@0.5 |
|---|---|
| Perfect | **1.000** |
| Boundaries jittered ±0.5 s | 0.925 |
| 25% of events missed | 0.650 |
| 25% spurious detections | 0.937 |
| Degraded until PLAIN matches Qwen2.5-Omni | **0.263** |
| Qwen2.5-Omni answering directly | **0.243** |

**Conditions are trivial given the events.** The entire gap from 1.000 to 0.26
is attributable to locating sounds. Three consequences, all acted on:

1. Weight fine-tuning towards PLAIN grounding (`--plain-ratio 0.6`).
2. Decomposition should win on relational types and lose on the two the model
   already handles — a falsifiable prediction.
3. Tune for **recall**: missing 25% costs 35 points; inventing 25% costs 6.

### 6.3 Testing the prediction on a real model

Held-out test split, 699 queries, Qwen2.5-Omni-7B.

| arm | ALL | PLAIN | ORDINAL | AFTER | BEFORE | NEXT_AFTER | WHILE | NOT_FOLLOWED |
|---|---|---|---|---|---|---|---|---|
| A direct | 0.207 | 0.276 | 0.188 | 0.220 | **0.323** | 0.101 | **0.175** | 0.146 |
| B decompose | **0.218** | **0.295** | **0.212** | **0.234** | 0.228 | **0.159** | 0.146 | **0.215** |

**Four of the six per-type predictions held.** NEXT_AFTER +57%, NOT_FOLLOWED
+47%, AFTER up, BEFORE lost as predicted. WHILE and ORDINAL went the other way.
The overall means nearly cancel — **the per-type structure, not the average, is
the finding**, and it is what a per-type hybrid should exploit.

### 6.4 The method: which error actually costs us

Degrading one failure mode at a time, full benchmark:

| rate | miss → f1 | false alarm → f1 |
|---|---|---|
| 0.05 | 0.913 | 0.987 |
| 0.15 | 0.780 | 0.961 |
| 0.25 | 0.650 | 0.937 |
| 0.30 | 0.590 | 0.923 |

```
cost per unit miss rate        1.411
cost per unit false-alarm rate 0.252
asymmetry                      5.60x   ->  beta = 2.37
```

**The cause is structural.** A missed occurrence removes a correct interval
*and* can silently change a relational answer — lose the car horn and "every
bark after the horn" becomes unanswerable, so one miss can corrupt every query
referencing it. A spurious detection usually adds one wrong interval and leaves
the rest intact.

**This is only measurable on a benchmark where conditions depend on a second
event.** That is the novelty chain: new task → new measurable quantity →
objective derived from it → method.

### 6.5 A negative result, reported as such

Simulation with independent noise said vote-and-merge at k=5 / 2 votes reaches
0.951. On the real model, k=3 / 2 votes scores **0.150 against 0.207 for plain
decoding**, with under-report rate rising from 0.177 to 0.538.

**Real sampling errors are correlated.** A model that mislocates an event
mislocates it the same way in every sample, so voting discards *real* events
rather than filtering spurious ones. The independence assumption was
load-bearing and it was wrong. The beta = 2.37 finding is untouched — it is a
property of the benchmark, not the decoder — but this mechanism is refuted and
reported that way.

---

## Part 7 — What has failed, honestly

| Attempt | Outcome | Cause |
|---|---|---|
| First full phase-2 run (5 h 34 m) | Arms A, B ran; C and E never trained | CUDA OOM in backward on a 15 GB T4 |
| Arm D in that run | Silently a copy of arm A | Only the test split was scored, so the val selection set was empty |
| Precision sweep | `NameError` at cell 1 | `time.time()` used, `time` never imported |
| Vote-and-merge decoding | 0.150 vs 0.207 | Correlated sampling errors |
| Re-launch via Kaggle API | SIGSEGV on every step | Kernel got a **P100 (sm_60)**; Kaggle's PyTorch supports sm_70+ only |

All are fixed or documented. The OOM diagnosis is worth carrying: the failing
allocation was **3.07 GiB, which is one `lm_head` output** — 152k vocabulary
entries × ~3.6k positions in fp16, plus gradient and fp32 softmax — **not the
weights**. Fixes: `--max-seq-len` bounds it, `paged_adamw_8bit` quarters the
optimiser state, and `--preflight` runs one forward+backward on the longest
example before committing, so an OOM costs a minute rather than the 159 minutes
it cost the first time.

**Status:** no fine-tuning arm has ever completed. Every number above is
zero-shot or training-free.

---

## Part 8 — Limitations

1. **Composed audio only.** Sounds mixed onto a synthetic timeline. Nothing
   here generalises to real recordings yet. This is the critical path.
2. **One prompt, one model.** No prompt-sensitivity study. TAG-Bench notes that
   an explicit "enumerate all intervals" instruction is an untested
   intervention; it should be baseline #1.
3. **No fine-tuned arm.** Arms C and E have never trained.
4. **`f1@0.5` conflates placement and duration** — mitigated by reporting
   centre error and duration ratio alongside, not solved.
5. **The 5.60× asymmetry is measured on this query mix.** On another
   distribution, re-derive beta rather than importing this number.
6. **The frozen audio encoder is a bet**, not a proven choice. If arm C does not
   beat arm A on PLAIN, that is the likely cause — and it should be reported as
   a limitation rather than triggering a month of encoder unfreezing on ~210
   clips.

---

## Part 9 — Future direction

### Months 1–3, the critical path

- **Fix the OOM, train arms C and E.** Everything is in place; nothing is proven.
- **Put the benchmark and the decomposition diagnosis on arXiv early.**
  TAG-Bench (Alibaba) appeared on 2 September 2026 and is one step from this
  task. Publishing timestamps the contribution.
- **Move to real recordings.** Derive conditional queries from DESED /
  AudioSet-strong / TAG-Bench audio using the same predicates, hand-verify
  ~300 clips, rerun every arm. **Start in month 2, not month 4.** This is what
  makes the work hard to scoop and is the honest answer to "does any of this
  survive outside synthetic mixtures?"

### Month 4, the method proper

- **F-beta as a verifiable reward under GRPO**, following Auto-AEG
  (arXiv:2607.04383), rather than offline preference pairs. The measured beta
  becomes a reward shaping term with a principled value.
- **Rerun recall-biased decoding with a real model** at several k and
  `min_votes`, now knowing that correlated errors weaken it.

### Months 5–6

Writing and defence.

### Explicitly out of scope, as future work

- Large-scale human annotation.
- Long-form audio, where a flat timestamp vocabulary grows linearly with
  duration and TimeAudio's anchor/offset scheme becomes necessary.
- **Whether the 5.60× asymmetry holds in video** — the most interesting
  extension, since it would generalise the claim beyond audio.

---

## Part 10 — Contributions, stated honestly

| Novel here | Not novel — cited, not claimed |
|---|---|
| The task and benchmark: interval-set answers to conditional temporal queries over audio, with exact programmatic ground truth and rejection queries | Atomic timestamp tokens — TEMPO, TimeAudio |
| The ceiling diagnostic separating "conditions are hard" from "grounding is bad", and the falsifiable per-type prediction it generates | Decompose-and-combine — CoMET-Agent, in video |
| The **5.60× miss/false-alarm asymmetry** and the objective derived from it — measurable only on a benchmark where conditions depend on a second event | LoRA fine-tuning for grounding — standard practice |
| Mixing-made `WHILE` conditions, impossible in concatenation-based work | Task framing is a port of CoMET-Bench from video |

---

## Part 11 — Questions to expect, and the answers

**"How do you know your composition matches your ground truth?"**
The oracle grounder through the decomposition agent scores 1.000 on every
condition type across all 4,404 queries. Two independent implementations of the
same semantics agreeing exactly. Any deviation would show up immediately.

**"Isn't this just prompt-template luck?"**
Two to three surface forms per condition type, and `template_idx` is recorded on
every query so the effect can be measured.

**"Why is f1@0.5 the headline if it conflates two failures?"**
It does, which is why centre error and duration ratio are reported alongside.
That is how Qwen2-Audio's failure was diagnosed as "durations 5× too short"
rather than just "low score".

**"How is this different from TREA?"**
Same ESC-50 source, but TREA concatenates and answers multiple choice over 600
items. This mixes — so concurrency exists at all — and answers with interval
sets over 4,404 queries.

**"Synthetic audio — why should I believe any of this?"**
You should believe the *relative* findings (which condition types are hard, the
ceiling, the asymmetry) and treat absolute numbers as benchmark-specific. Real
recordings are months 2–3 and are the critical path.

**"What is actually new, versus ported from video?"**
The task shape is a port and is cited as such. New: mixing-made concurrency,
the ceiling diagnostic, and the measured asymmetry with an objective derived
from it.

**"Your fine-tuning arms have never run. Is there a project here?"**
The benchmark, the diagnostic and the asymmetry stand without any fine-tuning —
they are measurements, not model results. Arms C and E test whether the
diagnosis translates into gains. If they fail, the diagnosis is still the
contribution, and the frozen audio encoder is the first suspect.

---

## Appendix — Glossary

| Term | Meaning |
|---|---|
| **IoU** | Intersection over Union of two intervals, in seconds |
| **f1@0.5** | F1 over one-to-one Hungarian-matched intervals, counting a match at IoU ≥ 0.5 |
| **union-IoU** | IoU between the *merged* predicted set and the merged gold set |
| **count accuracy** | Fraction of queries where `len(pred) == len(gold)` |
| **under-report rate** | Fraction of non-empty queries where the model returned fewer intervals than gold |
| **rejection F1** | F1 on queries whose correct answer is `[]` |
| **F-beta / beta=2.37** | F-measure weighting recall beta² = 5.60× precision, from the measured asymmetry |
| **LoRA / QLoRA** | Low-Rank Adaptation; QLoRA adds 4-bit quantised base weights |
| **GRPO** | Group Relative Policy Optimization, an RL method used with verifiable rewards |
| **Arm A–E** | direct prompting; decompose-and-combine; QLoRA text timestamps; hybrid; QLoRA atomic time tokens |

---

*Code: `github.com/Ani2512/mtech-project`, branch `compositional-temporal-grounding`.
Numbers: `docs/phase1_findings.md`, `docs/phase2_decomposition.md`,
`docs/recall_bias.md`. Citations: `docs/references.md`.*
