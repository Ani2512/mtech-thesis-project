# Speaker notes — one section per slide

Read these aloud or near enough. Times are a guide for a ~20-minute slot; cut
the *optional* paragraphs if you are running long. Numbers in **bold** are the
ones a committee member might write down — say them slowly.

---

## Slide 1 — Title (30 s)

Good morning. This is a project on compositional temporal audio grounding —
finding *when* a sound happens in a recording, when the question depends on
another sound in the same recording.

I'll show you the task, why it's open, what I built, what I measured, what
failed, and where it goes next. I'll be explicit throughout about what is
measured and what is not yet.

---

## Slide 2 — A timestamp question that needs two sounds (1 min)

Audio language models can now be asked *when* something happens, not just
*what*. But today that always means one sound on its own: "find the dog
barking."

Real questions are compositional. "The car horn *after* the siren." "The
*second* bark." "The dog barking *while* music plays." "The bark *not followed
by* a door slam." In every one of these, the answer depends on a second event,
on order, or on overlap.

Two things make this harder than it looks. First, the answer is a *set* of
intervals — the model has to decide how many, not just where. Second, the
correct answer can legitimately be *empty*. If no bark happens after the siren,
the right answer is "nothing," and a model that never says "nothing" is wrong by
construction on a quarter of the questions.

---

## Slide 3 — Temporal grounding in audio LLMs (1 min 30 s)

Here is the recent work on temporal grounding in audio — twelve papers, all
from the last ten months, all checked against arXiv this week.

They fall into a few families. Timestamp representations — TimeAudio,
SpotSound, the audio-side time prompt. Reinforcement learning with IoU rewards
— TEMPO, Auto-AEG, AudioMap. Long-form methods. And benchmarks — TAG-Bench,
which came out eleven days ago, and DCASE 2026 Task 6.

The column to look at is the last one. **Every single one of these localises one
sound named on its own.** None takes a query whose answer depends on a second
event.

*(Optional, if time:)* TAG-Bench is worth a sentence. It evaluated twenty-one
systems and found that on queries with multiple occurrences, **every model
under-reports** — their phrase is "models usually find one occurrence and stop."
Best count accuracy was **thirteen per cent**. So the field already knows models
can't enumerate. Nobody has asked what happens when you put a condition on top.

---

## Slide 4 — Compositional and relational queries (1 min 30 s)

The other half of the literature: work that *does* ask relational questions
about audio.

DAQA in 2019 and the Qualcomm NAACL paper generate before/after/ordinal
questions programmatically — but their answers are event names or free text,
never intervals. PolyBench and STAR-Bench ask about concurrency and order, but
as multiple choice. And the video community — CoMET-Bench, CompSTVG — has
built exactly this task, with conditions and interval outputs, but for video.

The row I want to draw attention to is **TREA**, Interspeech 2025. It builds its
audio the same way I do — by combining ESC-50 recordings. So it is the closest
neighbour, and I want to be upfront about that. Two differences. TREA
*concatenates*, so sounds never overlap and a "while" question is impossible to
ask. And TREA's answers are multiple choice over six hundred items, whereas here
the answer is a set of time intervals over four thousand queries.

---

## Slide 5 — The gap the two halves leave open (1 min)

Putting those together as a matrix. Three properties: interval-set output,
condition on a second event, and audio.

Audio grounding work has intervals and audio, no conditions. Audio relational
QA has conditions and audio, no intervals. Video compositional grounding has
intervals and conditions, no audio. **This project is the only row with all
three.**

I want to be honest that the task framing is a port of CoMET-Bench from video,
and a reviewer will say so. The defences are the things video cannot do or did
not do: conditions made by *mixing* sounds, rejection queries with exact ground
truth, a diagnostic that separates hearing from reasoning, and a measurement I'll
show you on slide twelve.

---

## Slide 6 — What was built (1 min 30 s)

The core idea is simple. If you *construct* the audio by placing known sounds
at known times, then the answer to any of these questions is a pure function of
the timeline you wrote. No annotation. No label noise. Exact.

So: a benchmark generator. Take clips from ESC-50 — dogs, sirens, car horns,
glass breaking. Trim each to its energy envelope so the labelled onset matches
what you hear. Place six events from three classes onto a twenty-second
timeline, and with probability 0.35 start one *inside* the previous one, so
overlap exists by construction. Add noise at 20 dB.

Then generate queries of eight types from that timeline, with two or three
phrasings each so results aren't a template artefact, and about a quarter of
the relational ones deliberately having an empty answer.

**Three hundred clips, four thousand four hundred and four queries, seventy-nine
seconds.** Metrics beyond F1: union IoU, count accuracy, under-report rate,
rejection F1, and centre error and duration ratio to separate *where* from *how
long*. Five experimental arms from one codebase. Forty-six tests.

---

## Slide 7 — Architecture, as it runs today (45 s)

One picture of the whole system, three layers.

Top row, **build**: ESC-50 clips go into the composer, which places them on a
timeline — and because it placed them, it knows the exact boundaries. The query
generator turns that timeline into eight kinds of question with exact answers,
and the splitter divides clips into train, validation and test by a stable hash
so nothing leaks.

Middle row, **answer**: the five arms. All of them read the same benchmark file.
The test split goes to every arm; the training split goes through `sft_data.py`
into the two fine-tuning arms; the validation split is what the hybrid selects on.

Bottom row, **score**: one metrics module scores every arm identically — IoU,
Hungarian F1, count, rejection, centre error, and F-beta — into per-type tables.

The green box on the left is the diagnostic: the oracle and the degraded
grounders feed arm B's composition logic directly, on CPU. That's how the
ceiling and the asymmetry were measured without a GPU.

*(If asked "what's not on this diagram?")* Real recordings — everything here is
composed audio. That's the months two-to-three work.

## Slide 8 — Only one of the two candidate backbones can ground at all (1 min 30 s)

Phase one: which open model can even attempt this? Two candidates, run zero-shot
on a hundred and fifty queries each.

Qwen2-Audio scores **0.094** on plain grounding and **exactly zero** on every
conditional type. Before accepting that I ruled out three artefacts. The audio
*is* reaching the model — it gives different answers to different questions.
The parser *was* partly at fault, and fixing it moved the number from 0.004 to
0.021 — the conclusion didn't change. And its intervals are about five times too
short — half a second where the truth is two and a half — which caps IoU at
about 0.2 whatever the placement. SpotSound reports the same failure mode for
this model.

*(Say this if asked, or pre-empt it:)* Another paper measures Qwen2-Audio at
0.37 mIoU on its own task, so I state this as "cannot ground on this benchmark
under this prompt," not as a property of the model.

Qwen2.5-Omni is the backbone. It scores **0.375** on plain grounding. It finds
the sound — and then drops the qualifier attached to it.

---

## Slide 9 — The hard part is not "conditions" — it is relating two events (1 min)

Look at *which* conditions fail. BEFORE and ORDINAL are handled — 0.38, 0.33.
WHILE, AFTER, NOT_FOLLOWED, NEXT_AFTER collapse — all between 0.09 and 0.15.

ORDINAL only needs counting within one sound. BEFORE only needs one comparison
against a single reference. The four that collapse **all require relating the
target to a second event and then selecting among the matches.** That is the
specific missing capability — not "conditions" in general.

One supporting number: on NOT_FOLLOWED the model under-reports on **sixty-nine
per cent** of queries, the highest of any type. It's collapsing a set of matches
down to one.

---

## Slide 10 — Is the model bad at conditions, or bad at hearing? (2 min)

This is the slide I'd most like you to take away.

When the model fails a conditional question, there are two possible reasons.
It can't hear where the sounds are. Or it can hear, but can't apply the
condition. The zero-shot numbers can't tell these apart.

So I separated them. I wrote a small agent that never asks the model to honour
a condition. It asks only "where are the barks?" and "where is the siren?" —
two plain groundings — and applies the condition itself in code, using the same
rules that define the ground truth.

Then I swapped in different grounders. A *perfect* grounder, reading straight
off the timeline: **1.000** on every condition type. So the conditions are
trivial, given the events. That also proves the agent's logic matches the
benchmark exactly.

Then I degraded that perfect grounder — jittered boundaries, dropped events,
added phantoms — until its plain-grounding score matched the real model's
0.375. With perfect condition logic on top of that: **0.263.** The real model
answering directly: 0.243.

**Essentially the entire gap from one point zero to point two-six is hearing,
not reasoning.** That reshaped the whole project: fine-tune for plain grounding
first, decomposition is a free win on relational types, and tune for recall.

---

## Slide 11 — Which produced a falsifiable prediction (45 s)

Because the ceiling experiment gave per-type numbers, it made a prediction.
Decomposition should *win* on the four types needing a second event — AFTER,
NEXT_AFTER, WHILE, NOT_FOLLOWED — and *lose* on ORDINAL and BEFORE, where the
model already copes and grounding a reference it doesn't need just adds error.

I want to stress: this prediction was made on CPU, from simulation, before any
of it ran on a GPU. The next slide is the test.

---

## Slide 12 — Tested on the real model: four of six predictions held (1 min 30 s)

Held-out test split, six hundred and ninety-nine queries, the real model.

Blue is direct prompting, orange is decompose-and-combine. Left of the dashed
line, decomposition was predicted to win. **NEXT_AFTER: up fifty-seven per cent.
NOT_FOLLOWED: up forty-seven per cent.** AFTER: up. WHILE flipped — direct won.
Right of the line, direct was predicted to win. BEFORE: held. ORDINAL flipped.

**Four of six.** I'm not going to call that a clean confirmation; it isn't. But
a prediction made from simulation that mostly survives contact with a real model
is worth more than a number that happened to be higher.

Notice the overall averages nearly cancel — 0.207 against 0.218. The per-type
structure is the finding, not the average, and it is what a per-type hybrid
should exploit.

---

## Slide 13 — Which grounding error actually costs us? (1 min 30 s)

If hearing is the bottleneck, the next question is: which *kind* of hearing
error? A model can miss a sound that's there, or invent one that isn't.

Same method: take the perfect grounder, inject one error type at a time at
increasing rates, over all 4,404 queries. Both responses are close to linear.
The slopes are very different. **Missing costs 1.411 per unit rate. False
alarms cost 0.252. That's a ratio of 5.6.**

The reason is structural. A missed occurrence removes a correct interval *and*
can silently change a relational answer — lose the siren and "every bark after
the siren" becomes unanswerable, so one miss can corrupt every question that
referenced it. A phantom usually adds one wrong interval and leaves the rest
intact.

*(If asked "isn't that obvious?")* On a plain question, no. Miss one of three
barks: F1 0.80. Invent one: 0.86. Nearly equal. The asymmetry only appears once
questions reference other events. That's why nobody had measured it.

---

## Slide 14 — So the objective is measured, not chosen (1 min)

F1 weights precision and recall equally. The data says that's wrong here. So
use F-beta, with beta squared set to the measured asymmetry.

The arithmetic is on the slide: beta squared equals the ratio of the two slopes,
1.411 over 0.252, which is 5.60. Beta is the square root, **2.37.** The F-beta
formula weights recall by beta squared relative to precision.

I want to be clear: nobody tuned this. It is the square root of a ratio of two
measured slopes, and there is a unit test that pins the value so it cannot
quietly become a hyperparameter later.

The worked example on the right shows what it does. Miss one of three: F1 says
0.80, F-2.37 says 0.70. Invent one: F1 says 0.86, F-2.37 says 0.95. Under F1 the
two mistakes look alike. Under the measured beta they don't — which matches
what they actually do downstream.

---

## Slide 15 — A negative result: the decoder did not survive contact (1 min)

I'm including this because it's the most instructive thing that happened.

From the asymmetry, a decoding trick follows: sample the model several times,
keep intervals that appear in at least two samples. In simulation with
independent noise, that reached **0.951.** On the real model: **0.150** — worse
than not doing it at all, with the under-report rate tripling.

The reason is that real sampling errors are *correlated*. A model that
mislocates an event mislocates it the same way in every sample. So voting
discards real events instead of filtering phantoms. The independence assumption
was load-bearing, and it was wrong.

The beta finding is untouched — it's a property of the benchmark, not the
decoder. But this mechanism is refuted, and I'm reporting it that way.

---

## Slide 16 — Where it stands (1 min 30 s)

Status as of this morning, honestly.

Arms A and B — direct prompting and decomposition — have run on the full test
split. Those are the numbers you've seen.

Arms C and E — the two fine-tuning arms — **have never trained.** A five-and-a-half
hour run yesterday completed the inference arms and lost both training arms to
out-of-memory in the backward pass. Arm D, the hybrid, was silently a copy of arm
A because no validation data had been scored.

All three causes are diagnosed and fixed. The out-of-memory was a single
allocation — the model's output over a hundred-and-fifty-thousand-word vocabulary
across thirty-six hundred positions — not the weights. Arm E was worse: the
standard way of training new vocabulary rows costs sixteen gigabytes to move
three hundred rows. That now costs four megabytes.

**The fixed run is on Kaggle as I speak,** past the point where every previous
attempt died.

---

## Slide 17 — What this evidence does not yet support (1 min)

Five things I don't want you to conclude from this.

It's composed audio only. Nothing here is validated on real recordings yet, and
that is the critical path, not an afterthought.

One prompt, one model. TAG-Bench notes that an explicit "list every interval"
instruction is untested; it should be baseline number one.

No fine-tuned arm. Every number today is zero-shot or training-free.

"Cannot ground" is benchmark-specific. Another paper gets 0.37 from the same
model on different data.

And the 5.6 is measured on *this* query mix. On another distribution, re-derive
beta rather than importing this number.

---

## Slide 18 — Six-month plan (1 min)

Month one: finish phase two on composed audio, and put the benchmark and the
diagnosis on arXiv early. TAG-Bench appeared on the second of September and is
one step from adding relational queries; publishing timestamps the contribution.

Months two and three — the critical path — real recordings. Derive the same
conditional queries from DESED and AudioSet-strong, hand-verify about three
hundred clips, rerun every arm. This starts in month two, not month four,
because it's what makes the work hard to scoop.

Month four: the method proper. Beta as a verifiable reward under GRPO, following
TEMPO and Auto-AEG, instead of the offline pairs and voting that we now know are
weak.

Months five and six: writing and defence.

Out of scope, and I'll say so: large-scale annotation, long-form audio, and
whether the asymmetry holds in *video* — which would be the most interesting
extension, because it would generalise the claim beyond audio.

---

## Slide 19 — Contributions, stated honestly (1 min)

Three things I'm claiming as new. The task and benchmark — interval-set answers
to conditional queries over audio, with exact programmatic ground truth and
rejection queries. The ceiling diagnostic that separates "conditions are hard"
from "grounding is bad," and the prediction it generated. And the measured
asymmetry with the objective derived from it — measurable only on a benchmark
where conditions depend on a second event.

Three things I'm *not* claiming. Timestamp tokens are TEMPO's and TimeAudio's.
Decompose-and-combine is CoMET-Agent's, in video. LoRA for grounding is
standard. And the task framing is a port from video; the defences are on the
left.

Thank you. I'm happy to take questions.

---

## If you get these questions

**"How do you know your composition matches your ground truth?"** — The oracle
grounder through the agent scores 1.000 on every type across all 4,404 queries.
Two independent implementations agreeing exactly.

**"Isn't this just prompt-template luck?"** — Two to three phrasings per type,
and the template index is recorded on every query.

**"Synthetic audio — why believe any of it?"** — Believe the *relative* findings:
which types are hard, the ceiling, the asymmetry. Treat absolute numbers as
benchmark-specific. Real recordings are months two and three.

**"How is this different from TREA?"** — Same ESC-50 source. They concatenate,
so concurrency can't exist; I mix. They answer multiple choice over 600 items;
I answer with interval sets over 4,404.

**"You haven't trained anything. Is there a project here?"** — The benchmark,
the diagnostic and the asymmetry are measurements; they stand without any
fine-tuning. The training arms test whether the diagnosis translates into
gains. That run is in progress. If it fails, the frozen audio encoder is the
first suspect, and the diagnosis is still the contribution.

**"Why should I trust 5.6?"** — Don't trust the number; trust the method. It's
the square root of a ratio of two measured slopes, unit-tested, reported beside
F1 everywhere. Re-run the two curves on your own data and you get your own
beta.
