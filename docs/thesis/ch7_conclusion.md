# 7 CONCLUSION

This thesis defined compositional temporal audio grounding, built the first
audio benchmark with time-window answers for relational, ordinal and
conditional questions, located where open audio language models fail on it,
and used that diagnosis to choose a method that works.

The diagnosis was that hearing where the sounds are, not reasoning about the
condition, is the bottleneck: given exact knowledge of the sounds a program
answers every question type perfectly, at the models' real accuracy nothing
built on top exceeds about 0.26, and a missed sound costs 5.6 times a false
one. The method that follows asks the model for one thing, the whole
timeline, and computes every question from it. On composed audio that moved
the overall score from 0.645 to 0.983, with every type above 0.96; the
control trained on the same data with the question as its target reached
0.951 on answerable questions and never once answered "nothing", so the gain
on the compositional types and on the empty answers belongs to the target, and
the rest to scale.

The composed number did not survive real recordings, and the thesis says so
with the same precision: 0.020 on DESED, below the untrained model, because
the model had learned the generator's vocabulary, window length and event
density. Each habit was traced to its cause and removed in turn. The last of
them, vocabulary, needed real labelled audio, and 784 clips of it were
enough: 0.671 event F1 and 0.582 across every question type on 692 real
domestic recordings, speech found in 80 percent of cases where it had never
been written before, absent sounds rejected at 0.993, and 0.772 kept on the
composed test. A second real test whose labels are exact by construction
scored higher on events (0.701) and the same across types (0.573), so the
real-audio result is not an artefact of annotation. The weaknesses that
remain are named rather than hidden: short, quiet, overlapping sounds
(dishes and alarm below 0.5 recall), long textures heard only in part
(blender and frying on the exact-label set), and "while" questions on brief
overlaps (0.34). Every number is reproducible from one tested code base,
and the paid GPU time for the whole programme was about fifty-five dollars,
with the rest on a free allowance.

Three things are worth stating as contributions beyond the numbers. The
benchmark construction, in which ground truth is a pure function of a
timeline the experimenter built, gives exact answers at any scale and makes
the ceiling of any method measurable on a CPU before a GPU hour is spent.
The empty answer, a quarter of the benchmark by construction, separates the
methods more sharply than any accuracy on answerable questions does, and the
timeline route gets it right for a structural reason. And the composed-to-real
gap, measured at its widest (0.983 to 0.020) and then closed to a stated
level, is the honest shape of synthetic training for this task: it teaches
format and habit, and only real data teaches the sounds.

What is left is in order of expected value. First, the real-data stage on
top of the 20,000-clip generated set instead of the 5,000-clip pilot, to
recover the composed accuracy the smaller set gave up without losing the real
gain; the run is queued. Second, targeted training on what the exact-label
test exposed, partial textures and brief overlaps, which the generator can
now produce on purpose. Third, a conventional sound-event-detection model on
the same clips as a ceiling, to show how much of the remaining gap is the
language model and how much the task. Fourth, free-text questions: the
decomposition is structural, so a parser from English to the eight types is
what stands between this system and a question typed by a person. Beyond the
thesis, whether the 5.6-to-1 cost asymmetry and the empty-answer finding hold
in video is the extension most worth making, since both were measured only
because the benchmark's conditions depend on a second event.
