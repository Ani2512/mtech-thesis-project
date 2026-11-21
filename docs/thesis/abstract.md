# ABSTRACT (draft for the final thesis)

Audio language models are AI systems that listen to a recording and answer
questions about it in text. They are now asked to say when a sound happens,
but only for one sound named on its own: "find the dog barking". This thesis
studies the compositional form of the task, in which the question carries a
condition that depends on a second sound: "every bark after the car horn",
"the second bark", "barking while the siren sounds", "a bark not followed by
footsteps within three seconds". The answer is a set of time windows, and
sometimes the correct answer is that no window qualifies.

No existing benchmark contains such questions, so one was built. Isolated
recordings from the ESC-50 collection are placed and mixed on a timeline, so
the correct answer to any question follows exactly from how the clip was
built, with no human labelling and at any scale: eight question types, 300
composed clips, 4,404 questions, a quarter of which correctly have no answer.
Open audio language models fail on it before training (Qwen2.5-Omni-7B
0.207), and two diagnostic experiments located the failure: given exact
knowledge of when every sound occurs, a program answers every question type
perfectly, while at the model's real accuracy in locating sounds even perfect
condition handling scores 0.26, and missing a sound costs 5.6 times as much as
reporting one that is not there. The bottleneck is hearing where the sounds
are, not reasoning about the condition.

That diagnosis set the method. Instead of training the model to answer seven
kinds of question, it is trained to answer one: write out every sound in the
recording with its start and end. Every question type is then computed from
that transcript by a program, and the decision to answer "nothing" is never
the model's. Trained on 20,000 generated clips, the model transcribes the
composed test clips with an event F1 of 0.989 and every question type scores
at least 0.96 (overall 0.983, against 0.645 for the best Phase II method and
0.951 for the same training recipe with the question as the target). The
comparison also showed that the question-trained model never answers
"nothing", scoring 0.733 against 0.982 once the empty-answer questions are
counted.

The composed number did not carry to real recordings. On 692 domestic clips
from the DESED public evaluation set the same model scored 0.020, below the
untrained model's 0.362, because it had learned three habits of the
generator: a fourteen-name vocabulary, a fixed 2.5 second window and twice the
true event density. Varying the generator removed two of the habits and could
not supply the vocabulary. Adding 784 real strongly labelled clips to the
training set did: the model then scores 0.671 event F1 and 0.582 across every
question type on the public set, finds speech, which it had never written
before, in 80 percent of cases, and rejects absent sounds with an F1 of
0.993, while keeping 0.772 on the composed test. A cross-check on 828
synthetic soundscapes built from real DESED recordings, whose labels are exact
by construction, scores higher still on events (0.701) and the same across
question types (0.573), so the real-audio number is not an artefact of human
annotation. The remaining weaknesses are named: short overlapping sounds,
long textures heard only in part, and "while" questions on brief overlaps.
Every number in this thesis is reproducible from the repository, and the
whole programme ran on roughly fifty-five dollars of rented GPU time.
