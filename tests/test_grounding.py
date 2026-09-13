import json
import random

from ctag.metrics import matched_f1, parse_intervals, score_query, summarize, union_iou
from ctag.queries import generate
from ctag.timeline import Event, Timeline

TL = Timeline(20.0, [
    Event("dog", 1.0, 2.0), Event("horn", 3.0, 4.0), Event("dog", 5.0, 6.0),
    Event("music", 7.0, 12.0), Event("dog", 8.0, 9.0), Event("steps", 9.5, 10.5), Event("dog", 14.0, 15.0),
])


def test_predicates():
    assert [e.interval for e in TL.plain("dog")] == [(1, 2), (5, 6), (8, 9), (14, 15)]
    assert [e.interval for e in TL.ordinal("dog", 2)] == [(5, 6)]
    assert [e.interval for e in TL.ordinal("dog", "last")] == [(14, 15)]
    assert TL.ordinal("dog", 9) == []
    assert [e.interval for e in TL.after("dog", "horn")] == [(5, 6), (8, 9), (14, 15)]
    assert [e.interval for e in TL.before("dog", "horn")] == [(1, 2)]
    assert [e.interval for e in TL.next_after("dog", "horn")] == [(5, 6)]
    assert [e.interval for e in TL.while_("dog", "music")] == [(8, 9)]
    assert [e.interval for e in TL.not_followed("dog", "steps", 3.0)] == [(1, 2), (5, 6), (14, 15)]
    assert TL.absent("cat") == []


def test_after_requires_unique_reference():
    import pytest
    with pytest.raises(ValueError):
        TL.after("horn", "dog")


def test_generate_is_consistent_with_predicates():
    qs = generate(TL, "c0", ["dog", "horn", "music", "steps", "cat"], random.Random(1), max_per_type=3)
    types = {q.qtype for q in qs}
    assert {"PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ABSENT"} <= types
    for q in qs:
        assert q.expects_empty == (len(q.answer) == 0)
        assert q.plain_intervals == [e.interval for e in TL.occ(q.x)]
        if q.qtype == "WHILE":
            assert q.answer == [e.interval for e in TL.while_(q.x, q.y)]
    # round-trips through JSON
    from ctag.queries import Query
    for q in qs:
        assert Query.from_dict(json.loads(json.dumps(q.to_dict()))) == q


def test_parse_intervals():
    assert parse_intervals("[[1.2, 2.0], [7.5, 8.1]]") == [(1.2, 2.0), (7.5, 8.1)]
    assert parse_intervals("Sure! Here: [[1.2, 2.0]]") == [(1.2, 2.0)]
    assert parse_intervals("[]") == []
    assert parse_intervals("There is no such event.") == []
    assert parse_intervals("The dog barks from 1.2 to 2.0 and 7.5-8.1 seconds.") == [(1.2, 2.0), (7.5, 8.1)]
    assert parse_intervals('[{"start": 1, "end": 2}]') == [(1.0, 2.0)]
    assert parse_intervals("I cannot help with that") is None


def test_metrics():
    assert union_iou([(1, 2)], [(1, 2)]) == 1.0
    assert union_iou([], []) == 1.0
    assert union_iou([(1, 2)], []) == 0.0
    assert abs(union_iou([(1, 3)], [(2, 4)]) - 1 / 3) < 1e-9
    p, r, f = matched_f1([(1, 2), (5, 6)], [(1, 2), (5, 6), (8, 9)], 0.5)
    assert (p, r) == (1.0, 2 / 3)
    s = score_query([(1, 2)], [(1, 2), (5, 6)], False)
    assert s["n_pred"] == 1 and not s["count_acc"]
    s = score_query(None, [(1, 2)], False)
    assert s["parse_fail"] and s["f1@0.5"] == 0.0


def test_summary_rejection_metrics():
    rows = [
        {"qtype": "ORDINAL", **score_query([], [], True)},           # correct rejection
        {"qtype": "ORDINAL", **score_query([(1, 2)], [], True)},     # missed rejection
        {"qtype": "ORDINAL", **score_query([], [(1, 2)], False)},    # false rejection
        {"qtype": "ORDINAL", **score_query([(1, 2)], [(1, 2)], False)},
    ]
    s = summarize(rows)["ORDINAL"]
    assert s["rejection_precision"] == 0.5 and s["rejection_recall"] == 0.5
    assert s["false_rejection_rate"] == 0.5 and s["count_acc"] == 0.5


def test_end_to_end_procedural(tmp_path):
    from ctag.build_benchmark import build
    from ctag.run_zeroshot import main as run

    counts = build("procedural", 8, tmp_path / "bench", seed=3)
    assert counts["WHILE"] > 0 and counts["ABSENT"] > 0
    assert len(list((tmp_path / "bench" / "wav").glob("*.wav"))) == 8
    for mode in ("oracle", "first_only", "ignore_condition"):
        run(["--model", f"mock:{mode}", "--bench", str(tmp_path / "bench" / "benchmark.jsonl"), "--out", str(tmp_path / mode)])
        s = json.loads((tmp_path / mode / "summary.json").read_text())["by_type"]
        if mode == "oracle":
            assert s["ALL"]["f1@0.5"] == 1.0 and s["ALL"]["count_acc"] == 1.0
        if mode == "first_only":
            assert s["PLAIN"]["under_report_rate"] > 0


def test_rejection_metrics_are_none_when_no_rejection_queries():
    """A type with no empty-answer queries was never asked to reject; reporting
    0.0 would read as a failure at a task that was never posed."""
    rows = [{"qtype": "PLAIN", **score_query([(1, 2)], [(1, 2)], False)},
            {"qtype": "PLAIN", **score_query([(5, 6)], [(5, 6)], False)}]
    s = summarize(rows)["PLAIN"]
    assert s["n_rejection_queries"] == 0
    assert s["rejection_f1"] is None and s["rejection_precision"] is None and s["rejection_recall"] is None
    assert s["false_rejection_rate"] == 0.0

    # and a false rejection still shows up in false_rejection_rate
    rows.append({"qtype": "PLAIN", **score_query([], [(9, 10)], False)})
    s = summarize(rows)["PLAIN"]
    assert s["rejection_f1"] is None
    assert abs(s["false_rejection_rate"] - 1 / 3) < 1e-9


def test_rejection_metrics_present_when_type_has_rejection_queries():
    rows = [{"qtype": "ABSENT", **score_query([], [], True)},
            {"qtype": "ABSENT", **score_query([(1, 2)], [], True)}]
    s = summarize(rows)["ABSENT"]
    assert s["rejection_recall"] == 0.5 and s["rejection_f1"] is not None


def test_parse_real_model_output_shapes():
    """Shapes actually emitted by Qwen2-Audio on this benchmark. Returning None
    for these would score the parser rather than the model."""
    # python-style single-quoted dicts with prefixed keys
    assert parse_intervals("[{'sneeze_start': '0.63', 'sneeze_end': '1.09'}, "
                           "{'sneeze_start': '4.21', 'sneeze_end': '4.62'}]") == [(0.63, 1.09), (4.21, 4.62)]
    # single-quoted plain start/end
    assert parse_intervals("[{'start': '16.39', 'end': '16.94'}]") == [(16.39, 16.94)]
    # a dict whose single value holds a range string
    assert parse_intervals("[{'sneeze': '1.96-2.34'}, {'glass_breaking': '18.87-20.00'}]") == \
        [(1.96, 2.34), (18.87, 20.0)]
    # a flat pair rather than a list of pairs
    assert parse_intervals("[19.43, 20.00]") == [(19.43, 20.0)]
    # onset/offset naming
    assert parse_intervals('[{"onset": 1.0, "offset": 2.0}]') == [(1.0, 2.0)]
    # still refuses genuine non-answers
    assert parse_intervals("I am unable to analyse this audio.") is None
    # and still reads explicit emptiness
    assert parse_intervals("[]") == []


def test_rescore_recovers_parse_failures(tmp_path):
    """A parser improvement must be applicable to a finished run without
    re-running the model."""
    import json as _json
    from ctag.rescore import rescore

    run = tmp_path / "run"
    run.mkdir()
    raws = ["[[1.0, 2.0]]", "[{'x_start': '1.0', 'x_end': '2.0'}]", "no idea"]
    with open(run / "predictions.jsonl", "w") as f:
        for i, raw in enumerate(raws):
            f.write(_json.dumps({
                "qid": f"q{i}", "qtype": "PLAIN", "text": "t", "answer": [[1.0, 2.0]],
                "raw": raw, "pred": None, "expects_empty": False}) + "\n")
    (run / "summary.json").write_text(_json.dumps(
        {"model": "m", "bench": "b", "by_type": summarize(
            [{"qtype": "PLAIN", **score_query(None, [(1.0, 2.0)], False)} for _ in raws])}))

    s = rescore(run, in_place=False)
    assert s["parse_recovered"] == 2                     # two of three now parse
    assert abs(s["by_type"]["ALL"]["parse_fail_rate"] - 1 / 3) < 1e-9
    assert abs(s["by_type"]["ALL"]["f1@0.5"] - 2 / 3) < 1e-9


def test_localisation_separates_place_from_duration():
    """A perfectly centred but too-short interval and a displaced one both score
    0 at IoU>=0.5; centre error and duration ratio tell them apart."""
    from ctag.metrics import localisation

    gold = [(10.0, 12.0)]
    short_but_centred = [(10.9, 11.1)]
    displaced_right_size = [(2.0, 4.0)]

    assert matched_f1(short_but_centred, gold, 0.5)[2] == 0.0
    assert matched_f1(displaced_right_size, gold, 0.5)[2] == 0.0

    d1, r1 = localisation(short_but_centred, gold)
    d2, r2 = localisation(displaced_right_size, gold)
    assert d1[0] < 0.1 and abs(r1 - 0.1) < 1e-9   # right place, a tenth the length
    assert d2[0] == 8.0 and r2 == 1.0         # right length, far away

    s = score_query(short_but_centred, gold, False)
    assert abs(s["duration_ratio"] - 0.1) < 1e-9 and s["centre_errors"][0] < 0.1
    agg = summarize([{"qtype": "PLAIN", **s}])["PLAIN"]
    assert abs(agg["duration_ratio_median"] - 0.1) < 1e-9 and agg["centre_within_1s"] == 1.0


def test_agent_combine_matches_timeline_predicates():
    """The agent must apply exactly the semantics that define the ground truth,
    otherwise it is solving a different task."""
    from ctag.agent import combine

    xs = [e.interval for e in TL.occ("dog")]
    horn = [e.interval for e in TL.occ("horn")]
    music = [e.interval for e in TL.occ("music")]
    steps = [e.interval for e in TL.occ("steps")]

    assert combine("PLAIN", xs, []) == [e.interval for e in TL.plain("dog")]
    assert combine("ORDINAL", xs, [], k=2) == [e.interval for e in TL.ordinal("dog", 2)]
    assert combine("ORDINAL", xs, [], k="last") == [e.interval for e in TL.ordinal("dog", "last")]
    assert combine("ORDINAL", xs, [], k=9) == []
    assert combine("AFTER", xs, horn) == [e.interval for e in TL.after("dog", "horn")]
    assert combine("BEFORE", xs, horn) == [e.interval for e in TL.before("dog", "horn")]
    assert combine("NEXT_AFTER", xs, horn) == [e.interval for e in TL.next_after("dog", "horn")]
    assert combine("WHILE", xs, music) == [e.interval for e in TL.while_("dog", "music")]
    assert combine("NOT_FOLLOWED", xs, steps, window=3.0) == \
        [e.interval for e in TL.not_followed("dog", "steps", 3.0)]


def test_agent_with_oracle_grounder_is_perfect(tmp_path):
    """With perfect grounding the agent must score 1.000, or its composition
    disagrees with the ground truth somewhere."""
    import json as _json
    from ctag.build_benchmark import build
    from ctag.run_agent import main as run_agent

    build("procedural", 15, tmp_path / "b", seed=5)
    run_agent(["--grounder", "oracle",
               "--bench", str(tmp_path / "b" / "benchmark.jsonl"),
               "--timelines", str(tmp_path / "b" / "timelines.jsonl"),
               "--out", str(tmp_path / "agent")])
    s = _json.loads((tmp_path / "agent" / "summary.json").read_text())["by_type"]
    assert s["ALL"]["f1@0.5"] == 1.0, s["ALL"]
    assert s["ALL"]["count_acc"] == 1.0
    assert s["ALL"]["parse_fail_rate"] == 0.0


def test_oracle_grounder_matches_multiword_labels():
    """Queries carry 'glass_breaking' while timelines and prompts may use
    'glass breaking'. Normalising only one side makes the oracle silently
    return nothing for every multi-word sound."""
    from ctag.agent import oracle_grounder

    g = oracle_grounder({"c1": {"events": [
        {"label": "glass_breaking", "onset": 1.0, "offset": 2.0},
        {"label": "dog", "onset": 5.0, "offset": 6.0}]}})
    assert g("/x/c1.wav", "glass_breaking") == [(1.0, 2.0)]
    assert g("/x/c1.wav", "glass breaking") == [(1.0, 2.0)]
    assert g("/x/c1.wav", "dog") == [(5.0, 6.0)]
    assert g("/x/c1.wav", "cat") == []
    # must not match a different sound by substring
    assert g("/x/c1.wav", "glass") == []


def test_split_is_clip_level_and_stable():
    """Splitting by query would leak: queries from one clip share audio and
    timeline. And adding clips later must not reshuffle existing assignments."""
    from ctag.split import assign

    a = {c: assign(c, 0, 0.7, 0.15) for c in [f"clip_{i:04d}" for i in range(400)]}
    assert set(a.values()) == {"train", "val", "test"}
    counts = {s: sum(v == s for v in a.values()) / len(a) for s in ("train", "val", "test")}
    assert 0.62 < counts["train"] < 0.78, counts
    assert 0.09 < counts["val"] < 0.21, counts
    # stable: same clip, same seed, same answer, regardless of what else exists
    assert assign("clip_0007", 0, 0.7, 0.15) == a["clip_0007"]
    # a different seed gives a different partition
    b = {c: assign(c, 1, 0.7, 0.15) for c in a}
    assert a != b


def test_split_benchmark_has_no_clip_overlap(tmp_path):
    import json as _json
    from ctag.build_benchmark import build
    from ctag.split import split_benchmark

    build("procedural", 40, tmp_path / "b", seed=2)
    stats = split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    assert sum(d["clips"] for d in stats.values()) == 40
    seen = {}
    for s in ("train", "val", "test"):
        for line in open(tmp_path / "b" / f"benchmark_{s}.jsonl"):
            cid = _json.loads(line)["clip_id"]
            assert seen.setdefault(cid, s) == s, f"{cid} appears in two splits"


def test_sft_mix_hits_the_requested_plain_share(tmp_path):
    """Grounding is the bottleneck, so the plain share is a deliberate knob and
    must actually be honoured."""
    import json as _json
    from ctag.build_benchmark import build as build_bench
    from ctag.sft_data import build as build_sft
    from ctag.split import split_benchmark

    build_bench("procedural", 40, tmp_path / "b", seed=4)
    split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    for ratio in (0.3, 0.5, 0.8):
        s = build_sft(tmp_path / "b" / "benchmark_train.jsonl", tmp_path / "b" / "timelines.jsonl",
                      tmp_path / f"sft_{ratio}.jsonl", plain_ratio=ratio, seed=0)
        assert abs(s["plain_share"] - ratio) < 0.05, (ratio, s)
        assert s["synthesised_plain"] > 0          # augmentation actually fired
        rows = [_json.loads(l) for l in open(tmp_path / f"sft_{ratio}.jsonl")]
        assert len(rows) == s["examples"]
        # prompt format must match inference exactly
        from ctag.models import SYSTEM
        assert rows[0]["messages"][0]["content"] == SYSTEM
        assert rows[0]["messages"][1]["content"].startswith("Locate:")
        _json.loads(rows[0]["target"])            # target is valid JSON intervals


def test_sft_targets_are_parseable_by_the_metric(tmp_path):
    """A target the scorer cannot read would train the model to emit unscoreable text."""
    import json as _json
    from ctag.build_benchmark import build as build_bench
    from ctag.sft_data import build as build_sft
    from ctag.split import split_benchmark

    build_bench("procedural", 20, tmp_path / "b", seed=6)
    split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    build_sft(tmp_path / "b" / "benchmark_train.jsonl", tmp_path / "b" / "timelines.jsonl",
              tmp_path / "sft.jsonl", plain_ratio=0.5, seed=0)
    for line in open(tmp_path / "sft.jsonl"):
        e = _json.loads(line)
        assert parse_intervals(e["target"]) is not None


def _stub_processor(expansion: int, answer_tokens: int):
    """Processor stub that EXPANDS the audio placeholder, like the real one.

    The first version of these tests used a stub with no expansion, so a mask
    computed from the text prompt length looked correct. On real audio the
    placeholder becomes hundreds of positions, the mask fell far short, and
    training produced NaN gradients. Any stub here must expand.
    """
    import torch

    class Tok:
        pad_token_id = 0
        eos_token = ""

        def __call__(self, text, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = list(range(len(text.split())))
            return r

    class P:
        tokenizer = Tok()

        def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
            return "PROMPT <audio> HERE"                 # 3 "words"

        def __call__(self, text, audio, sampling_rate, return_tensors, padding):
            rows = []
            for t in text:
                n_words = len(t.split())
                # placeholder expands into `expansion` audio positions
                ids = list(range(1, n_words + expansion))
                rows.append(ids)
            width = max(len(r) for r in rows)
            ii = torch.tensor([r + [0] * (width - len(r)) for r in rows])
            am = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
            return {"input_ids": ii, "attention_mask": am}

    return P()


def test_collator_masks_survive_audio_placeholder_expansion(tmp_path):
    """The real processor turns one audio placeholder into hundreds of positions.
    A mask measured from the text prompt would cover a fraction of them and leave
    the audio region as a training target, which is what produced NaN gradients."""
    import numpy as np
    import soundfile as sf
    import torch

    from ctag.train_lora import GroundingCollator

    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

    for expansion in (1, 50, 400):
        c = GroundingCollator(_stub_processor(expansion, 2))
        ex = {"audio": str(wav),
              "messages": [{"role": "user", "content": "Locate: every dog"}],
              "target": "A B"}                            # 2 answer tokens
        batch = c([ex])
        labels, ids, attn = batch["labels"], batch["input_ids"], batch["attention_mask"]
        end = int(attn[0].sum())
        kept = (labels[0] != -100).nonzero().flatten().tolist()
        # exactly the last two real tokens are trained on, whatever the expansion
        assert kept == [end - 2, end - 1], (expansion, kept, end)
        assert torch.equal(labels[0][kept], ids[0][kept])
        # everything before, including the whole expanded audio region, is masked
        assert (labels[0, :end - 2] == -100).all()


def test_collator_masks_each_row_of_a_batch_independently(tmp_path):
    """Rows have different answer lengths and different padding; a shared mask
    would leak tokens from one row into another's loss."""
    import numpy as np
    import soundfile as sf

    from ctag.train_lora import GroundingCollator

    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

    c = GroundingCollator(_stub_processor(30, 0))
    exs = [{"audio": str(wav), "messages": [{"role": "user", "content": "q"}], "target": "A"},
           {"audio": str(wav), "messages": [{"role": "user", "content": "q"}], "target": "A B C"}]
    out = c(exs)
    labels, attn = out["labels"], out["attention_mask"]
    for i, n_ans in enumerate((1, 3)):
        end = int(attn[i].sum())
        kept = (labels[i] != -100).nonzero().flatten().tolist()
        assert kept == list(range(end - n_ans, end)), (i, kept, end)
    # padded positions are never trained on
    for i in range(2):
        pad = attn[i] == 0
        if pad.any():
            assert (labels[i][pad] == -100).all()


def test_hybrid_selects_on_val_and_reports_on_test(tmp_path):
    """Choosing the arm on the same queries it is reported on would be taking
    the max of two noisy estimates and calling it a method."""
    import json as _json
    from ctag.hybrid import main as hybrid_main
    from ctag.split import assign

    # Construct two arms with a known, type-dependent winner.
    qids = [f"clip_{i:04d}_q{j}" for i in range(120) for j in range(2)]
    def write(run, better):
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "predictions.jsonl", "w") as f:
            for n, q in enumerate(qids):
                qtype = "AFTER" if n % 2 == 0 else "ORDINAL"
                good = (better == "agent") == (qtype == "AFTER")
                s = score_query([(1.0, 2.0)] if good else [(9.0, 9.5)], [(1.0, 2.0)], False)
                f.write(_json.dumps({"qid": q, "qtype": qtype, "answer": [[1.0, 2.0]], **s}) + "\n")
    write(tmp_path / "direct", "direct")
    write(tmp_path / "agent", "agent")

    hybrid_main(["--direct", str(tmp_path / "direct"), "--agent", str(tmp_path / "agent"),
                 "--out", str(tmp_path / "hyb")])
    res = _json.loads((tmp_path / "hyb" / "summary.json").read_text())
    assert res["choice"]["AFTER"] == "agent"
    assert res["choice"]["ORDINAL"] == "direct"
    # hybrid picks the winner for each type, so it beats both arms on test
    assert res["by_type"]["ALL"]["f1@0.5"] == 1.0
    assert res["arms"]["direct"]["ALL"]["f1@0.5"] < 1.0
    assert res["arms"]["agent"]["ALL"]["f1@0.5"] < 1.0
    # reported only on test clips
    n_test = sum(1 for q in qids if assign(q.rsplit("_q", 1)[0], 0, 0.7, 0.15) == "test")
    assert res["n_test"] == n_test


# ---------------------------------------------------------------- time tokens

def test_time_vocab_size_and_quantisation():
    from ctag.timetokens import EMPTY_TOKEN, TimeVocab

    v = TimeVocab(max_seconds=30.0, resolution=0.1)
    assert len(v.times) == 301 and len(v.tokens) == 302      # + the empty token
    assert v.tokens[0] == "<t=0.0>" and v.tokens[-2] == "<t=30.0>"
    assert v.tokens[-1] == EMPTY_TOKEN
    # quantisation error never exceeds half a step
    for t in (0.0, 0.04, 0.06, 16.757, 29.99, 30.0):
        assert abs(v.quantise(t) - t) <= v.resolution / 2 + 1e-9
    # out of range clamps rather than throwing
    assert v.quantise(-5) == 0.0 and v.quantise(999) == 30.0
    # halves round up consistently; round() would send 0.65 down to 0.6 and
    # float error makes 3.15/0.1 = 31.4999... which a plain round drops to 3.1
    assert v.quantise(0.65) == 0.7
    assert v.quantise(0.75) == 0.8
    assert v.quantise(3.15) == 3.2
    assert v.quantise(0.05) == 0.1


def test_time_token_roundtrip():
    from ctag.timetokens import TimeVocab

    v = TimeVocab(30.0, 0.1)
    got = v.decode(v.encode([(0.65, 3.15), (8.87, 11.37)]))
    assert got == [(0.7, 3.2), (8.9, 11.4)]                  # quantised, order kept
    assert v.encode([]) == "<t=none>"
    assert v.decode("<t=none>") == []
    assert v.decode("no tokens here") is None
    # a reversed pair is normalised, like the text parser does
    assert v.decode(v.encode([(3.0, 1.0)])) == [(1.0, 3.0)]


def test_time_tokens_are_one_token_each_for_the_real_tokenizer_contract():
    """The whole point is one categorical decision per timestamp. This checks the
    string form is atomic and unambiguous, which is what lets add_tokens make it
    a single id."""
    from ctag.timetokens import TimeVocab

    v = TimeVocab(20.0, 0.1)
    enc = v.encode([(1.2, 3.4)])
    assert enc == "<t=1.2><t=3.4>"
    assert len(v.decode(enc)) == 1
    assert len(set(v.tokens)) == len(v.tokens)               # no duplicates


def test_gaussian_soft_labels_reward_near_misses():
    """Cross-entropy on a one-hot target calls 0.1 s off exactly as wrong as
    10 s off. TEMPO's soft target is what makes the vocabulary ordinal."""
    from ctag.timetokens import TimeVocab

    v = TimeVocab(30.0, 0.1)
    q = v.soft_labels(10.0, sigma=0.3)
    assert abs(sum(q) - 1.0) < 1e-9
    assert q[-1] == 0.0                                      # empty token gets no mass
    peak = max(range(len(q)), key=lambda i: q[i])
    assert abs(v.times[peak] - 10.0) < 1e-9                  # peaks at the target
    # monotone decay with distance, and symmetric
    assert q[v.index(10.1)] > q[v.index(10.5)] > q[v.index(12.0)]
    assert abs(q[v.index(9.7)] - q[v.index(10.3)]) < 1e-9
    # a tighter sigma concentrates mass
    assert v.soft_labels(10.0, 0.1)[v.index(10.0)] > q[v.index(10.0)]


def test_soft_labels_survive_a_target_outside_the_range():
    from ctag.timetokens import TimeVocab

    v = TimeVocab(5.0, 0.1)
    q = v.soft_labels(500.0, sigma=0.3)
    assert abs(sum(q) - 1.0) < 1e-9
    assert q[v.index(500.0)] == 1.0                          # clamped to the last step


def test_time_token_answers_parse_through_the_scoring_path():
    """A fine-tuned model emits these; the runner must score them without
    special-casing, or the two arms are not comparable."""
    from ctag.timetokens import TimeVocab

    v = TimeVocab(30.0, 0.1)
    gold = [(1.0, 2.0), (5.0, 6.0)]
    pred = v.decode(v.encode(gold))
    s = score_query(pred, gold, False)
    assert s["f1@0.5"] == 1.0 and not s["parse_fail"]


def test_parse_intervals_reads_time_tokens_without_special_casing():
    """The runner must score a time-token model through the same path as a
    text model, or the two arms are not comparable."""
    assert parse_intervals("<t=1.2><t=3.4>") == [(1.2, 3.4)]
    assert parse_intervals("<t=none>") == []
    assert parse_intervals("<t=0.7><t=3.2><t=8.9><t=11.4>") == [(0.7, 3.2), (8.9, 11.4)]
    # plain text still works, unchanged
    assert parse_intervals("[[1.2, 2.0]]") == [(1.2, 2.0)]


def test_sft_data_can_emit_time_token_targets(tmp_path):
    import json as _json
    from ctag.build_benchmark import build as build_bench
    from ctag.sft_data import build as build_sft
    from ctag.split import split_benchmark

    build_bench("procedural", 20, tmp_path / "b", seed=7)
    split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    s = build_sft(tmp_path / "b" / "benchmark_train.jsonl", tmp_path / "b" / "timelines.jsonl",
                  tmp_path / "sft_tt.jsonl", plain_ratio=0.5, seed=0, time_tokens=True)
    assert s["time_tokens"] is True
    rows = [_json.loads(l) for l in open(tmp_path / "sft_tt.jsonl")]
    assert any("<t=" in r["target"] for r in rows)
    # every target must survive the scoring parser, empty ones included
    for r in rows:
        assert parse_intervals(r["target"]) is not None
    assert any(r["target"] == "<t=none>" for r in rows)


def test_time_loss_rewards_near_misses_and_ignores_non_timestamp_positions():
    """The distance-aware loss is the reason an ordinal vocabulary helps. If it
    reduced to plain cross-entropy the timestamp tokens would be no better than
    digits, and nothing downstream would reveal it."""
    import torch

    from ctag.timetokens import TimeVocab
    from ctag.train_lora import time_loss_terms

    v = TimeVocab(max_seconds=2.0, resolution=0.1)        # 21 times + empty
    T = len(v.times)
    time_ids = torch.arange(100, 100 + T)                 # pretend ids
    Q = torch.tensor([v.soft_labels(t, 0.3)[:-1] for t in v.times], dtype=torch.float)

    V = 200
    target_time_idx = v.index(1.0)
    target_id = int(time_ids[target_time_idx])

    def loss_for(pred_time):
        logits = torch.full((1, 3, V), -10.0)
        logits[0, 0, int(time_ids[v.index(pred_time)])] = 10.0   # position 0 predicts label 1
        labels = torch.tensor([[-100, target_id, -100]])
        return time_loss_terms(logits, labels, time_ids, Q)[0].item()

    exact = loss_for(1.0)
    near = loss_for(1.1)
    far = loss_for(2.0)
    assert exact < near < far, (exact, near, far)

    # positions whose label is not a timestamp contribute nothing
    logits = torch.zeros((1, 3, V))
    labels = torch.tensor([[-100, 5, -100]])              # id 5 is not a time token
    loss, n = time_loss_terms(logits, labels, time_ids, Q)
    assert n == 0 and float(loss) == 0.0

    # masked positions are excluded even when the id collides with a time token
    labels = torch.tensor([[-100, -100, -100]])
    assert time_loss_terms(logits, labels, time_ids, Q)[1] == 0


# ------------------------------------------------------------- recall bias

def test_f_beta_weights_recall_and_reduces_to_f1_at_beta_one():
    from ctag.recall_bias import MEASURED_BETA, f_beta

    gold = [(1.0, 2.0), (5.0, 6.0), (9.0, 10.0)]
    missing_one = [(1.0, 2.0), (5.0, 6.0)]                 # recall 2/3, precision 1
    spurious_one = gold + [(15.0, 16.0)]                   # recall 1, precision 3/4

    # beta=1 must agree with the f1 used everywhere else, so the two are comparable
    assert abs(f_beta(missing_one, gold, beta=1.0) - matched_f1(missing_one, gold, 0.5)[2]) < 1e-9
    assert abs(f_beta(spurious_one, gold, beta=1.0) - matched_f1(spurious_one, gold, 0.5)[2]) < 1e-9

    # the point of beta: the penalty for a miss grows relative to a false alarm
    gap_f1 = f_beta(spurious_one, gold, 1.0) - f_beta(missing_one, gold, 1.0)
    gap_b = f_beta(spurious_one, gold, MEASURED_BETA) - f_beta(missing_one, gold, MEASURED_BETA)
    assert gap_b > gap_f1 > 0, (gap_f1, gap_b)
    # and a miss is scored strictly worse than the same-sized false alarm
    assert f_beta(spurious_one, gold, MEASURED_BETA) > f_beta(missing_one, gold, MEASURED_BETA)
    assert f_beta(gold, gold, MEASURED_BETA) == 1.0
    assert f_beta([], gold, MEASURED_BETA) == 0.0
    assert f_beta([], [], MEASURED_BETA) == 1.0


def test_measured_beta_comes_from_the_degradation_slopes():
    """beta is read off the measured asymmetry rather than tuned; guard the value
    so a careless edit cannot quietly turn it into a hyperparameter."""
    from ctag.recall_bias import MEASURED_ASYMMETRY, MEASURED_BETA

    assert abs(MEASURED_ASYMMETRY - 5.60) < 0.01
    assert abs(MEASURED_BETA ** 2 - MEASURED_ASYMMETRY) < 1e-9
    assert 2.3 < MEASURED_BETA < 2.4


def test_union_decode_raises_recall_across_samples():
    from ctag.recall_bias import union_decode

    # three samples, each missing something different
    s1 = [(1.0, 2.0), (5.0, 6.0)]
    s2 = [(1.0, 2.0), (9.0, 10.0)]
    s3 = [(5.0, 6.0), (9.0, 10.0)]
    got = union_decode([s1, s2, s3], min_votes=1)
    assert got == [(1.0, 2.0), (5.0, 6.0), (9.0, 10.0)]    # full recall recovered

    # voting trades recall back for precision
    noisy = [s1, s2, s3, [(20.0, 21.0)]]
    assert (20.0, 21.0) in union_decode(noisy, min_votes=1)
    assert (20.0, 21.0) not in union_decode(noisy, min_votes=2)

    assert union_decode([None, []]) == []
    # near-duplicates merge rather than multiplying
    merged = union_decode([[(1.0, 2.0)], [(1.05, 2.05)]], min_votes=1)
    assert len(merged) == 1


def test_preference_pairs_always_reject_under_detection():
    """Over-detection must never be the negative: a false alarm costs a fifth of
    a miss, so penalising it would optimise the wrong direction."""
    import random as _random

    from ctag.metrics import parse_intervals
    from ctag.recall_bias import make_preference_pairs

    exs = [{"audio": "a.wav", "messages": [], "target": "[[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]]"},
           {"audio": "b.wav", "messages": [], "target": "[[1.0, 2.0]]"},
           {"audio": "c.wav", "messages": [], "target": "[]"}]
    pairs = make_preference_pairs(exs, _random.Random(0))

    assert all(p["n_rejected"] < p["n_gold"] for p in pairs)     # always fewer, never more
    assert not any(p["audio"] == "c.wav" for p in pairs)         # empty gold yields no pair
    single = [p for p in pairs if p["audio"] == "b.wav"][0]
    assert parse_intervals(single["rejected"]) == []             # the most severe miss
    for p in pairs:
        assert set(parse_intervals(p["rejected"])) < set(parse_intervals(p["chosen"]))


def test_preference_pairs_match_the_target_format():
    """chosen and rejected must differ only in content, not in notation."""
    import random as _random

    from ctag.recall_bias import make_preference_pairs

    exs = [{"audio": "a.wav", "messages": [], "target": "<t=1.0><t=2.0><t=5.0><t=6.0>"}]
    p = make_preference_pairs(exs, _random.Random(0))[0]
    assert "<t=" in p["rejected"] and "[[" not in p["rejected"]


def test_union_decoding_path_end_to_end(tmp_path):
    """--samples must union rather than overwrite, and must still report a genuine
    parse failure when every sample is unreadable."""
    import json as _json
    from ctag.build_benchmark import build
    from ctag.run_zeroshot import main as run

    build("procedural", 6, tmp_path / "b", seed=11)
    run(["--model", "mock:first_only", "--bench", str(tmp_path / "b" / "benchmark.jsonl"),
         "--out", str(tmp_path / "r"), "--samples", "3", "--min-votes", "1"])
    rows = [_json.loads(l) for l in open(tmp_path / "r" / "predictions.jsonl")]
    assert rows and all(isinstance(_json.loads(r["raw"]), list) for r in rows)
    # the mock is deterministic, so a union of three identical samples is unchanged
    for r in rows:
        assert r["pred"] == _json.loads(r["raw"])[0] and r["parse_fail"] is False \
            or r["pred"] == [] or r["pred"] is not None


def test_f_beta_is_reported_in_summaries():
    rows = [{"qtype": "PLAIN", **score_query([(1.0, 2.0)], [(1.0, 2.0), (5.0, 6.0)], False)}]
    s = summarize(rows)["PLAIN"]
    assert s["f_beta"] is not None
    # a miss scores lower under F-beta than under f1
    assert s["f_beta"] < s["f1@0.5"]


def test_every_called_name_in_the_package_resolves():
    """Catch a function that is called but no longer defined.

    A patch that spliced out a block of ctag/train_lora.py removed build_model
    while leaving main()'s call to it. The module still imported, every existing
    test still passed, and the failure only appeared on a GPU after the model had
    loaded. This is a cheap static check for that whole class of mistake.
    """
    import ast
    import builtins
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "ctag"
    problems = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        defined = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Import):
                defined.update((a.asname or a.name.split(".")[0]) for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                defined.update((a.asname or a.name) for a in node.names)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            elif isinstance(node, (ast.For, ast.comprehension)):
                tgt = getattr(node, "target", None)
                if isinstance(tgt, ast.Name):
                    defined.add(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    defined.update(e.id for e in tgt.elts if isinstance(e, ast.Name))
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                defined.add(node.optional_vars.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.Lambda):
                defined.update(a.arg for a in node.args.args)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in defined:
                    problems.append(f"{path.name}:{node.lineno} calls undefined {node.func.id}()")

    assert not problems, "undefined calls:\n  " + "\n  ".join(problems)


def test_bf16_is_rejected_when_only_emulated():
    """torch.cuda.is_bf16_supported() defaults to including_emulation=True, so a
    T4 answers yes and emulated bf16 runs about 5x slower than fp16. The check
    must look at compute capability instead."""
    import sys
    import types

    from ctag import train_lora

    real = sys.modules.get("torch")
    try:
        for major, expect, name in [(7, False, "T4 / sm_75"), (8, True, "A100 / sm_80"),
                                    (9, True, "H100 / sm_90"), (6, False, "P100 / sm_60")]:
            sys.modules["torch"] = types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    is_available=lambda: True,
                    get_device_capability=lambda m=major: (m, 0),
                    # deliberately claims support, as the emulating version does
                    is_bf16_supported=lambda **kw: True,
                )
            )
            assert train_lora._bf16_ok() is expect, name
        # no GPU at all
        sys.modules["torch"] = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False))
        assert train_lora._bf16_ok() is False
    finally:
        if real is not None:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)


# ---------------------------------------------------------------------------
# Memory fixes: only the new rows train, and long sequences are bounded
# ---------------------------------------------------------------------------


def test_new_rows_train_without_unfreezing_the_whole_vocabulary():
    """modules_to_save=["embed_tokens", "lm_head"] makes both full 152k x 3584
    matrices trainable -- about 16 GB of weights, gradients and Adam state, which
    is why arm E died on a T4 in under a minute. Only the timestamp rows need to
    move."""
    import torch
    from torch import nn

    from ctag.timetokens import _new_rows_modules

    NewRowsEmbedding, NewRowsLinear = _new_rows_modules()
    v_old, n_new, h = 64, 5, 8

    base = nn.Embedding(v_old + n_new, h)
    nn.init.normal_(base.weight, std=1.0)
    before = base.weight.data.clone()
    emb = NewRowsEmbedding(base, v_old)

    ids = torch.tensor([[7, v_old + 1, v_old + 4]])
    out = emb(ids)
    assert torch.allclose(out[0, 0], before[7])
    # the delta starts at zero, so TEMPO's mean-of-BPE initialisation is kept
    assert torch.allclose(out[0, 1], before[v_old + 1])

    emb(ids).pow(2).sum().backward()
    assert base.weight.grad is None, "base embedding must stay frozen"
    touched = (emb.delta.grad.abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert touched == [1, 4], f"only the ids actually used should move, got {touched}"

    head_base = nn.Linear(h, v_old + n_new, bias=False)
    nn.init.normal_(head_base.weight, std=1.0)
    kept = head_base.weight.data[:v_old].clone()
    head = NewRowsLinear(head_base, v_old)
    x = torch.randn(2, 3, h)
    logits = head(x)
    assert logits.shape == (2, 3, v_old + n_new)
    assert torch.allclose(logits[..., :v_old], x @ kept.T, atol=1e-5)
    # new-token logits are learned from zero rather than fighting a random init
    assert torch.allclose(logits[..., v_old:], torch.zeros(2, 3, n_new), atol=1e-6)
    logits.sum().backward()
    assert head_base.weight.grad is None, "base head must stay frozen"

    trainable = sum(p.numel() for p in [*emb.parameters(), *head.parameters()]
                    if p.requires_grad)
    assert trainable == 2 * n_new * h
    assert trainable < 2 * (v_old + n_new) * h


def test_timestamp_deltas_survive_a_save_and_load(tmp_path):
    """PEFT does not know about the deltas, so if they are not written beside the
    adapter the timestamp tokens stay at initialisation and arm E measures
    nothing -- the exact failure modules_to_save was there to prevent."""
    import torch
    from torch import nn

    from ctag.timetokens import DELTA_FILE, load_deltas, save_deltas, wrap_new_rows

    v_old, n_new, h = 32, 4, 6

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(v_old + n_new, h)
            self.lm_head = nn.Linear(h, v_old + n_new, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, m):
            self.embed_tokens = m

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, m):
            self.lm_head = m

    trained = Tiny()
    emb, head = wrap_new_rows(trained, v_old)
    with torch.no_grad():
        emb.delta.normal_()
        head.delta.normal_()
    assert save_deltas(trained, str(tmp_path)) is not None
    assert (tmp_path / DELTA_FILE).exists()

    fresh = Tiny()
    fresh.embed_tokens.weight.data = trained.embed_tokens.base.weight.data.clone()
    fresh.lm_head.weight.data = trained.lm_head.base.weight.data.clone()
    assert load_deltas(fresh, str(tmp_path)) is True
    ids = torch.tensor([[3, v_old + 2]])
    assert torch.allclose(fresh.embed_tokens(ids), trained.embed_tokens(ids), atol=1e-6)
    x = torch.randn(1, 2, h)
    assert torch.allclose(fresh.lm_head(x), trained.lm_head(x), atol=1e-5)

    # an adapter dir with no deltas must be reported, not silently accepted
    assert load_deltas(Tiny(), str(tmp_path / "empty")) is False


def test_collator_caps_sequence_length_but_keeps_the_answer(tmp_path):
    """One logit per vocabulary entry per position is what OOMs: 152k x 4096 is
    1.2 GB in fp16 before the gradient. Truncation has to take the front, because
    the supervised answer is at the end."""
    import numpy as np
    import soundfile as sf

    from ctag.train_lora import GroundingCollator

    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

    ex = {"audio": str(wav),
          "messages": [{"role": "user", "content": "Locate: every dog"}],
          "target": "A B"}

    uncapped = GroundingCollator(_stub_processor(400, 2), max_seq_len=None)([ex])
    full_len = uncapped["input_ids"].shape[1]
    assert full_len > 64

    capped = GroundingCollator(_stub_processor(400, 2), max_seq_len=64)([ex])
    assert capped["input_ids"].shape[1] == 64
    assert capped["labels"].shape == capped["input_ids"].shape
    kept = (capped["labels"][0] != -100).nonzero().flatten().tolist()
    end = int(capped["attention_mask"][0].sum())
    assert kept == [end - 2, end - 1], f"answer must survive truncation, got {kept}"


# ---------------------------------------------------------------------------
# The hybrid must select on validation clips, or say why it cannot
# ---------------------------------------------------------------------------


def _write_run(dirpath, clips, qtypes, score_for):
    """A minimal predictions.jsonl that summarize() will accept."""
    import json

    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "predictions.jsonl", "w", encoding="utf-8") as f:
        for c in clips:
            for j, t in enumerate(qtypes):
                s = score_for(t)
                f.write(json.dumps({
                    "qid": f"{c}_q{j}", "qtype": t, "expects_empty": False,
                    "pred_empty": False, "parse_fail": 0, "n_pred": 1, "n_gt": 1,
                    "union_iou": s, "f1@0.5": s, "f1@0.7": s, "f_beta": s,
                    "count_acc": 1, "centre_errors": [],
                }) + "\n")


def _clips_by_split(n=400, seed=0):
    from collections import defaultdict

    from ctag.split import assign

    out = defaultdict(list)
    for i in range(n):
        c = f"clip{i:04d}"
        out[assign(c, seed, 0.7, 0.15)].append(c)
    return out


REL = {"AFTER", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED"}
QTYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED"]


def test_hybrid_refuses_to_select_when_there_is_no_validation_data(tmp_path):
    """Scoring both arms on benchmark_test.jsonl leaves nothing to select on. The
    first full run did exactly that and the hybrid came out byte-identical to arm
    A, reported as a result. It has to fail instead."""
    import pytest

    from ctag import hybrid

    clips = _clips_by_split()
    _write_run(tmp_path / "d", clips["test"], QTYPES, lambda t: 0.2)
    _write_run(tmp_path / "a", clips["test"], QTYPES,
               lambda t: 0.3 if t in REL else 0.1)

    with pytest.raises(SystemExit) as e:
        hybrid.main(["--direct", str(tmp_path / "d"), "--agent", str(tmp_path / "a"),
                     "--out", str(tmp_path / "out")])
    assert "nothing to select on" in str(e.value)


def test_hybrid_selects_per_type_from_the_validation_runs(tmp_path):
    import json

    from ctag import hybrid

    clips = _clips_by_split()
    _write_run(tmp_path / "td", clips["test"], QTYPES, lambda t: 0.2)
    _write_run(tmp_path / "ta", clips["test"], QTYPES, lambda t: 0.3 if t in REL else 0.1)
    _write_run(tmp_path / "vd", clips["val"], QTYPES, lambda t: 0.2)
    _write_run(tmp_path / "va", clips["val"], QTYPES, lambda t: 0.3 if t in REL else 0.1)

    hybrid.main(["--direct", str(tmp_path / "td"), "--agent", str(tmp_path / "ta"),
                 "--direct-val", str(tmp_path / "vd"), "--agent-val", str(tmp_path / "va"),
                 "--out", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert {t: summary["choice"][t] for t in REL} == {t: "agent" for t in REL}
    assert summary["choice"]["PLAIN"] == "direct"
    assert summary["choice"]["BEFORE"] == "direct"
    # and the hybrid must actually beat direct on the types it switched
    assert summary["by_type"]["AFTER"]["f1@0.5"] > summary["arms"]["direct"]["AFTER"]["f1@0.5"]


def test_hybrid_rejects_validation_runs_that_overlap_the_test_runs(tmp_path):
    """Selecting on queries that are also reported is choosing the maximum of two
    noisy estimates and calling it a method."""
    import pytest

    from ctag import hybrid

    clips = _clips_by_split()
    _write_run(tmp_path / "td", clips["test"], QTYPES, lambda t: 0.2)
    _write_run(tmp_path / "ta", clips["test"], QTYPES, lambda t: 0.3 if t in REL else 0.1)

    with pytest.raises(SystemExit) as e:
        hybrid.main(["--direct", str(tmp_path / "td"), "--agent", str(tmp_path / "ta"),
                     "--direct-val", str(tmp_path / "td"), "--agent-val", str(tmp_path / "ta"),
                     "--out", str(tmp_path / "out")])
    assert "validation and test" in str(e.value)



def test_audio_flamingo_3_prompt_carries_the_same_instruction_as_the_other_backends():
    """AF3's template has no system role, so the system text is folded into the
    user turn. The words the model sees must be the ones every other backend
    sees, or its score measures a different prompt."""
    from ctag.models import SYSTEM, af3_conversation, get_backend, prompt_for

    conv = af3_conversation("/tmp/clip.wav", "every dog bark after the siren", 20.0)
    assert len(conv) == 1 and conv[0]["role"] == "user"
    parts = conv[0]["content"]
    assert parts[0] == {"type": "audio", "path": "/tmp/clip.wav"}
    assert parts[1]["type"] == "text"
    assert parts[1]["text"].startswith(SYSTEM)
    assert parts[1]["text"].endswith(prompt_for("every dog bark after the siren", 20.0))

    # registered under the name the runner uses; instantiating needs a GPU runtime
    import inspect
    src = inspect.getsource(get_backend)
    assert '"audio-flamingo-3"' in src


def test_hybrid_selects_on_the_reported_metric_not_the_row_mean(tmp_path):
    """summarize() reports f1@0.5 over non-rejection queries. The selector used
    to average the per-row f1 over every row, and a rejection query scores 1.0
    for an empty answer -- so an arm that answers "nothing" most of the time
    looked like it won every type on val while losing four of them by the
    reported number. The v4 hybrid chose the agent for all seven types that way."""
    import json

    from ctag import hybrid

    clips = _clips_by_split()

    def write(dirpath, clip_list, agent):
        dirpath.mkdir(parents=True, exist_ok=True)
        with open(dirpath / "predictions.jsonl", "w", encoding="utf-8") as f:
            for c in clip_list:
                for j in range(10):
                    rejection = j >= 6                    # 40 % rejection queries
                    if agent:
                        # answers empty always: right on rejection rows, wrong otherwise
                        s = score_query([], [] if rejection else [(1.0, 2.0)], rejection)
                    else:
                        # never empty: half right on real queries, wrong on rejection rows
                        pred = [(1.0, 2.0)] if j % 2 == 0 else [(8.0, 9.0)]
                        s = score_query(pred, [] if rejection else [(1.0, 2.0)], rejection)
                    f.write(json.dumps({"qid": f"{c}_q{j}", "qtype": "AFTER", **s}) + "\n")

    write(tmp_path / "vd", clips["val"], agent=False)
    write(tmp_path / "va", clips["val"], agent=True)
    write(tmp_path / "td", clips["test"], agent=False)
    write(tmp_path / "ta", clips["test"], agent=True)

    # sanity: the row mean would pick the agent (0.4 vs 0.3), the reported metric direct
    rows_a = [json.loads(l) for l in open(tmp_path / "va" / "predictions.jsonl")]
    rows_d = [json.loads(l) for l in open(tmp_path / "vd" / "predictions.jsonl")]
    assert sum(r["f1@0.5"] for r in rows_a) / len(rows_a) > sum(r["f1@0.5"] for r in rows_d) / len(rows_d)
    assert summarize(rows_a)["AFTER"]["f1@0.5"] < summarize(rows_d)["AFTER"]["f1@0.5"]

    hybrid.main(["--direct", str(tmp_path / "td"), "--agent", str(tmp_path / "ta"),
                 "--direct-val", str(tmp_path / "vd"), "--agent-val", str(tmp_path / "va"),
                 "--out", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["choice"]["AFTER"] == "direct"
    assert summary["by_type"]["AFTER"]["f1@0.5"] == summary["arms"]["direct"]["AFTER"]["f1@0.5"]


def test_lora_targets_only_the_language_model():
    """The bare names ["q_proj", ...] also match the audio tower's and the vision
    tower's attention projections. The first completed adapter had 192 audio and
    192 visual LoRA tensors next to the 392 language-model ones."""
    import re

    from ctag.train_lora import LM_TARGET_MODULES

    lm = ["model.layers.0.self_attn.q_proj", "model.layers.27.mlp.down_proj",
          "model.layers.3.self_attn.o_proj", "model.layers.12.mlp.gate_proj"]
    not_lm = ["audio_tower.layers.0.self_attn.q_proj", "visual.blocks.5.attn.qkv",
              "visual.blocks.5.attn.proj", "audio_tower.layers.31.fc1",
              "lm_head", "model.embed_tokens", "model.layers.0.self_attn.q_proj.weight"]
    assert all(re.fullmatch(LM_TARGET_MODULES, n) for n in lm)
    assert not any(re.fullmatch(LM_TARGET_MODULES, n) for n in not_lm)


def test_lora_subtree_count_refuses_encoder_leakage():
    from torch import nn

    from ctag.train_lora import lora_targets_by_subtree

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2)

    m = Fake()
    m.a.lora_A = nn.Linear(2, 1)
    names = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": 1,
             "base_model.model.audio_tower.layers.0.self_attn.q_proj.lora_A.weight": 1,
             "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": 1}

    class Named:
        def named_parameters(self):
            return [(n, None) for n in names]

    assert lora_targets_by_subtree(Named()) == {"model": 1, "audio_tower": 1}


def test_new_rows_delta_lives_where_the_base_rows_live():
    """Arm E's second death: the delta was born on the CPU while the 4-bit base sat
    on the GPU. Only the CPU case is testable here, but the device is read off
    the base weight rather than assumed."""
    import torch
    from torch import nn

    from ctag.timetokens import _new_rows_modules

    NewRowsEmbedding, NewRowsLinear = _new_rows_modules()
    base = nn.Embedding(10, 4)
    emb = NewRowsEmbedding(base, 8)
    assert emb.delta.device == base.weight.device
    head = NewRowsLinear(nn.Linear(4, 10, bias=False), 8)
    assert head.delta.device == head.base.weight.device
    out = emb(torch.tensor([[1, 9]]))
    assert out.shape == (1, 2, 4)


class _FakeTok:
    """Enough of a tokenizer for init_embeddings and vocab_from_tokenizer:
    digits and '.' are base pieces, timestamp tokens are added ids."""
    def __init__(self, vocab):
        self.base = {c: i for i, c in enumerate("0123456789.none")}
        self.added = {t: len(self.base) + i for i, t in enumerate(vocab.tokens)}
    def __len__(self):
        return len(self.base) + len(self.added)
    def convert_tokens_to_ids(self, t):
        return self.added.get(t, -1)
    def get_added_vocab(self):
        return dict(self.added)
    def __call__(self, text, add_special_tokens=False):
        class R: pass
        r = R(); r.input_ids = [self.base[c] for c in text if c in self.base]; return r


def test_inference_adds_the_delta_to_the_rows_it_was_trained_on(tmp_path):
    """resize_token_embeddings fills new rows from a fitted normal, not from the
    mean-of-BPE init training used. Loading the delta onto those random rows
    gives a different embedding for every timestamp token than the one the
    adapter learned against -- a silent way to make arm E measure nothing."""
    import torch
    from torch import nn

    from ctag.timetokens import TimeVocab, load_deltas, save_deltas, wrap_new_rows

    v = TimeVocab(1.0, 0.5)                     # 3 times + none = 4 new tokens
    tok = _FakeTok(v)
    base_size = len(tok.base); n_new = len(v.tokens); h = 6

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(base_size + n_new, h)
            self.head = nn.Linear(h, base_size + n_new, bias=False)
        def get_input_embeddings(self): return self.emb
        def set_input_embeddings(self, m): self.emb = m
        def get_output_embeddings(self): return self.head
        def set_output_embeddings(self, m): self.head = m

    torch.manual_seed(0)
    trained = Tiny()
    v.init_embeddings(tok, trained.emb.weight)          # what train_lora does
    emb_t, _ = wrap_new_rows(trained, base_size)
    with torch.no_grad():
        emb_t.delta.normal_()
    save_deltas(trained, tmp_path)
    ids = torch.tensor([[base_size, base_size + 2, 3]])
    want = emb_t(ids)

    # inference: same base rows, new rows filled with noise as resize() would
    fresh = Tiny()
    with torch.no_grad():
        fresh.emb.weight[:base_size].copy_(trained.emb.weight[:base_size] if hasattr(trained.emb, "weight") else emb_t.base.weight[:base_size])
        fresh.emb.weight[base_size:].normal_()
    assert load_deltas(fresh, tmp_path, tokenizer=None)
    assert torch.allclose(fresh.emb(ids), want), "saved base rows were not restored"

    # an older delta file without base rows: rebuilt from the tokenizer
    blob = torch.load(tmp_path / "time_deltas.pt")
    del blob["embedding"]["base_rows"]
    torch.save(blob, tmp_path / "time_deltas.pt")
    older = Tiny()
    with torch.no_grad():
        older.emb.weight[:base_size].copy_(emb_t.base.weight[:base_size])
        older.emb.weight[base_size:].normal_()
    assert load_deltas(older, tmp_path, tokenizer=tok)
    assert torch.allclose(older.emb(ids), want, atol=1e-6), "rebuilt base rows differ from training"

    # and with neither, refuse rather than measure noise
    import pytest
    with pytest.raises(RuntimeError):
        load_deltas(Tiny(), tmp_path, tokenizer=None)


def test_vocab_is_recovered_from_the_tokenizer():
    from ctag.timetokens import TimeVocab, vocab_from_tokenizer

    v = TimeVocab(30.0, 0.1)
    got = vocab_from_tokenizer(_FakeTok(v))
    assert got.tokens == v.tokens
