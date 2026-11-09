"""The hand-review page's CSV round trip (no network)."""
import csv
import sys

import pytest

sys.path.insert(0, "scripts")
import review_server as rs  # noqa: E402


def test_apply_verdict_round_trips_through_the_csv(tmp_path):
    p = tmp_path / "review.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rs.COLUMNS)
        w.writeheader()
        w.writerow({"qid": "c1_q1", "clip_id": "c1", "audio": "a.wav", "qtype": "PLAIN", "text": "every dog",
                    "answer": "[[1.0, 2.0]]", "expects_empty": "False", "ok": "", "corrected_answer": "", "note": ""})
        w.writerow({"qid": "c1_q2", "clip_id": "c1", "audio": "a.wav", "qtype": "ABSENT", "text": "every cat",
                    "answer": "[]", "expects_empty": "True", "ok": "", "corrected_answer": "", "note": ""})
    rows = rs.read_rows(p)
    assert rs.progress(rows) == {"total": 2, "done": 0, "yes": 0, "no": 0, "corrected": 0}
    rs.apply_verdict(rows, "c1_q1", "y", "[[1.05, 2.1]]", " late start ")
    rs.apply_verdict(rows, "c1_q2", "n", "", "")
    rs.write_rows(p, rows)
    back = rs.read_rows(p)
    assert back[0]["ok"] == "y" and back[0]["corrected_answer"] == "[[1.05, 2.1]]" and back[0]["note"] == "late start"
    assert back[1]["ok"] == "n" and rs.progress(back) == {"total": 2, "done": 2, "yes": 1, "no": 1, "corrected": 1}
    assert list(back[0].keys()) == rs.COLUMNS                     # apply-review's columns untouched
    rs.apply_verdict(back, "c1_q1", "", "", "")                   # a verdict can be cleared
    assert back[0]["ok"] == ""
    with pytest.raises(ValueError):
        rs.apply_verdict(back, "c1_q1", "maybe", "", "")
    with pytest.raises(ValueError):
        rs.apply_verdict(back, "c1_q1", "y", "[[2.0, 1.0]]", "")  # end before start
    with pytest.raises(KeyError):
        rs.apply_verdict(back, "nope", "y", "", "")


def test_apply_review_accepts_the_pages_output(tmp_path):
    """The CSV the page writes must be exactly what ctag.real_data apply-review reads."""
    import json
    from ctag.real_data import apply_review

    bench = tmp_path / "b.jsonl"
    rows = [{"qid": "c1_q1", "clip_id": "c1", "audio": "a.wav", "qtype": "PLAIN", "text": "every dog", "answer": [[1.0, 2.0]], "expects_empty": False},
            {"qid": "c1_q2", "clip_id": "c1", "audio": "a.wav", "qtype": "ABSENT", "text": "every cat", "answer": [], "expects_empty": True},
            {"qid": "c1_q3", "clip_id": "c1", "audio": "a.wav", "qtype": "PLAIN", "text": "every cow", "answer": [[3.0, 4.0]], "expects_empty": False}]
    bench.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = tmp_path / "review.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rs.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in ("qid", "clip_id", "audio", "qtype", "text")}, "answer": json.dumps(r["answer"]),
                        "expects_empty": str(r["expects_empty"]), "ok": "", "corrected_answer": "", "note": ""})
    rv = rs.read_rows(p)
    rs.apply_verdict(rv, "c1_q1", "y", "[[1.1, 2.0]]", "")
    rs.apply_verdict(rv, "c1_q2", "n", "", "unfair")
    rs.write_rows(p, rv)                                          # c1_q3 left blank: never passes
    out = tmp_path / "verified.jsonl"
    apply_review(bench, p, out)
    kept = [json.loads(l) for l in open(out)]
    assert [k["qid"] for k in kept] == ["c1_q1"] and kept[0]["answer"] == [[1.1, 2.0]]
