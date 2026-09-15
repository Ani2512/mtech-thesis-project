"""Real recordings: strong labels in, the same benchmark out.

Everything measured so far lives on composed ESC-50 audio. This module turns
existing strongly-labelled real datasets into the project's timeline format so
the *same* query generator and predicates produce conditional questions over
real audio, then supports the hand-verification pass those questions need.

    # 1. import strong labels (DESED: filename/onset/offset/event_label TSV)
    python -m ctag.real_data import --format desed --tsv DESED/metadata/validation/validation.tsv \
           --audio-root DESED/audio/validation --out data/desed
    #    AudioSet-strong: segment_id/start_time_seconds/end_time_seconds/label(mid) + mid map
    python -m ctag.real_data import --format audioset --tsv audioset_eval_strong.tsv \
           --mid-map mid_to_display_name.tsv --audio-root audioset/eval --out data/audioset_strong

    # 2. questions from the timelines (identical generator to the composed benchmark)
    python -m ctag.real_data queries --timelines data/desed/timelines.jsonl --out data/desed \
           --min-overlap 0.3 --max-per-type 2

    # 3. hand verification: export a CSV, mark ok / corrected_answer, apply it
    python -m ctag.real_data export-review --bench data/desed/benchmark.jsonl --out data/desed/review.csv
    python -m ctag.real_data apply-review --bench data/desed/benchmark.jsonl \
           --review data/desed/review.csv --out data/desed/benchmark_verified.jsonl

Clip ids are the audio file stems, as everywhere else, so `run_transcribe`,
`run_agent --grounder timeline` and the oracle work unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .queries import generate
from .timeline import Event, Timeline

DESED_COLS = ("filename", "onset", "offset", "event_label")
AUDIOSET_COLS = ("segment_id", "start_time_seconds", "end_time_seconds", "label")


def phrase(label: str) -> str:
    """'Electric_shaver_toothbrush' -> 'electric shaver toothbrush'; the same
    normalisation ctag.agent._norm applies when matching."""
    return label.strip().replace("_", " ").lower()


def read_mid_map(path: Path) -> dict[str, str]:
    """AudioSet ontology: mid -> display name (tab-separated, no header)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].startswith("/"):
            out[parts[0]] = parts[1]
    return out


def read_strong_tsv(tsv: Path, fmt: str, mid_map: dict[str, str] | None = None) -> dict[str, list[Event]]:
    """clip stem -> events. Rows with offset <= onset or an unknown mid are dropped
    and counted; the caller prints the counts so a bad file is visible."""
    events: dict[str, list[Event]] = defaultdict(list)
    dropped = Counter()
    with open(tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = DESED_COLS if fmt == "desed" else AUDIOSET_COLS
        missing = [c for c in cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{tsv}: expected columns {cols}, missing {missing}; got {reader.fieldnames}")
        for row in reader:
            fn, a, b, lab = (row[c] for c in cols)
            if fmt == "audioset":
                if mid_map is None or lab not in mid_map:
                    dropped["unknown_mid"] += 1
                    continue
                lab = mid_map[lab]
            stem = Path(fn).stem if fmt == "desed" else fn
            if not lab.strip():
                # DESED marks a clip with no events by one row with a blank label
                dropped["blank_label"] += 1
                events.setdefault(stem, [])
                continue
            try:
                a, b = float(a), float(b)
            except ValueError:
                dropped["bad_time"] += 1
                continue
            if b <= a:
                dropped["empty"] += 1
                continue
            events[stem].append(Event(lab.strip(), round(a, 3), round(b, 3)))
    if dropped:
        print(f"[real_data] dropped rows: {dict(dropped)}")
    return events


def _duration(audio: Path | None, events: list[Event], default: float | None) -> float:
    if audio and audio.exists():
        try:
            import soundfile as sf
            info = sf.info(str(audio))
            return round(info.frames / info.samplerate, 3)
        except Exception:
            pass
    if default:
        return default
    if not events:            # an empty clip with no audio on disk and no --duration
        return 0.0
    return round(max(e.offset for e in events) + 0.5, 3)


def import_strong(tsv: Path, audio_root: Path | None, out: Path, fmt: str = "desed",
                  mid_map: Path | None = None, duration: float | None = None, source: str | None = None,
                  audio_ext: str = ".wav", require_audio: bool = False) -> dict:
    """Write <out>/timelines.jsonl in the project's layout from a strong-label file."""
    mids = read_mid_map(mid_map) if mid_map else None
    per_clip = read_strong_tsv(tsv, fmt, mids)
    out.mkdir(parents=True, exist_ok=True)
    n_written = n_no_audio = 0
    labels = Counter()
    with open(out / "timelines.jsonl", "w", encoding="utf-8") as f:
        for stem in sorted(per_clip):
            audio = (audio_root / f"{stem}{audio_ext}") if audio_root else None
            if audio is not None and not audio.exists():
                n_no_audio += 1
                if require_audio:
                    continue
            tl = Timeline(_duration(audio, per_clip[stem], duration), per_clip[stem])
            row = {"clip_id": stem, "audio": str(audio) if audio else "", "source": source or fmt, **tl.to_dict()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            labels.update(e.label for e in tl.events)
    stats = {"clips": n_written, "clips_without_audio": n_no_audio, "events": sum(labels.values()),
             "labels": dict(sorted(labels.items())), "format": fmt}
    (out / "import_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ------------------------------------------------------------------ queries
def _max_overlap(e: Event, ys: list[Event]) -> float:
    return max((e.overlap(y) for y in ys), default=0.0)


def build_queries(timelines: Path, out: Path, seed: int = 0, max_per_type: int = 2, window: float = 3.0,
                  empty_frac: float = 0.25, min_overlap: float = 0.0, max_events: int | None = None,
                  vocab: list[str] | None = None) -> dict:
    """Run the composed benchmark's generator over real timelines.

    min_overlap: drop WHILE queries whose answer depends on an overlap shorter
    than this (the composer guarantees >= 0.3 s; real labels do not), so the
    condition is audible rather than an annotation artefact. Dropped queries
    are counted, not silently absent.
    """
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(timelines, encoding="utf-8")]
    vocab = vocab or sorted({e["label"] for r in rows for e in r["events"]})
    n_q, dropped = Counter(), Counter()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark.jsonl", "w", encoding="utf-8") as fq:
        for r in rows:
            tl = Timeline.from_dict(r)
            if max_events and len(tl.events) > max_events:
                dropped["clip_too_dense"] += 1
                continue
            if not tl.events:
                dropped["clip_empty"] += 1
                continue
            for q in generate(tl, r["clip_id"], vocab, rng, max_per_type, window, empty_frac):
                if q.qtype == "WHILE" and min_overlap > 0:
                    ys = tl.occ(q.y)
                    # ambiguous if any X event overlaps Y by a sliver, whether or not it is in the answer
                    if any(0 < _max_overlap(x, ys) < min_overlap for x in tl.occ(q.x)):
                        dropped["while_ambiguous"] += 1
                        continue
                for lab in vocab:
                    q.text = q.text.replace(lab, phrase(lab))
                d = q.to_dict()
                d["audio"], d["duration"] = r.get("audio", ""), r.get("duration")
                fq.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_q[q.qtype] += 1
    stats = {"clips": len(rows), "queries": sum(n_q.values()), "by_type": dict(n_q),
             "dropped": dict(dropped), "vocab": vocab, "min_overlap": min_overlap}
    (out / "query_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ------------------------------------------------------------------ review
REVIEW_COLS = ["qid", "clip_id", "audio", "qtype", "text", "answer", "expects_empty", "ok", "corrected_answer", "note"]


def export_review(bench: Path, out_csv: Path) -> int:
    """One row per query with blank `ok` / `corrected_answer` / `note` columns.
    Mark ok = y to keep, n to drop; put a JSON interval list in corrected_answer
    to override the automatic answer (an empty list [] is a valid correction)."""
    n = 0
    with open(bench, encoding="utf-8") as f, open(out_csv, "w", encoding="utf-8", newline="") as o:
        w = csv.DictWriter(o, fieldnames=REVIEW_COLS)
        w.writeheader()
        for line in f:
            d = json.loads(line)
            w.writerow({"qid": d["qid"], "clip_id": d["clip_id"], "audio": d.get("audio", ""), "qtype": d["qtype"],
                        "text": d["text"], "answer": json.dumps(d["answer"]), "expects_empty": d["expects_empty"],
                        "ok": "", "corrected_answer": "", "note": ""})
            n += 1
    return n


_YES = {"y", "yes", "1", "true", "ok"}
_NO = {"n", "no", "0", "false", "drop"}


def apply_review(bench: Path, review: Path, out: Path) -> dict:
    """Keep the queries marked ok, apply corrections, refuse ambiguity.

    A row whose `ok` is blank is *unreviewed* and is dropped with its own count,
    so a half-finished review never passes as a verified set."""
    marks = {}
    with open(review, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            marks[row["qid"]] = row
    kept, dropped, corrected, unreviewed, unknown = [], 0, 0, 0, 0
    for line in open(bench, encoding="utf-8"):
        d = json.loads(line)
        m = marks.get(d["qid"])
        if m is None:
            unreviewed += 1
            continue
        ok = (m.get("ok") or "").strip().lower()
        if ok in _NO:
            dropped += 1
            continue
        if ok not in _YES:
            if ok:
                unknown += 1
            else:
                unreviewed += 1
            continue
        corr = (m.get("corrected_answer") or "").strip()
        if corr:
            ans = json.loads(corr)
            if not isinstance(ans, list) or not all(isinstance(p, list) and len(p) == 2 for p in ans):
                raise ValueError(f"{d['qid']}: corrected_answer must be a JSON list of [start, end] pairs, got {corr!r}")
            d["answer"] = [[float(a), float(b)] for a, b in ans]
            d["expects_empty"] = len(ans) == 0
            d["verified"] = "corrected"
            corrected += 1
        else:
            d["verified"] = "confirmed"
        kept.append(d)
    if unknown:
        raise ValueError(f"{unknown} rows have an `ok` value that is neither yes nor no")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"kept": len(kept), "confirmed": len(kept) - corrected, "corrected": corrected,
            "dropped": dropped, "unreviewed": unreviewed,
            "by_type": dict(Counter(d["qtype"] for d in kept))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="strong-label TSV -> timelines.jsonl")
    p.add_argument("--tsv", required=True)
    p.add_argument("--format", choices=["desed", "audioset"], default="desed")
    p.add_argument("--mid-map", default=None, help="AudioSet mid_to_display_name.tsv")
    p.add_argument("--audio-root", default=None)
    p.add_argument("--audio-ext", default=".wav")
    p.add_argument("--duration", type=float, default=None, help="clip length if the audio is not on disk (DESED: 10)")
    p.add_argument("--require-audio", action="store_true", help="skip clips whose audio file is missing")
    p.add_argument("--source", default=None)
    p.add_argument("--out", required=True)
    q = sub.add_parser("queries", help="timelines.jsonl -> benchmark.jsonl with the composed generator")
    q.add_argument("--timelines", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--max-per-type", type=int, default=2)
    q.add_argument("--window", type=float, default=3.0)
    q.add_argument("--empty-frac", type=float, default=0.25)
    q.add_argument("--min-overlap", type=float, default=0.3)
    q.add_argument("--max-events", type=int, default=None, help="skip clips denser than this")
    e = sub.add_parser("export-review")
    e.add_argument("--bench", required=True)
    e.add_argument("--out", required=True)
    r = sub.add_parser("apply-review")
    r.add_argument("--bench", required=True)
    r.add_argument("--review", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "import":
        s = import_strong(Path(a.tsv), Path(a.audio_root) if a.audio_root else None, Path(a.out), a.format,
                          Path(a.mid_map) if a.mid_map else None, a.duration, a.source, a.audio_ext, a.require_audio)
    elif a.cmd == "queries":
        s = build_queries(Path(a.timelines), Path(a.out), a.seed, a.max_per_type, a.window, a.empty_frac,
                          a.min_overlap, a.max_events)
    elif a.cmd == "export-review":
        s = {"rows": export_review(Path(a.bench), Path(a.out))}
    else:
        s = apply_review(Path(a.bench), Path(a.review), Path(a.out))
    print(json.dumps({k: v for k, v in s.items() if k != "labels"}, indent=2))


if __name__ == "__main__":
    main()
