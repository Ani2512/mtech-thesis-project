"""DESED's synthetic evaluation soundscapes: exact labels, real sound sources.

DCASE 2020 task 4 shipped its synthetic evaluation set as Scaper "recipes"
(JAMS files, Zenodo record 6026841, DESED_synth_dcase20_eval_jams.tar.gz): for
each 10 s soundscape, which isolated recording from the DESED soundbank plays
when, at what level, with what pitch shift. The labels are therefore exact by
construction (the recipe *is* the ground truth), while the sounds are real
speech, dishes, frying ... in the same ten classes as the public set. That is
the "undisputed" real-sound test the hand review cannot be: no human drew the
boxes.

    # 1. labels straight from the recipes (no audio needed)
    python scripts/desed_synth.py labels --jams data/desed/synth/jams_eval --out data/desed/metadata/synth20_eval.tsv
    # 2. the audio, rendered from the soundbank exactly as DCASE did (scaper + sox)
    python scripts/desed_synth.py render --jams data/desed/synth/jams_eval --soundbank data/desed/synth/soundbank \
           --out data/desed/audio/synth20_eval --workers 6
    # 3. then the usual import + questions (docs/real_recordings.md)

`labels` applies DESED's own post-processing to the recipe events: clip to the
clip bounds, drop events shorter than 250 ms, merge same-label events separated
by less than 150 ms, strip the _nOn/_nOff source suffixes (desed.post_process;
the thresholds DESED used for its synthetic tsv files).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MIN_EVENT_S = 0.250
MERGE_GAP_S = 0.150


def recipe_events(jams_path: Path) -> tuple[float, list[tuple[str, float, float]]]:
    """(duration, [(label, onset, offset)]) for the foreground events of one recipe."""
    d = json.load(open(jams_path, encoding="utf-8"))
    dur = float(d["file_metadata"]["duration"])
    ann = next(a for a in d["annotations"] if a["namespace"] == "scaper")
    ev = []
    for o in ann["data"]:
        v = o["value"]
        if v.get("role") != "foreground":
            continue
        on, off = float(o["time"]), float(o["time"]) + float(o["duration"])
        # soundbank names mark sources cut at the clip edge: Frying_nOff, Vacuum_cleaner_nOn_nOff;
        # DESED strips the suffix for the labels (rm_nOn_nOff in desed.post_process)
        ev.append((str(v["label"]).split("_nO")[0], max(0.0, on), min(dur, off)))
    return dur, ev


def post_process(events: list[tuple[str, float, float]], min_event: float = MIN_EVENT_S,
                 merge_gap: float = MERGE_GAP_S) -> list[tuple[str, float, float]]:
    """DESED's rules for turning recipe events into strong labels: merge
    same-label events closer than merge_gap, then drop events shorter than
    min_event. Order matters (merge first), as in desed.post_process."""
    out = []
    by_label: dict[str, list[list[float]]] = {}
    for lab, on, off in sorted(events, key=lambda e: (e[0], e[1])):
        segs = by_label.setdefault(lab, [])
        if segs and on - segs[-1][1] < merge_gap:
            segs[-1][1] = max(segs[-1][1], off)
        else:
            segs.append([on, off])
    for lab, segs in by_label.items():
        for on, off in segs:
            if off - on >= min_event:
                out.append((lab, round(on, 3), round(off, 3)))
    return sorted(out, key=lambda e: (e[1], e[0]))


def write_labels(jams_dir: Path, out: Path) -> dict:
    files = sorted(jams_dir.rglob("*.jams"))
    if not files:
        raise SystemExit(f"no .jams under {jams_dir}")
    n_ev, n_raw, labels = 0, 0, {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "onset", "offset", "event_label"])
        for p in files:
            _, raw = recipe_events(p)
            n_raw += len(raw)
            ev = post_process(raw)
            name = p.stem + ".wav"
            if not ev:                                  # an empty soundscape: one blank-label row, as DESED does
                w.writerow([name, "", "", ""])
            for lab, on, off in ev:
                w.writerow([name, f"{on:.3f}", f"{off:.3f}", lab])
                labels[lab] = labels.get(lab, 0) + 1
                n_ev += 1
    return {"clips": len(files), "events": n_ev, "recipe_events": n_raw, "labels": dict(sorted(labels.items()))}


def _render_one(args):
    jams_path, out_wav, fg, bg = args
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf                                # scaper 1.6.5 still uses the name NumPy 2 removed
    import scaper                                      # needs sox; imported here so `labels` works without it
    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000:
        return jams_path, "exists"
    try:
        scaper.generate_from_jams(str(jams_path), str(out_wav), fg_path=str(fg), bg_path=str(bg),
                                  jams_outfile=None, save_isolated_events=False)
        return jams_path, "ok"
    except Exception as e:                             # one bad recipe must not kill the batch
        return jams_path, f"failed: {type(e).__name__}: {str(e)[:120]}"


def render(jams_dir: Path, soundbank: Path, out: Path, workers: int) -> dict:
    fg, bg = soundbank / "audio" / "eval" / "soundbank" / "foreground", soundbank / "audio" / "eval" / "soundbank" / "background"
    if not fg.is_dir() or not bg.is_dir():
        # the archive's layout varies by version; find the eval soundbank
        cands = [p for p in soundbank.rglob("foreground") if "eval" in str(p)]
        if not cands:
            raise SystemExit(f"eval soundbank not found under {soundbank} (expect .../eval/soundbank/foreground)")
        fg = cands[0]
        bg = fg.parent / "background"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, out / (p.stem + ".wav"), fg, bg) for p in sorted(jams_dir.rglob("*.jams"))]
    counts, failed = {}, []
    with ProcessPoolExecutor(workers) as pool:
        for k, fut in enumerate(as_completed([pool.submit(_render_one, j) for j in jobs]), 1):
            p, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failed.append((p.name, status))
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {counts}", flush=True)
    if failed:
        with open(out / "failed.tsv", "w", encoding="utf-8") as f:
            for name, why in failed:
                f.write(f"{name}\t{why}\n")
    return {"rendered": counts, "fg": str(fg), "bg": str(bg), "failed": len(failed)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("labels")
    a.add_argument("--jams", required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("--jams", required=True)
    r.add_argument("--soundbank", required=True, help="the unpacked DESED_synth_soundbank directory")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    if args.cmd == "labels":
        print(json.dumps(write_labels(Path(args.jams), Path(args.out)), indent=1))
    else:
        print(json.dumps(render(Path(args.jams), Path(args.soundbank), Path(args.out), args.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
