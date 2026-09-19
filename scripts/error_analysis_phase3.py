"""Error analysis of the phase 3 timeline transcriber (val + test).

Compares every predicted timeline with the gold timeline, classifies each
imperfect clip (dropped event / spurious event, where it sits, whether the
audio there is silent), and prints the tables used in
docs/error_analysis_phase3.md.  CPU only, a few seconds.

    python scripts/error_analysis_phase3.py --runs runs_nebius_phase3 --timelines data/esc50/timelines.jsonl
"""
import argparse
import collections
import json

import numpy as np
import soundfile as sf


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def max_concurrency(evs):
    pts = sorted([(e["onset"], 1) for e in evs] + [(e["offset"], -1) for e in evs])
    c = m = 0
    for _, d in pts:
        c += d
        m = max(m, c)
    return m


def rms(path, a, b):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    seg = y[int(a * sr): int(b * sr)]
    return float(np.sqrt(np.mean(seg ** 2)) + 1e-12), float(np.sqrt(np.mean(y ** 2)))


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_nebius_phase3")
    ap.add_argument("--timelines", default="data/esc50/timelines.jsonl")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    gold = {r["clip_id"]: r for r in load(args.timelines)}
    rows = []
    by_conc = collections.defaultdict(lambda: [0, 0])
    by_n = collections.defaultdict(lambda: [0, 0])
    hit = collections.Counter(); n_gold = collections.Counter(); n_pred = collections.Counter()
    matched_centre_err = []
    for split in args.splits:
        preds = load(f"{args.runs}/esc50/transcribe_{split}/pred_timelines.jsonl")
        for p in preds:
            g = sorted(gold[p["clip_id"]]["events"], key=lambda e: e["onset"])
            gl = [(e["label"].replace("_", " "), e["onset"], e["offset"]) for e in g]
            pl = [(e["label"], e["onset"], e["offset"]) for e in p["events"]]
            used = set()
            missed = []
            for x in gl:
                n_gold[x[0]] += 1
                ok = False
                for j, y in enumerate(pl):
                    if j not in used and y[0] == x[0] and iou(x[1:], y[1:]) >= args.iou:
                        used.add(j); hit[x[0]] += 1; ok = True
                        matched_centre_err.append(abs((x[1] + x[2]) / 2 - (y[1] + y[2]) / 2))
                        break
                if not ok:
                    missed.append(x)
            spurious = [(j, y) for j, y in enumerate(pl) if j not in used]
            for y in pl:
                n_pred[y[0]] += 1
            bad = bool(missed or spurious)
            mc = max_concurrency(g)
            by_conc[mc][0] += 1; by_conc[mc][1] += bad
            by_n[len(g)][0] += 1; by_n[len(g)][1] += bad
            if not bad:
                continue
            last_end = max(e["offset"] for e in g)
            for x in missed:
                overl = [z for z in gl if z != x and iou(x[1:], z[1:]) > 0]
                rows.append(dict(split=split, clip=p["clip_id"], kind="dropped", label=x[0],
                                 window=f"{x[1]:.2f}-{x[2]:.2f}", n_gold=len(g), maxconc=mc,
                                 note=f"overlaps {len(overl)} other event(s): " + ", ".join(z[0] for z in overl)))
            for j, y in spurious:
                r_seg, r_clip = rms(p["audio"], y[1], y[2])
                overl = [z for z in gl if iou(y[1:], z[1:]) > 0]
                where = "tail, after the last gold event" if y[1] >= last_end - 0.5 else \
                    f"on top of {', '.join(z[0] for z in overl)}" if overl else "gap between events"
                rows.append(dict(split=split, clip=p["clip_id"], kind="spurious", label=y[0],
                                 window=f"{y[1]:.2f}-{y[2]:.2f}", n_gold=len(g), maxconc=mc,
                                 note=f"{where}; last emitted={'yes' if j == len(pl) - 1 else 'no'}; "
                                      f"rms ratio {r_seg / r_clip:.2f}"))

    total = sum(v[0] for v in by_conc.values())
    print(f"clips scored: {total}; imperfect clips: {len({r['clip'] for r in rows})}; "
          f"dropped events: {sum(r['kind']=='dropped' for r in rows)}; spurious events: {sum(r['kind']=='spurious' for r in rows)}")
    print(f"matched events: {len(matched_centre_err)}, centre error median {np.median(matched_centre_err):.3f} s, "
          f"max {max(matched_centre_err):.3f} s, >0.2 s: {sum(e>0.2 for e in matched_centre_err)}")
    print("\n| split | clip | kind | label | window (s) | gold events | max concurrency | note |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['split']} | {r['clip']} | {r['kind']} | {r['label']} | {r['window']} | {r['n_gold']} | {r['maxconc']} | {r['note']} |")
    print("\n| max concurrency in the clip | clips | imperfect | rate |")
    print("|---|---|---|---|")
    for k in sorted(by_conc):
        c, b = by_conc[k]; print(f"| {k} | {c} | {b} | {b/c:.0%} |")
    print("\n| gold events in the clip | clips | imperfect | rate |")
    print("|---|---|---|---|")
    for k in sorted(by_n):
        c, b = by_n[k]; print(f"| {k} | {c} | {b} | {b/c:.0%} |")
    print("\n| sound | gold | predicted | hits | recall | precision |")
    print("|---|---|---|---|---|---|")
    for lab in sorted(n_gold, key=lambda k: (hit[k] / n_gold[k], hit[k] / max(n_pred[k], 1))):
        print(f"| {lab} | {n_gold[lab]} | {n_pred[lab]} | {hit[lab]} | {hit[lab]/n_gold[lab]:.3f} | {hit[lab]/max(n_pred[lab],1):.3f} |")


if __name__ == "__main__":
    main()
