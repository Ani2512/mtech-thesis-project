"""A bounded hand-review sample of the DESED public-eval questions.

review.csv has one row per question (4,948). Nobody listens to all of them;
docs/real_recordings.md targets about 300 verified questions. This picks a
stratified sample: the same number per question type (ABSENT half as many),
preferring questions whose target sound is not Speech, and spreading the rows
over clips so that no clip contributes more than --per-clip rows.

    python scripts/desed_review_sample.py --review data/desed/public/review.csv \
        --out data/desed/public/review_sample.csv --n 300 --seed 0

The output keeps review.csv's columns; fill ok / corrected_answer as before and
apply it with `ctag.real_data apply-review --review review_sample.csv`.
"""
import argparse
import collections
import csv
import random

TYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ABSENT"]


def sample(rows, n, seed=0, per_clip=2, avoid="speech"):
    rng = random.Random(seed)
    per_type = {t: (n // 15) if t == "ABSENT" else (2 * n // 15) for t in TYPES}   # 7 x 2 + 1 = 15 shares
    per_type["PLAIN"] += n - sum(per_type.values())                                 # the rounding remainder
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["qtype"]].append(r)
    chosen, used = [], collections.Counter()
    for t in TYPES:
        pool = by_type.get(t, [])
        rng.shuffle(pool)
        preferred = [r for r in pool if avoid not in r["text"].lower()]
        rest = [r for r in pool if avoid in r["text"].lower()]
        picked = 0
        for r in preferred + rest:
            if picked >= per_type[t]:
                break
            if used[r["clip_id"]] >= per_clip:
                continue
            chosen.append(r)
            used[r["clip_id"]] += 1
            picked += 1
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-clip", type=int, default=2)
    a = ap.parse_args(argv)
    with open(a.review, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    chosen = sample(rows, a.n, a.seed, a.per_clip)
    chosen.sort(key=lambda r: (r["clip_id"], r["qid"]))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(chosen)
    c = collections.Counter(r["qtype"] for r in chosen)
    print(f"{len(chosen)} questions over {len({r['clip_id'] for r in chosen})} clips -> {a.out}")
    print("  " + "  ".join(f"{t} {c[t]}" for t in TYPES))
    print(f"  with 'speech' in the question: {sum(1 for r in chosen if 'speech' in r['text'].lower())}")


if __name__ == "__main__":
    main()
