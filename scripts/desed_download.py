"""Fetch the DESED real clips whose audio is not on Zenodo (validation, weak) from YouTube.

DESED names each clip <Y><video id>_<start>_<end>.wav and ships only the labels;
the audio is a 10 s section of a Creative Commons YouTube video, fetched with
yt-dlp and cut with ffmpeg. Some videos are gone or region-locked; those are
recorded in <out>/missing.tsv and the importer (`ctag.real_data import
--require-audio`) skips them, so the usable set is whatever survives.

    python scripts/desed_download.py --tsv data/desed/metadata/validation.tsv \
           --out data/desed/audio/validation --workers 4

Idempotent: a clip whose wav exists is skipped, so rerun after an interruption.
Output: 44.1 kHz stereo wav as YouTube serves it; the project's loaders
resample. Needs `yt-dlp` (pip) and `ffmpeg` on PATH.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STEM = re.compile(r"^Y(?P<vid>.{11})_(?P<start>\d+\.\d+)_(?P<end>\d+\.\d+)$")


def clips(tsv: Path) -> list[tuple[str, str, float, float]]:
    seen = {}
    with open(tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stem = row["filename"][:-4] if row["filename"].endswith(".wav") else row["filename"]
            m = STEM.match(stem)
            if not m:
                raise ValueError(f"unexpected DESED file name: {row['filename']}")
            seen[stem] = (stem, m["vid"], float(m["start"]), float(m["end"]))
    return sorted(seen.values())


def fetch(stem: str, vid: str, start: float, end: float, out: Path, ytdlp: str,
          cookies_from_browser: str | None = None) -> tuple[str, str]:
    dest = out / f"{stem}.wav"
    if dest.exists() and dest.stat().st_size > 1000:
        return stem, "exists"
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ytdlp, "--quiet", "--no-warnings", "--no-playlist", "-f", "bestaudio/best",
               "--sleep-requests", "1", "--retries", "3", "--extractor-retries", "3",
               "--download-sections", f"*{start:.3f}-{end:.3f}", "--force-keyframes-at-cuts",
               "-x", "--audio-format", "wav", "--audio-quality", "0",
               "-o", f"{tmp}/clip.%(ext)s"] + (["--cookies-from-browser", cookies_from_browser] if cookies_from_browser else []) \
              + [f"https://www.youtube.com/watch?v={vid}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            # one stalled video must not take the whole run down (it did, at clip 274)
            return stem, "failed: timeout after 240 s"
        got = list(Path(tmp).glob("clip.wav"))
        if r.returncode != 0 or not got:
            err = (r.stderr.strip().splitlines() or ["no output"])[-1][:160]
            return stem, f"failed: {err}"
        # pad or trim to exactly end-start seconds so the labels line up with the file
        want = end - start
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(got[0]), "-t", f"{want:.3f}",
                            "-af", f"apad=whole_dur={want:.3f}", str(dest)], check=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            dest.unlink(missing_ok=True)
            return stem, f"failed: ffmpeg {type(e).__name__}"
    return stem, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N clips (a smoke test)")
    ap.add_argument("--ytdlp", default=shutil.which("yt-dlp") or "yt-dlp")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="e.g. chrome: use the browser's YouTube login when YouTube demands a sign-in (your account)")
    ap.add_argument("--retry-missing", action="store_true",
                    help="only the clips listed in <out>/missing.tsv whose reason was a sign-in or bot check")
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH (brew install ffmpeg)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = clips(Path(a.tsv))
    if a.retry_missing:
        miss = out / "missing.tsv"
        want = {l.split("\t")[0][:-4] for l in miss.read_text(encoding="utf-8").splitlines()[1:]
                if any(k in l.lower() for k in ("sign in", "not a bot", "timeout", "ffmpeg"))} if miss.exists() else set()
        todo = [c for c in todo if c[0] in want]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} clips -> {out}", flush=True)
    counts, missing, t0 = {"ok": 0, "exists": 0, "failed": 0}, [], time.time()
    (out / "missing.tsv").write_text("filename\treason\n", encoding="utf-8")
    with ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(fetch, *c, out, a.ytdlp, a.cookies_from_browser) for c in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            stem, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                missing.append((stem, status[8:]))
                with open(out / "missing.tsv", "a", encoding="utf-8") as f:
                    f.write(f"{stem}.wav\t{status[8:]}\n")
            if k % 25 == 0 or k == len(todo):
                print(f"{k}/{len(todo)}  ok {counts['ok']}  exists {counts['exists']}  failed {counts['failed']}  "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"done: {counts}; {len(missing)} missing listed in {out / 'missing.tsv'}")
    return 0 if counts["ok"] + counts["exists"] else 1


if __name__ == "__main__":
    sys.exit(main())
