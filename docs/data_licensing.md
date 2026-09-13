# Data sources and licences — 2026-09-11

| Source | Use | Licence | Notes |
|---|---|---|---|
| ESC-50 | isolated event bank for composition | CC BY-NC 3.0 | 2,000 5-s clips, 50 classes; download via GitHub release zip (~600 MB). Non-commercial only. |
| FSD50K | larger event bank (optional) | CC BY 4.0 (clip-level varies, mostly CC0/CC BY) | 51k clips, weak labels; use only classes with short single-event clips. |
| Procedural tones/noises | tests, dry runs | none | generated in `ctag/compose.py` |
| DESED real validation/eval | real-recording track | CC BY 4.0 (recordings from AudioSet/Freesound) | strong labels, 10-s domestic clips |
| TAG-Bench (2609.01542) | real-recording track, comparison to published numbers | CC BY 4.0 | 149.5 h drawn from AudioSet-strong, AudioCaps, Clotho, DESED, speech |
| AEGBench + Auto-AEG corpus (2607.04383) | real-recording track; phase 2 SFT data | CC BY 4.0, code MIT | 3,427 human-verified items |
| AudioSet-strong labels | reference only | CC BY 4.0 | audio is YouTube; do not depend on downloading it |

Decision: phase 1 runs on the **composed ESC-50 track** (exact ground truth,
`WHILE` by construction) plus the **procedural track** for pipeline tests.
Real-recording confirmation uses DESED and the TAG-Bench audio. The ESC-50
non-commercial clause is fine for a thesis; an FSD50K bank is the swap if a
CC BY release is wanted later.

| Audio Flamingo 3 (`nvidia/audio-flamingo-3-hf`) | third backbone, arms A/B | NVIDIA OneWay Noncommercial + Qwen Research License | research use only; weights not redistributed |
