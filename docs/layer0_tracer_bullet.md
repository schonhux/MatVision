# Layer 0 — Tracer Bullet: Results

> Fill in the "Real match run" section once `l0_tracer_bullet.py` has been run on real
> footage on the Mac. This file is the permanent record of whether Layer 0's acceptance
> gate passed, referenced from `BUILD_PLAN.md`.

## Acceptance criteria (from BUILD_PLAN.md)

- [x] Pipeline mechanics validated (synthetic smoke test) — see below.
- [ ] Runs end to end on a full real match without crashing.
- [ ] Two wrestlers keep stable track IDs through **>=80%** of active-wrestling time.
- [ ] Referee is separable from wrestlers in the overlay.
- [ ] Identity-switch and lost-track counts printed/logged.
- [ ] Overlay video reviewed and confirmed legible.

## What was built

- `ml/features/tracking_metrics.py` — pure-logic tracking-quality metrics (ID-hold %,
  identity switches, lost-track duration, referee heuristic classification, re-ID
  stitching for brief occlusion recovery). No torch dependency; fully unit-tested.
- `tests/pipeline/test_tracking_metrics.py` — 15 unit tests, all passing.
- `ml/notebooks/l0_tracer_bullet.py` — the real pipeline: YOLO person detection ->
  ByteTrack -> YOLOv8-pose -> overlay video + tracks/poses parquet + quality_report.json.
  Requires torch/ultralytics + a real video; run on the Mac.
- `ml/notebooks/l0_synthetic_smoketest.py` — OpenCV-only synthetic video (colored blobs
  simulating two wrestlers + a referee, with a simulated occlusion/scramble) run through
  the same ByteTrack + identity-resolution + re-ID logic, with **no torch required**.
  Validates pipeline plumbing (video I/O, tracker integration, identity seeding,
  re-ID recovery, overlay rendering, parquet export) end to end.

## Sandbox validation (mechanics only — not the real gate)

Environment: Linux CPU sandbox, no GPU, `torch`/`ultralytics` not installed (network
policy blocks `download.pytorch.org`; heavy installs also exceed the sandbox's 45s
per-command cap). `supervision` (ByteTrack) installs cleanly with no torch dependency,
which let us validate all tracking/metrics logic for real.

```
pytest tests/pipeline/test_tracking_metrics.py -v
15 passed in 0.82s

python3 ml/notebooks/l0_synthetic_smoketest.py
Overall ID-hold:  100.0%
  wrestler_a: hold=100.0% switches=0 lost_frames=0
  wrestler_b: hold=100.0% switches=0 lost_frames=0
PLUMBING OK — pipeline mechanics work end to end.
```

This confirms the tracker integration, identity seeding, re-ID stitching, and metrics
math are all correct. It does **not** confirm real-world tracking quality — that depends
entirely on how well YOLO detects entangled wrestlers on real footage, which this
synthetic test cannot simulate (color-blob detection is trivial compared to real person
detection under occlusion).

## Real match run (Mac) — PENDING

Run:
```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python notebooks/l0_tracer_bullet.py \
  --video /path/to/real_match.mp4 \
  --out notebooks/output \
  --seed-frame <frame where both wrestlers are clearly separated> \
  --wrestler-a-seed x1,y1,x2,y2 \
  --wrestler-b-seed x1,y1,x2,y2
```

Fill in once run:

| Metric | Value | Gate |
|---|---|---|
| Overall ID-hold | — | >= 80% |
| Wrestler A hold / switches / lost | — | — |
| Wrestler B hold / switches / lost | — | — |
| Video | — | — |
| Duration | — | — |
| Verdict | — | PASS / FAIL |

**If it fails the gate:** do not proceed to Layer 1. Options to try, in order: increase
detection confidence threshold tuning, adjust ByteTrack params (track buffer / match
threshold), tighten re-ID gap/distance params, try a larger YOLO variant (yolov8s
instead of yolov8n), or reconsider camera angle/distance requirements. Document what
was tried and the outcome here before any decision to change approach.
