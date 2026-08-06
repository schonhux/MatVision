# Layer 3 — CV Pipeline in Production: Results

Turns the Layer 0 tracer bullet into automatic worker stages that run on every
upload and produce structured motion data.

## Acceptance criteria (BUILD_PLAN.md M3)

- [x] detect/track/pose run as pipeline stages, writing tracks/poses/features parquet
- [x] Referee filtering (heuristics over per-track aggregate stats)
- [x] Track-quality report stored per match (presence ratio, switches, lost seconds)
- [x] Low-confidence intervals flagged, not guessed
- [x] Feature pipeline handles missing keypoints (masks + bounded interpolation)
- [ ] **Auto-tracks on 10 matches** — needs real footage
- [ ] **8-min match processes in < 15 min on the Mac** — needs a real timing run

## What was built

**`ml/features/wrestling_features.py`** — pose keypoints to wrestling-meaningful
numbers: hip height (the level-change signal), torso angle, stance width, knee bend,
center of mass, athlete distance, relative hip height, head position, plus temporal
features (velocity, closing speed, level-change rate).

**`ml/features/identity.py`** — binds tracker IDs to wrestler identities using the
Layer 2 click-to-identify seed box, with referee heuristics and longest-lived-track
fallback when no seed exists.

**`worker/app/vision/models.py`** — all torch/ultralytics contact isolated in one
module, imported lazily. Swapping YOLOv8-pose for RTMPose later (ADR-006) touches
only this file.

**Three new pipeline stages**, appended to `PIPELINE_STAGES`:
- `detect_track` — YOLO + ByteTrack at ~8fps coarse sampling, identity resolution,
  `tracks.parquet` + quality report
- `pose` — YOLOv8-pose on tracked wrestler boxes, IoU-matched back to identities,
  `poses.parquet`
- `features` — `features.parquet` with the full feature table

## Design decisions worth noting

**Missing data is never fabricated.** Every feature returns `None` — not `0.0` — when
the keypoints it needs aren't visible, and each frame carries `*_detected` flags and a
`*_visibility` fraction. Returning 0.0 for an occluded hip would teach a model that
occlusion means "hips on the floor," which is exactly backwards during a scramble
(the case that matters most). This is the single most important correctness property
in the layer.

**Interpolation is bounded to 3 frames (~0.4s at 8fps).** Brief dropouts are detection
misses and safe to bridge; long gaps are genuine occlusion, and filling them would
invent motion that never happened.

**Coarse sampling at 8fps, not 30.** Enough to follow position and identity at a
fraction of the compute. Layer 5 can re-run densely around candidate events if finer
temporal resolution proves necessary.

**Seed binding beats heuristics, always.** If the user clicked to identify themselves,
no heuristic overrides that. A wrong binding silently mislabels every feature in the
entire match.

**Heavy imports stay lazy, and a test enforces it.** `tests/worker/test_stage_imports.py`
asserts that importing any stage module does not pull in torch/ultralytics. Without
this guard, someone adds a top-level `from ultralytics import YOLO`, it passes on
their machine, and CI breaks for everyone else with a confusing ImportError.

## Testing

164 tests passing (65 new in Layer 3).

| Suite | Count | Covers |
|---|---|---|
| `test_wrestling_features.py` | 35 | Geometry correctness, occlusion → None behavior, low-confidence handling, temporal features, bounded interpolation, and an end-to-end "shot signature" check (hip drop + closing distance) |
| `test_identity.py` | 20 | IoU, seed binding (incl. refusing a bad click), referee heuristics, wrestler selection, track stats |
| `test_stage_imports.py` | 10 | The no-torch-at-import guard, stage/run() wiring, pipeline ordering |

**Test fixed during this layer:** `test_complete_upload_creates_jobs_and_enqueues`
hardcoded `["validate", "transcode"]` and broke when the pipeline grew. Rewritten to
assert against `PIPELINE_STAGES` — the test's job is "one job row per stage, in
order," not re-encoding the pipeline definition.

## Not yet verified (needs real footage / a Mac run)

The stages are correct in structure and their pure logic is thoroughly tested, but
detection and pose quality on real wrestling footage remains **unmeasured** — that's
still the Layer 0 gate, and it's the same open question it has been since day one.
The track-quality report exists specifically to answer it: run a real match through
the pipeline and read `quality.overall_presence_ratio` against the 0.80 gate.
