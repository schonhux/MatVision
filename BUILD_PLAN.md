# MatVision — Build Plan & Acceptance Criteria

> This is the execution plan: the order we build in, exactly what goes in each layer, and the **acceptance criteria + sanity checks** that prove each layer works before we move on. Companion to `SPEC.md` (scope) and `PROJECT_GUIDE.md` (deep explanation). If they conflict on scope, `SPEC.md` wins.

**Locked decisions (2026-07-30):** email/password auth (JWT) · Claude API for report prose · Redis + Dramatiq queue · triage tier designed-for but built last · start with the tracer bullet (Layer 0).

---

## How to read this

Each layer has:
- **Goal** — one plain-English sentence.
- **Build** — the concrete artifacts/code.
- **Depends on** — what must exist first.
- **Acceptance criteria** — checkable statements. All must be true to pass the layer.
- **How we verify** — the specific test/sanity check, and **who runs it**: `[sandbox]` = I can run it here in the Linux workspace; `[Mac]` = you run it on your machine (Docker/GPU/MPS/browser).

**Rule:** we do not start layer N+1 until layer N's acceptance criteria pass. A slipping layer cuts from STRETCH first, never from acceptance criteria.

---

## Two different meanings of "layer" (important)

You described three tiers — code generation/architecture, report generation, and triage. Those are **runtime AI workloads**, not build phases. Keeping them separate from the engineering layers avoids confusion:

**A. Engineering layers (how we build)** — Layers 0–6 below. This is the actual construction order.

**B. Runtime AI model tiers (what the product calls at runtime)** — mapped to models:
| Tier | Workload | Model | When built |
|---|---|---|---|
| Heavy | System architecture + code generation | (the dev process itself — us) | ongoing |
| Report | Structured match graph → grounded coach prose | **Claude (capable)** | Layer 6 |
| Triage | Parse user text queries · auto-tag dashboard metadata | **Claude Haiku (fast/cheap)** | designed-for now, built after M6 if time |

The architecture leaves a clean seam for the Triage tier (a single `llm.triage()` route) so adding it later touches one module, not the whole app.

---

## Locked tech stack

| Area | Choice | Notes |
|---|---|---|
| Frontend | Next.js 16 (App Router), TypeScript, Tailwind | Video dashboard, timeline, annotation, report |
| Video player | Plain `<video>` + custom canvas timeline | No heavyweight annotation lib (CVAT is overkill) |
| Backend API | FastAPI, Pydantic v2, SQLAlchemy 2, Alembic | Async, migrations from day one |
| Auth | Email/password, JWT, passlib+bcrypt | Simple, zero external dependency |
| Database | PostgreSQL 16 | Metadata, events, corrections, evidence graph |
| Object storage | MinIO (S3 API) via boto3 | Swap to AWS S3 later = config change |
| Queue / worker | Redis + Dramatiq | One worker process, all pipeline stages |
| Video processing | FFmpeg, OpenCV | Transcode, frame extraction, clip cutting |
| Detection | Ultralytics YOLO (person class) | Pretrained; MPS-friendly |
| Tracking | ByteTrack (via `supervision`/`boxmot`) | + color-histogram re-ID + user click |
| Pose | **YOLOv8-pose first**, RTMPose as upgrade | YOLO-pose avoids mmcv install pain on Mac |
| ML training | PyTorch (MPS), LightGBM, scikit-learn | Small models only, trained locally |
| Data | pandas, pyarrow (parquet artifacts) | Features/tracks stored as parquet |
| Experiments | MLflow (local) | Runs, metrics, model versions |
| Report LLM | Anthropic Claude API | Only paid dependency |
| Observability | OpenTelemetry + Prometheus + Grafana (light) | Pipeline timing; add when useful |
| Containers | Docker Compose | 6 services: web, api, worker, postgres, redis, minio |
| CI | GitHub Actions | Lint + tests on push |
| Testing | pytest (api/worker), Vitest + Playwright (web, light) | Golden-clip pipeline test |

---

## Build order (map to milestones)

```
L0 Tracer Bullet ──► L1 Platform ──► L2 Annotation ──► L3 CV Pipeline
      (M0)              (M1)            (M2)               (M3)
                                                            │
L6 Evidence+Report ◄── L5 Events ◄── L4 Match-State Model ◄─┘
      (M6)              (M5)            (M4)
```
Data labeling starts during L2 and runs continuously in the background through L5.

---

## Layer 0 — Tracer Bullet  (M0)

- **Goal:** Prove off-the-shelf CV can actually track two entangled wrestlers + a referee on real footage, before we build any product.
- **Build:** One Python script/notebook in `ml/notebooks/`. Real match → FFmpeg frames → YOLO person detection → ByteTrack → YOLOv8-pose → render an overlay MP4 (boxes + IDs + skeletons) and dump `tracks.parquet` + `poses.parquet`.
- **Depends on:** one real match video from you.
- **Acceptance criteria:**
  - [ ] Runs end to end on a full real match without crashing.
  - [ ] The two wrestlers keep stable track IDs through **≥80%** of active-wrestling time.
  - [ ] Referee is separable from wrestlers (by color/position) in the overlay.
  - [ ] Identity-switch and lost-track counts are printed and logged.
  - [ ] Overlay video is visually reviewable to confirm quality.
- **How we verify:**
  - `[sandbox]` I run the pipeline on a sample clip and compute ID-hold %, switch count, lost-track duration.
  - `[Mac]` You watch the overlay MP4 and confirm it "looks right."
  - **Gate:** if ID-hold < 80%, we fix approach here (sampling rate, tracker params, re-ID) — not later.

---

## Layer 1 — Platform Foundation  (M1)

- **Goal:** A working wrestling film-review web app with no AI yet — upload, watch, manually tag, cut clips.
- **Build:**
  - `infra/docker/compose.yml` — postgres, redis, minio, api, worker, web.
  - FastAPI: email/password auth (JWT), `matches` CRUD, **presigned direct-to-MinIO upload** flow, job status endpoints. Alembic migrations for the core schema.
  - Dramatiq worker with the **TRANSCODE** stage (FFmpeg → 720p/30fps + thumbnails), written as a resumable stage with a status row.
  - Next.js: signup/login, upload UI (direct to storage), match dashboard, `<video>` player, manual event tagging on a timeline, manual clip cut.
- **Depends on:** L0 confidence.
- **Acceptance criteria:**
  - [ ] `docker compose up` brings all 6 services healthy.
  - [ ] User can sign up, log in, and see only their own matches.
  - [ ] A ≤1GB video uploads **directly to MinIO** (not through the API) via presigned URL.
  - [ ] Upload enqueues a job; TRANSCODE produces a playable 720p copy + thumbnails.
  - [ ] Job status is visible and updates as stages complete; a killed worker **resumes** from the last completed stage.
  - [ ] User can manually tag an event and cut a clip that plays.
- **How we verify:**
  - `[sandbox]` pytest: auth flow, presigned-URL issuance, job state machine, resume-after-kill (simulated). FFmpeg transcode unit test on a 10s clip.
  - `[Mac]` `docker compose up`, run through upload→play→tag→clip in the browser once.

---

## Layer 2 — Annotation System  (M2)

- **Goal:** Turn the platform into the tool that builds our labeled dataset. Labeling begins.
- **Build:**
  - Frame-accurate stepping (◄ ► by frame), event labeling (type/initiator/outcome), match-state labeling, **click-a-body athlete identification**, boundary editing.
  - Dataset export (clean JSON/parquet) + leakage-safe split metadata (by match/athlete/venue).
  - Reviewer-agreement tracking (two labels on the same clip → agreement %).
- **Depends on:** L1 (player, storage, matches).
- **Acceptance criteria:**
  - [ ] Every Level-1 field (start, end, type, initiator, outcome) is labelable and persists.
  - [ ] Match states are labelable as segments over the timeline.
  - [ ] Export produces a schema-validated dataset file with split tags.
  - [ ] **5 matches fully labeled** and exported (kicks off the dataset).
  - [ ] Boundaries are editable to frame precision (±1 frame).
- **How we verify:**
  - `[sandbox]` schema validation on exported files; unit tests on split logic (assert no clip from one match spans train+test).
  - `[Mac]` you label 5 real matches; we review agreement numbers.

---

## Layer 3 — CV Pipeline in Production  (M3)

- **Goal:** The L0 notebook becomes automatic worker stages that run on every upload and store structured motion data.
- **Build:**
  - Worker stages: **DETECT+TRACK**, **POSE**, **FEATURES** (pose → wrestling features as parquet).
  - Referee filter (uniform color/position/movement heuristics).
  - Track-quality report (continuity, lost-track duration, ID switches, re-ID confidence) stored per match.
  - Identity-confirmation UI tying auto-tracks to the user's click.
  - Coarse-to-fine sampling (~8fps coarse pass; focused high-fps around candidate windows).
- **Depends on:** L0 (proven approach), L1 (worker/storage).
- **Acceptance criteria:**
  - [ ] Detect/track/pose run as pipeline stages on upload, writing `tracks.parquet` + `poses.parquet` + `features.parquet`.
  - [ ] Auto-tracks generated on **10 matches** with quality metrics stored.
  - [ ] Low-confidence intervals (occlusion/lost track) are flagged, not guessed.
  - [ ] Feature pipeline handles missing keypoints (confidence masks, short-gap interpolation).
  - [ ] 8-min match processes in **< 15 min** on the Mac.
- **How we verify:**
  - `[sandbox]` unit tests on feature math (angles, velocities) with synthetic keypoints; assert masks present where keypoints missing.
  - `[Mac]` run 10 matches, inspect quality report + a few overlays.

---

## Layer 4 — Match-State Model  (M4)

- **Goal:** Classify every moment as neutral / top / bottom / scramble / stopped — the context everything hangs on.
- **Build:**
  - Feature windowing + **LightGBM baseline**, then a **small TCN** (PyTorch/MPS).
  - **STATES** worker stage writing `state_segments`.
  - State-timeline visualization + per-state duration stats.
  - Evaluation harness (leakage-safe splits, macro-F1, transition accuracy, boundary error).
- **Depends on:** L2 (labels), L3 (features).
- **Acceptance criteria:**
  - [ ] Match-state **macro-F1 ≥ 0.75** on held-out athletes/matches (stretch 0.85).
  - [ ] State timeline renders and is clickable to seek.
  - [ ] Model version + metrics logged to MLflow and committed to `docs/`.
  - [ ] Degrades to bbox geometry/motion when pose is unreliable (verified on scramble clips).
- **How we verify:**
  - `[sandbox]` train baseline + eval; print per-class P/R, macro-F1, confusion matrix; assert split integrity.
  - `[Mac]` (if TCN training is slow) run training; else sandbox handles it.

---

## Layer 5 — Event Detection + Correction  (M5)

- **Goal:** Auto-find shots, takedowns, defended shots, escapes, restarts; place them on a clickable timeline with clips; every correction feeds the dataset.
- **Build:**
  - **EVENTS** rules engine (thresholds over features+state+motion), **CONSOLIDATE** (merge overlapping detections, reject impossible transitions), **CLIPS** stage.
  - Correction UI: confirm / relabel / adjust boundaries / mark outcome → writes `corrections` with `use_for_training`.
  - Stretch seam: pose-sequence TCN event classifier behind the same interface as the rules engine.
- **Depends on:** L4 (states), L2 (correction schema).
- **Acceptance criteria:**
  - [ ] 5 event types detected with start/peak/end, initiator, outcome, confidence.
  - [ ] Event **F1 ≥ 0.60** (rules baseline; stretch 0.75 with TCN), **median start error < 1.5s** on held-out matches.
  - [ ] Consolidation removes duplicates and rejects impossible state transitions.
  - [ ] Each event has an auto-cut clip; timeline click seeks + plays it.
  - [ ] Corrections persist and are flagged for the next training round.
- **How we verify:**
  - `[sandbox]` eval harness: event P/R/F1, temporal IoU, start/end error, duplicate rate; unit tests on the transition-rejection FSM.
  - `[Mac]` spot-check clips + make corrections in the UI; confirm they persist.

---

## Layer 6 — Evidence Graph + Grounded Report  (M6)

- **Goal:** Connect everything into an evidence graph, find patterns, and write the coach's note where every sentence is backed by clickable film.
- **Build:**
  - **OBSERVATIONS** stage (pattern detection over events/stats, attaches evidence event IDs) — evidence graph stored relationally in Postgres.
  - **REPORT** stage: `llm.report()` over structured evidence JSON via Claude API, with the enforced contract (cite timestamps, separate observation/interpretation, state uncertainty, one priority, no invented techniques, no injury claims).
  - **Validation pass** dropping any sentence without a matching evidence ID.
  - Report UI with clickable timestamps + feedback ratings.
  - Provider-agnostic `llm/` module with `report()` and a stubbed `triage()` seam.
- **Depends on:** L5 (events + stats).
- **Acceptance criteria:**
  - [ ] Report generates from evidence JSON only (LLM never sees raw video).
  - [ ] **0 unsupported claims** across 5 test matches (manual review).
  - [ ] Every observation cites at least one real timestamp; clicking it seeks the video.
  - [ ] Validation pass provably rejects a planted unsupported sentence (unit test).
  - [ ] Report quality rated (evidence-validity ≥ 4/5 on reviewed matches).
- **How we verify:**
  - `[sandbox]` unit test the validation pass with a fabricated claim → must be dropped; golden evidence JSON → deterministic-ish structure check.
  - `[Mac]` you + ideally a coach read 5 reports and score them.

---

## Layer 7 — Longitudinal (OUT for 2026) & Triage tier (build-last)

- **L7 Longitudinal:** deferred. Sits on top of L6, aggregating reports across matches. Not built this year; documented so it's not designed out.
- **Triage tier:** after M6 if time — natural-language search over events + auto-metadata tagging via Claude Haiku behind `llm.triage()`. Acceptance (if built): query "show my failed shots in period 2" returns correct events; tagging accuracy spot-checked.

---

## M7 — Polish & Demo (definition of done)

- **Acceptance criteria:**
  - [ ] A stranger can `git clone`, `docker compose up`, upload a match, and get a report.
  - [ ] `docs/` has architecture.md, dataset-card.md, model-card.md, limitations.md, demo-script.md.
  - [ ] README with architecture diagram + the eval table (honest numbers + failure cases).
  - [ ] 3-minute demo runs clean: upload → progress stages → dashboard → click event → clip → report → click timestamp → correct one label.

---

## Testing & sanity-check strategy (cross-cutting)

1. **Golden-clip pipeline test** — a committed 30s clip runs the whole pipeline end to end in CI; asserts each stage produces its artifact. This is the single most valuable regression guard.
2. **Per-stage unit tests** — feature math, FSM transitions, split integrity, validation pass.
3. **Model regression runs** — every model version re-evaluated on the held-out set; metrics committed to `docs/`. No silent regressions.
4. **Leakage guard** — automated assertion that no clip from one match appears in both train and test.
5. **What I can run here vs. what needs your Mac** — I run all `[sandbox]` checks (pytest, Python logic, feature math, small training, eval harness) in the Linux workspace. Docker Compose, MPS/GPU inference at scale, and browser walkthroughs are `[Mac]` — I'll give you the exact commands and expected output each time.

---

## Dev documentation (local only, never pushed)

`dev/` folder (gitignored) holds the running project memory:
- `dev/DEVLOG.md` — dated log of what we built, what broke, what we tried.
- `dev/DECISIONS.md` — ADR-style decision records (context → decision → consequences).
Updated as we go. `.gitignore` excludes `dev/` so it never reaches GitHub.

---

## Immediate next step

On your go, we start **Layer 0 (Tracer Bullet)**. I'll need one real folkstyle match video (stationary camera, full mat) dropped into the connected folder. I scaffold the notebook + tracking pipeline, run it on the clip here, and report ID-hold %, switch count, and an overlay you can watch on your Mac.
