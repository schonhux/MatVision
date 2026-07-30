# MatVision — Final Spec v1.0 (Draft for approval)

AI-powered wrestling film intelligence. Demo-ready by **Dec 31, 2026**.

**Constraints this spec is built around:**
- Solo developer, ~5 months (Aug–Dec 2026)
- Compute: Apple Silicon Mac (MPS inference, small-model training only)
- Infra: local-first, $0/month (Docker Compose)
- Data: own match footage + club footage (with permission) + purpose-recorded practice film
- Recruiting target: demonstrate end-to-end ML system ownership for summer 2027 internships

---

## 1. Scope decision — IN / OUT

### IN (must ship)
| # | Capability | Layer |
|---|-----------|-------|
| 1 | Upload → background processing → interactive match dashboard | 0 |
| 2 | Video player with event timeline + auto-generated clips | 0 |
| 3 | Annotation UI: frame-accurate event tagging, athlete ID, state labels, export | 1 |
| 4 | Wrestler detection + tracking + referee filtering (pretrained models) | 2 |
| 5 | Pose extraction + wrestling feature computation | 2 |
| 6 | Match-state classification: neutral / top / bottom / scramble / stopped | 3 |
| 7 | Event detection (rules baseline): shot attempt, takedown, defended shot, escape, restart | 4 |
| 8 | Match statistics: attempts, conversion, control time, scramble duration | 5-lite |
| 9 | Evidence-grounded report: rule-derived observations + constrained LLM writeup with timestamps | 6-lite |
| 10 | Correction UI: confirm / relabel / adjust boundaries → feeds dataset | 1+ |
| 11 | Evaluation harness with leakage-safe splits + metrics report | — |

### STRETCH (only if ahead of schedule)
- Pose-sequence event classifier (TCN) replacing/augmenting rules baseline
- Successful-vs-failed shot comparison (entry distance, setup detection)
- Cloud deployment for live demo link

### OUT (explicitly deferred — do not build in 2026)
- Freestyle/Greco (folkstyle only), handheld camera support (stationary only)
- RGB video classifiers (Video ResNet/MViT/Swin), multimodal fusion — not viable on Mac, not needed for demo
- Go control plane, Kafka, Kubernetes, gRPC, Terraform, GPU worker pools
- Separate microservices (report-generator, clip-generator as services) — one worker process does all
- Separate annotation-console app — annotation lives inside the main web app
- Longitudinal multi-match trends (Layer 7), advanced events (near fall, reversal, mat return, chain attacks)
- Graph database, ML-based scene validation (simple heuristics only), scoreboard OCR, audio analysis
- Multi-tenant coach accounts, teams, sharing

**Rule: nothing moves from OUT to IN without cutting something from IN.**

---

## 2. Architecture (MVP)

```
Next.js web app (upload, player, timeline, annotation, report)
        │ HTTP
FastAPI (auth, matches, presigned uploads, job status, events, reports)
        │
PostgreSQL ── metadata, events, annotations, corrections, observations
MinIO (S3 API) ── original video, 720p analysis copy, clips, artifacts
Redis + one Python worker (Dramatiq) ── pipeline stages
```

One repo, Docker Compose, containers: `web`, `api`, `worker`, `postgres`, `redis`, `minio`.
The S3 API contract via MinIO means swapping to real AWS later is a config change.

### Pipeline stages (single worker, resumable, per-stage artifacts)
```
VALIDATE → TRANSCODE (720p/30fps H.264, thumbnails)
→ DETECT+TRACK (YOLOv8/11 person + ByteTrack, ~8 fps sampling)
→ POSE (RTMPose on wrestler crops) → FEATURES (parquet)
→ STATES (classifier) → EVENTS (rules engine) → CONSOLIDATE
→ STATS → OBSERVATIONS → REPORT (LLM) → CLIPS (FFmpeg)
```
Each stage: idempotent, writes artifact to MinIO, updates status row. Failure resumes from last completed stage.

### Input limits
Folkstyle, stationary camera, full mat visible, MP4/MOV, ≤10 min, ≤1 GB, analysis at 720p.
Unsupported footage → clear rejection message from heuristic checks (brightness, person count, resolution), not garbage output.

---

## 3. ML plan (Mac-feasible)

| Component | Approach | Training needed |
|---|---|---|
| Detection | Pretrained YOLO (person class) | None |
| Tracking | ByteTrack + color-histogram re-ID + user click-to-identify | None |
| Referee filter | Heuristics: uniform color, position, movement pattern | None |
| Pose | Pretrained RTMPose (MMPose) or YOLO-pose | None |
| Match states | Baseline: gradient-boosted trees on windowed features (LightGBM — you know it). Then small TCN in PyTorch/MPS | Small — trains on Mac in minutes/hours |
| Events | Rules engine over pose/state/motion features with tunable thresholds | None (thresholds tuned on labeled data) |
| Events (stretch) | TCN/GRU pose-sequence classifier | Small — Mac OK; fallback $30–50 of Colab/RunPod if slow |
| Report | Claude API over structured evidence JSON | None |

Known hard case: pose quality collapses during ground contact/scrambles. **States and events lean on bounding-box geometry + motion first, keypoints second.** Confidence masks everywhere; low-confidence intervals marked uncertain, never guessed.

### Data plan
- Start labeling in **September** using the Layer 1 annotation UI (dogfooding is the point)
- Target: **20–30 matches / ~2.5–4 hrs labeled** by end of Nov (states + Level-1 event labels)
- Purpose-record practice footage for clean examples: shots, sprawls, escapes, restarts
- Splits by **match and athlete**, never by clip. Test set = athletes/venues absent from training
- Every correction made in-app becomes a training example (`use_for_training` flag)

### Report grounding contract
LLM receives only structured evidence JSON (stats, events, measurements, timestamps). Output must: cite timestamps, separate observation from interpretation, state uncertainty, give exactly one practical priority, never invent techniques or diagnose injuries. Validation pass rejects any claim without a matching evidence ID.

---

## 4. Core data model (PostgreSQL)

```
users(id, email, ...)
matches(id, user_id, style, duration, status, video_keys, meta jsonb)
jobs(id, match_id, stage, status, started_at, error, artifacts jsonb)
tracks(match_id, track_id, actor_type, identity, quality jsonb)
state_segments(match_id, state, start_ms, end_ms, confidence, source: model|human)
events(id, match_id, type, start_ms, peak_ms, end_ms, initiator, outcome,
       confidence, state_before, state_after, source, measurements jsonb)
corrections(id, event_id, field, old, new, corrected_by, reason, use_for_training)
observations(id, match_id, type, summary, evidence_event_ids[], stats jsonb)
reports(id, match_id, content jsonb, model_version, ratings jsonb)
```
Evidence graph = these relations. No graph DB.

---

## 5. Repository structure

```
matvision/
├── apps/web/            # Next.js + TypeScript (player, timeline, annotation, report)
├── apps/api/            # FastAPI
├── worker/              # pipeline stages (one process)
│   ├── stages/          # validate, transcode, track, pose, features, states, events, report, clips
│   └── rules/           # event rules engine
├── ml/
│   ├── datasets/        # export, splits (leakage-safe)
│   ├── features/        # pose→wrestling features
│   ├── training/        # state classifier, (stretch) event TCN
│   ├── evaluation/      # metrics harness, regression checks
│   └── notebooks/       # tracer bullet lives here first
├── packages/schemas/    # shared pydantic/zod: events, jobs, reports
├── infra/docker/        # compose + Dockerfiles
├── tests/               # unit, pipeline (golden 30s clip end-to-end)
└── docs/                # architecture.md, dataset-card.md, model-card.md, limitations.md, demo-script.md
```

---

## 6. Milestones (Aug → Dec 2026)

| Milestone | Dates | Deliverable | Gate to proceed |
|---|---|---|---|
| **M0 Tracer bullet** | Aug W1–2 | Notebook: real match → detect → track → pose → overlay video + tracks.parquet | Tracking survives real footage; wrestlers held ≥80% of active time. If not: fix here, not later |
| **M1 Platform** (L0) | Aug W3 – Sep W2 | Compose stack, auth, upload→MinIO, transcode job, player, match dashboard, manual tagging, clip cutting | Upload→playable→manually tagged match, end to end |
| **M2 Annotation** (L1) | Sep W3–4 | Frame-stepping, event/state labeling, athlete click-ID, export; **labeling begins** | 5 matches labeled by Oct 1 |
| **M3 CV in product** (L2) | Oct W1–3 | M0 pipeline productionized as worker stages; track quality report; identity UI | Auto-tracks on 10 matches with quality metrics stored |
| **M4 States** (L3) | Oct W4 – Nov W2 | Feature pipeline + state classifier + timeline visualization + eval | Macro-F1 ≥ 0.75 on held-out matches (0.85 = stretch) |
| **M5 Events** (L4) | Nov W2–4 | Rules engine (5 event types), consolidation, auto-clips, correction UI wired | Event F1 ≥ 0.6 rules baseline; corrections persist |
| **M6 Report** (L5/6-lite) | Dec W1–2 | Stats, observations, grounded LLM report, ratings | 0 unsupported claims on 5 test matches |
| **M7 Polish + demo** | Dec W3–4 | Eval writeup, model/dataset cards, limitations doc, README, 3-min demo script, demo video | A stranger can run `docker compose up` and process a match |

Buffer: stretch items only enter after M5 gate passes. If any milestone slips >1 week, cut from stretch first, then reduce event types to 3.

---

## 7. Evaluation gates (honest numbers > big numbers)

| Metric | Target | Measured on |
|---|---|---|
| Tracking: identity hold during active wrestling | ≥80% | 10 held-out matches |
| Match-state macro-F1 | ≥0.75 (stretch 0.85) | held-out athletes/matches |
| Event F1 (5 classes, rules) | ≥0.60 (stretch 0.75 w/ TCN) | held-out matches |
| Median event-start error | <1.5 s | held-out matches |
| Unsupported observation rate | <5% | manual review, 5 matches |
| 8-min match processing time (M2 Mac) | <15 min | pipeline timing |
| Upload→report success rate | ≥95% | all processed matches |

Every model version gets a regression run; results committed to `docs/`. Documented failure cases (scrambles, occlusion) are a feature of the writeup, not an embarrassment.

---

## 8. Demo definition (what "done" means)

3-minute recruiting demo: upload real match → progress stages visible → dashboard with state timeline → click event → auto-clip plays → open report → observation with timestamps → click timestamp, video seeks to the moment → correct one event label. Backed by README with architecture diagram, eval table, and limitations doc.

Resume bullets this produces (targets, wording finalized from real numbers):
- End-to-end CV pipeline (detection, tracking, pose, temporal classification) processing real competition footage
- Self-built annotation platform producing a leakage-controlled dataset of N matches
- Evidence-grounded LLM reporting with <5% unsupported-claim rate and human eval
- Async video pipeline: resumable stages, per-stage artifacts, observability

---

## 9. Open questions to finalize before build

1. **Auth**: simple email/password (fastest) vs Clerk/NextAuth OAuth? → recommend simple, it's a demo.
2. **LLM for reports**: Claude API (small monthly cost) — acceptable? Only paid dependency.
3. **Dramatiq vs Celery**: recommend Dramatiq (simpler). Any preference?
4. **Video annotation player**: build on `vidstack`/plain `<video>` + custom timeline (recommended) vs existing annotation lib (CVAT is overkill).
5. **Practice-footage recording session**: can you schedule one in August? Clean drilling clips accelerate M4–M5 significantly.
