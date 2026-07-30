# MatVision — Complete Project Guide

> **Purpose of this document.** This is the single source of truth for MatVision. It is written so that any engineer or AI agent joining the project can read it cold and understand *what we are building, why we are building it this way, how the whole system fits together, and exactly what belongs in each layer.* Every section has a plain-English explanation followed by the technical detail. If you are an agent picking up a task, read Sections 1–4 for context, then jump to the layer you own in Section 8. Do not deviate from the scope rules in Section 5 without an explicit human decision.

**Status:** Planning complete, pre-build. Target demo-ready: end of 2026.
**Owner:** Schon Huxley (solo dev + domain expert — competitive wrestler).
**Companion docs:** `SPEC.md` (the terse, decision-focused spec). This guide is the expanded narrative version; the two must stay consistent — if they conflict, `SPEC.md` wins on scope decisions and this guide wins on explanation.

---

## 1. What MatVision is (plain English)

MatVision turns a wrestling match video into a coach-style breakdown backed by evidence you can click and watch.

A wrestler uploads footage of a match, tells the system which wrestler they are, and gets back:

- An interactive timeline of what happened and when.
- Automatically cut video clips of each key moment.
- Offensive and defensive statistics (how many shots, how many finished, control time, etc.).
- Timestamped strengths and improvement areas.
- A short written report that reads like a coach's note — but every sentence points back to a specific moment in the video.

The emotional promise to the user: *"Every claim I make about your wrestling, I can show you on film."*

## 2. The one idea that makes this project real (read this twice)

There is a lazy version of this product and a real version. The difference is the entire point of MatVision.

**Lazy version:** "Upload a match, let a language model watch the video and describe it." This does not work reliably, cannot cite evidence, hallucinates events, and demonstrates nothing technically impressive.

**Real version — the architecture we are building:**

```
Video
  → Video preprocessing
  → Wrestler detection and tracking
  → Pose and motion extraction
  → Match-state recognition
  → Temporal event detection
  → Technique feature calculation
  → Structured evidence graph
  → Grounded coaching report
```

**The rule that follows from this:** *The computer-vision system decides what happened. The language model only explains findings that are already structured and proven.* The LLM never watches raw video and never invents events. It receives a JSON object of measured facts and writes prose around them. If a claim has no evidence ID behind it, it does not ship.

Everyone building any layer must protect this separation. It is what makes the project credible to ML teams and what makes the coaching trustworthy to athletes.

## 3. Who this is for and why it exists

- **Primary user:** a wrestler (initially Schon and teammates) who wants objective film feedback.
- **Secondary user:** a coach who reviews and corrects the system's output.
- **Project's real purpose:** a portfolio-grade demonstration of end-to-end ML system ownership — dataset creation, computer vision, temporal modeling, evaluation methodology, and grounded LLM generation — for internship recruiting. The wrestling domain is a genuine advantage: the builder has footage access, labeling ability, and expert judgment most people attempting this lack.

## 4. Guiding principles (apply to every layer)

1. **Evidence or it didn't happen.** Every statistic and every sentence of feedback traces to a timestamp and an event record.
2. **CV decides, LLM explains.** Never blur this line.
3. **Uncertainty is a first-class citizen.** Occlusion, lost tracks, and low-confidence intervals are marked as uncertain, never guessed. Saying "unsupported footage" is a correct output.
4. **Build the narrow thing well.** Folkstyle, stationary camera, neutral-position offense first. Depth over breadth.
5. **The product is the dataset tool.** The annotation UI is not a side script — it is how we build training data. Dogfooding is the strategy.
6. **Honest metrics beat impressive metrics.** A real number with a documented failure-case analysis is worth more than an inflated one.
7. **Every layer is independently demoable.** Even Layers 0–2 alone are a shippable "wrestling film platform with automated tracking."
8. **Idempotent, resumable, observable.** Every pipeline stage can rerun safely, resume after failure, and reports its own timing.

## 5. Scope — what is IN, STRETCH, and OUT

This is the guardrail. **Nothing moves from OUT to IN without cutting something from IN.**

### IN (must ship for the demo)
- Upload → background processing → interactive dashboard.
- Video player with event timeline + auto-generated clips.
- Annotation UI (frame-accurate event tagging, athlete identification, state labels, export).
- Wrestler detection + tracking + referee filtering (pretrained models).
- Pose extraction + wrestling-specific feature computation.
- Match-state classification: neutral / top / bottom / scramble / stopped.
- Event detection via rules baseline: shot attempt, takedown, defended shot, escape, restart.
- Match statistics (attempts, conversion, control time, scramble duration).
- Evidence-grounded report (rule-derived observations + constrained LLM writeup with timestamps).
- Correction UI (confirm / relabel / adjust boundaries) that feeds the dataset.
- Evaluation harness with leakage-safe splits.

### STRETCH (only after the M5 gate passes)
- Pose-sequence event classifier (TCN/GRU) augmenting the rules baseline.
- Successful-vs-failed shot comparison (entry distance, setup detection).
- Cloud deployment for a live demo link.

### OUT (explicitly deferred — do not build)
- Freestyle / Greco-Roman (folkstyle only). Handheld camera (stationary only).
- RGB video classifiers (Video ResNet, MViT, Swin) and multimodal fusion — not Mac-feasible, not needed for the demo.
- Go control plane, Kafka, Kubernetes, gRPC, Terraform, GPU worker pools.
- Separate microservices (report-generator, clip-generator as their own services) — one worker process does everything.
- Separate annotation-console app — annotation lives inside the main web app.
- Longitudinal multi-match trends and advanced events (near fall, reversal, mat return, chain attacks).
- Graph database, ML-based scene validation, scoreboard OCR, audio analysis.
- Multi-tenant coach accounts, teams, sharing.

## 6. Constraints that shaped every decision

- **Solo developer**, roughly Aug–Dec 2026.
- **Compute: Apple Silicon Mac only.** This is why all heavy models are *pretrained* (inference on MPS) and only *small* models (LightGBM, small TCN) are trained locally. Heavier training, if ever needed, is a small paid Colab/RunPod fallback.
- **Local-first infrastructure, $0/month.** Docker Compose with Postgres, Redis, and MinIO (S3-compatible). Because MinIO speaks the S3 API, moving to real AWS later is a config change, not a rewrite.
- **Data: own match footage + club footage (with permission) + purpose-recorded practice film.** There is no public wrestling event dataset — we build our own.

## 7. System architecture (the big picture)

```
┌─────────────────────────────────────────────────────────────┐
│                       Next.js web app                        │
│   Upload · Player · Timeline · Annotation · Report · Fixes   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────┐
│                          FastAPI                             │
│   Auth · Matches · Presigned uploads · Job status · Events   │
│   · Reports · Corrections                                    │
└───────────┬───────────────────────────────┬─────────────────┘
            │                               │
     ┌──────▼───────┐              ┌────────▼─────────┐
     │  PostgreSQL  │              │  MinIO (S3 API)  │
     │  metadata,   │              │  original video, │
     │  events,     │              │  720p copy,      │
     │  annotations,│              │  clips, artifacts│
     │  corrections │              └────────┬─────────┘
     └──────────────┘                       │
            ┌──────────────────────┐        │
            │  Redis job queue     │◄───────┘
            │  (Dramatiq)          │
            └──────────┬───────────┘
                       │
        ┌──────────────▼───────────────────────────┐
        │         Single Python worker              │
        │  runs all pipeline stages in sequence:    │
        │  validate → transcode → detect+track →    │
        │  pose → features → states → events →      │
        │  consolidate → stats → observations →     │
        │  report → clips                           │
        └───────────────────────────────────────────┘
```

**Why this shape.** One repo, one worker process, six containers. It is the smallest thing that is still a real distributed-ish system: async jobs, object storage, a queue, resumable stages. It intentionally omits the "mature" architecture (Go, Kafka, K8s, gRPC) because that is complexity for a scale we do not have. The mature diagram exists in `SPEC.md` as an aspiration, not a build target.

### The processing pipeline (stage by stage, plain English)

Each stage reads the previous stage's artifact from MinIO, does one job, writes its own artifact, and updates a status row. If the worker crashes, it resumes from the last completed stage.

1. **VALIDATE** — Is this usable footage? Heuristic checks for brightness, resolution, and whether at least two people are visible. Bad footage gets a clear rejection message, not garbage analysis.
2. **TRANSCODE** — FFmpeg standardizes the video to 720p, 30fps, H.264, fixes rotation, and generates thumbnails. Produces the predictable "analysis copy" everything downstream reads.
3. **DETECT + TRACK** — Pretrained YOLO finds people frame by frame (sampled ~8 fps for a first coarse pass); ByteTrack stitches those detections into continuous identities over time. Color histograms plus the user's click-to-identify pin down which track is "me," "opponent," and "referee."
4. **POSE** — Pretrained RTMPose extracts skeleton keypoints (head, shoulders, elbows, wrists, hips, knees, ankles) on the wrestler crops.
5. **FEATURES** — Raw keypoints become *wrestling-meaningful numbers*: torso angle, hip height, stance width, closing speed, distance between athletes, relative hip height, contact duration. Stored as parquet. Missing keypoints (guaranteed during body contact) carry confidence masks.
6. **STATES** — A temporal classifier labels every moment as neutral / top / bottom / scramble / stopped. This is the context everything else hangs on.
7. **EVENTS** — A rules engine reads features + state + motion to detect shot attempts, takedowns, defended shots, escapes, and restarts, each with a start/peak/end time, an initiator, and an outcome.
8. **CONSOLIDATE** — Because analysis windows overlap, the same event can be detected twice. This stage merges duplicates, aggregates confidence, and rejects impossible state transitions.
9. **STATS** — Aggregates events into match statistics.
10. **OBSERVATIONS** — Rules find patterns worth mentioning (e.g., "open-distance shots converted 0/3, contact-setup shots 2/3") and attach the supporting event IDs.
11. **REPORT** — The LLM receives the structured observations + evidence JSON and writes the coach's note. A validation pass rejects any sentence lacking a matching evidence ID.
12. **CLIPS** — FFmpeg cuts a short video for each event so the timeline is clickable.

## 8. The layered build — what goes in each layer

This is the heart of the document. The project is built in layers so that each one delivers a working, demoable product and de-risks the next. Layers map to milestones M0–M7 in `SPEC.md`. Each layer below lists: **plain-English goal**, **what to build**, **what it depends on**, **the gate that says it's done**, and **what NOT to do here.**

### Layer 0 — Tracer bullet (de-risk before anything else)

- **Plain English:** Before building any product, prove that off-the-shelf models can even see wrestling. Two entangled bodies plus a referee is close to the worst case for detection and tracking. We need to know in week one, not month three.
- **Build:** A single offline notebook. One real match video → FFmpeg frames → pretrained YOLO detection → ByteTrack tracking → pretrained pose → render an overlay video and dump `tracks.parquet`. No product, no training, no UI.
- **Depends on:** nothing.
- **Done when:** tracking holds wrestler identities through at least ~80% of active wrestling time on a real match. If it doesn't, we fix approach here (different detector, tighter sampling, re-ID tweaks) before investing in product plumbing.
- **Do NOT:** build upload flows, databases, or any web code yet. Resist it.

### Layer 1 — Product foundation (the platform)

- **Plain English:** A working wrestling film-review website with no AI yet. Upload a match, watch it, tag moments by hand, cut clips. This alone is already useful.
- **Build:** Docker Compose stack (Postgres, Redis, MinIO). FastAPI with auth, match records, presigned direct-to-storage uploads, and job status. Next.js frontend with upload UI, video player, a match dashboard, manual event tagging, and clip generation. The transcode stage as the first real background job.
- **Depends on:** Layer 0's confidence that CV is viable.
- **Done when:** a user can upload a match, watch it play, manually tag an event, and get a clip cut — fully end to end.
- **Do NOT:** add any ML. Do not build a separate annotation app.

**Upload flow detail (important — do not route big files through the API):**
```
1. Frontend requests an upload session.
2. API validates metadata, creates the match record.
3. API returns a presigned upload URL.
4. Browser uploads the video directly to MinIO/S3.
5. Browser notifies the API that upload finished.
6. API enqueues the processing job.
```

**Storage layout:**
```
matvision/users/{user_id}/matches/{match_id}/
  original/video.mov
  processed/analysis-720p.mp4
  thumbnails/
  clips/event-001.mp4 ...
  artifacts/poses.parquet, tracks.parquet, predictions.json, report.json
```

### Layer 2 — Annotation system (the dataset engine)

- **Plain English:** Turn the platform into the tool that builds our training data. This is the strategic core — without labeled data, no model layer can exist.
- **Build:** Frame-accurate stepping, event and state labeling, athlete identification by clicking a body in an early frame, boundary editing, dataset export, and reviewer-agreement tracking. Labeling of real matches begins here.
- **Depends on:** Layer 1 (player, storage, match records).
- **Done when:** 5 matches are fully labeled and exportable in a clean schema; labeling begins in September.
- **Do NOT:** reach for heavyweight tools like CVAT — a custom timeline on a plain `<video>` element is simpler and tailored to wrestling.

**Annotation levels (label depth increases as we go):**
- *Level 1:* start time, end time, event type, initiator, outcome.
- *Level 2:* state before, state after, opponent response, technique family.
- *Level 3:* setup type, entry quality, finish direction, head position, secondary attack, quality tags.

### Layer 3 — Detection, tracking, pose in production (CV in the pipeline)

- **Plain English:** Take the Layer 0 notebook and make it a real, automatic pipeline stage that runs on every uploaded match and stores structured motion data.
- **Build:** Productionize detect+track+pose as worker stages. Referee filtering (uniform color, position, movement heuristics). Track-quality reporting (continuity, lost-track duration, identity switches, re-ID confidence). Identity-confirmation UI that ties back to the user's click.
- **Depends on:** Layers 0 (proven approach) and 1 (worker + storage).
- **Done when:** auto-tracks exist for 10 matches with quality metrics stored, and low-confidence intervals are flagged.
- **Do NOT:** train anything yet — all models here are pretrained.

**Feature outputs (per athlete and relational):**
- *Individual:* torso angle, hip height, knee bend, stance width, center-of-mass estimate, horizontal/vertical velocity, body orientation.
- *Relational:* athlete distance, relative hip height, head position, shoulder alignment, closing speed, relative movement direction, contact duration, position around opponent.
- *Missing-data handling:* confidence masks, short-gap interpolation, explicit missing indicators. Never assume every joint is visible.

### Layer 4 — Match-state model (the context layer)

- **Plain English:** Teach the system to know whether wrestlers are on their feet, someone's on top, in a scramble, or the action is stopped. Every later judgment depends on knowing the position.
- **Build:** The feature pipeline feeding a state classifier. Baseline = gradient-boosted trees (LightGBM) on windowed features; then a small temporal convolutional network in PyTorch/MPS. A state timeline visualization and per-state duration stats. Full evaluation.
- **Depends on:** Layers 2 (labels) and 3 (features).
- **Done when:** match-state macro-F1 ≥ 0.75 on held-out *matches and athletes* (0.85 is the stretch target).
- **Do NOT:** rely on clean skeletons — during ground contact, lean on bounding-box geometry and motion more than keypoints.

**Why a TCN first (not a video transformer):** a temporal convolutional network classifies sequences well without the compute and data appetite of large video models — the right complexity for a Mac and a small dataset.

### Layer 5 — Event detection (the timeline)

- **Plain English:** Automatically find the moments that matter — shots, takedowns, sprawls, escapes, restarts — and place them on a clickable timeline with clips.
- **Build:** A rules engine over pose/state/motion thresholds (e.g., *rapid hip-height drop + closing distance + forward motion + neutral state = possible shot attempt*). Consolidation to merge overlapping detections and reject impossible transitions. Auto-clip generation. The correction UI wired so every fix becomes a training example.
- **Depends on:** Layer 4 (states) and Layer 2 (correction schema).
- **Done when:** event F1 ≥ 0.60 with the rules baseline and median start-time error under ~1.5s; corrections persist and are flagged for training.
- **Do NOT:** attempt advanced events (near fall, reversal, chain attacks) or the ML event classifier yet — those are stretch/out.

**Event record shape:**
```json
{
  "event_id": "evt_018", "type": "shot_attempt",
  "start_time": 142.3, "peak_time": 144.1, "end_time": 146.8,
  "initiator": "user", "defender": "opponent",
  "state_before": "neutral", "state_after": "scramble",
  "outcome": "failed", "confidence": 0.88
}
```

### Layer 6 — Evidence graph + grounded report (coaching intelligence)

- **Plain English:** Connect everything into a web of evidence, find patterns, and write the coach's note where every sentence is backed by clickable film.
- **Build:** The Match Evidence Graph (stored relationally in Postgres — no graph DB). Pattern detection over events and stats. The constrained LLM report. Feedback ratings on the report.
- **Depends on:** Layer 5 (events + stats).
- **Done when:** zero unsupported claims across 5 test matches, verified by manual review.
- **Do NOT:** let the LLM see raw video or invent techniques. It receives structured evidence only.

**The Match Evidence Graph** links: match → periods → athletes → events → match-states → measurements → outcomes → observations → clips → corrections → training priorities. Example: the observation "open-distance attacks were less successful" points to shots at 01:42 (no setup, failed), 03:18 (no setup, countered), 04:57 (wrist control, successful), plus the 0/3 vs 2/3 comparison.

**LLM contract (enforced):** cite timestamps; separate observation from interpretation; state uncertainty; give exactly one practical priority; never diagnose injuries; never claim to replace a coach; never invent techniques absent from the evidence. A validation pass drops any sentence without a matching evidence ID.

- *Strong output:* "Your attacks were more effective after establishing hand contact — two of three attempts after wrist control or an underhook finished, while all three open-distance attempts were stopped. Review 1:42, 3:18, 4:57. A useful film priority is identifying what let you close distance before the successful entries."
- *Weak output (banned):* "You need to be more aggressive and improve your shots."

### Layer 7 — Longitudinal analysis (OUT for 2026, documented for future agents)

- **Plain English:** Trends across many matches — is shot conversion improving, which situations recur, how do opponent styles compare.
- **Status:** deferred. Not built in 2026. Listed so future agents know where it fits: it sits on top of Layer 6, aggregating reports across matches into an athlete profile.

## 9. Data model (PostgreSQL)

The evidence graph is just these relations — no graph database.

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

## 10. ML approach summary (what's pretrained vs trained)

| Component | Approach | Training on Mac? |
|---|---|---|
| Detection | Pretrained YOLO (person) | None |
| Tracking | ByteTrack + color re-ID + user click | None |
| Referee filter | Heuristics | None |
| Pose | Pretrained RTMPose / YOLO-pose | None |
| Match states | LightGBM baseline → small TCN | Yes, small/fast |
| Events | Rules engine (tuned thresholds) | None |
| Events (stretch) | TCN/GRU pose-sequence classifier | Yes; Colab fallback if slow |
| Report | Claude API over evidence JSON | None |

**The consistently hard case:** pose quality collapses during ground contact and scrambles. Every layer that consumes pose must degrade gracefully to bounding-box geometry and motion, and mark uncertain intervals rather than fabricate.

## 11. Data strategy (the actual hardest part)

The web app is the easy part. Getting labeled data is the bottleneck.

- **Sources:** own footage, teammate footage (with permission), purpose-recorded practice/drilling, club footage with authorization. Public footage only where rights allow.
- **Purpose-recorded practice is gold:** clean, repeated examples of long-distance shots, setup shots, finishes (successful and failed), sprawls, escapes, restarts, and mat returns. Controlled sequences teach basic patterns before chaotic competition footage.
- **Leakage-safe splitting is non-negotiable.** Never split clips from the same match across train and test — same athletes, uniforms, mat, camera, and lighting cause leakage. Split by athlete, match, venue, competition, and camera setup. The strongest test set contains athletes and locations absent from training.
- **The correction flywheel:** model predicts → user/coach corrects → correction enters the reviewed dataset → new model trains → regression eval runs → improved model deploys.

## 12. Evaluation (honest numbers)

| Metric | Target | Measured on |
|---|---|---|
| Tracking identity hold (active wrestling) | ≥80% | 10 held-out matches |
| Match-state macro-F1 | ≥0.75 (stretch 0.85) | held-out athletes/matches |
| Event F1 (5 classes, rules) | ≥0.60 (stretch 0.75 w/ TCN) | held-out matches |
| Median event-start error | <1.5s | held-out matches |
| Unsupported observation rate | <5% | manual review, 5 matches |
| 8-min match processing (Mac) | <15 min | pipeline timing |
| Upload→report success rate | ≥95% | all processed matches |

Every model version gets a regression run committed to `docs/`. Documented failure cases (scrambles, occlusion) are part of the writeup, not hidden.

## 13. Repository structure

```
matvision/
├── apps/web/            # Next.js + TypeScript (player, timeline, annotation, report)
├── apps/api/            # FastAPI
├── worker/              # pipeline stages (one process)
│   ├── stages/          # validate, transcode, track, pose, features, states, events, report, clips
│   └── rules/           # event rules engine
├── ml/
│   ├── datasets/        # export, leakage-safe splits
│   ├── features/        # pose → wrestling features
│   ├── training/        # state classifier, (stretch) event TCN
│   ├── evaluation/      # metrics harness, regression checks
│   └── notebooks/       # tracer bullet lives here first
├── packages/schemas/    # shared pydantic/zod: events, jobs, reports
├── infra/docker/        # compose + Dockerfiles
├── tests/               # unit, pipeline (golden 30s clip end-to-end)
└── docs/                # architecture.md, dataset-card.md, model-card.md, limitations.md, demo-script.md
```

## 14. Tech stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend | Next.js, React, TypeScript | Video dashboard + annotation UI |
| API | FastAPI | Direct Python-pipeline integration |
| DB | PostgreSQL | Users, matches, events, corrections |
| Object storage | MinIO (S3 API) → AWS S3 later | Video, clips, artifacts; swap is config-only |
| Queue | Redis + Dramatiq | Simple async processing |
| Video | FFmpeg, OpenCV | Transcode, frame processing |
| ML | PyTorch, Torchvision, LightGBM | Inference + small-model training |
| Tracking | Pretrained detector + ByteTrack | Wrestler identity |
| Experiments | MLflow | Runs, metrics, model versions |
| Observability | OpenTelemetry, Prometheus, Grafana | Pipeline performance |
| Deployment | Docker Compose | Reproducibility |
| CI | GitHub Actions | Tests + build |

## 15. Milestones (Aug → Dec 2026)

| Milestone | Layer | Deliverable | Gate |
|---|---|---|---|
| M0 Tracer bullet | 0 | Notebook overlay + tracks.parquet | ≥80% identity hold on a real match |
| M1 Platform | 1 | Compose stack, upload, player, manual tagging, clips | Upload→playable→tagged end to end |
| M2 Annotation | 2 | Labeling UI + export; labeling begins | 5 matches labeled |
| M3 CV in product | 3 | Detect/track/pose as worker stages + quality report | Auto-tracks on 10 matches |
| M4 States | 4 | Feature pipeline + state model + timeline + eval | Macro-F1 ≥ 0.75 |
| M5 Events | 5 | Rules engine + consolidation + clips + corrections | Event F1 ≥ 0.60 |
| M6 Report | 6 | Stats + observations + grounded LLM report + ratings | 0 unsupported claims on 5 matches |
| M7 Polish + demo | — | Eval writeup, cards, limitations, README, demo video | Stranger can `docker compose up` and process a match |

Stretch items enter only after the M5 gate. If any milestone slips more than a week, cut from stretch first, then reduce event classes to three.

## 16. Definition of done (the demo)

A 3-minute recruiting demo: upload a real match → watch progress stages → open the dashboard with a state timeline → click an event → the auto-clip plays → open the report → an observation cites timestamps → click a timestamp and the video seeks to it → correct one event label. Backed by a README with an architecture diagram, the eval table, and a limitations doc.

## 17. How to work on this project (for agents)

- **Read Sections 1–5 before touching anything.** The CV-decides/LLM-explains rule and the scope rules are load-bearing.
- **Find your layer in Section 8.** Respect its "depends on," its gate, and its "do NOT."
- **Never ship a claim without evidence.** Applies to code and to coaching output alike.
- **Prefer pretrained over trained, rules over models, one process over many services** — until a gate proves you need more.
- **Keep `SPEC.md` and this guide in sync.** Scope changes go through the human owner.
- **When uncertain about wrestling meaning, flag it.** Some things we can measure directly; their coaching interpretation carries uncertainty and should be stated as such.

## 18. Open questions to finalize before build

1. Auth: simple email/password (recommended for a demo) vs OAuth.
2. LLM for reports: Claude API (small monthly cost) — the only paid dependency. Confirm acceptable.
3. Queue: Dramatiq (recommended, simpler) vs Celery.
4. Annotation player: custom timeline on `vidstack`/plain `<video>` (recommended) vs a heavier library.
5. Practice-footage recording session in August — clean drilling clips meaningfully accelerate Layers 4–5.

---

*End of guide. This document plus `SPEC.md` fully define MatVision. Build in layers, gate honestly, and keep every claim tied to film.*
