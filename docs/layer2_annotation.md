# Layer 2 — Annotation System: Results

Turns the platform into the tool that builds the training dataset. Per
PROJECT_GUIDE.md: "the product is the dataset tool."

## Acceptance criteria (BUILD_PLAN.md M2)

- [x] Every Level-1 field (start, end, type, initiator, outcome) is labelable and persists
- [x] Match states labelable as segments over the timeline
- [x] Export produces a schema-validated dataset file with split tags
- [x] Boundaries editable to frame precision (30fps frame-stepping, PATCH endpoints)
- [ ] **5 matches fully labeled and exported** — requires real footage + labeling time

## What was built

**Data model** (`api/app/models.py`, migration `0002`)
- `Event` gains Level-1/Level-2 label fields: `initiator`, `outcome`, `state_before`,
  `state_after`, `opponent_response`, `technique`, free-form `detail` JSON, plus
  `annotator_id` for reviewer-agreement tracking.
- `StateSegment` — hand-labeled position spans (neutral/top/bottom/scramble/stopped).
  Layer 4's classifier writes to this same table with `source='model:<version>'`, so
  predictions and ground truth are directly comparable at eval time.
- `MatchAthlete` — who is who, plus the seed bbox from click-to-identify that Layer 3's
  tracker will bind a track ID to.
- `Match` gains `venue` and `annotation_complete` (both required for leakage-safe splits).

**Leakage-safe splitting** (`ml/datasets/splits.py`) — the most important correctness
guarantee in the project. Groups matches transitively by shared athlete or venue using
union-find, then assigns whole groups to train/val/test. Deterministic (stable hashing,
not `random`) so evaluation stays comparable across runs. Includes an *independent*
`verify_no_leakage()` implemented without reusing the grouping code, so a bug in one
doesn't silently pass the other.

**API** — state segment CRUD with overlap rejection, athlete upsert-by-role, event
PATCH/DELETE for boundary editing and relabeling, match annotation metadata, dataset
export, and dataset stats.

**Frontend** — the annotation console (`/matches/[id]/annotate`): frame-accurate
stepping (30fps, keyboard `,`/`.`, shift for 10 frames), in/out marking (`i`/`o`),
event labeling constrained to controlled vocabularies, state segment labeling, athlete
identification, and live lists with seek-on-click. Plus a dataset progress page
(`/dataset`) showing M2-gate progress and a one-click export download.

## Design decisions worth noting

**Controlled vocabularies, not free strings.** `initiator`, `outcome`, and state names
are Pydantic `Literal` types. A typo becomes a 422 at label time instead of a
silently-corrupt training example discovered weeks later.

**Overlapping state segments are rejected (409).** A wrestler can't be in two positions
at once; overlapping ground truth would make Layer 4's training targets ambiguous.

**`top`/`bottom` require `controlling`.** The state is meaningless without knowing who
is on top.

**Export refuses to ship leaky data.** After computing splits, the export re-verifies
independently and returns HTTP 500 rather than emitting a dataset with leakage — a
silently leaky dataset is worse than no dataset, because it produces inflated metrics
that look like success.

**Marking a match complete is gated.** Requires at least the `user` athlete identified
and one state segment labeled, so an empty match can't quietly enter the training set.

## Testing

99 tests passing (55 new in Layer 2).

| Suite | Count | Covers |
|---|---|---|
| `tests/pipeline/test_splits.py` | 23 | Grouping (incl. transitive), determinism, name normalization, and deliberately-broken splits that the verifier must catch |
| `tests/api/test_annotations.py` | 22 | State CRUD, overlap rejection, top/bottom validation, athlete upsert, bbox validation, event labeling/editing, cross-match scoping |
| `tests/api/test_datasets.py` | 10 | Export shape, split correctness, determinism, owner scoping, M2 gate stats |

**Bug caught during development:** migration `0002` originally used a bare
`op.create_foreign_key`, which SQLite cannot execute (no ALTER for constraints) —
it failed partway through, leaving a half-applied schema. Production runs Postgres so
it would have worked there, but a migration that only runs on one database is a trap.
Fixed with `batch_alter_table`, and verified upgrade → downgrade → upgrade all succeed.

## Next

Labeling real footage is now the bottleneck, and it gates Layers 4 and 5. Layer 3 (CV
pipeline in production) does not depend on labels and can proceed in parallel.
