# Layer 6 — Evidence Graph + Grounded Report: Results

Connects stats, rule-detected patterns, and events into an evidence graph, then
writes a coach's note where every claim is either backed by a real event ID or
dropped before the athlete ever sees it.

## Acceptance criteria (BUILD_PLAN.md M6)

- [x] Report generates from evidence JSON only (the LLM never sees raw video —
      `ml/reporting/evidence.py` is the entire payload)
- [x] Validation pass provably rejects a planted unsupported sentence (unit test:
      `tests/pipeline/test_reporting_llm.py::test_planted_unsupported_statement_is_dropped`)
- [x] Every surviving observation cites at least one real timestamp; clicking it
      seeks the video (report section on the match page)
- [ ] **0 unsupported claims across 5 test matches** — needs a Mac run against real
      match data with `ANTHROPIC_API_KEY` set
- [ ] **Report quality rated (evidence-validity >= 4/5)** — needs human review of
      real reports; the rating UI exists (5-star, posts to `/report/rating`)

## What was built

**`ml/reporting/stats.py`** — the numbers everything else stands on: shot
attempts, conversion rate, control time, scramble duration, escapes, conceded
takedowns — computed per athlete from persisted events + state segments. Pure
function, no I/O.

**`ml/reporting/observations.py`** — rule-based pattern detection over those
stats: low conversion, strong finishing, defense leaks, control-time
imbalance, long scrambles. Every observation that's traceable to specific
events attaches their real IDs; the one pattern that isn't (long scramble,
derived from state-segment duration alone) carries no evidence IDs on purpose
— see "Design decisions" below.

**`ml/reporting/evidence.py`** — assembles the exact JSON sent to Claude: match
metadata, stats, observations, and events with IDs/timestamps/measurements.
Nothing else is in scope; this function *is* the grounding contract's "LLM
never sees raw video" clause, made literal.

**`ml/reporting/llm.py`** — builds the system prompt (grounding rules + the
`coach_tone` instruction from the existing `tone.py`), calls Claude, parses the
JSON response, and — the important part — `validate_report()` strips any
statement whose evidence IDs don't all resolve to real events in this match. A
statement with zero valid IDs is dropped entirely; a statement with some valid
and some fabricated IDs keeps only the valid ones. The priority gets the same
treatment. Nothing ever reaches the athlete without a citation that traces back
to an actual event.

**Three new pipeline stages** (`stats` -> `observations` -> `report`), appended
to `PIPELINE_STAGES` after `clips`. `stats` caches its output on
`Match.stats_summary` so `observations` and `report` don't recompute it.
`report` upserts a single `Report` row per match (a corrected event set makes
the old report stale, not historically interesting).

**API**: `GET /matches/{id}/stats`, `GET /matches/{id}/observations`,
`GET /matches/{id}/report`, `POST /matches/{id}/report/rating`, and
`POST /matches/{id}/report/regenerate` — the last one resets just the
stats/observations/report job rows to `pending` and re-enqueues, so a
regenerate after correcting events (or switching `coach_tone`) doesn't repeat
the CV pipeline.

**Coach tone / intensity setting**: `Match.coach_tone` (`balanced` / `hard` /
`extreme`) already existed from Layer 5 scaffolding (`ml/reporting/tone.py`,
the settings toggle on the match page). This layer wires it all the way
through: the tone instruction is embedded directly in the system prompt, and
every report is generated in that voice — subject to the same evidence rules
regardless of intensity. "Extreme" is instructed to be forceful and direct
without insulting the athlete's identity, inventing a technique, diagnosing an
injury, or making an uncited claim; the validation pass enforces the last one
mechanically rather than trusting the model to comply.

**Frontend**: a report section on the match page — summary, statements
(labeled observation vs. interpretation), one priority, and a clickable
timestamp chip per citation that seeks the video to that moment (the exact
interaction from SPEC.md's demo definition). Shows how many statements were
filtered by validation, a regenerate button, and a 5-star evidence-validity
rating.

## Design decisions worth noting

**Grounding is enforced in code, not just prompted.** The system prompt tells
Claude to cite evidence IDs, but `validate_report()` doesn't trust that it did
so correctly — it re-checks every citation against the real event ID set for
that match and drops anything that doesn't check out. This is the same
philosophy as Layer 2's independently-implemented `verify_no_leakage()`: the
thing that catches the bug can't share the same logic as the thing that might
produce it.

**Not every observation carries evidence, and that's fine.** Control-time
imbalance and long-scramble patterns come from aggregate state-segment
duration, not any single event, so they cite the nearest bounding
takedown/escape events where one exists, or nothing at all when it doesn't.
An ungrounded observation can still shape the report's framing and tone, but
`validate_report()` guarantees it can never become an ungrounded *statement*
in what the athlete reads.

**`report` regenerates independently of the CV pipeline.** Correcting an event
in the annotation console, or just switching `coach_tone`, shouldn't mean
re-running detection/tracking/pose. `POST /report/regenerate` resets exactly
the three Layer 6 job rows to `pending`, and the existing resumable runner
(`worker/app/stages/runner.py`, unchanged since Layer 1) picks up from there —
no new resume logic needed, because the pipeline was already built to support
this.

**The Anthropic client import is lazy**, same pattern as torch in Layer 3
(ADR-007). `ml/reporting/llm.py` only imports `anthropic` inside
`generate_report_content()`, so every other module in `ml/reporting/` —
stats, observations, evidence assembly, and the validation pass — is fully
testable without the package installed or a network connection. The one
function that does need it raises a clear `ReportGenerationError` when
`ANTHROPIC_API_KEY` isn't set, which the worker stage turns into a
`StageError` the UI already knows how to display.

## Testing

249 tests passing total (43 new in Layer 6).

| Suite | Count | Covers |
|---|---|---|
| `test_reporting_stats.py` | 8 | Duration-by-state math, control time, conversion rate, conceded takedowns, scramble stats, empty-input safety |
| `test_reporting_observations.py` | 8 | Each pattern rule in isolation, evidence-ID correctness, threshold boundaries, an end-to-end run against real `compute_match_stats` output |
| `test_reporting_llm.py` | 9 | Prompt construction, the missing-API-key error path, and — the important ones — a planted unsupported statement being dropped, an ungrounded priority being dropped, partial-citation statements keeping only valid IDs, and an all-ungrounded response producing an empty (not crashed) report |
| `test_report_stages.py` (worker) | 6 | stats -> observations -> report wiring, stage-ordering guards (`observations` fails clearly without `stats` first), the report stage validating a mocked LLM response and upserting a single row per match |
| `test_reports.py` (api) | 10 | Every endpoint, 404s before data exists, rating validation (1-5 range), owner scoping, and the regenerate endpoint's job-reset + enqueue behavior |

`tests/worker/test_stage_imports.py` was extended: `stats`/`observations`/`report`
added to the no-heavy-import guard (now also checking `anthropic`, not just
torch/ultralytics), plus a new `test_report_stages_are_registered` asserting
pipeline order (`events` < `stats` < `observations` < `report`).

Migration `0004_layer6_report.py` was verified with a real upgrade -> downgrade
-> upgrade round-trip against SQLite in the sandbox, the same check every prior
migration got (Layer 2's migration caught a SQLite-only incompatibility this
way — see ADR in `dev/DECISIONS.md`).

## Not yet verified (needs a Mac run with a real API key)

Every piece of logic that can be tested without the network or a live API key
has been. What's left is inherently a Mac-side check: set
`ANTHROPIC_API_KEY` in `.env`, run `docker compose up`, process a real match
through the full pipeline, and read the actual reports — checking the
unsupported-claim rate and rating evidence validity by hand across the 5 test
matches the BUILD_PLAN gate calls for. The mechanism that would catch a
regression here (`validate_report`) is proven; what's unmeasured is real-world
prompt quality, i.e. how often Claude produces well-grounded statements in the
first place versus how often the validation pass has to intervene.
