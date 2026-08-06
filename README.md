# MatVision

MatVision turns a wrestling match video into a coach-style breakdown that's actually backed by the film. You upload a match, tell it which wrestler you are, and it gives you back an interactive timeline of what happened, short clips of the key moments, stats on your offense and defense, and a short written report pointing out what's working and what to fix — with every claim linked to a timestamp you can click and watch for yourself.

The system doesn't just point a language model at raw video and ask it to describe the match — that doesn't work reliably and can't back up what it says. Instead, computer vision does the actual work of figuring out what happened: it tracks the two wrestlers (and filters out the referee), follows their body position and movement, recognizes what stage of the match they're in (neutral, top, bottom, scrambling), and detects specific moments like shots and takedowns. Only after all of that is figured out and measured does a language model get involved, and its only job is to explain those already-verified facts in plain English. It never watches the video itself and never gets to invent something that isn't backed by the data.

## How it works, roughly

```
video
  → detect and track the wrestlers
  → extract body pose and motion
  → figure out what position/state the match is in
  → detect specific events (shots, takedowns, escapes, etc.)
  → turn events into stats and patterns
  → generate a written report explaining the patterns, citing timestamps
```

## Stack

- **Frontend:** Next.js web app — upload, video player, timeline, annotation tools, the report view.
- **Backend:** FastAPI — handles auth, matches, uploads, and job tracking.
- **Database:** PostgreSQL — stores match data, detected events, and corrections.
- **Storage:** S3-compatible object storage (MinIO locally) — holds video files, clips, and processed data.
- **Background processing:** Redis + a Python worker — does the actual video analysis in stages, so a long match doesn't block the app.
- **Computer vision:** pretrained detection and pose-estimation models, plus a lightweight custom model for recognizing match states.
- **Report generation:** Claude API, constrained to only describe facts it's given.

## Status

In active development. See `PROJECT_GUIDE.md` for the full technical writeup and `BUILD_PLAN.md` for how it's being built and tested.
