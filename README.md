# MatVision

AI-powered wrestling film intelligence. Computer vision determines what happened in a match; a grounded LLM report explains it — every claim tied to a timestamp.

**Status:** In development, building layer by layer. See `PROJECT_GUIDE.md` for the full technical writeup, `SPEC.md` for scope, and `BUILD_PLAN.md` for the build order and acceptance criteria per layer.

## Layers
- L0 Tracer Bullet — prove detection/tracking survives real wrestling footage
- L1 Platform Foundation — upload, storage, player, manual tagging
- L2 Annotation System — labeling tools, dataset export
- L3 CV Pipeline — detection, tracking, pose in production
- L4 Match-State Model — neutral/top/bottom/scramble classification
- L5 Event Detection — shots, takedowns, escapes + corrections
- L6 Evidence Graph + Report — grounded coaching report generation

Built layer by layer on separate branches, merged to `main` after each layer's acceptance criteria pass.
