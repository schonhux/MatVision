"""
Guards the ADR-007 constraint: worker stage modules must be importable WITHOUT
torch/ultralytics installed.

Why this matters enough to test: the moment someone adds `from ultralytics import
YOLO` at the top of a stage module, the entire test suite becomes unrunnable in CI
and in the agent sandbox — but only on machines that lack torch, so it'll pass
locally for whoever wrote it and break for everyone else. This test catches that
immediately and explains why.

The fix when this fails is always the same: move the heavy import inside the
function that uses it (see the `run()` bodies in detect_track.py / pose.py, or
app/vision/models.py).
"""

import importlib
import sys

import pytest

STAGE_MODULES = [
    "validate", "transcode", "clips",
    "detect_track", "pose", "features",
    "runner",
]

HEAVY_MODULES = ["torch", "ultralytics", "torchvision"]


@pytest.mark.parametrize("stage", STAGE_MODULES)
def test_stage_imports_without_heavy_deps(merged_worker_app, stage):
    already_loaded = {m for m in HEAVY_MODULES if m in sys.modules}

    importlib.import_module(f"app.stages.{stage}")

    newly_loaded = {m for m in HEAVY_MODULES if m in sys.modules} - already_loaded
    assert not newly_loaded, (
        f"app.stages.{stage} imported {newly_loaded} at module level. "
        "Move that import inside the function that uses it — see ADR-007."
    )


def test_vision_models_module_imports_without_torch(merged_worker_app):
    """app/vision/models.py is the designated home for heavy imports, but even it
    must not import them at module scope — only inside its functions.
    """
    already_loaded = {m for m in HEAVY_MODULES if m in sys.modules}
    importlib.import_module("app.vision.models")
    newly_loaded = {m for m in HEAVY_MODULES if m in sys.modules} - already_loaded
    assert not newly_loaded, (
        f"app/vision/models.py imported {newly_loaded} at module level; "
        "model loading must stay lazy."
    )


def test_pipeline_stages_match_available_stage_modules(merged_worker_app):
    """Every stage named in PIPELINE_STAGES must have a corresponding module with a
    run() function — otherwise the runner fails at execution time on a real match
    rather than here.
    """
    from app.models import PIPELINE_STAGES

    for stage in PIPELINE_STAGES:
        module = importlib.import_module(f"app.stages.{stage}")
        assert hasattr(module, "run"), f"app.stages.{stage} has no run() function"
        assert callable(module.run)


def test_layer3_stages_are_registered():
    """Layer 3 wired its stages into the pipeline — a guard against building stages
    that never actually get run.
    """
    import importlib
    import sys as _sys

    # Import fresh via the merged app in the fixture-independent way used above.
    from app.models import PIPELINE_STAGES

    for expected in ("detect_track", "pose", "features"):
        assert expected in PIPELINE_STAGES, f"{expected} missing from PIPELINE_STAGES"
    # Order matters: features depends on pose, which depends on detect_track.
    assert PIPELINE_STAGES.index("detect_track") < PIPELINE_STAGES.index("pose")
    assert PIPELINE_STAGES.index("pose") < PIPELINE_STAGES.index("features")
