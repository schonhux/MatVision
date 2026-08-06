"""
Model loading, isolated behind a small interface.

Why this module exists separately from the stages: `torch`/`ultralytics` are heavy,
platform-specific, and (per ADR-007) can't be installed in the CI/agent sandbox at
all. Keeping every import of them inside function bodies here means:

  - The stage modules can be imported and unit-tested anywhere, torch or not.
  - Swapping YOLOv8-pose for RTMPose later (ADR-006) touches only this file.
  - Tests inject a fake detector rather than mocking `ultralytics` internals.

Models are cached at module level because loading weights takes seconds and the
worker processes many matches per run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_detector = None
_pose_model = None

DEFAULT_DETECT_MODEL = "yolov8n.pt"
DEFAULT_POSE_MODEL = "yolov8n-pose.pt"


@dataclass
class Detection:
    """One person detected in one frame. Deliberately plain data — no torch tensors
    escape this module, so everything downstream is numpy/python and testable.
    """
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    track_id: int | None = None
    keypoints: np.ndarray | None = None  # (17, 3) COCO, filled by the pose stage


def get_device() -> str:
    """Prefers Apple Silicon's MPS, then CUDA, then CPU. The project targets a Mac
    (SPEC.md constraints), so MPS is the expected path.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_detector(model_name: str = DEFAULT_DETECT_MODEL):
    global _detector
    if _detector is None:
        from ultralytics import YOLO

        logger.info("Loading detector %s", model_name)
        _detector = YOLO(model_name)
    return _detector


def load_pose_model(model_name: str = DEFAULT_POSE_MODEL):
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO

        logger.info("Loading pose model %s", model_name)
        _pose_model = YOLO(model_name)
    return _pose_model


def detect_people(frame: np.ndarray, conf: float = 0.35) -> list[Detection]:
    """Person detections for a single frame (COCO class 0 only)."""
    model = load_detector()
    result = model(frame, classes=[0], conf=conf, verbose=False, device=get_device())[0]

    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, c in zip(boxes, confs):
            detections.append(
                Detection(bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                          confidence=float(c))
            )
    return detections


def estimate_pose(frame: np.ndarray, conf: float = 0.35) -> list[Detection]:
    """Pose estimates for a single frame. Returns Detections whose `keypoints` are
    populated; the caller matches them to tracked boxes by IoU.
    """
    model = load_pose_model()
    result = model(frame, conf=conf, verbose=False, device=get_device())[0]

    detections = []
    if result.keypoints is not None and result.boxes is not None:
        kpts_all = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for kpts, box, c in zip(kpts_all, boxes, confs):
            detections.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(c),
                    keypoints=kpts,
                )
            )
    return detections


def make_tracker(fps: float):
    """ByteTrack instance. Isolated here because `supervision` deprecated the class
    in 0.28 (still functional in 0.29) — when it's finally removed, this is the one
    place that changes.
    """
    import supervision as sv

    return sv.ByteTrack(frame_rate=int(round(fps)))
