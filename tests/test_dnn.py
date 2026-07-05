"""Unit tests for the DNN (SSD/ResNet) face detector (Issue #50).

The confidence-filter / box-scaling logic is tested purely (a synthetic detection
tensor, no model needed). Real detection runs the bundled caffe model on a
procedurally-drawn face and is skipped where the model files aren't present (e.g. CI
before ``scripts/download_face_model.py``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video.dnn import DnnFaceDetector

_PROTOTXT = "models/deploy.prototxt"
_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"


def _model_available() -> bool:
    return (
        hasattr(cv2.dnn, "readNetFromCaffe")
        and os.path.exists(_PROTOTXT)
        and os.path.exists(_MODEL)
    )


requires_model = pytest.mark.skipif(
    not _model_available(), reason="OpenCV DNN face model not downloaded"
)


def _cfg(confidence_threshold=0.5, prototxt=_PROTOTXT, model=_MODEL) -> dict:
    return {
        "face_detector": {
            "dnn_prototxt": prototxt,
            "dnn_model": model,
            "confidence_threshold": confidence_threshold,
        }
    }


def _synthetic_face() -> np.ndarray:
    img = np.full((300, 300, 3), 200, np.uint8)
    cv2.ellipse(img, (150, 160), (80, 100), 0, 0, 360, (150, 150, 150), -1)
    cv2.circle(img, (120, 140), 12, (50, 50, 50), -1)
    cv2.circle(img, (180, 140), 12, (50, 50, 50), -1)
    cv2.ellipse(img, (150, 175), (8, 18), 0, 0, 360, (110, 110, 110), -1)
    cv2.ellipse(img, (150, 210), (30, 12), 0, 0, 360, (70, 70, 70), -1)
    return img


def _detections(*rows) -> np.ndarray:
    """Build an SSD ``(1, 1, N, 7)`` tensor from (confidence, x1, y1, x2, y2) rows."""
    det = np.zeros((1, 1, len(rows), 7), np.float32)
    for i, (conf, x1, y1, x2, y2) in enumerate(rows):
        det[0, 0, i] = [0, 1, conf, x1, y1, x2, y2]
    return det


# ---------------------------------------------------------------------------
# _parse_detections — confidence filter + box scaling (pure, no model)
# ---------------------------------------------------------------------------


def test_parse_respects_confidence_threshold() -> None:
    det = _detections(
        (0.90, 0.1, 0.1, 0.5, 0.5),  # keep
        (0.40, 0.2, 0.2, 0.6, 0.6),  # drop (< 0.5)
        (0.99, 0.0, 0.0, 1.0, 1.0),  # keep
    )
    faces = DnnFaceDetector._parse_detections(det, 100, 200, 0.5)
    assert len(faces) == 2


def test_parse_scales_normalized_boxes_to_pixels() -> None:
    det = _detections((0.9, 0.1, 0.1, 0.5, 0.5))
    x, y, w, h = DnnFaceDetector._parse_detections(det, 100, 200, 0.5)[0]
    assert (x, y, w, h) == (10, 20, 40, 80)  # 0.1*100, 0.1*200, (0.5-0.1)*100, ...


def test_parse_higher_threshold_keeps_fewer() -> None:
    det = _detections((0.6, 0, 0, 0.4, 0.4), (0.95, 0.5, 0.5, 0.9, 0.9))
    assert len(DnnFaceDetector._parse_detections(det, 100, 100, 0.5)) == 2
    assert len(DnnFaceDetector._parse_detections(det, 100, 100, 0.9)) == 1


def test_parse_empty_when_all_below_threshold() -> None:
    det = _detections((0.2, 0, 0, 1, 1), (0.3, 0, 0, 1, 1))
    assert DnnFaceDetector._parse_detections(det, 100, 100, 0.5) == []


# ---------------------------------------------------------------------------
# fail-loud (no model needed)
# ---------------------------------------------------------------------------


def test_missing_model_fails_loud() -> None:
    with pytest.raises(FileNotFoundError):
        DnnFaceDetector(_cfg(model="/no/such/model.caffemodel"))


# ---------------------------------------------------------------------------
# real detection (needs the downloaded model)
# ---------------------------------------------------------------------------


@requires_model
def test_detects_face_in_sample_image() -> None:
    detector = DnnFaceDetector(_cfg())
    faces = detector.detect(_synthetic_face())
    assert len(faces) >= 1
    x, y, w, h = faces[0]
    assert w > 0 and h > 0


@requires_model
def test_blank_image_returns_empty() -> None:
    detector = DnnFaceDetector(_cfg())
    assert detector.detect(np.zeros((300, 300, 3), np.uint8)) == []


@requires_model
def test_bad_frame_raises_value_error() -> None:
    detector = DnnFaceDetector(_cfg())
    with pytest.raises(ValueError):
        detector.detect(np.zeros((100, 100), np.uint8))  # grayscale, no channel axis
