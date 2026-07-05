"""Unit tests for the face-detector dispatch (Issue #50)."""

from __future__ import annotations

import os

import pytest

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video.factory import build_face_detector

_CASCADE = "haarcascade_frontalface_default.xml"
_PROTOTXT = "models/deploy.prototxt"
_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"


def _cascade_available() -> bool:
    return hasattr(cv2, "CascadeClassifier") and os.path.exists(
        os.path.join(cv2.data.haarcascades, _CASCADE)
    )


def _model_available() -> bool:
    return (
        hasattr(cv2.dnn, "readNetFromCaffe")
        and os.path.exists(_PROTOTXT)
        and os.path.exists(_MODEL)
    )


def _cfg(backend: str) -> dict:
    return {
        "face_detector": {
            "backend": backend,
            "haar_cascade": _CASCADE,
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": [30, 30],
            "dnn_prototxt": _PROTOTXT,
            "dnn_model": _MODEL,
            "confidence_threshold": 0.5,
        }
    }


@pytest.mark.skipif(not _cascade_available(), reason="Haar cascade unavailable")
def test_dispatch_haar() -> None:
    assert type(build_face_detector(_cfg("haar"))).__name__ == "HaarFaceDetector"


@pytest.mark.skipif(not _model_available(), reason="DNN model not downloaded")
def test_dispatch_dnn() -> None:
    assert type(build_face_detector(_cfg("dnn"))).__name__ == "DnnFaceDetector"


def test_dispatch_unknown_backend_fails_loud() -> None:
    # dispatch validates the name before constructing, so no model files are needed.
    with pytest.raises(ValueError):
        build_face_detector(_cfg("teleport"))
