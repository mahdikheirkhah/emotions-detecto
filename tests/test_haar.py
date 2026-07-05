"""Unit tests for the Haar-cascade face detector (Issue #49).

Real detection uses OpenCV's bundled ``haarcascade_frontalface_default.xml`` on a
procedurally-drawn face (no committed image needed) and a blank frame. A fake cascade
covers the output-format / config-param plumbing deterministically. Skipped where
OpenCV or its bundled cascade is unavailable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video.haar import HaarFaceDetector


def _cascade_available() -> bool:
    return hasattr(cv2, "CascadeClassifier") and os.path.exists(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )


requires_cascade = pytest.mark.skipif(
    not _cascade_available(), reason="OpenCV Haar cascade unavailable"
)


def _cfg(scale_factor=1.05, min_neighbors=3, min_size=(30, 30), cascade=None) -> dict:
    return {
        "face_detector": {
            "haar_cascade": cascade or "haarcascade_frontalface_default.xml",
            "scale_factor": scale_factor,
            "min_neighbors": min_neighbors,
            "min_size": list(min_size),
        }
    }


def _synthetic_face() -> np.ndarray:
    """A drawn face (oval head + eyes + nose + mouth) the Haar cascade detects."""
    img = np.full((300, 300, 3), 200, np.uint8)
    cv2.ellipse(img, (150, 160), (80, 100), 0, 0, 360, (150, 150, 150), -1)
    cv2.circle(img, (120, 140), 12, (50, 50, 50), -1)  # left eye
    cv2.circle(img, (180, 140), 12, (50, 50, 50), -1)  # right eye
    cv2.ellipse(img, (150, 175), (8, 18), 0, 0, 360, (110, 110, 110), -1)  # nose
    cv2.ellipse(img, (150, 210), (30, 12), 0, 0, 360, (70, 70, 70), -1)  # mouth
    return img


class _FakeCascade:
    """Records detectMultiScale kwargs and returns a preset detection array."""

    def __init__(self, faces) -> None:
        self._faces = faces
        self.kwargs = None

    def empty(self) -> bool:
        return False

    def detectMultiScale(self, gray, **kwargs):  # noqa: N802 — mirrors cv2 API
        self.kwargs = kwargs
        return self._faces


# ---------------------------------------------------------------------------
# real cascade detection
# ---------------------------------------------------------------------------


@requires_cascade
def test_detects_face_in_sample_image() -> None:
    detector = HaarFaceDetector(_cfg())
    faces = detector.detect(_synthetic_face())
    assert len(faces) >= 1
    x, y, w, h = faces[0]
    assert w > 0 and h > 0 and 0 <= x < 300 and 0 <= y < 300  # box within the frame


@requires_cascade
def test_blank_image_returns_empty() -> None:
    detector = HaarFaceDetector(_cfg())
    assert detector.detect(np.zeros((200, 200, 3), np.uint8)) == []


@requires_cascade
def test_missing_cascade_fails_loud() -> None:
    with pytest.raises(FileNotFoundError):
        HaarFaceDetector(_cfg(cascade="/no/such/cascade.xml"))


# ---------------------------------------------------------------------------
# output format + config plumbing (fake cascade)
# ---------------------------------------------------------------------------


@requires_cascade
def test_detect_returns_int_tuples() -> None:
    detector = HaarFaceDetector(_cfg())
    detector._cascade = _FakeCascade(np.array([[10, 20, 30, 40]]))
    faces = detector.detect(np.zeros((100, 100, 3), np.uint8))
    assert faces == [(10, 20, 30, 40)]
    assert all(isinstance(v, int) for v in faces[0])  # python ints, not np.int64


@requires_cascade
def test_no_faces_returns_empty_list() -> None:
    detector = HaarFaceDetector(_cfg())
    detector._cascade = _FakeCascade(())  # cv2 returns () when nothing found
    assert detector.detect(np.zeros((100, 100, 3), np.uint8)) == []


@requires_cascade
def test_config_params_passed_to_detectmultiscale() -> None:
    detector = HaarFaceDetector(
        _cfg(scale_factor=1.3, min_neighbors=7, min_size=(48, 48))
    )
    fake = _FakeCascade(())
    detector._cascade = fake
    detector.detect(np.zeros((100, 100, 3), np.uint8))
    assert fake.kwargs["scaleFactor"] == 1.3
    assert fake.kwargs["minNeighbors"] == 7
    assert fake.kwargs["minSize"] == (48, 48)


# ---------------------------------------------------------------------------
# bad input
# ---------------------------------------------------------------------------


@requires_cascade
@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((100, 100), np.uint8),  # grayscale (no channel axis)
        np.zeros((100, 100, 4), np.uint8),  # 4 channels
        "not an array",
    ],
)
def test_bad_frame_raises_value_error(bad) -> None:
    detector = HaarFaceDetector(_cfg())
    with pytest.raises(ValueError):
        detector.detect(bad)
