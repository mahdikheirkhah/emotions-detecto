"""Unit + functional tests for the real-time sampling glue (Issue #53).

Layers:
  * **Pure sampling logic** — a fake source + fake preprocessor make the cadence, the
    FPS-derived timestamps, the no-face skip, and the config-driven rate deterministic
    (no cv2, no model).
  * **Functional** — a real 3-second synthetic-face video through ``VideoSource`` +
    ``FacePreprocessor`` emits ≥3 preprocessed tensors; skipped where the Haar cascade
    or the video codec is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.emotion_detector.video.preprocess import NoFaceError
from src.emotion_detector.video.stream import preprocessed_frames


def _cfg(predictions_per_second=1) -> dict:
    return {
        "stages": {"preprocessing": True},
        "preprocessing": {"image_size": 48, "normalization": "rescale"},
        "video": {"predictions_per_second": predictions_per_second},
        "face_detector": {
            "backend": "haar",
            "haar_cascade": "haarcascade_frontalface_default.xml",
            "scale_factor": 1.05,
            "min_neighbors": 3,
            "min_size": [30, 30],
        },
    }


class _FakeSource:
    """Duck-typed VideoSource: yields ``n`` sentinel frames at a fixed ``fps``."""

    def __init__(self, n: int, fps: float = 10.0) -> None:
        self._n = n
        self.fps = fps

    def frames(self):
        for i in range(self._n):
            yield np.full(
                (8, 8, 3), i % 256, np.uint8
            )  # sentinel: value == frame index


class _FakePreprocessor:
    """Returns a 48×48 tensor tagged with the frame's first pixel; NoFace on a set."""

    def __init__(self, no_face_indices=()) -> None:
        self._no_face = set(no_face_indices)

    def to_model_input(self, frame):
        idx = int(frame[0, 0, 0])
        if idx in self._no_face:
            raise NoFaceError("no face")
        return np.full((48, 48), idx, np.float32)


# ---------------------------------------------------------------------------
# sampling cadence + timestamps (pure)
# ---------------------------------------------------------------------------


def test_three_second_source_emits_at_least_three() -> None:
    # 30 frames @ 10 fps = 3.0s of stream; 1 pred/sec → samples at t=0,1,2.
    out = list(
        preprocessed_frames(_FakeSource(30, fps=10.0), _cfg(1), _FakePreprocessor())
    )
    assert len(out) >= 3
    timestamps = [round(ts, 3) for ts, _ in out]
    assert timestamps == [0.0, 1.0, 2.0]


def test_timestamps_are_fps_derived() -> None:
    out = list(
        preprocessed_frames(_FakeSource(30, fps=10.0), _cfg(1), _FakePreprocessor())
    )
    # Each yielded tensor is tagged with its frame index → index == ts * fps.
    for ts, tensor in out:
        assert float(tensor[0, 0]) == pytest.approx(ts * 10.0)


def test_predictions_per_second_is_config_driven() -> None:
    src = _FakeSource(30, fps=10.0)
    out = list(
        preprocessed_frames(src, _cfg(predictions_per_second=2), _FakePreprocessor())
    )
    # 2/sec over 3s → t=0,0.5,1.0,...,2.5 → 6 samples.
    assert len(out) == 6
    assert [round(ts, 2) for ts, _ in out] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def test_faster_than_fps_emits_every_frame() -> None:
    # predictions_per_second above fps can't exceed the frame rate → one per frame.
    out = list(
        preprocessed_frames(_FakeSource(5, fps=1.0), _cfg(10), _FakePreprocessor())
    )
    assert len(out) == 5


# ---------------------------------------------------------------------------
# no-face handling (skip, never crash)
# ---------------------------------------------------------------------------


def test_no_face_sample_is_skipped_not_fatal() -> None:
    # Sampled frames are indices 0,10,20; make index 10 face-less → it's skipped.
    src = _FakeSource(30, fps=10.0)
    out = list(
        preprocessed_frames(src, _cfg(1), _FakePreprocessor(no_face_indices=[10]))
    )
    ts = [round(t, 2) for t, _ in out]
    assert ts == [0.0, 2.0]  # the t=1.0 sample had no face → dropped, loop continued


def test_all_faceless_yields_nothing() -> None:
    src = _FakeSource(30, fps=10.0)
    pre = _FakePreprocessor(no_face_indices=range(30))
    assert list(preprocessed_frames(src, _cfg(1), pre)) == []


# ---------------------------------------------------------------------------
# config validation + logging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_rate_fails_loud(bad) -> None:
    gen = preprocessed_frames(_FakeSource(3), _cfg(bad), _FakePreprocessor())
    with pytest.raises(ValueError):
        next(gen)


def test_logs_preprocessing_per_emitted_frame() -> None:
    from src.emotion_detector.utils.logging import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        out = list(
            preprocessed_frames(_FakeSource(30, fps=10.0), _cfg(1), _FakePreprocessor())
        )
    finally:
        logger.remove(sink_id)
    preprocessing_logs = [m for m in messages if "Preprocessing" in m]
    assert len(preprocessing_logs) == len(
        out
    )  # one "Preprocessing ..." per emitted frame


# ---------------------------------------------------------------------------
# functional — real VideoSource + FacePreprocessor on a 3s face video
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video.capture import VideoSource
from src.emotion_detector.video.preprocess import FacePreprocessor


def _draw_face(img) -> None:
    cv2.ellipse(img, (150, 160), (80, 100), 0, 0, 360, (150, 150, 150), -1)
    cv2.circle(img, (120, 140), 12, (50, 50, 50), -1)
    cv2.circle(img, (180, 140), 12, (50, 50, 50), -1)
    cv2.ellipse(img, (150, 175), (8, 18), 0, 0, 360, (110, 110, 110), -1)
    cv2.ellipse(img, (150, 210), (30, 12), 0, 0, 360, (70, 70, 70), -1)


def _make_face_video(path: Path, seconds: int, fps: float) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (300, 300)
    )
    for _ in range(int(seconds * fps)):
        img = np.full((300, 300, 3), 200, np.uint8)
        _draw_face(img)
        writer.write(img)
    writer.release()


def _video_io_works(tmp_path: Path) -> bool:
    probe = tmp_path / "_probe.mp4"
    try:
        _make_face_video(probe, seconds=1, fps=3.0)
        cap = cv2.VideoCapture(str(probe))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return bool(ok)
    except Exception:
        return False


def _cascade_available() -> bool:
    return hasattr(cv2, "CascadeClassifier") and os.path.exists(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )


def _source_cfg(cfg: dict, path: str) -> dict:
    cfg = dict(cfg)
    cfg["video"] = dict(cfg["video"], source=path, fallback_path="", frame_skip=1)
    return cfg


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_functional_three_second_video_emits_three_tensors(tmp_path: Path) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_face_video(video, seconds=3, fps=10.0)
    cfg = _source_cfg(_cfg(1), str(video))

    with VideoSource(cfg) as src:
        out = list(preprocessed_frames(src, cfg, FacePreprocessor(cfg)))

    assert len(out) >= 3  # ≥1 tensor/sec over a 3s clip
    for ts, tensor in out:
        assert tensor.shape == (48, 48)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min() and tensor.max() <= 1.0  # rescaled, model-ready
