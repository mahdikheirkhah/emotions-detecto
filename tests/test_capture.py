"""Unit tests for the video capture source + recorded-video fallback (Issue #51).

Two layers:
  * **Real I/O** — write a tiny sample video with ``cv2.VideoWriter`` and prove
    ``VideoSource`` yields its frames, exposes FPS, and that a bad webcam index falls
    back to that file. Skipped where the OpenCV video backend can't round-trip a file.
  * **Pure logic** — a fake ``cv2.VideoCapture`` proves the primary→fallback selection,
    release, and fail-loud paths deterministically, with no codec dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video import capture as capture_mod
from src.emotion_detector.video.capture import VideoSource

_SAMPLE_FPS = 10.0
_SAMPLE_FRAMES = 8


def _make_sample_video(path: Path, frames: int = _SAMPLE_FRAMES) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _SAMPLE_FPS, (64, 64)
    )
    for i in range(frames):
        writer.write(np.full((64, 64, 3), (i * 20) % 256, np.uint8))
    writer.release()


def _video_io_works(tmp_path: Path) -> bool:
    """True only if this OpenCV build can both write and read back a small video."""
    probe = tmp_path / "_probe.mp4"
    try:
        _make_sample_video(probe, frames=4)
        cap = cv2.VideoCapture(str(probe))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return bool(ok)
    except Exception:
        return False


def _cfg(source, fallback_path="", frame_skip=2) -> dict:
    return {
        "video": {
            "source": source,
            "fallback_path": fallback_path,
            "frame_skip": frame_skip,
        }
    }


# ---------------------------------------------------------------------------
# real video I/O
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    path = tmp_path / "sample.mp4"
    _make_sample_video(path)
    return path


def test_frames_yielded_from_file(sample_video: Path) -> None:
    src = VideoSource(_cfg(str(sample_video)))
    frames = list(src.frames())
    src.release()
    assert len(frames) == _SAMPLE_FRAMES
    assert all(f.shape == (64, 64, 3) for f in frames)
    assert not src.used_fallback  # a file source is not a "fallback"


def test_fps_exposed(sample_video: Path) -> None:
    with VideoSource(_cfg(str(sample_video))) as src:
        assert src.fps == pytest.approx(_SAMPLE_FPS)


def test_bad_index_falls_back_to_file(sample_video: Path) -> None:
    # A webcam index that won't open (headless/no camera) must degrade to the file.
    src = VideoSource(_cfg(999, fallback_path=str(sample_video)))
    try:
        assert src.used_fallback is True
        assert src.active_source == str(sample_video)
        assert len(list(src.frames())) == _SAMPLE_FRAMES
    finally:
        src.release()


def test_context_manager_releases(sample_video: Path) -> None:
    with VideoSource(_cfg(str(sample_video))) as src:
        assert src._capture is not None
    assert src._capture is None  # __exit__ released it
    with pytest.raises(RuntimeError):  # released source can't be reused
        list(src.frames())


# ---------------------------------------------------------------------------
# fallback selection / lifecycle — pure logic, no real codec needed
# ---------------------------------------------------------------------------


class _FakeCapture:
    """Stand-in for cv2.VideoCapture: opens iff *ok*, yields *n_frames* then stops."""

    def __init__(self, source, ok: bool, n_frames: int = 3, fps: float = 24.0) -> None:
        self.source = source
        self._ok = ok
        self._left = n_frames
        self._fps = fps
        self.released = False

    def isOpened(self):  # noqa: N802 — mirrors cv2 API
        return self._ok

    def read(self):
        if self._left <= 0:
            return False, None
        self._left -= 1
        return True, np.zeros((8, 8, 3), np.uint8)

    def get(self, prop):
        return self._fps

    def release(self):
        self.released = True


def _fake_opener(available: dict) -> "callable":
    """A cv2.VideoCapture replacement: ``available`` maps source → opens? (bool)."""

    def _open(source):
        return _FakeCapture(source, ok=available.get(source, False))

    return _open


def test_uses_primary_when_available(monkeypatch) -> None:
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _fake_opener({0: True}))
    src = VideoSource(_cfg(0, fallback_path="fallback.mp4"))
    assert src.active_source == 0
    assert src.used_fallback is False


def test_falls_back_when_primary_unavailable(monkeypatch) -> None:
    opener = _fake_opener({0: False, "fallback.mp4": True})
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", opener)
    src = VideoSource(_cfg(0, fallback_path="fallback.mp4"))
    assert src.used_fallback is True
    assert src.active_source == "fallback.mp4"
    assert len(list(src.frames())) == 3


def test_fail_loud_when_both_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _fake_opener({}))
    with pytest.raises(RuntimeError):
        VideoSource(_cfg(0, fallback_path="fallback.mp4"))


def test_fail_loud_when_primary_unavailable_and_no_fallback(monkeypatch) -> None:
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _fake_opener({0: False}))
    with pytest.raises(RuntimeError):
        VideoSource(_cfg(0, fallback_path=""))


def test_release_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _fake_opener({0: True}))
    src = VideoSource(_cfg(0))
    cap = src._capture
    src.release()
    src.release()  # second call is a no-op, must not raise
    assert cap.released is True
    assert src._capture is None


def test_fps_defaults_when_source_reports_zero(monkeypatch) -> None:
    def _open(source):
        return _FakeCapture(source, ok=True, fps=0.0)

    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _open)
    src = VideoSource(_cfg(0))
    assert src.fps == capture_mod._DEFAULT_FPS


def test_frame_skip_exposed(monkeypatch) -> None:
    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _fake_opener({0: True}))
    src = VideoSource(_cfg(0, frame_skip=5))
    assert src.frame_skip == 5


def test_open_error_degrades_to_fallback(monkeypatch) -> None:
    # cv2.VideoCapture itself raising (bad source) must fall back, not propagate.
    def _open(source):
        if source == "bad://source":
            raise cv2.error("cannot open")
        return _FakeCapture(source, ok=True)

    monkeypatch.setattr(capture_mod.cv2, "VideoCapture", _open)
    src = VideoSource(_cfg("bad://source", fallback_path="fallback.mp4"))
    assert src.used_fallback is True
