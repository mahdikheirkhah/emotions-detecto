"""Unit + integration tests for the live-stream deliverable script (#54, #55).

The output-format helpers are the audit contract, so they're pinned exactly. The
end-to-end ``run_live_stream`` / ``run_display_stream`` run a real 3s synthetic clip
(VideoSource + FacePreprocessor) with an injected fake classifier (and a fake GUI for
the dashboard), exercising the whole path without TensorFlow or a display; skipped where
Haar cascade or the video codec is unavailable.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pytest

# The script imports cv2 at module load, so guard before loading it.
cv2 = pytest.importorskip("cv2")

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "predict_live_stream.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("predict_live_stream", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()

_PRED_LINE = re.compile(r"^\d\d:\d\d:\d\ds : \w+ , \d+%$")


# ---------------------------------------------------------------------------
# output-format contract (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00:00"),
        (1.0, "00:00:01"),
        (1.9, "00:00:01"),  # truncates, not rounds
        (65, "00:01:05"),
        (3661, "01:01:01"),
    ],
)
def test_format_timestamp(seconds, expected) -> None:
    assert script.format_timestamp(seconds) == expected


def test_format_prediction_matches_required_contract() -> None:
    assert script.format_prediction(1.0, "Happy", 0.734) == "00:00:01s : Happy , 73%"
    assert script.format_prediction(65, "Sad", 0.6) == "00:01:05s : Sad , 60%"


# ---------------------------------------------------------------------------
# end-to-end run_live_stream / run_display_stream (real video + fakes)
# ---------------------------------------------------------------------------


def _cfg(source, predictions_per_second=1) -> dict:
    return {
        "stages": {"preprocessing": True},
        "preprocessing": {"image_size": 48, "normalization": "rescale"},
        "video": {
            "source": source,
            "fallback_path": "",
            "frame_skip": 1,
            "predictions_per_second": predictions_per_second,
            "display": False,
            "window_title": "Test Window",
        },
        "face_detector": {
            "backend": "haar",
            "haar_cascade": "haarcascade_frontalface_default.xml",
            "scale_factor": 1.05,
            "min_neighbors": 3,
            "min_size": [30, 30],
        },
    }


class _FakeClassifier:
    """Injected predictor: a fixed (label, confidence), no model needed."""

    def predict_top(self, tensor):
        assert tensor.shape == (48, 48)  # the model-ready tensor from #53
        return "Happy", 0.9


class _FakeGui:
    """Headless stand-in for OpenCV's GUI: records shown frames, can trigger quit."""

    def __init__(self, quit_after=None) -> None:
        self.shown = []
        self.key_calls = 0
        self.destroyed = False
        self._quit_after = quit_after

    def show(self, window, frame):
        self.shown.append((window, frame.copy()))

    def wait_key(self, delay_ms):
        self.key_calls += 1
        if self._quit_after is not None and self.key_calls >= self._quit_after:
            return ord("q")
        return -1

    def destroy(self):
        self.destroyed = True


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


def _make_blank_video(path: Path, seconds: int, fps: float) -> None:
    """Faceless frames — the detector finds nothing (for the no-face display path)."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (300, 300)
    )
    for i in range(int(seconds * fps)):
        writer.write(np.full((300, 300, 3), (i * 7) % 256, np.uint8))
    writer.release()


def _has_green_box(frames) -> bool:
    """True if any shown frame contains a confidence-green (0,200,0) pixel."""
    return any((frame == (0, 200, 0)).all(axis=2).any() for _, frame in frames)


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


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_run_live_stream_prints_the_contract(tmp_path: Path, capsys) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_face_video(video, seconds=3, fps=10.0)

    count = script.run_live_stream(
        _cfg(source=0), video_path=str(video), classifier=_FakeClassifier()
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Reading video stream ..."  # exact header, first line
    assert count >= 3  # >= 1 prediction per second over a 3s clip

    prediction_lines = [ln for ln in lines if _PRED_LINE.match(ln)]
    assert len(prediction_lines) == count
    assert prediction_lines[0] == "00:00:00s : Happy , 90%"
    assert lines.count("Preprocessing ...") == count  # one per emitted frame


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_timestamps_increase_about_one_per_second(tmp_path: Path, capsys) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_face_video(video, seconds=3, fps=10.0)

    script.run_live_stream(
        _cfg(source=0), video_path=str(video), classifier=_FakeClassifier()
    )
    stamps = [
        ln.split("s :")[0]
        for ln in capsys.readouterr().out.splitlines()
        if _PRED_LINE.match(ln)
    ]
    assert stamps[:3] == ["00:00:00", "00:00:01", "00:00:02"]


# ---------------------------------------------------------------------------
# #55 dashboard — run_display_stream (fake GUI, no real window)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_display_shows_every_frame_predicts_on_cadence(tmp_path: Path, capsys) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_face_video(video, seconds=3, fps=10.0)  # 30 frames
    gui = _FakeGui()

    count = script.run_display_stream(
        _cfg(source=0), video_path=str(video), classifier=_FakeClassifier(), gui=gui
    )

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Reading video stream ..."
    assert gui.destroyed  # window released even on normal exit
    assert count >= 3  # predictions on the ~1/sec cadence
    assert len(gui.shown) > count  # displayed far more frames than predictions
    assert all(w == "Test Window" for w, _ in gui.shown)  # config window title
    prediction_lines = [ln for ln in out if _PRED_LINE.match(ln)]
    assert len(prediction_lines) == count
    assert _has_green_box(gui.shown)  # confident (0.9) → green box drawn


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_display_quit_key_stops_the_loop(tmp_path: Path) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_face_video(video, seconds=3, fps=10.0)  # 30 frames available
    gui = _FakeGui(quit_after=5)  # 'q' on the 5th waitKey

    script.run_display_stream(
        _cfg(source=0), video_path=str(video), classifier=_FakeClassifier(), gui=gui
    )
    assert len(gui.shown) == 5  # loop broke at the quit key, not after all 30 frames
    assert gui.destroyed


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_display_no_face_still_shows_frames(tmp_path: Path) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "blank.mp4"
    _make_blank_video(video, seconds=1, fps=10.0)  # 10 faceless frames
    gui = _FakeGui()

    count = script.run_display_stream(
        _cfg(source=0), video_path=str(video), classifier=_FakeClassifier(), gui=gui
    )
    assert count == 0  # no face → no predictions, no crash
    assert len(gui.shown) == 10  # every frame still displayed
    assert gui.destroyed
    assert not _has_green_box(gui.shown)  # nothing annotated


# ---------------------------------------------------------------------------
# CLI parsing + display routing
# ---------------------------------------------------------------------------


def test_cli_defaults() -> None:
    args = script._parse_args([])
    assert args.video_path is None
    assert args.config == "config.yaml"
    assert args.display is None  # unset → main falls back to config


def test_cli_display_and_no_display_flags() -> None:
    assert script._parse_args(["--display"]).display is True
    assert script._parse_args(["--no-display"]).display is False


def test_cli_positional_video_path() -> None:
    args = script._parse_args(["clip.mp4", "--display"])
    assert args.video_path == "clip.mp4"
    assert args.display is True


def test_main_routes_display_flag(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(script, "load_config", lambda p: {"video": {"display": False}})
    monkeypatch.setattr(script, "setup_logging", lambda cfg: None)
    monkeypatch.setattr(
        script, "run_display_stream", lambda cfg, vp=None: calls.append("display")
    )
    monkeypatch.setattr(
        script, "run_live_stream", lambda cfg, vp=None: calls.append("print")
    )

    script.main(display=True)
    script.main(display=False)
    script.main(display=None)  # None → config's video.display (False here)
    assert calls == ["display", "print", "print"]
