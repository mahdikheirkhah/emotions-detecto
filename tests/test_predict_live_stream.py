"""Unit + integration tests for the live-stream deliverable script (#54).

The output-format helpers are the audit contract, so they're pinned exactly. The
end-to-end ``run_live_stream`` runs a real 3s synthetic-face video (VideoSource +
FacePreprocessor) with an injected fake classifier, exercising the whole print path
without TensorFlow; skipped where the Haar cascade or the video codec is unavailable.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "predict_live_stream.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("predict_live_stream", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


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
# end-to-end run_live_stream (real video + fake classifier)
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")

_PRED_LINE = re.compile(r"^\d\d:\d\d:\d\ds : \w+ , \d+%$")


def _cfg(source, predictions_per_second=1) -> dict:
    return {
        "stages": {"preprocessing": True},
        "preprocessing": {"image_size": 48, "normalization": "rescale"},
        "video": {
            "source": source,
            "fallback_path": "",
            "frame_skip": 1,
            "predictions_per_second": predictions_per_second,
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
