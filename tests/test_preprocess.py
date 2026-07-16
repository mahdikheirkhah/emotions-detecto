"""Unit + functional tests for live-frame preprocessing (Issue #52).

Layers:
  * **Pure logic** — a fake detector drives ``FacePreprocessor``: output shape/dtype,
    largest-face selection, square-centered cropping (incl. edge padding), the
    training-matched normalizer, and the no-face / bad-frame branches (CONTRIBUTING §9).
  * **Deliverable loop** — ``scripts/preprocess.py`` sampling a tiny video with an
    injected preprocessor yields ≥20 saved 48×48 grayscale images (deterministic).
  * **Functional (§9)** — the real Haar detector on a synthetic-face video produces the
    same ≥20 face crops; skipped where the cascade / video codec is unavailable.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video.preprocess import FacePreprocessor, NoFaceError

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "preprocess.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("preprocess_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(size=48, normalization="rescale", preprocessing_stage=True) -> dict:
    return {
        "stages": {"preprocessing": preprocessing_stage},
        "preprocessing": {
            "image_size": size,
            "normalization": normalization,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": 8,
        },
        "paths": {"preprocessing_test_dir": "results/preprocessing_test/"},
        "preprocessing_test": {
            "input_video": "data/sample_video.mp4",
            "sample_fps": 1,
            "min_images": 20,
        },
        "video": {"source": 0, "fallback_path": "", "frame_skip": 1},
        "face_detector": {
            "backend": "haar",
            "haar_cascade": "haarcascade_frontalface_default.xml",
            "scale_factor": 1.05,
            "min_neighbors": 3,
            "min_size": [30, 30],
        },
    }


class _FakeDetector:
    """Returns preset boxes; validates frame shape like the real detectors (§9)."""

    def __init__(self, *boxes) -> None:
        self._boxes = list(boxes)

    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"bad frame shape {getattr(frame, 'shape', None)}")
        return list(self._boxes)


# ---------------------------------------------------------------------------
# FacePreprocessor — output contract
# ---------------------------------------------------------------------------


def test_process_frame_returns_48x48_gray_uint8() -> None:
    pre = FacePreprocessor(_cfg(), detector=_FakeDetector((20, 20, 40, 40)))
    out = pre.process_frame(np.zeros((100, 100, 3), np.uint8))
    assert out.shape == (48, 48)  # single-channel grayscale, configured size
    assert out.dtype == np.uint8


def test_output_size_is_config_driven() -> None:
    pre = FacePreprocessor(_cfg(size=64), detector=_FakeDetector((10, 10, 50, 50)))
    assert pre.process_frame(np.zeros((100, 100, 3), np.uint8)).shape == (64, 64)


def test_picks_largest_face() -> None:
    # White where the large box is, black elsewhere → the crop is white only if the
    # *larger* box (not the small one at the origin) was chosen.
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[50:90, 50:90] = 255
    pre = FacePreprocessor(
        _cfg(), detector=_FakeDetector((0, 0, 10, 10), (50, 50, 40, 40))
    )
    out = pre.process_frame(frame)
    assert out.mean() > 200  # the bright (large-box) region


def test_no_face_raises_no_face_error() -> None:
    pre = FacePreprocessor(_cfg(), detector=_FakeDetector())  # no boxes
    with pytest.raises(NoFaceError):
        pre.process_frame(np.zeros((100, 100, 3), np.uint8))


def test_bad_frame_raises_value_error() -> None:
    pre = FacePreprocessor(_cfg(), detector=_FakeDetector((0, 0, 10, 10)))
    with pytest.raises(ValueError):
        pre.process_frame(np.zeros((100, 100), np.uint8))  # grayscale, no channel axis


def test_to_model_input_applies_rescale_normalizer() -> None:
    pre = FacePreprocessor(
        _cfg(normalization="rescale"), detector=_FakeDetector((20, 20, 40, 40))
    )
    frame = np.full((100, 100, 3), 255, np.uint8)
    out = pre.to_model_input(frame)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0  # rescaled to [0, 1]


def test_no_face_error_is_not_a_value_error() -> None:
    # The sampling loop catches NoFaceError to *skip*; it must not also swallow a
    # genuine ValueError (bad frame). Keep the two hierarchies distinct.
    assert not issubclass(NoFaceError, ValueError)


# ---------------------------------------------------------------------------
# _square_crop — centering + edge padding
# ---------------------------------------------------------------------------


def test_square_crop_inside_frame_is_square_and_centered() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    crop = FacePreprocessor._square_crop(frame, (30, 30, 20, 20))
    assert crop.shape[0] == crop.shape[1] == 20  # max(w, h), no padding needed


def test_square_crop_pads_when_box_runs_off_edge() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    # cx=15, cy=10, half=15 -> y spans [-5, 25] -> 5px top pad restores a 30x30 square.
    crop = FacePreprocessor._square_crop(frame, (0, 0, 30, 20))
    assert crop.shape[0] == crop.shape[1] == 30


def test_square_crop_uses_longer_side() -> None:
    frame = np.zeros((200, 200, 3), np.uint8)
    crop = FacePreprocessor._square_crop(frame, (50, 50, 20, 60))  # h > w
    assert crop.shape[0] == crop.shape[1] == 60


# ---------------------------------------------------------------------------
# sample_step
# ---------------------------------------------------------------------------


def test_sample_step_one_per_second() -> None:
    script = _load_script()
    assert script.sample_step(30.0, 1) == 30  # 30 fps, 1/sec → every 30th frame
    assert script.sample_step(10.0, 2) == 5
    assert script.sample_step(1.0, 5) == 1  # never below 1


# ---------------------------------------------------------------------------
# deliverable loop — tiny video + injected preprocessor (deterministic)
# ---------------------------------------------------------------------------


def _make_video(path: Path, frames: int, fps: float, draw=None) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (300, 300)
    )
    for i in range(frames):
        img = np.full((300, 300, 3), 40, np.uint8)
        if draw is not None:
            draw(img, i)
        writer.write(img)
    writer.release()


def _video_io_works(tmp_path: Path) -> bool:
    probe = tmp_path / "_probe.mp4"
    try:
        _make_video(probe, frames=3, fps=5.0)
        cap = cv2.VideoCapture(str(probe))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return bool(ok)
    except Exception:
        return False


class _FixedPreprocessor:
    """Injected stand-in: every frame yields one 48×48 gray face."""

    def process_frame(self, frame):
        return np.full((48, 48), 128, np.uint8)


def test_run_preprocess_saves_at_least_20_images(tmp_path: Path) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_video(video, frames=22, fps=1.0)  # 22s @ 1 fps
    cfg = _cfg()
    cfg["paths"]["preprocessing_test_dir"] = str(tmp_path / "out")

    script = _load_script()
    saved = script.run_preprocess(
        cfg, video_path=str(video), preprocessor=_FixedPreprocessor()
    )

    assert saved >= 20
    out = tmp_path / "out"
    images = sorted(out.glob("image*.png"))
    assert len(images) == saved
    assert (out / "image0.png").exists() and (out / f"image{saved - 1}.png").exists()
    for img_path in images:  # each is a 48×48 single-channel grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        assert img.shape == (48, 48)


def test_run_preprocess_does_not_mutate_caller_cfg(tmp_path: Path) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "clip.mp4"
    _make_video(video, frames=5, fps=1.0)
    cfg = _cfg()
    cfg["paths"]["preprocessing_test_dir"] = str(tmp_path / "out")
    original_source = cfg["video"]["source"]

    script = _load_script()
    script.run_preprocess(cfg, video_path=str(video), preprocessor=_FixedPreprocessor())
    assert cfg["video"]["source"] == original_source  # deepcopy, not in-place edit


# ---------------------------------------------------------------------------
# functional §9 — real Haar detector end-to-end
# ---------------------------------------------------------------------------


def _cascade_available() -> bool:
    return hasattr(cv2, "CascadeClassifier") and os.path.exists(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )


def _draw_face(img, i) -> None:
    """A drawn face the Haar cascade reliably detects, with a per-frame off-face mark.

    Haar detection on this synthetic face is position-sensitive, so the face stays at a
    fixed (proven-detectable) location; the frame index only tints a small corner block,
    well away from the face, so successive frames differ without breaking detection.
    """
    cv2.ellipse(img, (150, 160), (80, 100), 0, 0, 360, (150, 150, 150), -1)
    cv2.circle(img, (120, 140), 12, (50, 50, 50), -1)
    cv2.circle(img, (180, 140), 12, (50, 50, 50), -1)
    cv2.ellipse(img, (150, 175), (8, 18), 0, 0, 360, (110, 110, 110), -1)
    cv2.ellipse(img, (150, 210), (30, 12), 0, 0, 360, (70, 70, 70), -1)
    v = (i * 10) % 256
    cv2.rectangle(img, (0, 0), (12, 12), (v, v, v), -1)  # per-frame marker, off-face


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_process_frame_real_detector_on_synthetic_face() -> None:
    img = np.full((300, 300, 3), 200, np.uint8)
    _draw_face(img, 0)
    pre = FacePreprocessor(_cfg())  # real Haar detector
    out = pre.process_frame(img)
    assert out.shape == (48, 48) and out.dtype == np.uint8
    assert out.std() > 0  # a real face crop, not a blank square


@pytest.mark.skipif(not _cascade_available(), reason="OpenCV Haar cascade unavailable")
def test_functional_preprocessing_test_deliverable(tmp_path: Path) -> None:
    if not _video_io_works(tmp_path):
        pytest.skip("OpenCV video backend can't round-trip a file here")
    video = tmp_path / "face_clip.mp4"
    _make_video(video, frames=24, fps=1.0, draw=_draw_face)  # 24s of a visible face
    cfg = _cfg()
    cfg["paths"]["preprocessing_test_dir"] = str(tmp_path / "out")

    script = _load_script()
    saved = script.run_preprocess(cfg, video_path=str(video))  # real FacePreprocessor

    assert saved >= 20  # the §9 contract: ≥20 images from a ≥20s clip
    for img_path in (tmp_path / "out").glob("image*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        assert img.shape == (48, 48)
        assert img.std() > 0
