"""Preprocess a face video into per-frame 48×48 grayscale images (the deliverable).

The functional ``preprocessing_test`` (CONTRIBUTING §9): take a **>=20-second** face
video, sample ~``preprocessing_test.sample_fps`` frames per second, and for each sampled
frame detect -> crop -> center -> 48x48-grayscale the face, saving ``image0.png`` ..
``imageN.png`` to ``paths.preprocessing_test_dir`` (>=20 images from a 20-second clip).
Proves the live preprocessing turns real frames into FER-2013-format samples.

The frame source (``VideoSource``, #51), the detector, the image size, and the
transforms all come from ``config.yaml`` (the same knobs training uses), so the same
pipeline runs on live or recorded input (Ablation §3).

    python scripts/preprocess.py                  # uses preprocessing_test.input_video
    python scripts/preprocess.py path/to/face_video.mp4
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

import cv2

# Make `src` importable when run as `python scripts/preprocess.py` from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.video.capture import VideoSource
from src.emotion_detector.video.preprocess import FacePreprocessor, NoFaceError


def sample_step(fps: float, sample_fps: float) -> int:
    """Frames to advance between samples: ``round(fps / sample_fps)`` (≥ 1).

    At 30 fps source and 1 fps sampling this is 30 — one saved frame per second, which
    is exactly the "≥1 prediction/second" cadence the audit asks for.
    """
    return max(1, int(round(fps / max(float(sample_fps), 1e-9))))


def _source_config(cfg: dict, video_path: Optional[str]) -> dict:
    """A cfg copy whose ``video.source`` is the clip to preprocess (cfg untouched)."""
    source = video_path or cfg["preprocessing_test"]["input_video"]
    src_cfg = copy.deepcopy(cfg)
    src_cfg["video"]["source"] = source
    return src_cfg


def run_preprocess(
    cfg: dict,
    video_path: Optional[str] = None,
    preprocessor: Optional[FacePreprocessor] = None,
) -> int:
    """Sample the video, save each detected face, and return the image count.

    Args:
        cfg: Loaded config.
        video_path: Optional override for the input clip (else ``input_video``).
        preprocessor: Optional injected ``FacePreprocessor`` (tests); else from cfg.

    Returns:
        The number of face images written.
    """
    pt = cfg["preprocessing_test"]
    out_dir = Path(cfg["paths"]["preprocessing_test_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = preprocessor if preprocessor is not None else FacePreprocessor(cfg)
    saved = 0
    with VideoSource(_source_config(cfg, video_path)) as src:
        step = sample_step(src.fps, pt["sample_fps"])
        logger.info(
            f"Preprocessing '{src.active_source}' (fps={src.fps:.1f}) - sampling every "
            f"{step} frame(s) -> {out_dir}."
        )
        for i, frame in enumerate(src.frames()):
            if i % step != 0:
                continue
            try:
                face = pre.process_frame(frame)
            except NoFaceError:
                continue  # no face this frame — a normal skip, not an error (§9)
            cv2.imwrite(str(out_dir / f"image{saved}.png"), face)
            saved += 1

    logger.info(f"Saved {saved} face image(s) to {out_dir}.")
    if saved < pt["min_images"]:
        logger.warning(
            f"Only {saved} image(s) saved; deliverable wants >= {pt['min_images']}. "
            "Use a longer (>=20s) clip with a visible face, or raise sample_fps."
        )
    return saved


def main(config_path: str = "config.yaml", video_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    run_preprocess(cfg, video_path)


if __name__ == "__main__":
    # Optional first arg: a face-video path overriding preprocessing_test.input_video.
    main(video_path=sys.argv[1] if len(sys.argv) > 1 else None)
