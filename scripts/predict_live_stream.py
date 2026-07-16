"""Real-time emotion detection — the project's headline deliverable (#54, #55).

Wires the whole live path end-to-end: capture (``VideoSource``, #51) -> sample ~1/sec +
detect + preprocess (#52/#53) -> classify (#54) -> **print** the timestamped prediction.
With ``--display`` (or ``video.display: true``) it also opens the #55 OpenCV dashboard:
a window showing the feed with a rectangle around the face and the emotion + confidence
overlaid. With no webcam it falls back to the recorded ``video.fallback_path``
automatically (inherited from #51), so it runs headless.

This is one of only two scripts allowed to ``print`` (CONTRIBUTING §5): the audit greps
this exact stdout, so the format is fixed --

    Reading video stream ...
    Preprocessing ...
    00:00:00s : Happy , 73%
    Preprocessing ...
    00:00:01s : Neutral , 68%

Everything diagnostic still goes through Loguru (stderr). Model path, detector, pred
rate, and display are all config values, so the same pipeline runs on live or recorded
input, with or without the window (Ablation §3).

    python scripts/predict_live_stream.py                       # print only
    python scripts/predict_live_stream.py --display             # + dashboard window
    python scripts/predict_live_stream.py path/to/video.mp4     # recorded source
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Optional

import cv2

# Make `src` importable when running `python scripts/predict_live_stream.py` from root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.models.classifier import KerasEmotionClassifier
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import setup_logging
from src.emotion_detector.video.capture import VideoSource
from src.emotion_detector.video.overlay import annotate
from src.emotion_detector.video.preprocess import FacePreprocessor, NoFaceError
from src.emotion_detector.video.stream import preprocessed_frames

_QUIT_KEYS = {ord("q"), 27}  # 'q' or ESC closes the dashboard


def format_timestamp(seconds: float) -> str:
    """Elapsed stream seconds → ``HH:MM:SS`` (zero-padded)."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_prediction(seconds: float, label: str, confidence: float) -> str:
    """The exact audit line: ``HH:MM:SSs : Emotion , XX%``."""
    return f"{format_timestamp(seconds)}s : {label} , {confidence:.0%}"


def _source_config(cfg: dict, video_path: Optional[str]) -> dict:
    """A cfg copy whose ``video.source`` is *video_path* (cfg untouched); else cfg."""
    if not video_path:
        return cfg
    src_cfg = copy.deepcopy(cfg)
    src_cfg["video"]["source"] = video_path
    return src_cfg


def run_live_stream(
    cfg: dict,
    video_path: Optional[str] = None,
    classifier: Optional[Any] = None,
) -> int:
    """Print a timestamped emotion prediction for every sampled face; return the count.

    Args:
        cfg: Loaded config.
        video_path: Optional explicit source overriding ``video.source`` (else the
            configured webcam index, with the recorded fallback).
        classifier: Optional injected predictor exposing ``predict_top`` (tests); else a
            ``KerasEmotionClassifier`` built from cfg.

    Returns:
        The number of predictions printed.
    """
    predictor = classifier if classifier is not None else KerasEmotionClassifier(cfg)
    src_cfg = _source_config(cfg, video_path)

    print("Reading video stream ...")
    count = 0
    with VideoSource(src_cfg) as source:
        for timestamp, tensor in preprocessed_frames(source, src_cfg):
            print("Preprocessing ...")
            label, confidence = predictor.predict_top(tensor)
            print(format_prediction(timestamp, label, confidence))
            count += 1
    return count


class _CvGui:
    """Thin wrapper over OpenCV GUI calls, injected so the loop is testable headless."""

    def show(self, window: str, frame: Any) -> None:
        cv2.imshow(window, frame)

    def wait_key(self, delay_ms: int) -> int:
        return cv2.waitKey(delay_ms)

    def destroy(self) -> None:
        cv2.destroyAllWindows()


def run_display_stream(
    cfg: dict,
    video_path: Optional[str] = None,
    classifier: Optional[Any] = None,
    preprocessor: Optional[FacePreprocessor] = None,
    gui: Optional[Any] = None,
) -> int:
    """The #55 dashboard: show every frame with the face box + emotion overlay.

    Displays at the capture frame rate (interactive) but predicts on the
    ``predictions_per_second`` cadence (#53), holding the last label between preds, so
    the box tracks the face smoothly while the CNN runs ~1/sec. Still prints the same
    ``HH:MM:SSs`` lines as ``run_live_stream`` (the audit stdout is unchanged). Quits on
    'q'/ESC and always releases the window.

    Args:
        cfg: Loaded config (``video.predictions_per_second`` + ``video.window_title``).
        video_path: Optional source override (else webcam, with the recorded fallback).
        classifier / preprocessor / gui: Optional injected collaborators (tests); each
            defaults to the real component built from cfg.

    Returns:
        The number of predictions printed.
    """
    predictor = classifier if classifier is not None else KerasEmotionClassifier(cfg)
    pre = preprocessor if preprocessor is not None else FacePreprocessor(cfg)
    gui = gui if gui is not None else _CvGui()
    window = cfg["video"]["window_title"]

    src_cfg = _source_config(cfg, video_path)
    pps = float(src_cfg["video"].get("predictions_per_second", 1))
    interval = 1.0 / pps if pps > 0 else 1.0

    print("Reading video stream ...")
    count = 0
    last_label: Optional[str] = None
    last_confidence = 0.0
    next_prediction = 0.0
    with VideoSource(src_cfg) as source:
        fps = float(source.fps) or 30.0
        for index, frame in enumerate(source.frames()):
            timestamp = index / fps
            try:
                box, tensor = pre.locate_and_prepare(frame)
            except NoFaceError:
                box = None

            if box is not None and timestamp + 1e-9 >= next_prediction:
                next_prediction = timestamp + interval
                last_label, last_confidence = predictor.predict_top(tensor)
                print("Preprocessing ...")
                print(format_prediction(timestamp, last_label, last_confidence))
                count += 1

            if box is not None and last_label is not None:
                annotate(frame, box, last_label, last_confidence)

            gui.show(window, frame)
            if (gui.wait_key(1) & 0xFF) in _QUIT_KEYS:
                break
    gui.destroy()
    return count


def main(
    config_path: str = "config.yaml",
    video_path: Optional[str] = None,
    display: Optional[bool] = None,
) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    show = cfg["video"].get("display", False) if display is None else display
    if show:
        run_display_stream(cfg, video_path)
    else:
        run_live_stream(cfg, video_path)


def _parse_args(argv: Optional[list] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time emotion detection (#54/#55)."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Recorded-video path overriding video.source (optional).",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path.")
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--display",
        dest="display",
        action="store_true",
        default=None,
        help="Show the OpenCV dashboard window (#55).",
    )
    display.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Force print-only, ignoring video.display.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config, video_path=args.video_path, display=args.display)
