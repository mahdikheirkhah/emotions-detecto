"""Real-time emotion detection — the project's headline deliverable (#54).

Wires the whole live path end-to-end: capture (``VideoSource``, #51) -> sample ~1/sec +
detect + preprocess (``preprocessed_frames``, #53) -> classify (#54) -> **print** the
timestamped prediction. On a machine with no webcam it falls back to the recorded
``video.fallback_path`` automatically (inherited from #51), so it runs headless.

This is one of only two scripts allowed to ``print`` (CONTRIBUTING §5): the audit greps
this exact stdout, so the format is fixed --

    Reading video stream ...
    Preprocessing ...
    00:00:00s : Happy , 73%
    Preprocessing ...
    00:00:01s : Neutral , 68%

Everything diagnostic still goes through Loguru (stderr). Model path, detector, and
prediction rate all come from ``config.yaml``, so this runs any trained model on live or
recorded input unchanged (Ablation §3).

    python scripts/predict_live_stream.py
    python scripts/predict_live_stream.py path/to/video.mp4    # recorded source
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Optional

# Make `src` importable when running `python scripts/predict_live_stream.py` from root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.models.classifier import KerasEmotionClassifier
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import setup_logging
from src.emotion_detector.video.capture import VideoSource
from src.emotion_detector.video.stream import preprocessed_frames


def format_timestamp(seconds: float) -> str:
    """Elapsed stream seconds → ``HH:MM:SS`` (zero-padded)."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_prediction(seconds: float, label: str, confidence: float) -> str:
    """The exact audit line: ``HH:MM:SSs : Emotion , XX%``."""
    return f"{format_timestamp(seconds)}s : {label} , {confidence:.0%}"


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

    src_cfg = cfg
    if video_path:
        src_cfg = copy.deepcopy(cfg)
        src_cfg["video"]["source"] = video_path

    print("Reading video stream ...")
    count = 0
    with VideoSource(src_cfg) as source:
        for timestamp, tensor in preprocessed_frames(source, src_cfg):
            print("Preprocessing ...")
            label, confidence = predictor.predict_top(tensor)
            print(format_prediction(timestamp, label, confidence))
            count += 1
    return count


def main(config_path: str = "config.yaml", video_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    run_live_stream(cfg, video_path)


if __name__ == "__main__":
    # Optional first arg: a recorded-video path overriding video.source.
    main(video_path=sys.argv[1] if len(sys.argv) > 1 else None)
