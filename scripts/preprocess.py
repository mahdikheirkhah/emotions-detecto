"""Preprocess a face video into per-frame 48×48 grayscale images.

Reads a video file, applies face detection, crops and resizes each detected
face to 48×48 grayscale, and saves the results under
results/preprocessing_test/ as image0.png … imageN.png.
"""
from __future__ import annotations

from emotion_detector.utils.config import load_config
from emotion_detector.utils.logging import setup_logging


def main() -> None:
    cfg = load_config("config.yaml")
    setup_logging(cfg)
    raise NotImplementedError("Implemented in later issues.")


if __name__ == "__main__":
    main()
