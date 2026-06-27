"""Run emotion prediction on a single image or directory of images.

Loads the saved .keras model, detects faces, and prints the predicted emotion
label to stdout (required audit output — print() is intentional here).
"""
from __future__ import annotations

from emotion_detector.utils.config import load_config
from emotion_detector.utils.logging import setup_logging


def main() -> None:
    cfg = load_config("config.yaml")
    setup_logging(cfg)
    # print() is intentional in this script — audit requires exact stdout output.
    raise NotImplementedError("Implemented in later issues.")


if __name__ == "__main__":
    main()
