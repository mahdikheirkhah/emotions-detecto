"""Plot and save training/validation loss and accuracy curves.

Reads the TensorBoard event logs produced by train.py and renders the
learning curves to results/model/learning_curves.png.
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
