"""Train the CNN emotion classifier on FER-2013.

Loads config.yaml, builds the dataset pipeline, constructs the model, runs the
training loop with early stopping and model checkpointing, then saves the final
.keras model and architecture .txt to results/model/.
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
