"""Optimize the trained model → sparser (pruned) and/or smaller (quantized) artifacts.

Reproduces the exact training preprocessing, then optionally:
  * **prunes** (``optimization.pruning.enabled``) → fine-tunes → saves a sparse
    ``pruned_emotion_model.keras`` (#48), and
  * **quantizes** (``optimization.enabled``) the (pruned-or-original) model → a
    ``.tflite`` (#47).
The two chain (prune → quantize) and each measures size/accuracy before shipping. Does
nothing when both are off.

    python scripts/optimize.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.data.cleaning import clean_dataset
from src.emotion_detector.data.fer2013 import Fer2013Fetcher
from src.emotion_detector.data.pipeline import to_tensors
from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.data.splits import make_splits
from src.emotion_detector.models.optimize import (
    optimize_model,
    prune_and_report,
    save_tflite,
)
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.utils.seeding import set_global_seed

_SPLITS = ("Training", "PublicTest", "PrivateTest")


def _load_all_rows(cfg: dict):
    fetcher = Fer2013Fetcher(cfg)
    images, labels, usage = [], [], []
    for split in _SPLITS:
        Xi, yi = fetcher.fetch(split)
        images.append(Xi)
        labels.append(yi)
        usage.append(np.full(len(yi), split))
    return np.concatenate(images), np.concatenate(labels), np.concatenate(usage)


def _load_keras_model(path: str) -> Any:
    """Load a saved ``.keras`` model (import kept local + patchable for tests)."""
    from tensorflow.keras.models import load_model

    return load_model(path)


def build_eval_and_representative(cfg: dict):
    """Return (X_test, y_test, X_representative, X_train, y_train_onehot).

    Normalizer is fit on the cleaned TRAIN split (never on test); the representative set
    for int8 calibration and the pruning fine-tune set both come from that same
    cleaned+normalized train split (fine-tune labels one-hot for the CE loss).
    """
    set_global_seed(cfg["global"]["seed"])
    X, y, usage = _load_all_rows(cfg)
    X_train, y_train, _, _, X_test, y_test = make_splits(cfg, X, y, usage)

    X_train, y_train = clean_dataset(cfg, X_train, y_train)
    normalizer = build_normalizer(cfg).fit(X_train)
    num_classes = cfg["model"]["num_classes"]

    X_test_t, _ = to_tensors(
        normalizer.transform(X_test), y_test, num_classes=num_classes
    )
    X_train_t, y_train_oh = to_tensors(
        normalizer.transform(X_train), y_train, num_classes=num_classes
    )
    n_rep = int(cfg["optimization"].get("representative_samples", 100))
    representative = np.asarray(X_train_t)[:n_rep]
    return (
        np.asarray(X_test_t),
        y_test,
        representative,
        np.asarray(X_train_t),
        np.asarray(y_train_oh),
    )


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)

    opt = cfg["optimization"]
    prune_on = opt.get("pruning", {}).get("enabled", False)
    quantize_on = opt.get("enabled", False)
    if not prune_on and not quantize_on:
        logger.info(
            "optimization off (pruning + quantization disabled) — nothing to do."
        )
        return

    model_path = cfg["paths"]["model_save_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run scripts/train.py first."
        )
    model = _load_keras_model(model_path)

    X_eval, y_eval, representative, X_ft, y_ft = build_eval_and_representative(cfg)

    # 1) Prune (+ fine-tune) first, so quantization compounds on the sparse model.
    if prune_on:
        prune_and_report(model, cfg, X_eval, y_eval, X_ft, y_ft)
        pruned_path = cfg["paths"]["pruned_model_save_path"]
        Path(pruned_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(pruned_path)
        logger.info(f"Saved pruned model → {pruned_path}")

    # 2) Quantize the (possibly pruned) model to TFLite.
    if quantize_on:
        report = optimize_model(
            model, cfg, X_eval, y_eval, representative_data=representative
        )
        save_tflite(report["tflite_bytes"], cfg["paths"]["tflite_model_save_path"])


if __name__ == "__main__":
    main()
