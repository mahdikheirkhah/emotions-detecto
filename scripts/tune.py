"""Run the hyperparameter search (#43 space), pick the winner, close the loop.

Pipeline: reproduce the exact training data pipeline (seed → load → split → clean →
normalize → imbalance → tf.data), run ``tuner.search`` scoring on **validation
only**, log a sorted trials table and persist it under ``results/tuning/``, then
**promote the winning values into ``config.yaml``** as the new defaults (the searched
arrays stay in ``tuning.search_space`` as the permanent record). Finally retrain once
with the promoted config and confirm accuracy on the untouched test set.

    python scripts/tune.py

The official audit number still comes from ``scripts/predict.py``; the confirmation
here is logged (not printed) so this script never emits the graded stdout line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make `src` importable when run as `python scripts/tune.py` from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.data.cleaning import clean_dataset
from src.emotion_detector.data.fer2013 import Fer2013Fetcher
from src.emotion_detector.data.imbalance import resolve_imbalance
from src.emotion_detector.data.pipeline import make_dataset
from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.data.splits import make_splits
from src.emotion_detector.models.tuning import (
    best_hyperparameters,
    make_tuner,
    results_table,
    save_results_table,
)
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.config_writer import promote_values
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.utils.seeding import set_global_seed

_SPLITS = ("Training", "PublicTest", "PrivateTest")


def _load_all_rows(cfg: dict):
    """Fetch every split and stack into (X, y, usage) for make_splits."""
    fetcher = Fer2013Fetcher(cfg)
    images, labels, usage = [], [], []
    for split in _SPLITS:
        Xi, yi = fetcher.fetch(split)
        images.append(Xi)
        labels.append(yi)
        usage.append(np.full(len(yi), split))
    return np.concatenate(images), np.concatenate(labels), np.concatenate(usage)


def build_search_datasets(cfg: dict):
    """Reproduce train.py's pipeline up to the tf.data train/val sets + class_weight.

    Same discipline as training: split BEFORE fitting, normalizer fit on the cleaned
    TRAIN split only, imbalance remedy on train only. The tuner never sees test.
    """
    X, y, usage = _load_all_rows(cfg)
    X_train, y_train, X_val, y_val, _, _ = make_splits(cfg, X, y, usage)

    X_train, y_train = clean_dataset(cfg, X_train, y_train)
    normalizer = build_normalizer(cfg).fit(X_train)
    X_train = normalizer.transform(X_train)
    X_val = normalizer.transform(X_val)

    X_train, y_train, class_weight = resolve_imbalance(cfg, X_train, y_train)
    train_ds = make_dataset(X_train, y_train, cfg, training=True)
    val_ds = make_dataset(X_val, y_val, cfg, training=False)
    return train_ds, val_ds, class_weight


def _search_callbacks(cfg: dict) -> list:
    """Callbacks for the *search* — EarlyStopping only.

    Deliberately NOT the full training bundle: Keras Tuner owns per-trial model
    checkpointing and its own logging, so a ModelCheckpoint/TensorBoard here would
    fight it across trials. EarlyStopping is the useful one — it prunes a weak trial
    the moment ``callbacks.monitor`` stalls, so the budget goes to promising configs.
    """
    from tensorflow import keras

    cb = cfg["callbacks"]
    if not cb.get("early_stopping", True):
        return []
    return [
        keras.callbacks.EarlyStopping(
            monitor=cb["monitor"],
            patience=cb.get("early_stopping_patience", 10),
            restore_best_weights=True,
        )
    ]


def run_search(cfg: dict) -> Any:
    """Run the tuner over the search space and return the searched tuner."""
    set_global_seed(cfg["global"]["seed"])
    logger.info("=== Hyperparameter search start ===")

    train_ds, val_ds, class_weight = build_search_datasets(cfg)
    tuner = make_tuner(cfg)
    tuner.search_space_summary()

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["tuning"]["tune_epochs"],
        callbacks=_search_callbacks(cfg),
        class_weight=class_weight,
        verbose=2,
    )
    return tuner


def _retrain_and_confirm(config_path: str) -> None:
    """Reload the promoted config, retrain once, and log test accuracy.

    Loaded as modules (not imports) because train.py / predict.py are scripts, not a
    package. The audit-graded accuracy line remains predict.py's job — here we only
    log the confirmation.
    """
    train = _load_script("train")
    predict = _load_script("predict")

    best_cfg = load_config(config_path)  # now carries the promoted defaults
    logger.info("Retraining once with the promoted best config…")
    train.run(best_cfg)

    model = predict._load_keras_model(best_cfg["paths"]["model_save_path"])
    scores = predict.score_test_set(best_cfg, model)
    logger.info(
        f"Confirmed on test set — accuracy={scores['accuracy']:.4f} "
        f"(run scripts/predict.py for the official audit line)."
    )


def _load_script(name: str) -> Any:
    """Import a sibling script module (train / predict) by file path."""
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)

    tuner = run_search(cfg)

    df = results_table(tuner)
    logger.info("Trials (best first):\n" + df.to_string(index=False))
    save_results_table(df, cfg)

    best = best_hyperparameters(tuner)
    logger.info(f"Best hyperparameters (by validation): {best}")

    updated = promote_values(config_path, "model", best)
    logger.info(
        f"Promoted {updated} into {config_path} — searched arrays kept as record."
    )

    _retrain_and_confirm(config_path)


if __name__ == "__main__":
    main()
