"""Train the CNN emotion classifier on FER-2013 — the thin orchestrator.

Reads the entire run from ``config.yaml`` and wires the ``src/`` components:
seed -> load -> split (no leakage) -> clean -> normalize (fit on train) ->
imbalance -> tf.data -> build+compile model -> fit with callbacks -> save the
best model + serialized history.

All *logic* lives in ``src/``; this file only conducts. Flip any stage toggle in
config and re-run for a complete ablation experiment:

    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make `src` importable when run as `python scripts/train.py` from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.data.cleaning import clean_dataset
from src.emotion_detector.data.fer2013 import Fer2013Fetcher
from src.emotion_detector.data.imbalance import resolve_imbalance
from src.emotion_detector.data.pipeline import make_dataset
from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.data.splits import make_splits
from src.emotion_detector.models.builders import build_model
from src.emotion_detector.models.callbacks import build_callbacks
from src.emotion_detector.models.classifier import resolve_history_path
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.utils.seeding import set_global_seed
from src.emotion_detector.utils.stages import is_stage_on

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
    return (
        np.concatenate(images),
        np.concatenate(labels),
        np.concatenate(usage),
    )


def _search_callbacks(cfg: dict) -> list:
    """EarlyStopping-only callbacks for the search (prune weak trials early).

    Not the full training bundle: Keras Tuner owns per-trial checkpointing/logging,
    so ModelCheckpoint/TensorBoard here would fight it across trials.
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


def _run_tuning(cfg: dict, train_ds: Any, val_ds: Any, class_weight: Any) -> dict:
    """Search the #43 space on the given datasets and return a config with the winner.

    Selects by VALIDATION score (never test), applies the winning hyperparameters to a
    copy of ``cfg`` (in-memory — this does NOT rewrite config.yaml; run scripts/tune.py
    to persist), and logs + saves the trials table. Returns the tuned config.
    """
    from src.emotion_detector.models.tuning import (
        apply_hyperparameters,
        best_hyperparameters,
        make_tuner,
        results_table,
        save_results_table,
    )

    logger.info("stages.tuning ON — searching hyperparameters before the final train…")
    tuner = make_tuner(cfg)
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["tuning"]["tune_epochs"],
        callbacks=_search_callbacks(cfg),
        class_weight=class_weight,
        verbose=2,
    )
    try:
        save_results_table(results_table(tuner), cfg)
    except Exception as exc:  # persistence is best-effort; never block training
        logger.warning(f"Could not save tuning results table: {exc}")

    best = best_hyperparameters(tuner)
    logger.info(f"Tuning done — best hyperparameters (by validation): {best}")
    return apply_hyperparameters(cfg, best)


def run(cfg: dict) -> Any:
    """Execute one training run from a loaded config dict.

    Returns the Keras ``History`` object. Every stage self-gates on its toggle
    (cleaning, preprocessing, augmentation, tuning, …), so this conductor just calls
    them in order.
    """
    set_global_seed(cfg["global"]["seed"])
    logger.info("=== Training run start ===")

    if is_stage_on(cfg, "decomposition"):
        logger.warning(
            "stages.decomposition is ON but ignored by the CNN trainer — PCA "
            "flattens the spatial structure a CNN needs (data.md §6.2)."
        )

    # 1. Load everything, then split BEFORE fitting anything (no leakage).
    X, y, usage = _load_all_rows(cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(cfg, X, y, usage)

    # 2. Clean train; fit the normalizer on train only, apply to train+val.
    X_train, y_train = clean_dataset(cfg, X_train, y_train)
    normalizer = build_normalizer(cfg).fit(X_train)
    X_train = normalizer.transform(X_train)
    X_val = normalizer.transform(X_val)

    # 3. Imbalance remedy on the training split only (class_weight or resample).
    X_train, y_train, class_weight = resolve_imbalance(cfg, X_train, y_train)

    # 4. tf.data pipelines (shuffle + augment on train only).
    train_ds = make_dataset(X_train, y_train, cfg, training=True)
    val_ds = make_dataset(X_val, y_val, cfg, training=False)

    # 4b. Optional hyperparameter search (stages.tuning) — pick the best config on
    # VALIDATION first, then train the final model with it. Rebuild the datasets so a
    # tuned batch_size takes effect. Off by default → normal runs are unchanged.
    if is_stage_on(cfg, "tuning"):
        cfg = _run_tuning(cfg, train_ds, val_ds, class_weight)
        train_ds = make_dataset(X_train, y_train, cfg, training=True)
        val_ds = make_dataset(X_val, y_val, cfg, training=False)

    # 5. Build + compile the model; assemble callbacks.
    model = build_model(cfg)
    callbacks = build_callbacks(cfg)

    _write_summary(model, cfg)

    # 6. Fit. Every choice above came from config → this run is fully reproducible.
    logger.info(
        f"Fitting for up to {cfg['model']['epochs']} epochs "
        f"(class_weight={'yes' if class_weight else 'no'})"
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["model"]["epochs"],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    # 7. Persist the best model + serialized history (for the #39 curves).
    _save_outputs(model, history, cfg)
    return history


def _artifact_paths(cfg: dict) -> tuple[str, str]:
    """Return (model_save_path, arch_txt_path), routed for transfer architectures.

    A ``transfer_*`` run writes to the separate ``pretrained_*`` paths so it never
    overwrites the from-scratch ``final_emotion_model.keras`` (Issue #46).
    """
    paths = cfg["paths"]
    if cfg["model"]["architecture"].startswith("transfer"):
        return paths["pretrained_model_save_path"], paths["pretrained_arch_txt_path"]
    return paths["model_save_path"], paths["arch_txt_path"]


def _write_summary(model: Any, cfg: dict) -> None:
    """Write model.summary() to the architecture txt path (transfer-aware)."""
    _, arch_txt_path = _artifact_paths(cfg)
    path = Path(arch_txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Architecture summary → {path}")


def _save_outputs(model: Any, history: Any, cfg: dict) -> None:
    """Save the final model and the history dict as JSON."""
    model_save_path, _ = _artifact_paths(cfg)
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)  # early stopping already restored the best weights

    # Transfer-aware (Issue #46): a transfer_* run writes pre_trained_history.json so it
    # never overwrites the from-scratch run's history.json / learning curves.
    history_path = Path(resolve_history_path(cfg))
    serializable = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    history_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    final_val = {
        k: round(v[-1], 4) for k, v in history.history.items() if k.startswith("val_")
    }
    logger.info(f"Saved model → {model_path}")
    logger.info(f"Saved history → {history_path}")
    logger.info(f"Final validation metrics: {final_val}")


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    run(cfg)


if __name__ == "__main__":
    # Optional config path arg, e.g. `python scripts/train.py config_transfer.yaml`
    # to train the transfer model without touching the default config.yaml.
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
