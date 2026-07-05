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


def run(cfg: dict) -> Any:
    """Execute one training run from a loaded config dict.

    Returns the Keras ``History`` object. Every stage self-gates on its toggle
    (cleaning, preprocessing, augmentation, …), so this conductor just calls them
    in order.
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

    history_path = model_path.parent / "history.json"
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
    main()
