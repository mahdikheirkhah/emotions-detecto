"""Config-driven training callbacks — the anti-overfitting toolkit (CONTRIBUTING §8).

Builds the three callbacks the subject requires:

* **EarlyStopping** — halt when validation loss stops improving, and restore the
  best weights (regularization: stop before the model memorizes the train set).
* **ModelCheckpoint** — persist the best model to disk, so a crash or a late-epoch
  overfit never loses the good weights.
* **ReduceLROnPlateau** — drop the learning rate when val loss plateaus, which can
  unlock further improvement by taking finer steps.

All thresholds/patience come from ``config.yaml``. Early stopping can be **ablated**
(``callbacks.early_stopping: false``) to *see* the overfitting it prevents.
The TensorBoard callback is added in #37. TensorFlow is imported lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from src.emotion_detector.utils.logging import logger


def build_callbacks(cfg: dict) -> List[Any]:
    """Assemble the training callbacks from config.

    Args:
        cfg: Loaded config dict (reads the ``callbacks`` block and
             ``paths.model_save_path``).

    Returns:
        A list of ``keras.callbacks.Callback`` — EarlyStopping (unless ablated),
        ModelCheckpoint, and ReduceLROnPlateau.

    Raises:
        KeyError: if a required callbacks / paths config key is missing.
    """
    from tensorflow.keras import callbacks as cb

    try:
        c = cfg["callbacks"]
        monitor = c.get("monitor", "val_loss")
        use_early_stopping = c.get("early_stopping", True)
        es_patience = c["early_stopping_patience"]
        lr_patience = c["reduce_lr_patience"]
        lr_factor = c["reduce_lr_factor"]
        min_lr = c["min_lr"]
        save_path = Path(cfg["paths"]["model_save_path"])
    except KeyError as exc:
        raise KeyError(
            f"Missing callbacks/paths config key: {exc}. "
            "Check the 'callbacks:' and 'paths:' sections in config.yaml."
        ) from exc

    save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks: List[Any] = []

    if use_early_stopping:
        callbacks.append(
            cb.EarlyStopping(
                monitor=monitor,
                patience=es_patience,
                restore_best_weights=True,  # keep the best epoch, not the last
                verbose=1,
            )
        )
    else:
        logger.warning(
            "Early stopping is OFF (callbacks.early_stopping=false) — training runs "
            "all epochs; expect the val-loss curve to diverge (overfitting)."
        )

    callbacks.append(
        cb.ModelCheckpoint(
            filepath=str(save_path),
            monitor=monitor,
            save_best_only=True,  # only overwrite when the monitored metric improves
            verbose=1,
        )
    )
    callbacks.append(
        cb.ReduceLROnPlateau(
            monitor=monitor,
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        )
    )

    logger.info(
        f"Callbacks — early_stopping={use_early_stopping} (patience={es_patience}), "
        f"checkpoint→{save_path}, reduce_lr(factor={lr_factor}, "
        f"patience={lr_patience}), monitor={monitor}"
    )
    return callbacks
