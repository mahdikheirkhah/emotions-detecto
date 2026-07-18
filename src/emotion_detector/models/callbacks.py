"""Config-driven training callbacks — the anti-overfitting toolkit (CONTRIBUTING §8).

Builds the three callbacks the subject requires:

* **EarlyStopping** — halt when validation loss stops improving, and restore the
  best weights (regularization: stop before the model memorizes the train set).
* **ModelCheckpoint** — persist the best model to disk, so a crash or a late-epoch
  overfit never loses the good weights. The target is transfer-aware
  (``resolve_model_path``): a ``transfer_*`` run checkpoints to its own
  ``pretrained_*`` path and never overwrites the from-scratch model.
* **ReduceLROnPlateau** — drop the learning rate when val loss plateaus, which can
  unlock further improvement by taking finer steps.

Also wires in **TensorBoard** (mandatory per the subject): each run logs scalars
(loss/accuracy/LR), the model graph, and weight histograms to a per-run,
config-tagged directory, so runs overlay as a side-by-side ablation dashboard.

Launch the viewer with::

    tensorboard --logdir logs/tensorboard

All thresholds/patience come from ``config.yaml``. Early stopping can be **ablated**
(``callbacks.early_stopping: false``) to *see* the overfitting it prevents.
TensorFlow is imported lazily.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List

from src.emotion_detector.models.classifier import resolve_model_path
from src.emotion_detector.utils.logging import logger


def _run_name(cfg: dict) -> str:
    """Build a timestamped, config-tagged run name for the TensorBoard log dir.

    Tagging with the active config (architecture, normalization, aug/clean on-off)
    makes each run self-describing, so ablation runs overlay cleanly in TensorBoard.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stages = cfg.get("stages", {})
    tags = [
        cfg["model"]["architecture"],
        f"norm-{cfg['preprocessing']['normalization']}",
        "aug" if stages.get("augmentation") else "noaug",
        "clean" if stages.get("cleaning") else "noclean",
    ]
    return f"{ts}_{'_'.join(tags)}"


def build_callbacks(cfg: dict) -> List[Any]:
    """Assemble the training callbacks from config.

    Args:
        cfg: Loaded config dict. Reads the ``callbacks`` block, ``paths.*``, and —
             transfer-aware via ``resolve_model_path`` — ``model.architecture`` to
             route the checkpoint: a ``transfer_*`` run checkpoints to
             ``pretrained_model_save_path``, never the from-scratch model.

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
        use_tensorboard = c.get("tensorboard", True)
        es_patience = c["early_stopping_patience"]
        lr_patience = c["reduce_lr_patience"]
        lr_factor = c["reduce_lr_factor"]
        min_lr = c["min_lr"]
        # Transfer-aware: a transfer_* run checkpoints to its own pretrained_* path,
        # so it never overwrites the from-scratch final_emotion_model.keras (Issue #46).
        save_path = Path(resolve_model_path(cfg))
        tb_root = Path(cfg["paths"]["tensorboard_dir"])
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

    if use_tensorboard:
        run_dir = tb_root / _run_name(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            cb.TensorBoard(
                log_dir=str(run_dir),
                histogram_freq=1,  # weight histograms each epoch
                write_graph=True,
            )
        )
        logger.info(f"TensorBoard → {run_dir}")
        logger.info(f"  launch: tensorboard --logdir {tb_root}")

    logger.info(
        f"Callbacks — early_stopping={use_early_stopping} (patience={es_patience}), "
        f"checkpoint→{save_path}, reduce_lr(factor={lr_factor}, "
        f"patience={lr_patience}), tensorboard={use_tensorboard}, monitor={monitor}"
    )
    return callbacks
