"""Plot train/validation loss & accuracy learning curves.

Loads the ``history.json`` saved by ``scripts/train.py`` and renders
``results/model/learning_curves.png``: loss and accuracy vs epoch, with the
early-stopping epoch (the val-loss minimum that ``restore_best_weights`` keeps)
marked. This plot is the audit evidence that training stopped **before**
validation loss diverged (CONTRIBUTING §8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

# Make `src` importable when run as `python scripts/validation_loss_accuracy.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.models.classifier import resolve_history_path
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging


def _best_epoch(history: dict) -> Optional[int]:
    """Epoch index (0-based) of minimum val_loss — where early stopping restored."""
    val_loss = history.get("val_loss")
    if not val_loss:
        return None
    return int(min(range(len(val_loss)), key=lambda i: val_loss[i]))


def plot_learning_curves(history: dict, out_path) -> Path:
    """Render loss + accuracy learning curves to *out_path* and return the path.

    Args:
        history: Dict of metric-name -> per-epoch list (Keras ``History.history``).
        out_path: Destination PNG path.

    Returns:
        The written PNG ``Path``.

    Raises:
        ValueError: if *history* has no ``loss`` series to plot.
    """
    if "loss" not in history:
        raise ValueError("history has no 'loss' series to plot.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    best = _best_epoch(history)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- loss ---
    axes[0].plot(epochs, history["loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")

    # --- accuracy (Keras names it 'accuracy'/'val_accuracy') ---
    if "accuracy" in history:
        axes[1].plot(epochs, history["accuracy"], label="train")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")

    # --- mark the early-stopping epoch on both panels ---
    if best is not None:
        for ax in axes:
            ax.axvline(
                best + 1,
                ls="--",
                color="gray",
                label=f"early stop (best val_loss, epoch {best + 1})",
            )
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:  # only if the panel has plotted data
            ax.legend()

    fig.suptitle("Learning curves — training vs validation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)

    history_path = Path(resolve_history_path(cfg))  # transfer-aware (Issue #46)
    if not history_path.exists():
        raise FileNotFoundError(
            f"No history at {history_path}. Run scripts/train.py first."
        )
    history = json.loads(history_path.read_text(encoding="utf-8"))

    out = plot_learning_curves(history, cfg["paths"]["learning_curves_path"])
    best = _best_epoch(history)
    logger.info(f"Saved learning curves → {out}")
    if best is not None:
        logger.info(
            f"Best epoch (min val_loss): {best + 1} — early stopping restored these "
            "weights (training stopped before val loss diverged)."
        )


if __name__ == "__main__":
    main()
