"""Config-driven evaluation — the honest scorecard for an imbalanced 7-class task.

Reports **more than accuracy**: overall accuracy, macro-F1 (every emotion weighted
equally, so a model can't hide by favouring "Happy"), a confusion matrix, and a
per-class precision/recall report. Which metrics to compute is driven by
``evaluation.metrics`` so the same scorecard is produced across ablation runs.

Only ``sklearn``/``numpy``/``matplotlib`` are used — the model is called through a
duck-typed ``.predict``, so this module needs no TensorFlow import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.emotion_detector.models.labels import FER_EMOTIONS  # single source of truth
from src.emotion_detector.utils.logging import logger


def _to_indices(y: NDArray) -> NDArray:
    """Turn one-hot ``(N, C)`` or integer ``(N,)`` labels into class indices."""
    y = np.asarray(y)
    return y.argmax(axis=1) if y.ndim == 2 else y.astype(int)


def _class_names(num_classes: int) -> List[str]:
    return FER_EMOTIONS if num_classes == 7 else [str(i) for i in range(num_classes)]


def plot_confusion_matrix(cm: NDArray, out_path, class_names: List[str]) -> Path:
    """Render a labelled confusion-matrix heatmap to *out_path*."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate(
    model: Any,
    X_test: NDArray,
    y_test: NDArray,
    cfg: dict,
    plot: bool = True,
) -> dict:
    """Score *model* on the test set and return a metrics dict.

    Computes the metrics listed in ``cfg["evaluation"]["metrics"]`` plus a
    per-class classification report. The confusion matrix is always built over
    all ``num_classes`` so it stays square even if a class is absent from the
    predictions.

    Args:
        model:  Anything with ``predict(X) -> (N, C)`` softmax scores.
        X_test: Test images.
        y_test: Test labels — one-hot ``(N, C)`` or integer ``(N,)``.
        cfg:    Loaded config dict.
        plot:   If True and ``confusion_matrix`` is requested, save the heatmap.

    Returns:
        Dict with the requested metrics, a ``classification_report`` (per-class
        precision/recall/F1), and — if plotted — ``confusion_matrix_path``.
    """
    num_classes = cfg["model"]["num_classes"]
    labels = list(range(num_classes))
    names = _class_names(num_classes)
    requested = cfg["evaluation"]["metrics"]

    y_true = _to_indices(y_test)
    probs = np.asarray(model.predict(X_test))
    y_pred = probs.argmax(axis=1)

    results: dict = {}
    if "accuracy" in requested:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "f1_macro" in requested:
        results["f1_macro"] = float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if "confusion_matrix" in requested:
        results["confusion_matrix"] = cm.tolist()

    # Per-class precision/recall/F1 is always reported (CONTRIBUTING §8).
    results["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )

    if plot and "confusion_matrix" in requested:
        cm_path = Path(cfg["paths"]["model_save_path"]).parent / "confusion_matrix.png"
        results["confusion_matrix_path"] = str(
            plot_confusion_matrix(cm, cm_path, names)
        )

    acc = results.get("accuracy")
    f1m = results.get("f1_macro")
    logger.info(
        f"Evaluation — accuracy={acc:.4f} macro-F1={f1m:.4f}"
        if acc is not None and f1m is not None
        else f"Evaluation metrics: {sorted(results)}"
    )
    return results
