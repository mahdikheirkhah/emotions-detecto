"""Optional, config-toggleable stratified k-fold cross-validation.

A single train/val split gives an estimate that depends on *which* samples landed
in validation. k-fold rotates the validation fold k times and averages → a
lower-variance estimate, with **stratified** folds preserving class ratios (vital
for FER-2013's imbalance).

For a deep CNN this is k× the training cost, so it defaults **off**
(``evaluation.cross_validation: false``) and is primarily for the lighter models.

The model is rebuilt fresh per fold via ``build_fn`` and scored with the #40
``evaluate``; only ``sklearn`` is required here (TF enters through ``build_fn``).
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Callable, Dict, List

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from src.emotion_detector.data.pipeline import to_tensors
from src.emotion_detector.models.evaluation import evaluate
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.seeding import set_global_seed


def _log_aggregate(results: List[Dict[str, Any]]) -> None:
    """Log mean ± std of accuracy / macro-F1 across the folds."""
    for key in ("accuracy", "f1_macro"):
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            m = mean(values)
            s = pstdev(values) if len(values) > 1 else 0.0
            logger.info(f"CV {key}: {m:.4f} ± {s:.4f} over {len(values)} folds")


def cross_validate(
    build_fn: Callable[[], Any],
    X: NDArray,
    y: NDArray,
    cfg: dict,
) -> List[Dict[str, Any]]:
    """Run stratified k-fold CV, returning one metrics dict per fold.

    Gated by ``evaluation.cross_validation``; when off, returns ``[]`` (skipped).

    Args:
        build_fn: Zero-arg factory returning a **fresh compiled** model each call
                  (e.g. ``lambda: build_model(cfg)``). Rebuilt per fold so folds
                  don't share weights.
        X:        Image array ``(N, H, W)`` / ``(N, H, W, 1)``.
        y:        Integer labels ``(N,)`` (used to stratify).
        cfg:      Loaded config dict.

    Returns:
        ``[{"fold": i, "accuracy": ..., "f1_macro": ...}, ...]`` — one per fold,
        or ``[]`` if cross-validation is disabled.
    """
    ev = cfg["evaluation"]
    if not ev.get("cross_validation", False):
        logger.info(
            "Cross-validation off (evaluation.cross_validation=false) — skipped."
        )
        return []

    k = ev.get("cv_folds", 5)
    seed = cfg["global"]["seed"]
    num_classes = cfg["model"]["num_classes"]
    epochs = cfg["model"]["epochs"]
    batch_size = cfg["model"]["batch_size"]

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    results: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        # Seed per fold (distinct but reproducible) so weight init + fit are
        # deterministic → the whole CV run is reproducible (CONTRIBUTING §8).
        set_global_seed(seed + fold)
        model = build_fn()  # fresh model per fold
        X_tr, y_tr = to_tensors(X[train_idx], y[train_idx], num_classes=num_classes)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        X_va, _ = to_tensors(X[val_idx], y[val_idx], num_classes=num_classes)
        scores = evaluate(model, X_va, y[val_idx], cfg, plot=False)

        entry = {
            "fold": fold,
            "accuracy": scores.get("accuracy"),
            "f1_macro": scores.get("f1_macro"),
        }
        logger.info(
            f"Fold {fold}/{k}: accuracy={entry['accuracy']:.4f} "
            f"macro-F1={entry['f1_macro']:.4f}"
        )
        results.append(entry)

    _log_aggregate(results)
    return results
