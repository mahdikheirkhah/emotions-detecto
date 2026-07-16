"""Config-driven class-imbalance remedies (see data.md §3.2).

Four strategies behind the Ablation-Driven dispatch: ``none``, ``class_weight``,
``oversample``, ``undersample``. All apply to the **training split only** — never
validation/test. ``class_weight`` returns a weight dict for ``model.fit``; the
resamplers return a rebalanced training set.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_class_weight

from src.emotion_detector.data.base import BaseImbalanceStrategy
from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.logging import logger


def _class_counts(y: NDArray) -> Dict[int, int]:
    """Return an ordered ``{class: count}`` map for logging."""
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}


class NoResample(BaseImbalanceStrategy):
    """Leave the training set untouched (raw-distribution baseline)."""

    def apply(
        self, X: NDArray, y: NDArray
    ) -> Tuple[NDArray, NDArray, Optional[Dict[int, float]]]:
        logger.info(
            f"Imbalance strategy 'none' — class counts unchanged: {_class_counts(y)}"
        )
        return X, y, None


class ClassWeightStrategy(BaseImbalanceStrategy):
    """Compute inverse-frequency class weights; leave the data unchanged.

    Uses sklearn's ``compute_class_weight("balanced")`` which sets each weight to
    ``n_samples / (n_classes * count_c)`` — so rarer classes get proportionally
    larger weights, rebalancing the per-sample loss without changing the data.
    """

    def apply(
        self, X: NDArray, y: NDArray
    ) -> Tuple[NDArray, NDArray, Optional[Dict[int, float]]]:
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info(f"Imbalance strategy 'class_weight' — weights: {class_weight}")
        return X, y, class_weight


class Oversampler(BaseImbalanceStrategy):
    """Oversample every minority class up to the majority count (seeded).

    Keeps all original samples and draws extra minority samples *with
    replacement* until every class matches the largest class. Risk: duplicated
    minority rows can encourage overfitting (data.md §3.2).
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def apply(
        self, X: NDArray, y: NDArray
    ) -> Tuple[NDArray, NDArray, Optional[Dict[int, float]]]:
        classes, counts = np.unique(y, return_counts=True)
        target = int(counts.max())
        parts = []
        for c in classes:
            idx_c = np.where(y == c)[0]
            n_extra = target - len(idx_c)
            if n_extra > 0:
                extra = self._rng.choice(idx_c, size=n_extra, replace=True)
                parts.append(np.concatenate([idx_c, extra]))
            else:
                parts.append(idx_c)
        indices = np.concatenate(parts)
        self._rng.shuffle(indices)
        X_out, y_out = X[indices], y[indices]
        logger.info(
            f"Imbalance strategy 'oversample' — every class → {target}; "
            f"{len(y):,} → {len(y_out):,} samples."
        )
        return X_out, y_out, None


class Undersampler(BaseImbalanceStrategy):
    """Undersample every majority class down to the minority count (seeded).

    Draws ``min_count`` samples *without replacement* from each class. Risk:
    discards real majority samples → information loss (data.md §3.2).
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def apply(
        self, X: NDArray, y: NDArray
    ) -> Tuple[NDArray, NDArray, Optional[Dict[int, float]]]:
        classes, counts = np.unique(y, return_counts=True)
        target = int(counts.min())
        parts = []
        for c in classes:
            idx_c = np.where(y == c)[0]
            chosen = self._rng.choice(idx_c, size=target, replace=False)
            parts.append(chosen)
        indices = np.concatenate(parts)
        self._rng.shuffle(indices)
        X_out, y_out = X[indices], y[indices]
        logger.info(
            f"Imbalance strategy 'undersample' — every class → {target}; "
            f"{len(y):,} → {len(y_out):,} samples."
        )
        return X_out, y_out, None


def resolve_imbalance(
    cfg: dict, X_train: NDArray, y_train: NDArray
) -> Tuple[NDArray, NDArray, Optional[Dict[int, float]]]:
    """Apply the configured imbalance remedy to the **training split only**.

    Reads ``cleaning.handle_imbalance`` (master toggle) and
    ``cleaning.imbalance_strategy`` (which remedy), and seeds resampling from
    ``global.seed`` for reproducibility. Never call this with validation/test
    data — reweighting or resampling them would corrupt the evaluation.

    Args:
        cfg:     Loaded config dict.
        X_train: Training image array.
        y_train: Training label array.

    Returns:
        ``(X_out, y_out, class_weight)`` — pass ``class_weight`` straight to
        ``model.fit`` (it is ``None`` for the resampling strategies).

    Raises:
        ValueError: if ``imbalance_strategy`` is not a known option.
        KeyError:   if required config keys are missing.
    """
    try:
        handle = cfg["cleaning"]["handle_imbalance"]
        strategy_name = cfg["cleaning"]["imbalance_strategy"]
        seed = cfg["global"]["seed"]
    except KeyError as exc:
        raise KeyError(
            f"Missing imbalance config key: {exc}. "
            "Check the 'cleaning:' / 'global:' sections in config.yaml."
        ) from exc

    if not handle:
        logger.info("handle_imbalance is off — training distribution left as-is.")
        return NoResample().apply(X_train, y_train)

    registry = {
        "none": NoResample,
        "class_weight": ClassWeightStrategy,
        "oversample": lambda: Oversampler(seed),
        "undersample": lambda: Undersampler(seed),
    }
    strategy = dispatch(strategy_name, registry)
    return strategy.apply(X_train, y_train)
