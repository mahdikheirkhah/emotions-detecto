"""Canonical, no-leakage train / validation / test splits (CONTRIBUTING §8).

The test set is **fixed by the FER-2013 ``Usage`` column** (PublicTest +
PrivateTest = the audit's ``test_with_emotions.csv``); the validation set is carved
**stratified** out of the Training rows. The split is seeded and fixed so every
ablation trains and evaluates on identical data.

This module only *partitions* the data. Fitting any transform (standardization
stats, PCA) happens **after** the split, on the training split only — computing
statistics on the full set would leak test information into training.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from src.emotion_detector.utils.logging import logger

_TRAINING = "Training"


def _class_balance(y: NDArray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(c): f"{n} ({n / total:.1%})" for c, n in zip(classes, counts)}


def make_splits(
    cfg: dict, X: NDArray, y: NDArray, usage: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Partition ``(X, y)`` into train / validation / test with no leakage.

    Test = every row whose ``usage`` is **not** ``"Training"`` (i.e. PublicTest +
    PrivateTest). Validation = a stratified ``split.val_size`` fraction carved out
    of the Training rows; train = the remaining Training rows.

    Args:
        cfg:   Loaded config dict (reads ``split.val_size``, ``split.stratify``,
               ``global.seed``).
        X:     Image array of shape ``(N, ...)`` for all rows.
        y:     Integer label array of shape ``(N,)``.
        usage: String array of shape ``(N,)`` with the ``Usage`` of each row.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.

    Raises:
        ValueError: if the inputs have mismatched lengths, if there are no
            Training rows to split, or if a required config key is missing.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    usage = np.asarray(usage)
    if not (len(X) == len(y) == len(usage)):
        raise ValueError(f"X/y/usage length mismatch: {len(X)}, {len(y)}, {len(usage)}")

    try:
        val_size = cfg["split"]["val_size"]
        stratify_on = cfg["split"]["stratify"]
        seed = cfg["global"]["seed"]
    except KeyError as exc:
        raise ValueError(
            f"Missing split config key: {exc}. "
            "Check the 'split:' / 'global:' sections in config.yaml."
        ) from exc

    # Split on indices so we can assert the partitions are disjoint.
    all_idx = np.arange(len(X))
    test_idx = all_idx[usage != _TRAINING]
    pool_idx = all_idx[usage == _TRAINING]
    if len(pool_idx) == 0:
        raise ValueError("No rows with Usage == 'Training' to build train/val from.")

    stratify = y[pool_idx] if stratify_on else None
    train_idx, val_idx = train_test_split(
        pool_idx, test_size=val_size, random_state=seed, stratify=stratify
    )

    # No-leakage guarantee: the three index sets must be pairwise disjoint.
    assert set(train_idx).isdisjoint(val_idx), "train/val overlap"
    assert set(train_idx).isdisjoint(test_idx), "train/test overlap"
    assert set(val_idx).isdisjoint(test_idx), "val/test overlap"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(
        f"Splits — train={len(train_idx):,}, val={len(val_idx):,}, "
        f"test={len(test_idx):,} (test fixed by Usage; val stratified, "
        f"val_size={val_size}, seed={seed})"
    )
    logger.info(f"train class balance: {_class_balance(y_train)}")
    logger.info(f"val   class balance: {_class_balance(y_val)}")

    return X_train, y_train, X_val, y_val, X_test, y_test
