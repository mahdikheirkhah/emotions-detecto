"""Unit tests for the config-driven class-imbalance strategies."""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.base import BaseImbalanceStrategy
from src.emotion_detector.data.imbalance import (
    ClassWeightStrategy,
    NoResample,
    Oversampler,
    Undersampler,
    resolve_imbalance,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(strategy: str = "class_weight", handle: bool = True, seed: int = 42) -> dict:
    return {
        "global": {"seed": seed},
        "cleaning": {
            "handle_imbalance": handle,
            "imbalance_strategy": strategy,
        },
    }


def _imbalanced():
    """3 classes: 0 has 6 samples, 1 has 3, 2 has 1 (imbalance ratio 6)."""
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2])
    X = np.arange(len(y) * 4).reshape(len(y), 2, 2).astype(np.uint8)
    return X, y


def _counts(y):
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# base is abstract
# ---------------------------------------------------------------------------


def test_base_imbalance_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseImbalanceStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# NoResample
# ---------------------------------------------------------------------------


def test_no_resample_returns_data_unchanged() -> None:
    X, y = _imbalanced()
    X_out, y_out, cw = NoResample().apply(X, y)
    assert np.array_equal(X_out, X)
    assert np.array_equal(y_out, y)
    assert cw is None


# ---------------------------------------------------------------------------
# ClassWeightStrategy
# ---------------------------------------------------------------------------


def test_class_weight_leaves_data_unchanged() -> None:
    X, y = _imbalanced()
    X_out, y_out, cw = ClassWeightStrategy().apply(X, y)
    assert np.array_equal(X_out, X)
    assert np.array_equal(y_out, y)
    assert cw is not None


def test_class_weight_inverse_frequency() -> None:
    X, y = _imbalanced()  # counts: {0:6, 1:3, 2:1}, n=10, n_classes=3
    _, _, cw = ClassWeightStrategy().apply(X, y)
    # weight_c = n_samples / (n_classes * count_c)
    assert cw[0] == pytest.approx(10 / (3 * 6))
    assert cw[1] == pytest.approx(10 / (3 * 3))
    assert cw[2] == pytest.approx(10 / (3 * 1))
    # rarest class gets the largest weight
    assert cw[2] > cw[1] > cw[0]


# ---------------------------------------------------------------------------
# Oversampler
# ---------------------------------------------------------------------------


def test_oversample_balances_to_majority() -> None:
    X, y = _imbalanced()  # majority count = 6
    X_out, y_out, cw = Oversampler(seed=0).apply(X, y)
    assert cw is None
    assert _counts(y_out) == {0: 6, 1: 6, 2: 6}
    assert len(X_out) == len(y_out) == 18


def test_oversample_keeps_all_originals() -> None:
    X, y = _imbalanced()
    X_out, y_out, _ = Oversampler(seed=0).apply(X, y)
    # every original sample value still present
    orig_first = set(X.reshape(len(X), -1)[:, 0].tolist())
    out_first = set(X_out.reshape(len(X_out), -1)[:, 0].tolist())
    assert orig_first.issubset(out_first)


def test_oversample_is_seeded_reproducible() -> None:
    X, y = _imbalanced()
    a = Oversampler(seed=7).apply(X, y)[1]
    b = Oversampler(seed=7).apply(X, y)[1]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Undersampler
# ---------------------------------------------------------------------------


def test_undersample_balances_to_minority() -> None:
    X, y = _imbalanced()  # minority count = 1
    X_out, y_out, cw = Undersampler(seed=0).apply(X, y)
    assert cw is None
    assert _counts(y_out) == {0: 1, 1: 1, 2: 1}
    assert len(X_out) == len(y_out) == 3


def test_undersample_no_replacement() -> None:
    X, y = _imbalanced()
    X_out, _, _ = Undersampler(seed=0).apply(X, y)
    # all chosen rows are distinct (sampled without replacement)
    flat = X_out.reshape(len(X_out), -1)
    unique_rows = np.unique(flat, axis=0)
    assert len(unique_rows) == len(flat)


# ---------------------------------------------------------------------------
# resolve_imbalance dispatch
# ---------------------------------------------------------------------------


def test_resolve_class_weight() -> None:
    X, y = _imbalanced()
    X_out, y_out, cw = resolve_imbalance(_cfg("class_weight"), X, y)
    assert cw is not None
    assert np.array_equal(X_out, X)


def test_resolve_oversample() -> None:
    X, y = _imbalanced()
    _, y_out, cw = resolve_imbalance(_cfg("oversample"), X, y)
    assert cw is None
    assert _counts(y_out) == {0: 6, 1: 6, 2: 6}


def test_resolve_undersample() -> None:
    X, y = _imbalanced()
    _, y_out, cw = resolve_imbalance(_cfg("undersample"), X, y)
    assert cw is None
    assert _counts(y_out) == {0: 1, 1: 1, 2: 1}


def test_resolve_none_returns_unchanged() -> None:
    X, y = _imbalanced()
    X_out, y_out, cw = resolve_imbalance(_cfg("none"), X, y)
    assert np.array_equal(y_out, y)
    assert cw is None


def test_resolve_handle_imbalance_off_is_noop() -> None:
    X, y = _imbalanced()
    # strategy says oversample, but master toggle is off → no change
    X_out, y_out, cw = resolve_imbalance(_cfg("oversample", handle=False), X, y)
    assert np.array_equal(y_out, y)
    assert cw is None


def test_resolve_unknown_strategy_raises() -> None:
    X, y = _imbalanced()
    with pytest.raises(ValueError, match="Unknown option"):
        resolve_imbalance(_cfg("smote"), X, y)


def test_resolve_missing_key_raises() -> None:
    X, y = _imbalanced()
    bad = {"cleaning": {"handle_imbalance": True}}  # missing keys
    with pytest.raises(KeyError, match="Missing imbalance config key"):
        resolve_imbalance(bad, X, y)


def test_resolve_seeded_from_config() -> None:
    X, y = _imbalanced()
    a = resolve_imbalance(_cfg("oversample", seed=5), X, y)[1]
    b = resolve_imbalance(_cfg("oversample", seed=5), X, y)[1]
    assert np.array_equal(a, b)
