"""Unit tests for the no-leakage train/val/test split."""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.splits import make_splits

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(val_size: float = 0.2, stratify: bool = True, seed: int = 42) -> dict:
    return {
        "split": {"val_size": val_size, "stratify": stratify},
        "global": {"seed": seed},
    }


def _data(n_train: int = 100, n_test: int = 40):
    """Images whose [0,0] pixel is a unique id so we can track where each row lands."""
    n = n_train + n_test
    assert n < 256, "keep ids in uint8 range"
    X = np.stack([np.full((4, 4), i, dtype=np.uint8) for i in range(n)])
    # imbalanced but with >=2 per class in the training pool for stratification
    y_train = np.array([i % 5 for i in range(n_train)])
    y_test = np.array([i % 5 for i in range(n_test)])
    y = np.concatenate([y_train, y_test])
    usage = np.array(["Training"] * n_train + ["PublicTest"] * n_test)
    return X, y, usage


def _ids(X):
    return set(int(img[0, 0]) for img in X)


def _ratios(y):
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): n / counts.sum() for c, n in zip(classes, counts)}


# ---------------------------------------------------------------------------
# split composition
# ---------------------------------------------------------------------------


def test_test_set_comes_from_usage() -> None:
    X, y, usage = _data(n_train=100, n_test=40)
    _, _, _, _, X_test, _ = make_splits(_cfg(), X, y, usage)
    assert len(X_test) == 40  # all non-Training rows


def test_train_val_carved_from_training() -> None:
    X, y, usage = _data(n_train=100, n_test=40)
    X_tr, _, X_val, _, _, _ = make_splits(_cfg(val_size=0.2), X, y, usage)
    assert len(X_tr) == 80
    assert len(X_val) == 20  # 0.2 * 100


# ---------------------------------------------------------------------------
# no leakage — disjoint + complete
# ---------------------------------------------------------------------------


def test_splits_are_disjoint() -> None:
    X, y, usage = _data()
    X_tr, _, X_val, _, X_test, _ = make_splits(_cfg(), X, y, usage)
    tr, val, test = _ids(X_tr), _ids(X_val), _ids(X_test)
    assert tr.isdisjoint(val)
    assert tr.isdisjoint(test)
    assert val.isdisjoint(test)


def test_splits_cover_all_rows() -> None:
    X, y, usage = _data()
    X_tr, _, X_val, _, X_test, _ = make_splits(_cfg(), X, y, usage)
    union = _ids(X_tr) | _ids(X_val) | _ids(X_test)
    assert union == _ids(X)  # nothing lost or duplicated


# ---------------------------------------------------------------------------
# stratification
# ---------------------------------------------------------------------------


def test_stratification_preserves_class_ratios() -> None:
    X, y, usage = _data(n_train=100, n_test=40)
    _, y_tr, _, y_val, _, _ = make_splits(_cfg(stratify=True), X, y, usage)
    tr, val = _ratios(y_tr), _ratios(y_val)
    for cls in tr:
        assert abs(tr[cls] - val[cls]) < 0.05  # ratios nearly identical


def test_stratify_off_still_runs() -> None:
    X, y, usage = _data()
    result = make_splits(_cfg(stratify=False), X, y, usage)
    assert len(result[3]) == 20  # y_val


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_deterministic_under_fixed_seed() -> None:
    X, y, usage = _data()
    a = make_splits(_cfg(seed=7), X, y, usage)
    b = make_splits(_cfg(seed=7), X, y, usage)
    assert _ids(a[0]) == _ids(b[0])  # identical train ids
    assert _ids(a[2]) == _ids(b[2])  # identical val ids


def test_different_seed_changes_val_membership() -> None:
    X, y, usage = _data()
    a = make_splits(_cfg(seed=1), X, y, usage)
    b = make_splits(_cfg(seed=2), X, y, usage)
    assert _ids(a[2]) != _ids(b[2])  # different val split


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    X, y, usage = _data()
    with pytest.raises(ValueError, match="length mismatch"):
        make_splits(_cfg(), X, y[:-1], usage)


def test_no_training_rows_raises() -> None:
    X, y, usage = _data(n_train=100, n_test=40)
    usage = np.array(["PublicTest"] * len(usage))  # no Training rows
    with pytest.raises(ValueError, match="No rows with Usage"):
        make_splits(_cfg(), X, y, usage)


def test_missing_config_key_raises() -> None:
    X, y, usage = _data()
    with pytest.raises(ValueError, match="Missing split config key"):
        make_splits({"split": {"val_size": 0.2}, "global": {}}, X, y, usage)
