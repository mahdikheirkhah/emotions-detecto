"""Unit tests for optional stratified k-fold cross-validation (skipped w/o TF)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras

from src.emotion_detector.models.cross_val import cross_validate


def _build_fn():
    """Tiny fresh compiled model — GAP head keeps folds fast."""
    model = keras.Sequential(
        [
            keras.Input((48, 48, 1)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(7, activation="softmax"),
        ]
    )
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model


def _cfg(cross_validation=True, cv_folds=3) -> dict:
    return {
        "global": {"seed": 42},
        "model": {"num_classes": 7, "epochs": 1, "batch_size": 16},
        "evaluation": {
            "metrics": ["accuracy", "f1_macro", "confusion_matrix"],
            "cross_validation": cross_validation,
            "cv_folds": cv_folds,
        },
        "paths": {"model_save_path": "/tmp/unused/final.keras"},
    }


def _data(per_class=10):
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(7), per_class)
    X = rng.integers(0, 256, (len(y), 48, 48), dtype=np.uint8)
    return X, y


# ---------------------------------------------------------------------------
# toggle
# ---------------------------------------------------------------------------


def test_disabled_returns_empty() -> None:
    X, y = _data()
    assert cross_validate(_build_fn, X, y, _cfg(cross_validation=False)) == []


# ---------------------------------------------------------------------------
# k folds
# ---------------------------------------------------------------------------


def test_produces_k_result_entries() -> None:
    X, y = _data()
    results = cross_validate(_build_fn, X, y, _cfg(cv_folds=3))
    assert len(results) == 3
    assert [r["fold"] for r in results] == [1, 2, 3]


def test_each_entry_has_accuracy_and_f1() -> None:
    X, y = _data()
    results = cross_validate(_build_fn, X, y, _cfg(cv_folds=2))
    for r in results:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["f1_macro"] <= 1.0


def test_cv_folds_is_config_driven() -> None:
    X, y = _data()
    assert len(cross_validate(_build_fn, X, y, _cfg(cv_folds=2))) == 2
    assert len(cross_validate(_build_fn, X, y, _cfg(cv_folds=5))) == 5


def test_deterministic_folds_under_seed() -> None:
    X, y = _data()
    a = cross_validate(_build_fn, X, y, _cfg(cv_folds=3))
    b = cross_validate(_build_fn, X, y, _cfg(cv_folds=3))
    # same seed → StratifiedKFold produces the same partition → same accuracies
    assert [r["accuracy"] for r in a] == [r["accuracy"] for r in b]
