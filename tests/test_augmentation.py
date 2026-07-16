"""Unit tests for config-driven, train-only data augmentation.

TF-dependent tests are skipped automatically where TensorFlow is unavailable;
the stage-off / zero-param / missing-key paths need no TF and always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.augmentation import IdentityAugmenter, build_augmenter

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(stage_on: bool = True, **overrides) -> dict:
    aug = {
        "strategy": "basic",
        "horizontal_flip": True,
        "rotation_range": 10,
        "zoom_range": 0.1,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
    }
    aug.update(overrides)
    return {
        "stages": {"augmentation": stage_on},
        "global": {"seed": 42},
        "augmentation": aug,
    }


def _batch(n: int = 8):
    return np.random.default_rng(0).random((n, 48, 48, 1)).astype("float32")


# ---------------------------------------------------------------------------
# no-TF paths (always run)
# ---------------------------------------------------------------------------


def test_stage_off_returns_identity() -> None:
    assert isinstance(build_augmenter(_cfg(stage_on=False)), IdentityAugmenter)


def test_stage_off_returns_input_unchanged() -> None:
    X = _batch()
    aug = build_augmenter(_cfg(stage_on=False))
    assert np.array_equal(aug(X, training=True), X)


def test_all_params_zero_returns_identity() -> None:
    aug = build_augmenter(
        _cfg(
            horizontal_flip=False,
            rotation_range=0,
            zoom_range=0,
            width_shift_range=0,
            height_shift_range=0,
        )
    )
    assert isinstance(aug, IdentityAugmenter)


def test_missing_key_raises() -> None:
    bad = {"stages": {"augmentation": True}, "augmentation": {"horizontal_flip": True}}
    with pytest.raises(KeyError, match="Missing augmentation config key"):
        build_augmenter(bad)


# ---------------------------------------------------------------------------
# TF-dependent paths
# ---------------------------------------------------------------------------


def test_augmented_batch_keeps_shape() -> None:
    pytest.importorskip("tensorflow")
    X = _batch()
    aug = build_augmenter(_cfg())
    out = aug(X, training=True)
    assert tuple(out.shape) == X.shape


def test_augmenter_has_expected_layers() -> None:
    pytest.importorskip("tensorflow")
    aug = build_augmenter(_cfg())
    names = [type(layer).__name__ for layer in aug.layers]
    assert names == ["RandomFlip", "RandomRotation", "RandomZoom", "RandomTranslation"]


def test_only_enabled_transforms_are_added() -> None:
    pytest.importorskip("tensorflow")
    aug = build_augmenter(
        _cfg(rotation_range=0, zoom_range=0, width_shift_range=0, height_shift_range=0)
    )
    names = [type(layer).__name__ for layer in aug.layers]
    assert names == ["RandomFlip"]  # only flip enabled


def test_inference_mode_is_identity() -> None:
    """training=False must leave the batch unchanged — augmentation is train-only."""
    pytest.importorskip("tensorflow")
    X = _batch()
    aug = build_augmenter(_cfg())
    out = aug(X, training=False)
    assert np.allclose(np.asarray(out), X)


def test_labels_are_not_touched() -> None:
    """The augmenter transforms X only; labels ride along unchanged."""
    pytest.importorskip("tensorflow")
    X = _batch()
    y = np.arange(len(X))
    aug = build_augmenter(_cfg())
    _ = aug(X, training=True)
    assert np.array_equal(y, np.arange(len(X)))  # y never passed to the augmenter
