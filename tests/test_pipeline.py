"""Unit tests for the tf.data input pipeline.

The whole module is skipped where TensorFlow is unavailable (both to_tensors and
make_dataset need Keras/tf.data).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.emotion_detector.data.pipeline import make_dataset, to_tensors


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(batch_size: int = 16, aug: bool = False, buffer: int = 1000) -> dict:
    return {
        "stages": {"augmentation": aug},
        "global": {"seed": 42},
        "model": {"batch_size": batch_size, "num_classes": 7},
        "pipeline": {"shuffle_buffer": buffer, "cache": True},
        "augmentation": {
            "strategy": "basic",
            "horizontal_flip": True,
            "rotation_range": 10,
            "zoom_range": 0.1,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
        },
    }


def _data(n: int = 50):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (n, 48, 48), dtype=np.uint8)
    y = rng.integers(0, 7, n)
    return X, y


def _labels_in_order(ds):
    return np.concatenate([by.numpy().argmax(axis=1) for _, by in ds])


# ---------------------------------------------------------------------------
# to_tensors
# ---------------------------------------------------------------------------

def test_to_tensors_adds_channel_axis() -> None:
    X, y = _data()
    images, _ = to_tensors(X, y)
    assert images.shape == (50, 48, 48, 1)
    assert images.dtype == np.float32


def test_to_tensors_one_hot_encodes() -> None:
    X, y = _data()
    _, onehot = to_tensors(X, y, num_classes=7)
    assert onehot.shape == (50, 7)
    assert np.allclose(onehot.sum(axis=1), 1.0)      # each row sums to 1
    assert np.array_equal(onehot.argmax(axis=1), y)  # argmax recovers the label


def test_to_tensors_passthrough_4d() -> None:
    X = np.zeros((5, 48, 48, 1), dtype=np.uint8)
    y = np.arange(5)
    images, _ = to_tensors(X, y)
    assert images.shape == (5, 48, 48, 1)


def test_to_tensors_bad_shape_raises() -> None:
    with pytest.raises(ValueError, match="3-D or 4-D"):
        to_tensors(np.zeros((5, 48)), np.arange(5))


def test_to_tensors_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        to_tensors(np.zeros((5, 48, 48)), np.arange(4))


# ---------------------------------------------------------------------------
# make_dataset — shapes
# ---------------------------------------------------------------------------

def test_train_batch_shape() -> None:
    X, y = _data()
    ds = make_dataset(X, y, _cfg(batch_size=16), training=True)
    bx, by = next(iter(ds))
    assert tuple(bx.shape) == (16, 48, 48, 1)
    assert tuple(by.shape) == (16, 7)


def test_eval_batch_shape() -> None:
    X, y = _data()
    ds = make_dataset(X, y, _cfg(batch_size=10), training=False)
    bx, by = next(iter(ds))
    assert tuple(bx.shape) == (10, 48, 48, 1)
    assert tuple(by.shape) == (10, 7)


def test_batch_size_is_config_driven() -> None:
    X, y = _data()
    ds = make_dataset(X, y, _cfg(batch_size=25), training=False)
    bx, _ = next(iter(ds))
    assert bx.shape[0] == 25


# ---------------------------------------------------------------------------
# make_dataset — shuffle behavior
# ---------------------------------------------------------------------------

def test_eval_dataset_not_shuffled() -> None:
    X, y = _data()
    ds = make_dataset(X, y, _cfg(), training=False)
    assert np.array_equal(_labels_in_order(ds), y)  # original order preserved


def test_train_dataset_is_shuffled() -> None:
    X, y = _data()
    ds = make_dataset(X, y, _cfg(buffer=1000), training=True)
    order = _labels_in_order(ds)
    assert not np.array_equal(order, y)  # order changed
    # same multiset → it's a permutation, nothing lost
    assert np.array_equal(np.sort(order), np.sort(y))


def test_train_augmentation_preserves_batch_shape() -> None:
    X, y = _data()
    ds = make_dataset(X, y, _cfg(batch_size=16, aug=True), training=True)
    bx, by = next(iter(ds))
    assert tuple(bx.shape) == (16, 48, 48, 1)
    assert tuple(by.shape) == (16, 7)
