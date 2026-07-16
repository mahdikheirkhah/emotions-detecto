"""Unit tests for global reproducibility seeding (``set_global_seed``).

Guards the silent-bug class that hurts most in ML: a run that can't be reproduced. If
seeding regresses, two "identical" runs diverge and every ablation comparison becomes
noise, so we pin that the same seed reproduces the same draws (and a different seed does
not).
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from src.emotion_detector.utils.seeding import set_global_seed


def test_same_seed_reproduces_numpy_and_random() -> None:
    # Warm-up: the first-ever set_global_seed imports TensorFlow, whose one-time module
    # init advances Python's global random state *after* random.seed() runs. Trigger the
    # import once up front so both measured seedings start from the same clean state.
    set_global_seed(0)

    set_global_seed(123)
    np_a, py_a = np.random.rand(5), [random.random() for _ in range(5)]
    set_global_seed(123)
    np_b, py_b = np.random.rand(5), [random.random() for _ in range(5)]
    np.testing.assert_array_equal(np_a, np_b)
    assert py_a == py_b


def test_different_seed_changes_draws() -> None:
    set_global_seed(1)
    first = np.random.rand(10)
    set_global_seed(2)
    second = np.random.rand(10)
    assert not np.array_equal(first, second)


def test_sets_pythonhashseed_env() -> None:
    set_global_seed(77)
    assert os.environ["PYTHONHASHSEED"] == "77"


def test_seeds_tensorflow_when_available() -> None:
    tf = pytest.importorskip("tensorflow")
    set_global_seed(9)
    first = tf.random.uniform((6,)).numpy()
    set_global_seed(9)
    second = tf.random.uniform((6,)).numpy()
    np.testing.assert_array_equal(first, second)
