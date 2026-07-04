"""Unit tests for the config-driven normalization strategies."""
from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.preprocessing import (
    IdentityPreprocessor,
    RescalePreprocessor,
    StandardizePreprocessor,
    build_normalizer,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(strategy: str = "rescale", stage_on: bool = True) -> dict:
    return {
        "stages": {"preprocessing": stage_on},
        "preprocessing": {"normalization": strategy},
    }


def _images(seed: int = 0, n: int = 20):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 8, 8), dtype=np.uint8)


# ---------------------------------------------------------------------------
# IdentityPreprocessor
# ---------------------------------------------------------------------------

def test_identity_returns_values_unchanged() -> None:
    X = _images()
    out = IdentityPreprocessor().fit(X).transform(X)
    assert np.array_equal(out, X.astype(np.float32))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# RescalePreprocessor
# ---------------------------------------------------------------------------

def test_rescale_maps_to_unit_range() -> None:
    X = _images()
    out = RescalePreprocessor().fit(X).transform(X)
    assert out.max() <= 1.0
    assert out.min() >= 0.0
    assert out.dtype == np.float32


def test_rescale_exact_values() -> None:
    X = np.array([[[0, 255], [128, 64]]], dtype=np.uint8)
    out = RescalePreprocessor().transform(X)
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[0, 0, 1] == pytest.approx(1.0)
    assert out[0, 1, 0] == pytest.approx(128 / 255)


def test_rescale_fit_is_noop_returns_self() -> None:
    p = RescalePreprocessor()
    assert p.fit(_images()) is p


# ---------------------------------------------------------------------------
# StandardizePreprocessor
# ---------------------------------------------------------------------------

def test_standardize_zero_mean_unit_std_on_train() -> None:
    X = _images()
    out = StandardizePreprocessor().fit(X).transform(X)
    assert out.mean() == pytest.approx(0.0, abs=1e-4)
    assert out.std() == pytest.approx(1.0, abs=1e-4)


def test_standardize_uses_train_stats_on_val() -> None:
    """val is transformed with TRAIN mean/std → val mean is generally not 0."""
    X_train = _images(seed=1) // 2  # darker train (mean ~64)
    X_val = _images(seed=2)  # brighter val (mean ~128)
    p = StandardizePreprocessor().fit(X_train)
    val_out = p.transform(X_val)
    # if train stats are reused, brighter val standardizes to a positive mean
    assert val_out.mean() > 0.5


def test_standardize_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        StandardizePreprocessor().transform(_images())


def test_standardize_constant_data_no_divide_by_zero() -> None:
    X = np.full((5, 8, 8), 100, dtype=np.uint8)  # std == 0
    out = StandardizePreprocessor().fit(X).transform(X)
    assert np.all(np.isfinite(out))  # no inf/nan
    assert np.allclose(out, 0.0)  # (100 - 100) / 1.0


def test_standardize_fit_returns_self() -> None:
    p = StandardizePreprocessor()
    assert p.fit(_images()) is p


# ---------------------------------------------------------------------------
# build_normalizer dispatch
# ---------------------------------------------------------------------------

def test_build_normalizer_none() -> None:
    assert isinstance(build_normalizer(_cfg("none")), IdentityPreprocessor)


def test_build_normalizer_rescale() -> None:
    assert isinstance(build_normalizer(_cfg("rescale")), RescalePreprocessor)


def test_build_normalizer_standardize() -> None:
    assert isinstance(build_normalizer(_cfg("standardize")), StandardizePreprocessor)


def test_build_normalizer_stage_off_returns_identity() -> None:
    # strategy says standardize, but the stage is off → raw-pixel passthrough
    norm = build_normalizer(_cfg("standardize", stage_on=False))
    assert isinstance(norm, IdentityPreprocessor)


def test_build_normalizer_unknown_option_raises() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        build_normalizer(_cfg("minmax"))


def test_build_normalizer_missing_key_raises() -> None:
    bad = {"stages": {"preprocessing": True}, "preprocessing": {}}
    with pytest.raises(KeyError, match="Missing preprocessing config key"):
        build_normalizer(bad)
