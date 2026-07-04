"""Unit tests for the optional PCA decomposition stage."""
from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.base import BaseDecomposer
from src.emotion_detector.data.decomposition import (
    IdentityReducer,
    PcaReducer,
    build_decomposer,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(stage_on: bool = False, n_components=0.95) -> dict:
    return {
        "stages": {"decomposition": stage_on},
        "global": {"seed": 42},
        "decomposition": {"method": "pca", "n_components": n_components},
    }


def _images(n: int = 60, size: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, size, size)).astype(np.uint8)


# ---------------------------------------------------------------------------
# base is abstract
# ---------------------------------------------------------------------------

def test_base_decomposer_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseDecomposer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# IdentityReducer
# ---------------------------------------------------------------------------

def test_identity_returns_data_unchanged() -> None:
    X = _images()
    out = IdentityReducer().fit(X).transform(X)
    assert np.array_equal(out, X)
    assert out.shape == X.shape  # keeps 2-D layout for the CNN


def test_identity_fit_returns_self() -> None:
    r = IdentityReducer()
    assert r.fit(_images()) is r


# ---------------------------------------------------------------------------
# PcaReducer
# ---------------------------------------------------------------------------

def test_pca_int_components_shape() -> None:
    X = _images()  # (60, 8, 8) → 64 features
    reducer = PcaReducer(n_components=5).fit(X)
    out = reducer.transform(X)
    assert out.shape == (60, 5)
    assert reducer.n_components_ == 5


def test_pca_float_components_keeps_variance() -> None:
    X = _images()
    reducer = PcaReducer(n_components=0.9).fit(X)
    assert reducer.explained_variance_ratio_.sum() >= 0.9


def test_pca_fit_on_train_transforms_val_same_width() -> None:
    X_train = _images(seed=1)
    X_val = _images(n=20, seed=2)
    reducer = PcaReducer(n_components=5).fit(X_train)
    val_out = reducer.transform(X_val)
    assert val_out.shape == (20, 5)  # val projected onto TRAIN components


def test_pca_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        PcaReducer(n_components=5).transform(_images())


def test_pca_explained_variance_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="after fit"):
        _ = PcaReducer(n_components=5).explained_variance_ratio_


def test_pca_fit_returns_self() -> None:
    r = PcaReducer(n_components=5)
    assert r.fit(_images()) is r


def test_pca_flattens_images() -> None:
    X = _images()  # 3-D input
    out = PcaReducer(n_components=3).fit(X).transform(X)
    assert out.ndim == 2  # (N, n_components)


# ---------------------------------------------------------------------------
# build_decomposer dispatch
# ---------------------------------------------------------------------------

def test_build_decomposer_stage_off_returns_identity() -> None:
    assert isinstance(build_decomposer(_cfg(stage_on=False)), IdentityReducer)


def test_build_decomposer_stage_on_returns_pca() -> None:
    assert isinstance(build_decomposer(_cfg(stage_on=True, n_components=5)), PcaReducer)


def test_build_decomposer_off_passes_data_through() -> None:
    X = _images()
    reducer = build_decomposer(_cfg(stage_on=False))
    assert np.array_equal(reducer.fit(X).transform(X), X)


def test_build_decomposer_missing_key_raises() -> None:
    bad = {
        "stages": {"decomposition": True},
        "global": {"seed": 42},
        "decomposition": {},
    }
    with pytest.raises(KeyError, match="Missing decomposition config key"):
        build_decomposer(bad)
