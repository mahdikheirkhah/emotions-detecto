"""Unit tests for histogram-equalization / CLAHE preprocessing strategies."""
from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.preprocessing import (
    ClaheEqualizer,
    HistogramEqualizer,
    RescalePreprocessor,
    SequentialPreprocessor,
    build_normalizer,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _low_contrast_batch(n: int = 4, size: int = 48, seed: int = 0):
    """A batch of low-contrast 48x48 images (intensities squeezed into [100, 140])."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 41, (n, size, size)) + 100).astype(np.uint8)


def _hist_entropy(img) -> float:
    """Shannon entropy of the 256-bin intensity histogram (higher = flatter)."""
    counts, _ = np.histogram(img, bins=256, range=(0, 256))
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _cfg(strategy: str, clip=2.0, tile=8) -> dict:
    return {
        "stages": {"preprocessing": True},
        "preprocessing": {
            "normalization": strategy,
            "clahe_clip_limit": clip,
            "clahe_tile_grid": tile,
        },
    }


# ---------------------------------------------------------------------------
# HistogramEqualizer
# ---------------------------------------------------------------------------

def test_histogram_equalizer_preserves_shape_and_dtype() -> None:
    X = _low_contrast_batch()
    out = HistogramEqualizer().transform(X)
    assert out.shape == (4, 48, 48)
    assert out.dtype == np.uint8


def test_histogram_equalizer_flattens_histogram() -> None:
    X = _low_contrast_batch()
    out = HistogramEqualizer().transform(X)
    # a flatter histogram has higher entropy (closer to uniform)
    assert _hist_entropy(out) > _hist_entropy(X)


def test_histogram_equalizer_widens_range() -> None:
    X = _low_contrast_batch()  # squeezed into [100, 140]
    out = HistogramEqualizer().transform(X)
    assert (out.max() - out.min()) > (int(X.max()) - int(X.min()))


def test_histogram_equalizer_single_image() -> None:
    X = _low_contrast_batch()[0]  # (48, 48)
    out = HistogramEqualizer().transform(X)
    assert out.shape == (48, 48)
    assert out.dtype == np.uint8


def test_histogram_equalizer_fit_returns_self() -> None:
    p = HistogramEqualizer()
    assert p.fit(_low_contrast_batch()) is p


# ---------------------------------------------------------------------------
# ClaheEqualizer
# ---------------------------------------------------------------------------

def test_clahe_preserves_shape_and_dtype() -> None:
    X = _low_contrast_batch()
    out = ClaheEqualizer().transform(X)
    assert out.shape == (4, 48, 48)
    assert out.dtype == np.uint8


def test_clahe_increases_contrast() -> None:
    X = _low_contrast_batch()
    out = ClaheEqualizer(clip_limit=4.0, tile_grid_size=8).transform(X)
    # per-image contrast (std) should rise on low-contrast input
    assert out.std() > X.std()


def test_clahe_clip_limit_is_configurable() -> None:
    X = _low_contrast_batch()
    low = ClaheEqualizer(clip_limit=1.0).transform(X)
    high = ClaheEqualizer(clip_limit=4.0).transform(X)
    # a higher clip limit amplifies contrast more aggressively
    assert high.std() >= low.std()


# ---------------------------------------------------------------------------
# unsupported shape
# ---------------------------------------------------------------------------

def test_equalizer_rejects_bad_shape() -> None:
    bad = np.zeros((2, 2, 2, 2), dtype=np.uint8)  # 4D
    with pytest.raises(ValueError, match="expects 2D .* or 3D"):
        HistogramEqualizer().transform(bad)


# ---------------------------------------------------------------------------
# SequentialPreprocessor + dispatch
# ---------------------------------------------------------------------------

def test_sequential_chains_equalize_then_rescale() -> None:
    X = _low_contrast_batch()
    pipe = SequentialPreprocessor([HistogramEqualizer(), RescalePreprocessor()])
    out = pipe.fit(X).transform(X)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_build_normalizer_histogram_eq_outputs_unit_float() -> None:
    X = _low_contrast_batch()
    norm = build_normalizer(_cfg("histogram_eq"))
    out = norm.fit(X).transform(X)
    assert out.dtype == np.float32
    assert out.shape == (4, 48, 48)
    assert out.max() <= 1.0


def test_build_normalizer_clahe_outputs_unit_float() -> None:
    X = _low_contrast_batch()
    norm = build_normalizer(_cfg("clahe"))
    out = norm.fit(X).transform(X)
    assert out.dtype == np.float32
    assert out.max() <= 1.0


def test_build_normalizer_still_supports_prior_options() -> None:
    # never delete prior options — rescale/standardize/none must still dispatch
    from src.emotion_detector.data.preprocessing import (
        IdentityPreprocessor,
        RescalePreprocessor as R,
        StandardizePreprocessor,
    )
    assert isinstance(build_normalizer(_cfg("none")), IdentityPreprocessor)
    assert isinstance(build_normalizer(_cfg("rescale")), R)
    assert isinstance(build_normalizer(_cfg("standardize")), StandardizePreprocessor)
