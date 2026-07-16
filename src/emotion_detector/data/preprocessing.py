"""Config-driven pixel-value normalization strategies (feature engineering A).

Three selectable ``BaseImagePreprocessor`` strategies behind the Ablation-Driven
dispatch: ``none`` (raw-pixel baseline), ``rescale`` (÷255 → [0, 1]), and
``standardize`` (per-dataset z-score). Standardization statistics are fit on the
**training split only** and reused on val/test to avoid leakage (CONTRIBUTING §8).
The whole step is gated by the ``stages.preprocessing`` toggle.
"""

from __future__ import annotations

from typing import Callable, List

import cv2
import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.data.base import BaseImagePreprocessor
from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.stages import is_stage_on


def _to_uint8(img: NDArray) -> NDArray:
    """Return *img* as uint8, clipping to [0, 255] only if it is not already."""
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def _per_image(X: NDArray, fn: Callable[[NDArray], NDArray], name: str) -> NDArray:
    """Apply a single-image cv2 op to a 2D image or a 3D ``(N, H, W)`` batch.

    Supports both shapes so the identical transform runs on a training batch and
    on a single webcam face at inference (#52).

    Raises:
        ValueError: on an unsupported shape or an underlying ``cv2.error``.
    """
    arr = np.asarray(X)
    try:
        if arr.ndim == 2:
            return fn(_to_uint8(arr))
        if arr.ndim == 3:
            return np.stack([fn(_to_uint8(arr[i])) for i in range(arr.shape[0])])
    except cv2.error as exc:
        raise ValueError(f"{name} failed: {exc}") from exc
    raise ValueError(
        f"{name} expects 2D (H, W) or 3D (N, H, W) grayscale; got shape {arr.shape}"
    )


class IdentityPreprocessor(BaseImagePreprocessor):
    """No normalization — returns pixels unchanged (raw-pixel baseline).

    Used for ``normalization: none`` and whenever the preprocessing stage is off,
    so a raw-pixel ablation and a disabled stage behave identically. Values are
    cast to float32 for a consistent downstream dtype, but not scaled.
    """

    def fit(self, X: NDArray) -> "IdentityPreprocessor":
        return self

    def transform(self, X: NDArray) -> NDArray:
        return X.astype(np.float32)


class RescalePreprocessor(BaseImagePreprocessor):
    """Rescale pixel intensities from ``[0, 255]`` to ``[0, 1]`` (÷255).

    Stateless — ``fit`` is a no-op. Keeping inputs in a small, fixed range keeps
    gradients well-conditioned and speeds convergence.
    """

    def fit(self, X: NDArray) -> "RescalePreprocessor":
        return self

    def transform(self, X: NDArray) -> NDArray:
        return X.astype(np.float32) / 255.0


class StandardizePreprocessor(BaseImagePreprocessor):
    """Per-dataset z-score: ``(x - mean) / std`` using scalar train statistics.

    ``fit`` computes a single global mean and std over the **training** pixels;
    ``transform`` reuses them on any split. Because the statistics come from train
    only, applying the same transform to val/test introduces no leakage.
    """

    def __init__(self) -> None:
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, X: NDArray) -> "StandardizePreprocessor":
        Xf = X.astype(np.float32)
        self._mean = float(Xf.mean())
        std = float(Xf.std())
        # Guard against constant data (std == 0) → avoid divide-by-zero.
        self._std = std if std > 0 else 1.0
        logger.info(
            f"StandardizePreprocessor fit on train — mean={self._mean:.3f}, "
            f"std={self._std:.3f}"
        )
        return self

    def transform(self, X: NDArray) -> NDArray:
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "StandardizePreprocessor.transform called before fit(). "
                "Fit on the training split first."
            )
        return (X.astype(np.float32) - self._mean) / self._std


class HistogramEqualizer(BaseImagePreprocessor):
    """Global histogram equalization (``cv2.equalizeHist``), uint8 → uint8.

    Remaps intensities so the cumulative histogram becomes ~linear, spreading
    contrast across the full ``[0, 255]`` range — great for low-contrast faces,
    but it can over-amplify noise on already-balanced images (hence the toggle).

    Runs on **raw uint8 grayscale, before any rescale**. Stateless (``fit`` is a
    no-op); the identical transform runs in the live webcam pipeline (#52).
    """

    def fit(self, X: NDArray) -> "HistogramEqualizer":
        return self

    def transform(self, X: NDArray) -> NDArray:
        return _per_image(X, cv2.equalizeHist, "HistogramEqualizer")


class ClaheEqualizer(BaseImagePreprocessor):
    """CLAHE — Contrast-Limited Adaptive Histogram Equalization, uint8 → uint8.

    Equalizes in local ``tile_grid_size × tile_grid_size`` tiles with a
    ``clip_limit`` cap on contrast amplification. Because it works locally and
    clips, it enhances local detail without global equalization's tendency to
    blow up noise in flat regions.

    Runs on **raw uint8 grayscale, before any rescale**.

    Args:
        clip_limit:     Contrast clip limit (higher = stronger, more noise).
        tile_grid_size: Number of tiles per axis (an ``N × N`` grid).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        self._clip_limit = float(clip_limit)
        self._tile_grid_size = int(tile_grid_size)
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=(self._tile_grid_size, self._tile_grid_size),
        )

    def fit(self, X: NDArray) -> "ClaheEqualizer":
        return self

    def transform(self, X: NDArray) -> NDArray:
        return _per_image(X, self._clahe.apply, "ClaheEqualizer")


class SequentialPreprocessor(BaseImagePreprocessor):
    """Chain preprocessors so an equalizer (uint8) runs *before* a rescale.

    ``histogram_eq`` and ``clahe`` are contrast steps that output uint8; the CNN
    still wants ``[0, 1]`` floats, so the dispatch wraps each equalizer with a
    following ``RescalePreprocessor``. This class encodes that documented order:
    equalize on uint8 → rescale to ``[0, 1]``.
    """

    def __init__(self, steps: List[BaseImagePreprocessor]) -> None:
        self._steps = steps

    def fit(self, X: NDArray) -> "SequentialPreprocessor":
        cur = np.asarray(X)
        for step in self._steps:
            step.fit(cur)
            cur = step.transform(cur)
        return self

    def transform(self, X: NDArray) -> NDArray:
        cur = np.asarray(X)
        for step in self._steps:
            cur = step.transform(cur)
        return cur


def build_normalizer(cfg: dict) -> BaseImagePreprocessor:
    """Return the configured normalizer (the dispatch step).

    When ``stages.preprocessing`` is off, returns ``IdentityPreprocessor`` so the
    pipeline falls back to a raw-pixel baseline. Otherwise dispatches on
    ``preprocessing.normalization``.

    Args:
        cfg: Loaded config dict.

    Returns:
        A ``BaseImagePreprocessor`` — the caller fits it on train and transforms
        every split.

    Raises:
        KeyError:   if the ``preprocessing.normalization`` config key is missing.
        ValueError: if the option string is not a known normalizer.
    """
    if not is_stage_on(cfg, "preprocessing"):
        return IdentityPreprocessor()

    try:
        strategy = cfg["preprocessing"]["normalization"]
    except KeyError as exc:
        raise KeyError(
            f"Missing preprocessing config key: {exc}. "
            "Check the 'preprocessing:' section in config.yaml."
        ) from exc

    clip = cfg["preprocessing"].get("clahe_clip_limit", 2.0)
    tile = cfg["preprocessing"].get("clahe_tile_grid", 8)

    registry = {
        "none": IdentityPreprocessor,
        "rescale": RescalePreprocessor,
        "standardize": StandardizePreprocessor,
        # Contrast enhancers run on uint8 first, then rescale to [0, 1].
        "histogram_eq": lambda: SequentialPreprocessor(
            [HistogramEqualizer(), RescalePreprocessor()]
        ),
        "clahe": lambda: SequentialPreprocessor(
            [ClaheEqualizer(clip, tile), RescalePreprocessor()]
        ),
    }
    return dispatch(strategy, registry)
