"""Config-driven pixel-value normalization strategies (feature engineering A).

Three selectable ``BaseImagePreprocessor`` strategies behind the Ablation-Driven
dispatch: ``none`` (raw-pixel baseline), ``rescale`` (÷255 → [0, 1]), and
``standardize`` (per-dataset z-score). Standardization statistics are fit on the
**training split only** and reused on val/test to avoid leakage (CONTRIBUTING §8).
The whole step is gated by the ``stages.preprocessing`` toggle.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.data.base import BaseImagePreprocessor
from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.stages import is_stage_on


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

    registry = {
        "none": IdentityPreprocessor,
        "rescale": RescalePreprocessor,
        "standardize": StandardizePreprocessor,
    }
    return dispatch(strategy, registry)
