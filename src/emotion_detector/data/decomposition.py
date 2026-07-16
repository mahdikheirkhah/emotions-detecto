"""Optional, toggleable PCA dimensionality reduction (see data.md §6).

Defaults **off**: a CNN keeps raw 48×48 images because convolutions exploit the
2-D spatial layout that flattening-to-components would destroy. PCA is valuable
for the MNIST logistic-regression baseline and as a teaching ablation
(``stages.decomposition``). Components are fit on the **training split only**.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from src.emotion_detector.data.base import BaseDecomposer
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.stages import is_stage_on


def _flatten(X: NDArray) -> NDArray:
    """Flatten ``(N, H, W[, C])`` images to ``(N, D)`` feature vectors."""
    arr = np.asarray(X)
    return arr.reshape(arr.shape[0], -1)


class IdentityReducer(BaseDecomposer):
    """No decomposition — returns the data unchanged (stage off).

    Keeps the raw image tensor so a CNN sees the full 2-D spatial layout.
    """

    def fit(self, X: NDArray) -> "IdentityReducer":
        return self

    def transform(self, X: NDArray) -> NDArray:
        return X


class PcaReducer(BaseDecomposer):
    """Project flattened images onto the top principal components.

    ``fit`` flattens the **training** images to ``(N, D)`` and fits a scikit-learn
    ``PCA``; ``transform`` flattens and projects any split to ``(N, n_components)``.
    Because the components come from train only, applying them to val/test adds no
    leakage.

    Args:
        n_components: Passed straight to ``sklearn.decomposition.PCA``. An ``int``
            keeps that many components; a ``float`` in ``(0, 1)`` keeps the fewest
            components explaining at least that fraction of variance.
        seed: Random seed for the (randomized) SVD solver.
    """

    def __init__(self, n_components=0.95, seed: int = 42) -> None:
        self._n_components = n_components
        self._pca = PCA(n_components=n_components, random_state=seed)
        self._fitted = False

    def fit(self, X: NDArray) -> "PcaReducer":
        self._pca.fit(_flatten(X))
        self._fitted = True
        kept = self._pca.n_components_
        var = float(self._pca.explained_variance_ratio_.sum())
        logger.info(
            f"PcaReducer fit on train — kept {kept} components "
            f"(from {self._pca.n_features_in_}), explaining {var:.1%} of variance."
        )
        return self

    def transform(self, X: NDArray) -> NDArray:
        if not self._fitted:
            raise RuntimeError(
                "PcaReducer.transform called before fit(). Fit on the training "
                "split first."
            )
        return self._pca.transform(_flatten(X))

    @property
    def explained_variance_ratio_(self) -> NDArray:
        """Per-component fraction of variance explained (after ``fit``)."""
        if not self._fitted:
            raise RuntimeError("explained_variance_ratio_ available only after fit().")
        return self._pca.explained_variance_ratio_

    @property
    def n_components_(self) -> int:
        """Number of components actually kept (after ``fit``)."""
        if not self._fitted:
            raise RuntimeError("n_components_ available only after fit().")
        return int(self._pca.n_components_)


def build_decomposer(cfg: dict) -> BaseDecomposer:
    """Return the configured decomposer (the dispatch step).

    When ``stages.decomposition`` is off (the default), returns
    :class:`IdentityReducer` so the data passes through unchanged. Otherwise
    returns a :class:`PcaReducer` built from ``decomposition.n_components``.

    Args:
        cfg: Loaded config dict.

    Returns:
        A ``BaseDecomposer`` — the caller fits it on train and transforms splits.

    Raises:
        KeyError: if the ``decomposition`` config is missing when the stage is on.
    """
    if not is_stage_on(cfg, "decomposition"):
        return IdentityReducer()

    try:
        n_components = cfg["decomposition"]["n_components"]
    except KeyError as exc:
        raise KeyError(
            f"Missing decomposition config key: {exc}. "
            "Check the 'decomposition:' section in config.yaml."
        ) from exc

    seed = cfg.get("global", {}).get("seed", 42)
    return PcaReducer(n_components=n_components, seed=seed)
