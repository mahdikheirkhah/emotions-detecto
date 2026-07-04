"""Abstract base classes for dataset fetching and image preprocessing."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BaseDatasetFetcher(abc.ABC):
    """Contract for downloading / locating a dataset and returning it as arrays.

    Concrete subclasses (e.g. ``Fer2013Fetcher``, ``MnistFetcher``) implement
    ``fetch`` for a specific source.  Callers always type-hint against this
    base so the concrete source can be swapped via config + dispatch.
    """

    @abc.abstractmethod
    def fetch(self, data_dir: Path) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Download (if needed) and load the dataset.

        Args:
            data_dir: Directory where raw data files are stored / cached.

        Returns:
            Six arrays ``(X_train, y_train, X_val, y_val, X_test, y_test)``
            where X arrays are image arrays and y arrays are integer class labels.

        Raises:
            FileNotFoundError: if the dataset cannot be found and cannot be
                downloaded automatically.
        """


class BaseImagePreprocessor(abc.ABC):
    """Contract for a stateless or fit-then-transform image preprocessing step.

    Concrete subclasses (e.g. ``RescalePreprocessor``, ``StandardizePreprocessor``,
    ``HistogramEqualizer``) implement one preprocessing strategy.  The caller
    dispatches to a subclass based on ``cfg["preprocessing"]["normalization"]``
    and always receives a ``BaseImagePreprocessor`` back.
    """

    @abc.abstractmethod
    def fit(self, X: NDArray) -> "BaseImagePreprocessor":
        """Compute any statistics needed for the transform (e.g. mean/std).

        Must be called on *training data only* to prevent data leakage.

        Args:
            X: Image array of shape ``(N, H, W)`` or ``(N, H, W, C)``.

        Returns:
            ``self`` so calls can be chained: ``preprocessor.fit(X_train).transform(X_train)``.
        """

    @abc.abstractmethod
    def transform(self, X: NDArray) -> NDArray:
        """Apply the preprocessing strategy to *X*.

        Args:
            X: Image array with the same shape convention as ``fit``.

        Returns:
            Preprocessed array of the same shape as *X*, dtype ``float32``.
        """


class BaseCleaner(abc.ABC):
    """Contract for one deterministic dataset-cleaning step.

    Concrete subclasses (e.g. ``DuplicateRemover``, ``CorruptImageRemover``)
    each remove one class of bad rows.  The orchestrator chains cleaners
    selected via ``cfg["cleaning"]`` (see data.md §3).

    Two invariants every cleaner must satisfy:

    * **Idempotent** — applying it twice equals applying it once
      (a second pass finds nothing left to remove).
    * **Order-independent** — because each cleaner selects a row-subset from a
      per-image property, chaining cleaners in any order yields the same result.
    """

    @abc.abstractmethod
    def clean(self, images: NDArray, labels: NDArray) -> Tuple[NDArray, NDArray]:
        """Return *(images, labels)* with this step's bad rows removed.

        Args:
            images: Image array of shape ``(N, H, W)`` or ``(N, H, W, C)``.
            labels: Integer label array of shape ``(N,)`` aligned with *images*.

        Returns:
            A ``(images, labels)`` pair containing only the kept rows, in the
            original order.
        """


class BaseImbalanceStrategy(abc.ABC):
    """Contract for one class-imbalance remedy (see data.md §3.2).

    Concrete subclasses (``NoResample``, ``ClassWeightStrategy``,
    ``Oversampler``, ``Undersampler``) each implement one strategy, selected via
    ``cfg["cleaning"]["imbalance_strategy"]`` + dispatch.

    **Train-only:** every strategy is applied to the *training split only* —
    resampling or reweighting the validation/test sets would corrupt the
    evaluation (CONTRIBUTING §8).

    The unified return shape ``(X, y, class_weight)`` lets the trainer treat all
    four strategies identically::

        X_tr, y_tr, class_weight = strategy.apply(X_train, y_train)
        model.fit(X_tr, y_tr, class_weight=class_weight)

    Resampling strategies return the resampled arrays with ``class_weight=None``;
    ``ClassWeightStrategy`` returns the data unchanged with a weight dict.
    """

    @abc.abstractmethod
    def apply(
        self, X: NDArray, y: NDArray
    ) -> Tuple[NDArray, NDArray, Optional[Dict[int, float]]]:
        """Apply the imbalance remedy to a *training* split.

        Args:
            X: Training image array of shape ``(N, ...)``.
            y: Training integer label array of shape ``(N,)``.

        Returns:
            ``(X_out, y_out, class_weight)`` — ``class_weight`` is a
            ``{class_index: weight}`` dict for ``model.fit`` or ``None``.
        """


class BaseDecomposer(abc.ABC):
    """Contract for an optional dimensionality-reduction step (see data.md §6).

    Concrete subclasses (``IdentityReducer``, ``PcaReducer``) implement one
    strategy, selected via the ``stages.decomposition`` toggle + config. Like the
    preprocessors, statistics are fit on the **training split only** and reused on
    val/test (CONTRIBUTING §8).

    ``IdentityReducer`` (stage off) returns images unchanged so a CNN keeps the
    2-D spatial layout its convolutions rely on; ``PcaReducer`` (stage on) flattens
    and projects onto the top principal components — useful for a linear baseline.
    """

    @abc.abstractmethod
    def fit(self, X: NDArray) -> "BaseDecomposer":
        """Fit any decomposition statistics on the **training** data.

        Args:
            X: Training image array of shape ``(N, H, W)`` or ``(N, D)``.

        Returns:
            ``self`` so calls can be chained.
        """

    @abc.abstractmethod
    def transform(self, X: NDArray) -> NDArray:
        """Project *X* onto the fitted representation.

        Args:
            X: Image/feature array with the same convention as ``fit``.

        Returns:
            Either the unchanged array (identity) or a reduced ``(N, n_components)``
            feature array (PCA).
        """
