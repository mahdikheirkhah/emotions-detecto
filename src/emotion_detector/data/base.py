"""Abstract base classes for dataset fetching and image preprocessing."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Tuple

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
