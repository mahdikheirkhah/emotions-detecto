"""Abstract base classes for model building and emotion classification."""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class BaseModelBuilder(abc.ABC):
    """Contract for constructing a compiled Keras model.

    Concrete subclasses (e.g. ``SimpleCnnBuilder``, ``VggSmallBuilder``,
    ``TransferVgg16Builder``) implement ``build`` for one architecture.
    The trainer always holds a ``BaseModelBuilder`` reference so the
    architecture is selected via ``cfg["model"]["architecture"]`` + dispatch.
    """

    @abc.abstractmethod
    def build(self, input_shape: Tuple[int, int, int], num_classes: int) -> object:
        """Construct and compile a Keras model for the given input shape.

        Args:
            input_shape: ``(height, width, channels)`` — for FER-2013 this is
                ``(48, 48, 1)`` (grayscale).
            num_classes: Number of output classes (7 for FER-2013).

        Returns:
            A compiled ``tf.keras.Model`` ready to call ``.fit()`` on.
        """


class BaseEmotionClassifier(abc.ABC):
    """Contract for running emotion inference on a preprocessed face crop.

    Concrete subclasses wrap a loaded ``.keras`` model and handle the
    preprocessing → prediction → label mapping pipeline for a single face.
    """

    @abc.abstractmethod
    def predict(self, face: NDArray) -> str:
        """Return the predicted emotion label for a single face crop.

        Args:
            face: Grayscale face image of shape ``(48, 48)`` or ``(48, 48, 1)``,
                pixel values normalised to ``[0, 1]``.

        Returns:
            One of ``"Happy"``, ``"Sad"``, ``"Angry"``, ``"Surprise"``,
            ``"Fear"``, ``"Disgust"``, ``"Neutral"``.

        Raises:
            ValueError: if *face* does not have the expected shape.
        """

    @abc.abstractmethod
    def predict_proba(self, face: NDArray) -> NDArray:
        """Return the raw softmax probability vector for a single face crop.

        Args:
            face: Same constraints as ``predict``.

        Returns:
            Float32 array of shape ``(7,)`` summing to 1.0.
        """
