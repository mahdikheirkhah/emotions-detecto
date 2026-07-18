"""Emotion inference on a preprocessed face — wraps a trained ``.keras`` model (#54).

The last stop of the live pipeline: a ``(48, 48)`` model-ready tensor from
``FacePreprocessor`` (#52) goes in, a softmax over the 7 FER-2013 emotions comes out,
and ``predict_top`` collapses it to the ``(label, confidence)`` the live stream prints.

Kept deliberately thin: all the geometry/normalization already happened upstream (#52),
so this only adds the batch + channel axes the CNN expects, runs one forward pass, and
reads ``argmax``/``max`` off the probability vector. TensorFlow is imported lazily in
``_load`` so importing this module (and unit-testing the pure logic with a fake model)
costs nothing. Which ``.keras`` file to load is config-driven (``paths.*`` +
``model.architecture``), so any trained model runs the same way (Ablation §3).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.models.base import BaseEmotionClassifier
from src.emotion_detector.models.labels import emotion_labels
from src.emotion_detector.utils.logging import logger


def resolve_model_path(cfg: dict) -> str:
    """The trained-model path for the configured architecture (transfer-aware).

    A ``transfer_*`` run saved to the separate ``pretrained_*`` path, so route there;
    otherwise the from-scratch ``model_save_path``. Mirrors the trainer's routing.
    """
    paths = cfg["paths"]
    if cfg["model"]["architecture"].startswith("transfer"):
        return paths["pretrained_model_save_path"]
    return paths["model_save_path"]


def resolve_history_path(cfg: dict) -> str:
    """The training-history JSON path for the configured architecture (transfer-aware).

    Sits in the model directory next to the ``.keras`` and mirrors ``resolve_model_path``:
    a ``transfer_*`` run writes ``pre_trained_history.json`` so it never overwrites the
    from-scratch run's ``history.json`` (Issue #46). Both ``train.py`` (writer) and
    ``validation_loss_accuracy.py`` (learning-curve reader) route through this.
    """
    model_dir = os.path.dirname(resolve_model_path(cfg))
    name = (
        "pre_trained_history.json"
        if cfg["model"]["architecture"].startswith("transfer")
        else "history.json"
    )
    return os.path.join(model_dir, name)


class KerasEmotionClassifier(BaseEmotionClassifier):
    """Run a trained Keras model on a single preprocessed face crop.

    Args:
        cfg: Loaded config (reads ``model.num_classes`` for the labels and, when no
            ``model`` is injected, ``paths.*`` + ``model.architecture`` for the file).
        model: Optional pre-loaded model (injected for tests); any object with a
            ``predict(x)`` returning the class-probability array. Defaults to loading
            the configured ``.keras`` file.
    """

    def __init__(self, cfg: dict, model: Optional[Any] = None) -> None:
        self._labels = emotion_labels(cfg)
        self._model = model if model is not None else self._load(cfg)

    @staticmethod
    def _load(cfg: dict) -> Any:
        """Load the configured ``.keras`` model (TensorFlow imported lazily)."""
        path = resolve_model_path(cfg)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model at '{path}'. Run scripts/train.py first, or point "
                "paths.model_save_path at a trained .keras file."
            )
        from tensorflow.keras.models import load_model  # heavy; lazy on real load

        logger.info(f"Loading emotion model from {path} ...")
        return load_model(path)

    def predict_proba(self, face: NDArray) -> NDArray:
        """Softmax probability vector ``(num_classes,)`` for a single face."""
        probs = np.asarray(self._model.predict(self._as_batch(face), verbose=0))
        return probs.reshape(-1).astype(np.float32)

    def predict(self, face: NDArray) -> str:
        """The single most likely emotion label for *face*."""
        return self._labels[int(np.argmax(self.predict_proba(face)))]

    def predict_top(self, face: NDArray) -> Tuple[str, float]:
        """``(label, confidence)`` — the argmax emotion and its softmax probability.

        This is what the live stream prints as ``"Happy , 73%"`` (#54).
        """
        probs = self.predict_proba(face)
        idx = int(np.argmax(probs))
        return self._labels[idx], float(probs[idx])

    def _as_batch(self, face: NDArray) -> NDArray:
        """Add the batch + channel axes: ``(H, W)`` or ``(H, W, 1)`` → ``(1, H, W, 1)``.

        Raises:
            ValueError: if *face* is not a 2-D or single-channel 3-D array.
        """
        arr = np.asarray(face, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim != 3 or arr.shape[-1] != 1:
            raise ValueError(
                "Expected a (H, W) or (H, W, 1) grayscale face; got shape "
                f"{np.asarray(face).shape}."
            )
        return arr[np.newaxis, ...]
