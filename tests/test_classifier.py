"""Unit tests for the emotion classifier (#54) — fake model, no TensorFlow.

The forward-pass logic (batching, argmax label, confidence, shape checks) is tested with
a fake ``predict``; loading a real ``.keras`` model is a VM concern, so only the
fail-loud missing-file branch is exercised (it checks the path before importing TF).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.models.classifier import (
    KerasEmotionClassifier,
    resolve_model_path,
)
from src.emotion_detector.models.labels import FER_EMOTIONS, emotion_labels


def _cfg(architecture="vgg_small", num_classes=7) -> dict:
    return {
        "model": {"architecture": architecture, "num_classes": num_classes},
        "paths": {
            "model_save_path": "results/model/final_emotion_model.keras",
            "pretrained_model_save_path": "results/model/pre_trained_model.keras",
        },
    }


class _FakeModel:
    """Duck-typed Keras model: records the input shape, returns preset probabilities."""

    def __init__(self, probs) -> None:
        self._probs = np.asarray(probs, np.float32)
        self.last_input_shape = None

    def predict(self, x, **kwargs):
        self.last_input_shape = x.shape
        return self._probs.reshape(1, -1)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def test_labels_are_the_seven_fer_emotions() -> None:
    assert emotion_labels(_cfg()) == FER_EMOTIONS
    assert FER_EMOTIONS[3] == "Happy"  # index 3 is Happy (FER-2013 column order)


def test_label_count_mismatch_fails_loud() -> None:
    with pytest.raises(ValueError):
        emotion_labels(_cfg(num_classes=5))


# ---------------------------------------------------------------------------
# resolve_model_path — transfer routing
# ---------------------------------------------------------------------------


def test_resolve_scratch_model_path() -> None:
    assert resolve_model_path(_cfg("vgg_small")).endswith("final_emotion_model.keras")


def test_resolve_transfer_model_path() -> None:
    assert resolve_model_path(_cfg("transfer_vgg16")).endswith(
        "pre_trained_model.keras"
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def test_predict_returns_argmax_label() -> None:
    probs = [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]  # index 3 → Happy
    clf = KerasEmotionClassifier(_cfg(), model=_FakeModel(probs))
    assert clf.predict(np.zeros((48, 48), np.float32)) == "Happy"


def test_predict_top_returns_label_and_confidence() -> None:
    probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.73, 0.27]  # index 5 → Surprise, max 0.73
    clf = KerasEmotionClassifier(_cfg(), model=_FakeModel(probs))
    label, confidence = clf.predict_top(np.zeros((48, 48), np.float32))
    assert label == "Surprise"
    assert confidence == pytest.approx(0.73)


def test_predict_proba_shape_and_dtype() -> None:
    clf = KerasEmotionClassifier(
        _cfg(), model=_FakeModel([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    )
    probs = clf.predict_proba(np.zeros((48, 48), np.float32))
    assert probs.shape == (7,)
    assert probs.dtype == np.float32


def test_batches_with_channel_axis() -> None:
    fake = _FakeModel([0.1] * 7)
    KerasEmotionClassifier(_cfg(), model=fake).predict(np.zeros((48, 48), np.float32))
    assert fake.last_input_shape == (1, 48, 48, 1)  # (batch, H, W, channel)


def test_accepts_already_channelled_face() -> None:
    fake = _FakeModel([0.1] * 7)
    KerasEmotionClassifier(_cfg(), model=fake).predict(
        np.zeros((48, 48, 1), np.float32)
    )
    assert fake.last_input_shape == (1, 48, 48, 1)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((48, 48, 3), np.float32),  # 3 channels
        np.zeros((48,), np.float32),  # 1-D
        np.zeros((2, 48, 48, 1), np.float32),  # already batched
    ],
)
def test_bad_face_shape_raises(bad) -> None:
    clf = KerasEmotionClassifier(_cfg(), model=_FakeModel([0.1] * 7))
    with pytest.raises(ValueError):
        clf.predict(bad)


def test_missing_model_file_fails_loud() -> None:
    cfg = _cfg()
    cfg["paths"]["model_save_path"] = "/no/such/model.keras"
    with pytest.raises(FileNotFoundError):
        KerasEmotionClassifier(cfg)  # no injected model → attempts to load
