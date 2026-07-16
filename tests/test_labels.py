"""Unit tests for the canonical FER-2013 emotion labels (``models/labels.py``).

The label order IS a silent-bug hotspot: if index → name ever drifts from the FER-2013
``emotion`` column the trainer one-hot-encodes, every prediction is mislabelled while
the model looks fine. These pin the order and the fail-loud count check.
"""

from __future__ import annotations

import pytest

from src.emotion_detector.models.labels import FER_EMOTIONS, emotion_labels


def test_fer_emotions_order_and_count() -> None:
    assert FER_EMOTIONS == [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    ]
    assert len(FER_EMOTIONS) == 7  # matches model.num_classes


def test_emotion_labels_returns_a_copy() -> None:
    labels = emotion_labels({"model": {"num_classes": 7}})
    assert labels == FER_EMOTIONS
    labels.append("Bored")  # mutating the return must not corrupt the canonical list
    assert len(FER_EMOTIONS) == 7


def test_emotion_labels_mismatch_fails_loud() -> None:
    # A num_classes disagreeing with the label count would silently mislabel; reject it.
    with pytest.raises(ValueError):
        emotion_labels({"model": {"num_classes": 6}})
