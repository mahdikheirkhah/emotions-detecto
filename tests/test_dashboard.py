"""Unit tests for the dashboard's analysis + state logic (#56).

The Streamlit *view* (``scripts/dashboard.py``) needs the streamlit runtime + a camera,
so it isn't unit-tested; all the real logic lives in the ``dashboard`` module and is
exercised here with fakes — no streamlit, no TensorFlow, no GUI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.emotion_detector.dashboard import (
    EmotionHistory,
    FrameAnalysis,
    analyze_frame,
    probabilities_frame,
)
from src.emotion_detector.models.labels import FER_EMOTIONS
from src.emotion_detector.video.preprocess import NoFaceError


class _FakePreprocessor:
    """locate_and_prepare → (box, tensor), or NoFaceError when no face is configured."""

    def __init__(self, box=(10, 20, 30, 30), has_face=True) -> None:
        self._box = box
        self._has_face = has_face

    def locate_and_prepare(self, frame):
        if not self._has_face:
            raise NoFaceError("no face")
        return self._box, np.zeros((48, 48), np.float32)


class _FakeClassifier:
    def __init__(self, probs) -> None:
        self._probs = np.asarray(probs, np.float32)

    def predict_proba(self, tensor):
        return self._probs


# ---------------------------------------------------------------------------
# analyze_frame
# ---------------------------------------------------------------------------


def test_analyze_frame_returns_top_label_box_and_vector() -> None:
    probs = [0.0, 0.0, 0.0, 0.7, 0.1, 0.2, 0.0]  # index 3 → Happy
    result = analyze_frame(
        np.zeros((100, 100, 3), np.uint8),
        _FakePreprocessor(box=(5, 6, 7, 8)),
        _FakeClassifier(probs),
    )
    assert isinstance(result, FrameAnalysis)
    assert result.box == (5, 6, 7, 8)
    assert result.label == "Happy"
    assert result.confidence == pytest.approx(0.7)
    assert result.probabilities.shape == (7,)


def test_analyze_frame_returns_none_without_a_face() -> None:
    result = analyze_frame(
        np.zeros((100, 100, 3), np.uint8),
        _FakePreprocessor(has_face=False),
        _FakeClassifier([0.1] * 7),
    )
    assert result is None


# ---------------------------------------------------------------------------
# probabilities_frame
# ---------------------------------------------------------------------------


def test_probabilities_frame_has_all_seven_emotions() -> None:
    probs = [0.05, 0.05, 0.05, 0.6, 0.1, 0.1, 0.05]
    df = probabilities_frame(probs)
    assert list(df.index) == FER_EMOTIONS
    assert list(df.columns) == ["probability"]
    assert df.loc["Happy", "probability"] == pytest.approx(0.6)


def test_probabilities_frame_wrong_length_fails_loud() -> None:
    with pytest.raises(ValueError):
        probabilities_frame([0.5, 0.5])  # not 7


# ---------------------------------------------------------------------------
# EmotionHistory
# ---------------------------------------------------------------------------


def test_history_appends_and_reports_length() -> None:
    hist = EmotionHistory(maxlen=10)
    hist.add(0.0, "Happy", 0.9)
    hist.add(1.0, "Sad", 0.6)
    assert len(hist) == 2
    assert hist.records[0] == (0.0, "Happy", 0.9)


def test_history_is_bounded_by_maxlen() -> None:
    hist = EmotionHistory(maxlen=3)
    for i in range(5):
        hist.add(float(i), "Neutral", 0.5)
    assert len(hist) == 3  # oldest two evicted
    assert [t for t, _, _ in hist.records] == [2.0, 3.0, 4.0]


def test_history_unknown_label_fails_loud() -> None:
    with pytest.raises(ValueError):
        EmotionHistory().add(0.0, "Ecstatic", 0.9)  # not a FER emotion


def test_history_to_dataframe_columns() -> None:
    hist = EmotionHistory()
    hist.add(0.5, "Fear", 0.4)
    df = hist.to_dataframe()
    assert list(df.columns) == ["time", "emotion", "confidence"]
    assert df.iloc[0]["emotion"] == "Fear"


def test_timeline_encodes_emotions_as_fer_codes() -> None:
    hist = EmotionHistory()
    hist.add(0.0, "Angry", 0.9)  # code 0
    hist.add(1.0, "Happy", 0.8)  # code 3
    hist.add(2.0, "Neutral", 0.7)  # code 6
    timeline = hist.timeline()
    assert list(timeline["emotion_code"]) == [0, 3, 6]
    assert timeline.index.name == "time"
    assert list(timeline.index) == [0.0, 1.0, 2.0]


def test_timeline_empty_history_is_empty_frame() -> None:
    timeline = EmotionHistory().timeline()
    assert "emotion_code" in timeline.columns
    assert len(timeline) == 0


def test_counts_reindexes_to_all_labels() -> None:
    hist = EmotionHistory()
    for label in ["Happy", "Happy", "Sad"]:
        hist.add(0.0, label, 0.5)
    counts = hist.counts()
    assert list(counts.index) == FER_EMOTIONS  # every emotion present, 0-filled
    assert counts["Happy"] == 2
    assert counts["Sad"] == 1
    assert counts["Angry"] == 0


def test_counts_empty_history_all_zero() -> None:
    counts = EmotionHistory().counts()
    assert list(counts.index) == FER_EMOTIONS
    assert (counts == 0).all()
