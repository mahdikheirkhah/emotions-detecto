"""Dashboard analysis + state — the UI-agnostic core of the Streamlit app (#56).

**No streamlit import here.** Everything is pure numpy/pandas so the app's real logic
(per-frame analysis, the 7-class probability chart data, and the rolling emotion
history/timeline) is unit-tested headless, and ``scripts/dashboard.py`` stays a thin
view over it (CONTRIBUTING §7: reuse the pipeline, keep logic out of the script). It
reuses the pipeline unchanged: ``FacePreprocessor.locate_and_prepare`` (#52/#55) for
one detection giving the box + model-ready tensor, and the classifier's softmax (#54).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.emotion_detector.models.labels import FER_EMOTIONS
from src.emotion_detector.video.base import FaceRect
from src.emotion_detector.video.preprocess import NoFaceError


@dataclass(frozen=True)
class FrameAnalysis:
    """One frame's result: where the face is, the top emotion, and the full vector."""

    box: FaceRect
    label: str
    confidence: float
    probabilities: NDArray  # (num_classes,) softmax, in FER_EMOTIONS order


def analyze_frame(
    frame: NDArray,
    preprocessor: Any,
    classifier: Any,
    labels: Sequence[str] = FER_EMOTIONS,
) -> Optional[FrameAnalysis]:
    """Detect + classify one BGR frame → ``FrameAnalysis``, or ``None`` if no face.

    A face-less frame returns ``None`` (not an error) so the UI loop just skips it and
    keeps streaming. One detection (``locate_and_prepare``) yields both the box to draw
    and the tensor to classify.
    """
    try:
        box, tensor = preprocessor.locate_and_prepare(frame)
    except NoFaceError:
        return None
    probs = np.asarray(classifier.predict_proba(tensor), dtype=float).reshape(-1)
    idx = int(np.argmax(probs))
    return FrameAnalysis(
        box=box, label=labels[idx], confidence=float(probs[idx]), probabilities=probs
    )


def probabilities_frame(
    probabilities: NDArray, labels: Sequence[str] = FER_EMOTIONS
) -> pd.DataFrame:
    """A one-column ``emotion → probability`` frame for ``st.bar_chart`` (all 7 shown).

    Surfacing the whole distribution, not just the argmax, is the honest/clinical view:
    a hedged 40% "Happy" reads very differently from a confident 95%.

    Raises:
        ValueError: if the vector length does not match the label count.
    """
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.shape[0] != len(labels):
        raise ValueError(f"expected {len(labels)} probabilities, got {probs.shape[0]}.")
    return pd.DataFrame({"probability": probs}, index=list(labels))


class EmotionHistory:
    """A fixed-length rolling log of detections for the timeline (in session_state).

    ``maxlen`` bounds memory so the app never grows unbounded across Streamlit reruns.
    The UI stores one instance in ``st.session_state`` so it survives the top-to-bottom
    rerun that every widget interaction triggers.
    """

    def __init__(self, maxlen: int = 100, labels: Sequence[str] = FER_EMOTIONS) -> None:
        self._records: Deque[Tuple[float, str, float]] = deque(maxlen=int(maxlen))
        self._labels = list(labels)
        self._code = {name: i for i, name in enumerate(self._labels)}

    def add(self, timestamp: float, label: str, confidence: float) -> None:
        """Append a detection; the oldest is evicted once ``maxlen`` is reached.

        Raises:
            ValueError: if *label* is not one of the configured emotions.
        """
        if label not in self._code:
            raise ValueError(f"unknown emotion '{label}'.")
        self._records.append((float(timestamp), str(label), float(confidence)))

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[Tuple[float, str, float]]:
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """The raw history as ``[time, emotion, confidence]`` rows (a table view)."""
        return pd.DataFrame(self._records, columns=["time", "emotion", "confidence"])

    def timeline(self) -> pd.DataFrame:
        """Numeric emotion-code-over-time (indexed by timestamp) for ``st.line_chart``.

        Emotions are categorical, so the line plots their FER index (0–6) against time;
        the reader maps codes back via label order. Empty until the first detection.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame({"emotion_code": pd.Series(dtype=float)})
        return pd.DataFrame(
            {"emotion_code": [self._code[e] for e in df["emotion"]]},
            index=pd.Index(df["time"], name="time"),
        )

    def counts(self) -> pd.Series:
        """Per-emotion counts, reindexed to all labels (0-filled, in label order)."""
        df = self.to_dataframe()
        if df.empty:
            return pd.Series(0, index=self._labels, name="count")
        return (
            df["emotion"]
            .value_counts()
            .reindex(self._labels, fill_value=0)
            .rename("count")
        )
