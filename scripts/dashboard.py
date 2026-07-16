"""Streamlit emotion dashboard (#56): live feed + 7-class probabilities + timeline.

    streamlit run scripts/dashboard.py

The mental-health-app view: the webcam feed with the detected face boxed, the current
emotion, a bar chart over **all seven** probabilities (honest about uncertainty, not
just the top label), and a rolling timeline of emotions over the session. Every bit of
inference is the existing pipeline (``FacePreprocessor`` #52 + the trained model #54 +
the #55 overlay); this file is only the *view* — the analysis/state logic lives in
``src/emotion_detector/dashboard.py`` and is unit-tested there (CONTRIBUTING §7).

Model, video source, and refresh rate all come from ``config.yaml`` (Ablation §3), so
the dashboard runs whichever model an experiment produced. Loguru stays behind the UI;
the recorded-video fallback (#51) means it still works without a webcam.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Tuple

import cv2
import streamlit as st

# Make `src` importable under `streamlit run scripts/dashboard.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.dashboard import (
    EmotionHistory,
    analyze_frame,
    probabilities_frame,
)
from src.emotion_detector.models.classifier import KerasEmotionClassifier
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.video.capture import VideoSource
from src.emotion_detector.video.overlay import annotate
from src.emotion_detector.video.preprocess import FacePreprocessor


@st.cache_resource
def load_pipeline(config_path: str = "config.yaml") -> Tuple[dict, Any, Any]:
    """Load config + preprocessor + model once, cached across Streamlit reruns.

    ``st.cache_resource`` is what stops the (heavy) ``.keras`` load from re-running on
    every widget interaction — the model is built a single time per session.
    """
    cfg = load_config(config_path)
    setup_logging(cfg)
    logger.info("Dashboard pipeline loading (model + preprocessor) ...")
    return cfg, FacePreprocessor(cfg), KerasEmotionClassifier(cfg)


def _render_timeline(slot: Any, history: EmotionHistory) -> None:
    """Draw the emotion-over-time line + a per-emotion count bar in *slot*."""
    with slot.container():
        if len(history) == 0:
            st.info("No detections yet — the timeline fills as faces are seen.")
            return
        st.line_chart(history.timeline(), y="emotion_code")
        st.bar_chart(history.counts())


def main() -> None:
    st.set_page_config(page_title="Emotion Detecto", page_icon="🧠", layout="wide")
    st.title("🧠 Emotion Detecto — live dashboard")
    st.caption(
        "Real-time facial-emotion detection with per-class probabilities and history."
    )

    cfg, preprocessor, classifier = load_pipeline()
    dash = cfg["dashboard"]
    history_length = int(dash["history_length"])
    interval = 1.0 / max(float(dash["refresh_rate"]), 1e-6)

    # Persist the rolling history across reruns — the whole point of session_state.
    if "history" not in st.session_state:
        st.session_state.history = EmotionHistory(maxlen=history_length)

    run = st.sidebar.toggle("Run camera", value=False)
    if st.sidebar.button("Clear history"):
        st.session_state.history = EmotionHistory(maxlen=history_length)
    history: EmotionHistory = st.session_state.history

    feed_col, probs_col = st.columns([2, 1])
    feed_col.subheader("Live feed")
    frame_slot = feed_col.empty()
    label_slot = feed_col.empty()
    probs_col.subheader("Current probabilities")
    probs_slot = probs_col.empty()
    st.subheader("Emotion timeline")
    timeline_slot = st.empty()

    if not run:
        frame_slot.info("Toggle **Run camera** in the sidebar to start streaming.")
        _render_timeline(timeline_slot, history)
        return

    # While the toggle is on, stream frames; unchecking triggers a rerun that skips this
    # loop. VideoSource gives the webcam (or the recorded fallback) with clean release.
    start = time.time()
    with VideoSource(cfg) as source:
        for frame in source.frames():
            result = analyze_frame(frame, preprocessor, classifier)
            if result is not None:
                annotate(frame, result.box, result.label, result.confidence)
                history.add(time.time() - start, result.label, result.confidence)
                label_slot.metric(
                    "Current emotion", result.label, f"{result.confidence:.0%}"
                )
                probs_slot.bar_chart(probabilities_frame(result.probabilities))
            else:
                label_slot.metric("Current emotion", "—", "no face")
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            _render_timeline(timeline_slot, history)
            time.sleep(interval)


if __name__ == "__main__":
    main()
