"""Streamlit emotion dashboard (#56): upload a video, or use the webcam — live or snapshot.

    streamlit run scripts/dashboard.py

Everything renders INSIDE the browser: the frame with the detected face boxed, the
current emotion, a bar chart over **all seven** probabilities (honest about uncertainty,
not just the top label), and a rolling emotion timeline. Pick one of three sources in
the sidebar:

* **Upload a video** — choose an .mp4/.mov/.avi and the pipeline processes it frame by
  frame. The most reliable option: no camera permission, works headless.
* **Webcam · live** — stream the OS webcam via OpenCV and predict each refresh tick.
* **Webcam · snapshot** — grab one photo through the *browser* camera
  (``st.camera_input``); the Mac-friendly path when the live OpenCV loop misbehaves.

All inference is the existing pipeline (``FacePreprocessor`` #52 + the trained model #54
+ the #55 overlay); this file is only the *view* — the analysis/state logic lives in
``src/emotion_detector/dashboard.py`` and is unit-tested there (CONTRIBUTING §7). Model,
source, and refresh rate come from ``config.yaml`` (Ablation §3); Loguru stays behind the
UI, and the recorded-video fallback (#51) means live mode still works without a webcam.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
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


class _Slots:
    """The UI placeholders a stream writes into (feed image, label, probs, timeline)."""

    def __init__(self, frame: Any, label: Any, probs: Any, timeline: Any) -> None:
        self.frame = frame
        self.label = label
        self.probs = probs
        self.timeline = timeline


def _render_timeline(slot: Any, history: EmotionHistory) -> None:
    """Draw the emotion-over-time line + a per-emotion count bar in *slot*."""
    with slot.container():
        if len(history) == 0:
            st.info("No detections yet — the timeline fills as faces are seen.")
            return
        st.line_chart(history.timeline(), y="emotion_code")
        st.bar_chart(history.counts())


def _render_frame(
    slots: _Slots, frame: Any, result: Any, history: EmotionHistory, elapsed: float
) -> None:
    """Annotate + push one processed BGR *frame* (and its result) into the UI slots."""
    if result is not None:
        annotate(frame, result.box, result.label, result.confidence)
        history.add(elapsed, result.label, result.confidence)
        slots.label.metric("Current emotion", result.label, f"{result.confidence:.0%}")
        slots.probs.bar_chart(probabilities_frame(result.probabilities))
    else:
        slots.label.metric("Current emotion", "—", "no face")
    slots.frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    _render_timeline(slots.timeline, history)


def _source_cfg(cfg: dict, source: Any) -> dict:
    """A cfg copy whose ``video.source`` is *source* (uploaded clip or webcam index)."""
    src = copy.deepcopy(cfg)
    src["video"]["source"] = source
    return src


def _stream(
    source_cfg: dict,
    preprocessor: Any,
    classifier: Any,
    history: EmotionHistory,
    slots: _Slots,
    interval: float,
    frame_skip: int = 1,
    progress: Any = None,
    total: int = 0,
) -> None:
    """Iterate a ``VideoSource``, predicting + rendering each (skipped) frame in-browser.

    Shared by the uploaded-video and live-webcam paths: ``frame_skip`` thins the work
    (process every Nth frame), ``interval`` paces live playback, and ``progress`` (with
    ``total`` frames) drives the upload progress bar.
    """
    with VideoSource(source_cfg) as source:
        fps = float(source.fps) or 30.0
        for i, frame in enumerate(source.frames()):
            if frame_skip > 1 and i % frame_skip:
                continue
            result = analyze_frame(frame, preprocessor, classifier)
            _render_frame(slots, frame, result, history, i / fps)
            if progress is not None and total:
                progress.progress(min((i + 1) / total, 1.0))
            if interval > 0:
                time.sleep(interval)


def _frame_count(path: str) -> int:
    """Total frames in *path* (for the upload progress bar); 0 if unknown."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return max(n, 0)


def _process_upload(
    cfg: dict,
    preprocessor: Any,
    classifier: Any,
    history: EmotionHistory,
    slots: _Slots,
) -> None:
    """Upload-a-video source: save the clip to a temp file and stream it through."""
    upload = st.sidebar.file_uploader(
        "Video file", type=["mp4", "mov", "avi", "mkv", "m4v"]
    )
    go = st.sidebar.button("▶️ Process video", type="primary", disabled=upload is None)
    if upload is None:
        slots.frame.info("⬆️ Upload a face video in the sidebar, then press Process.")
        _render_timeline(slots.timeline, history)
        return
    if not go:
        slots.frame.info("Press **▶️ Process video** in the sidebar to start.")
        _render_timeline(slots.timeline, history)
        return

    # cv2 needs a real path, so persist the uploaded bytes to a temp file.
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(upload.name).suffix or ".mp4"
    ) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = tmp.name
    try:
        progress = st.progress(0.0)
        _stream(
            _source_cfg(cfg, tmp_path),
            preprocessor,
            classifier,
            history,
            slots,
            interval=0.0,  # process as fast as the CPU allows
            frame_skip=max(1, int(cfg["video"].get("frame_skip", 1))),
            progress=progress,
            total=_frame_count(tmp_path),
        )
        progress.empty()
        st.success("Done — the timeline below is the session summary.")
    except Exception as exc:  # bad codec / unreadable clip — show it, don't crash
        logger.exception("Upload processing failed")
        st.error(f"Could not process the video: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _run_webcam_live(
    cfg: dict,
    preprocessor: Any,
    classifier: Any,
    history: EmotionHistory,
    slots: _Slots,
    interval: float,
) -> None:
    """Live-webcam source: stream the OS camera (or recorded fallback) until toggled off."""
    run = st.sidebar.toggle("Run camera", value=False)
    if not run:
        slots.frame.info("Toggle **Run camera** in the sidebar to start streaming.")
        _render_timeline(slots.timeline, history)
        return
    # Unchecking the toggle triggers a rerun that skips this (blocking) loop. VideoSource
    # gives the webcam or the recorded fallback (#51) with a clean release.
    _stream(cfg, preprocessor, classifier, history, slots, interval=interval)


def _snapshot(
    cfg: dict,
    preprocessor: Any,
    classifier: Any,
    history: EmotionHistory,
    slots: _Slots,
) -> None:
    """Snapshot source: classify one photo from the BROWSER camera (reliable on macOS)."""
    photo = st.sidebar.camera_input("Take a photo")
    if photo is None:
        slots.frame.info("📸 Use **Take a photo** in the sidebar to classify a frame.")
        _render_timeline(slots.timeline, history)
        return
    frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    result = analyze_frame(frame, preprocessor, classifier)
    _render_frame(slots, frame, result, history, float(len(history)))


def main() -> None:
    st.set_page_config(page_title="Emotion Detecto", page_icon="🧠", layout="wide")
    st.title("🧠 Emotion Detecto — live dashboard")
    st.caption(
        "Upload a video or use your webcam; see the boxed face, the current emotion, "
        "all-seven probabilities, and an emotion timeline."
    )

    cfg, preprocessor, classifier = load_pipeline()
    dash = cfg["dashboard"]
    history_length = int(dash["history_length"])
    interval = 1.0 / max(float(dash["refresh_rate"]), 1e-6)

    # Persist the rolling history across reruns — the whole point of session_state.
    if "history" not in st.session_state:
        st.session_state.history = EmotionHistory(maxlen=history_length)
    if st.sidebar.button("🧹 Clear history"):
        st.session_state.history = EmotionHistory(maxlen=history_length)
    history: EmotionHistory = st.session_state.history

    source = st.sidebar.radio(
        "Source", ("Upload a video", "Webcam · live", "Webcam · snapshot")
    )
    st.sidebar.divider()

    feed_col, probs_col = st.columns([2, 1])
    with feed_col:
        st.subheader("Feed")
        frame_slot = st.empty()
        label_slot = st.empty()
    with probs_col:
        st.subheader("Current probabilities")
        probs_slot = st.empty()
    st.subheader("Emotion timeline")
    slots = _Slots(frame_slot, label_slot, probs_slot, st.empty())

    if source == "Upload a video":
        _process_upload(cfg, preprocessor, classifier, history, slots)
    elif source == "Webcam · live":
        _run_webcam_live(cfg, preprocessor, classifier, history, slots, interval)
    else:
        _snapshot(cfg, preprocessor, classifier, history, slots)


if __name__ == "__main__":
    main()
