"""Frame annotation for the live dashboard — draw the face box + emotion label (#55).

Pure drawing on a BGR frame (no GUI, no capture), so the whole visual layer is unit-
testable headless: ``annotate`` stamps a rectangle around the detected face and writes
``"Happy 73%"`` above it, colour-coded by confidence. Keeping it separate from the
render loop (``scripts/predict_live_stream.py``) means the loop orchestrates read ->
predict -> **annotate** -> show, and the pixel-level drawing is verified on its own.

**Confidence UX.** The box + text colour encodes how sure the model is — green when it's
confident, amber when it's hedging, red when it's barely above chance — so a clinician
reads certainty at a glance, not just the label (CONTRIBUTING: show probability).
"""

from __future__ import annotations

from typing import Tuple

import cv2
from numpy.typing import NDArray

from src.emotion_detector.video.base import FaceRect

# BGR (OpenCV's channel order), not RGB.
_GREEN = (0, 200, 0)
_AMBER = (0, 190, 235)
_RED = (0, 0, 220)

_HIGH_CONFIDENCE = 0.66  # >= this → green (model is sure)
_LOW_CONFIDENCE = 0.33  # < this → red (barely above 1/7 chance); between → amber

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_FONT_THICKNESS = 2
_BOX_THICKNESS = 2


def confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Map a softmax confidence to a BGR colour: green (sure) → amber → red (unsure)."""
    if confidence >= _HIGH_CONFIDENCE:
        return _GREEN
    if confidence >= _LOW_CONFIDENCE:
        return _AMBER
    return _RED


def format_overlay_text(label: str, confidence: float) -> str:
    """The overlay caption, e.g. ``"Happy 73%"`` (``:.0%`` matches the printed line)."""
    return f"{label} {confidence:.0%}"


def draw_face_box(
    frame: NDArray, box: FaceRect, color: Tuple[int, int, int]
) -> NDArray:
    """Draw the face rectangle in place and return *frame*."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, _BOX_THICKNESS)
    return frame


def draw_label(
    frame: NDArray, box: FaceRect, text: str, color: Tuple[int, int, int]
) -> NDArray:
    """Write *text* just above the box (or below it if the box hugs the top edge)."""
    x, y, _, h = box
    above = y - 10
    text_y = above if above > 15 else y + h + 25  # keep the caption on-screen
    cv2.putText(
        frame,
        text,
        (x, text_y),
        _FONT,
        _FONT_SCALE,
        color,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def annotate(frame: NDArray, box: FaceRect, label: str, confidence: float) -> NDArray:
    """Draw the confidence-coloured box + ``"label conf%"`` caption on *frame*."""
    color = confidence_color(confidence)
    draw_face_box(frame, box, color)
    draw_label(frame, box, format_overlay_text(label, confidence), color)
    return frame
