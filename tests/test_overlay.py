"""Unit tests for the dashboard drawing layer (#55) — pure, headless (no GUI).

Every function draws on a NumPy frame, so we assert on pixels directly: the confidence
colour mapping, the caption text, and that the rectangle/label actually mark the frame.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.emotion_detector.video import overlay

_GREEN = (0, 200, 0)
_AMBER = (0, 190, 235)
_RED = (0, 0, 220)


# ---------------------------------------------------------------------------
# confidence colour + caption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.95, _GREEN),
        (0.66, _GREEN),  # boundary: >= 0.66 is green
        (0.65, _AMBER),
        (0.50, _AMBER),
        (0.33, _AMBER),  # boundary: >= 0.33 is amber
        (0.32, _RED),
        (0.05, _RED),
    ],
)
def test_confidence_color_bands(confidence, expected) -> None:
    assert overlay.confidence_color(confidence) == expected


def test_format_overlay_text() -> None:
    assert overlay.format_overlay_text("Happy", 0.734) == "Happy 73%"
    assert overlay.format_overlay_text("Sad", 0.6) == "Sad 60%"


# ---------------------------------------------------------------------------
# drawing marks the frame
# ---------------------------------------------------------------------------


def _blank() -> np.ndarray:
    return np.zeros((200, 200, 3), np.uint8)


def test_draw_face_box_draws_in_place() -> None:
    frame = _blank()
    out = overlay.draw_face_box(frame, (50, 60, 40, 40), _GREEN)
    assert out is frame  # mutates and returns the same array
    assert tuple(int(c) for c in frame[60, 50]) == _GREEN  # box corner is coloured


def test_draw_label_writes_pixels() -> None:
    frame = _blank()
    overlay.draw_label(frame, (50, 60, 40, 40), "Happy 73%", _GREEN)
    assert frame.any()  # text was rendered somewhere


def test_draw_label_stays_on_screen_near_top_edge() -> None:
    # A box hugging the top would push text off-screen (negative y); it must drop below.
    frame = _blank()
    overlay.draw_label(frame, (50, 5, 40, 40), "Hi", _GREEN)
    assert not frame[:5].any()  # nothing drawn above the near-top box
    assert frame.any()  # caption rendered below it instead


def test_annotate_boxes_and_labels_by_confidence() -> None:
    frame = _blank()
    out = overlay.annotate(frame, (50, 60, 40, 40), "Happy", 0.9)
    assert out is frame
    assert tuple(int(c) for c in frame[60, 50]) == _GREEN  # confident → green box


def test_annotate_low_confidence_is_red() -> None:
    frame = _blank()
    overlay.annotate(frame, (50, 60, 40, 40), "Fear", 0.2)
    assert tuple(int(c) for c in frame[60, 50]) == _RED
