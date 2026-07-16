"""Canonical FER-2013 emotion labels — the one class-index → name mapping.

The model's softmax output index ``i`` maps to ``FER_EMOTIONS[i]``. This order is the
FER-2013 ``emotion`` column (0-6) the trainer one-hot-encodes, so it is the one source
of truth shared by evaluation (confusion-matrix axes) and live inference (#54), never
re-typed in two places.
"""

from __future__ import annotations

from typing import List

FER_EMOTIONS: List[str] = [
    "Angry",  # 0
    "Disgust",  # 1
    "Fear",  # 2
    "Happy",  # 3
    "Sad",  # 4
    "Surprise",  # 5
    "Neutral",  # 6
]


def emotion_labels(cfg: dict) -> List[str]:
    """Return the label list for the configured ``model.num_classes``.

    Raises:
        ValueError: if ``num_classes`` does not match the FER-2013 label count — a
            mismatch means the label mapping and the model head disagree, which would
            silently mislabel every prediction.
    """
    n = int(cfg["model"]["num_classes"])
    if n != len(FER_EMOTIONS):
        raise ValueError(
            f"model.num_classes={n} but {len(FER_EMOTIONS)} FER-2013 labels are "
            "defined; the label mapping and the model output would disagree."
        )
    return list(FER_EMOTIONS)
