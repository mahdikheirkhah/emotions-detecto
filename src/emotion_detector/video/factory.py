"""Face-detector dispatch — ``face_detector.backend`` selects the backend.

``haar`` (fast, classic Viola–Jones) vs ``dnn`` (SSD/ResNet, robust to pose/lighting)
is a one-line config switch, so the two can be ablated on the same video to measure
detection quality's effect on emotion accuracy (Ablation §3). Backends are imported
lazily so selecting one never imports the other's dependencies.
"""

from __future__ import annotations

from typing import Any

from src.emotion_detector.utils.dispatch import dispatch


def build_face_detector(cfg: dict) -> Any:
    """Build the ``BaseFaceDetector`` selected by ``face_detector.backend``.

    Args:
        cfg: Loaded config (reads ``face_detector.backend``: ``haar`` | ``dnn``).

    Returns:
        A ready ``HaarFaceDetector`` or ``DnnFaceDetector``.

    Raises:
        ValueError: if ``face_detector.backend`` is not a known option.
    """

    def _haar():
        from src.emotion_detector.video.haar import HaarFaceDetector

        return HaarFaceDetector(cfg)

    def _dnn():
        from src.emotion_detector.video.dnn import DnnFaceDetector

        return DnnFaceDetector(cfg)

    registry = {"haar": _haar, "dnn": _dnn}
    return dispatch(cfg["face_detector"]["backend"], registry)
