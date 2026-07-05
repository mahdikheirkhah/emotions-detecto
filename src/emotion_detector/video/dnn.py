"""DNN face detection via OpenCV's ``cv2.dnn`` — an SSD/ResNet caffe model.

A **deep** detector: a single-shot detector (SSD) with a ResNet base, trained on many
faces, so it generalizes to pose/lighting far better than Haar's fixed rectangle
features — at a higher compute cost. OpenCV's ``cv2.dnn`` acts as a *runtime*: it loads
a model trained in another framework (Caffe here) and runs inference, no Caffe needed.

The pipeline each frame:
  1. ``blobFromImage`` — resize to 300x300, subtract the training mean (104, 177, 123)
     per BGR channel, and pack into the network's NCHW input tensor (the "blob").
  2. ``net.setInput`` + ``net.forward`` — one forward pass emits an ``(1, 1, N, 7)``
     tensor: N candidate boxes, each ``[img, class, confidence, x1, y1, x2, y2]`` with
     coordinates normalized to ``[0, 1]``.
  3. Keep boxes whose ``confidence >= face_detector.confidence_threshold`` and scale the
     normalized corners back to pixel ``(x, y, w, h)``. A higher threshold trades recall
     (misses) for precision (fewer false positives).

Second dispatch option (``face_detector.backend: dnn``) — a direct ablation of detection
quality against ``haar`` on the same video (Ablation §3).
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.utils.logging import logger
from src.emotion_detector.video.base import BaseFaceDetector, FaceRect

_INPUT_SIZE = (300, 300)
_MEAN = (104.0, 177.0, 123.0)  # BGR training mean the model expects subtracted
_SCALE = 1.0


class DnnFaceDetector(BaseFaceDetector):
    """Detect faces with OpenCV's SSD/ResNet caffe model (``cv2.dnn``)."""

    def __init__(self, cfg: dict) -> None:
        fd = cfg["face_detector"]
        self._confidence = float(fd.get("confidence_threshold", 0.5))
        prototxt = self._require(fd["dnn_prototxt"], "prototxt")
        model = self._require(fd["dnn_model"], "caffemodel")
        try:
            self._net = cv2.dnn.readNetFromCaffe(prototxt, model)
        except cv2.error as exc:  # unreadable / mismatched files
            raise FileNotFoundError(
                f"Failed to load DNN model from '{prototxt}' + '{model}': {exc}"
            ) from exc
        logger.info(
            f"DnnFaceDetector ready — model={os.path.basename(model)}, "
            f"confidence_threshold={self._confidence}."
        )

    @staticmethod
    def _require(path: str, kind: str) -> str:
        """Return *path* if it exists, else FileNotFoundError with a download hint."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DNN {kind} not found at '{path}'. Run "
                "`python scripts/download_face_model.py` to fetch it, or point "
                f"face_detector.dnn_{'prototxt' if kind == 'prototxt' else 'model'} "
                "at an existing file."
            )
        return path

    @staticmethod
    def _parse_detections(
        detections: NDArray, width: int, height: int, threshold: float
    ) -> List[FaceRect]:
        """Filter the SSD ``(1,1,N,7)`` tensor by confidence → pixel boxes.

        Rows are ``[img, class, confidence, x1, y1, x2, y2]`` with normalized corners;
        boxes are scaled to the frame and clamped to its bounds. Pure (no cv2) so the
        confidence/box logic is testable without the model.
        """
        faces: List[FaceRect] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def detect(self, frame: NDArray) -> List[FaceRect]:
        """``(x, y, w, h)`` boxes for every face in a BGR *frame* (``[]`` if none).

        Raises:
            ValueError: if *frame* is not an ``(H, W, 3)`` array, or OpenCV fails on it.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected a BGR (H, W, 3) frame; got "
                f"{type(frame).__name__} shape {getattr(frame, 'shape', None)}."
            )
        height, width = frame.shape[:2]
        try:
            blob = cv2.dnn.blobFromImage(frame, _SCALE, _INPUT_SIZE, _MEAN)
            self._net.setInput(blob)
            detections = self._net.forward()
        except cv2.error as exc:
            raise ValueError(f"OpenCV DNN could not process the frame: {exc}") from exc
        return self._parse_detections(detections, width, height, self._confidence)
