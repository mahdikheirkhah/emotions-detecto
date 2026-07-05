"""Haar-cascade face detection (Viola–Jones) via OpenCV — the workhorse detector.

**How it works.** A sliding window scans the frame at many positions and scales; each
window is classified face / not-face. The classifier uses **Haar features** — sums of
pixels in light vs dark rectangles — computed in O(1) from an **integral image**
(a running-sum table), so evaluating a window is cheap. The **cascade** chains many
such classifiers of increasing strictness: a non-face window is rejected by an early,
cheap stage, so the detector spends real work only on face-like regions.

Key parameters (from ``config.yaml``), all precision/recall knobs:
  * ``scaleFactor`` — how much the window grows each scale (smaller = finer scan, more
    detections but slower and more false positives).
  * ``minNeighbors`` — how many overlapping detections a region needs to count (higher =
    fewer false positives, but more misses).
  * ``minSize`` — smallest face to look for (skips tiny noise).

One ``model.architecture``-style dispatch option (``face_detector.backend: haar``); the
cascade + params live entirely in config (Ablation §3).
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.utils.logging import logger
from src.emotion_detector.video.base import BaseFaceDetector, FaceRect


class HaarFaceDetector(BaseFaceDetector):
    """Detect faces with an OpenCV Haar cascade (``cv2.CascadeClassifier``)."""

    def __init__(self, cfg: dict) -> None:
        fd = cfg["face_detector"]
        self._scale_factor = float(fd.get("scale_factor", 1.1))
        self._min_neighbors = int(fd.get("min_neighbors", 5))
        self._min_size = tuple(fd.get("min_size", [30, 30]))

        cascade_path = self._resolve_cascade(fd["haar_cascade"])
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():  # unreadable / not a valid cascade XML
            raise FileNotFoundError(
                f"Failed to load Haar cascade from '{cascade_path}' (empty classifier)."
            )
        logger.info(
            f"HaarFaceDetector ready — cascade={os.path.basename(cascade_path)}, "
            f"scaleFactor={self._scale_factor}, minNeighbors={self._min_neighbors}, "
            f"minSize={self._min_size}."
        )

    @staticmethod
    def _resolve_cascade(name: str) -> str:
        """Resolve a cascade name to a path: an explicit path, else OpenCV's bundle.

        Raises:
            FileNotFoundError: if it is neither an existing path nor a bundled cascade.
        """
        if os.path.isabs(name) or os.path.sep in name:
            if os.path.exists(name):
                return name
            raise FileNotFoundError(f"Haar cascade not found at path: '{name}'.")
        bundled = os.path.join(cv2.data.haarcascades, name)
        if os.path.exists(bundled):
            return bundled
        raise FileNotFoundError(
            f"Haar cascade '{name}' not found (checked '{name}' and OpenCV's bundle "
            f"'{bundled}'). Give an absolute path or a filename OpenCV ships."
        )

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
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
            )
        except cv2.error as exc:  # malformed frame that passed the shape check
            raise ValueError(f"OpenCV could not process the frame: {exc}") from exc

        # detectMultiScale returns () when empty, else an (N, 4) int array.
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
