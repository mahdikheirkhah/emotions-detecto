"""Abstract base class for face detection backends."""

from __future__ import annotations

import abc
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

# A detected face is (x, y, w, h) in pixel coordinates.
FaceRect = Tuple[int, int, int, int]


class BaseFaceDetector(abc.ABC):
    """Contract for detecting faces in a single image frame.

    Concrete subclasses (e.g. ``HaarFaceDetector``, ``DnnFaceDetector``)
    implement ``detect`` for a specific OpenCV backend.  The pipeline always
    holds a ``BaseFaceDetector`` reference so the backend can be swapped via
    ``cfg["face_detector"]["backend"]`` + dispatch.
    """

    @abc.abstractmethod
    def detect(self, frame: NDArray) -> List[FaceRect]:
        """Detect all faces in *frame* and return their bounding boxes.

        Args:
            frame: BGR image array of shape ``(H, W, 3)``, as returned by
                ``cv2.VideoCapture.read()``.

        Returns:
            List of ``(x, y, w, h)`` tuples — one per detected face — in
            pixel coordinates.  Returns an empty list when no face is found.

        Raises:
            ValueError: if *frame* has an unexpected shape or dtype.
        """
