"""Live-frame preprocessing — make a webcam frame look like an FER-2013 sample.

The model was trained on FER-2013: **48×48, grayscale, face-centered**. A raw webcam
frame is none of those, so feeding it straight to the classifier collapses accuracy —
this is the *train/inference gap*. ``FacePreprocessor`` closes it, turning any frame
into the exact format the model saw during training:

    detect the (largest) face  →  square-crop centered on the box  →
    resize to 48×48 (area interpolation when downscaling)  →  BGR→grayscale

``process_frame`` returns that 48×48 grayscale crop — the viewable image saved by the
functional ``preprocessing_test`` deliverable (Issue #52). ``to_model_input`` then
applies the **same fitted normalizer** used in training (rescale / equalization /
standardize — reused, never re-derived), so the live tensor matches the training tensor
bit-for-bit. Detector, image size, and transforms all come from ``config.yaml`` (the
same knobs training uses), so the pipeline is one config switch away from any ablation
(§3).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.video.base import BaseFaceDetector, FaceRect
from src.emotion_detector.video.factory import build_face_detector


class NoFaceError(Exception):
    """No face was found in a frame — the caller should skip it, not crash.

    Distinct from ``ValueError`` (a *malformed* frame): an empty frame is a normal,
    expected event in a live stream (the person looked away), so the sampling loop
    skips it, while a bad-shape frame is a real bug that must surface (CONTRIBUTING §9).
    """


class FacePreprocessor:
    """Detect → crop → center → 48×48 grayscale, with training-matched normalization.

    Args:
        cfg: Loaded config. Reads ``preprocessing.image_size`` (output side) and builds
            the ``face_detector.*`` backend + the ``preprocessing.normalization`` step.
        detector: Optional pre-built ``BaseFaceDetector`` (injected for tests); defaults
            to ``build_face_detector(cfg)``.
        normalizer: Optional pre-built normalizer (injected for tests); defaults to
            ``build_normalizer(cfg)`` — the *same* class training fits.
    """

    def __init__(
        self,
        cfg: dict,
        detector: Optional[BaseFaceDetector] = None,
        normalizer: Optional[Any] = None,
    ) -> None:
        self._size = int(cfg["preprocessing"]["image_size"])
        self._detector = detector if detector is not None else build_face_detector(cfg)
        self._normalizer = (
            normalizer if normalizer is not None else build_normalizer(cfg)
        )

    def _detect_largest(self, frame: NDArray) -> FaceRect:
        """The largest detected face box (the subject); NoFaceError if none.

        Raises:
            NoFaceError: if the detector finds no face (a frame to skip).
            ValueError: if *frame* is malformed (propagated from the detector).
        """
        faces = self._detector.detect(frame)  # raises ValueError on a bad frame (§9)
        if not faces:
            raise NoFaceError("No face detected in frame.")
        return max(faces, key=lambda b: b[2] * b[3])  # largest by area = the subject

    def _crop_to_model_size(self, frame: NDArray, box: FaceRect) -> NDArray:
        """Square-crop *box*, resize to ``image_size``, gray -> ``(size, size)`` uint8.

        Raises:
            ValueError: if OpenCV fails on a degenerate crop.
        """
        crop = self._square_crop(frame, box)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            interp = cv2.INTER_AREA if crop.shape[0] >= self._size else cv2.INTER_LINEAR
            return cv2.resize(gray, (self._size, self._size), interpolation=interp)
        except cv2.error as exc:  # degenerate crop that slipped past the checks
            raise ValueError(f"Could not preprocess the face crop: {exc}") from exc

    def process_frame(self, frame: NDArray) -> NDArray:
        """Return the FER-format ``(size, size)`` uint8 grayscale crop of the face.

        Picks the **largest** detected face (the subject in frame), squares the box
        around its centre so the face isn't stretched, resizes to ``image_size`` (area
        interpolation down, linear up), and converts to grayscale.

        Raises:
            NoFaceError: if the detector finds no face (a frame to skip).
            ValueError: if *frame* is malformed (propagated from the detector) or OpenCV
                fails on the crop.
        """
        return self._crop_to_model_size(frame, self._detect_largest(frame))

    def to_model_input(self, frame: NDArray) -> NDArray:
        """The classifier-ready array: ``process_frame`` + the fitted normalizer.

        Applies the exact ``preprocessing.normalization`` used in training (rescale to
        ``[0, 1]`` by default), so the live input distribution matches training's.
        """
        return self._normalizer.transform(self.process_frame(frame))

    def locate_and_prepare(self, frame: NDArray) -> Tuple[FaceRect, NDArray]:
        """Detect once, returning ``(face_box, model_ready_tensor)`` for the dashboard.

        The overlay (#55) needs the **box** to draw the rectangle *and* the normalized
        tensor to classify; doing both off a single detection avoids detecting twice.

        Returns:
            ``((x, y, w, h), tensor)`` — the largest face's pixel box and its normalized
            ``(size, size)`` array (same as ``to_model_input``).

        Raises:
            NoFaceError: if no face is detected.
            ValueError: if *frame* is malformed or the crop fails.
        """
        box = self._detect_largest(frame)
        tensor = self._normalizer.transform(self._crop_to_model_size(frame, box))
        return box, tensor

    @staticmethod
    def _square_crop(frame: NDArray, box: FaceRect) -> NDArray:
        """Square, centered crop around *box*, padded (replicate) if it hits an edge.

        Squaring keeps the face's aspect ratio through the resize; padding rather than
        clamping keeps the face **centered** even when the box runs off the frame.
        """
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        half = max(w, h) / 2.0
        x1, y1 = int(round(cx - half)), int(round(cy - half))
        x2, y2 = int(round(cx + half)), int(round(cy + half))

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w_frame, x2), min(h_frame, y2)
        if x2c <= x1c or y2c <= y1c:
            raise ValueError(f"Face box {box} lies outside the {frame.shape} frame.")

        crop = frame[y1c:y2c, x1c:x2c]
        top, bottom = y1c - y1, y2 - y2c
        left, right = x1c - x1, x2 - x2c
        if top or bottom or left or right:  # box ran off an edge → pad back to square
            crop = cv2.copyMakeBorder(
                crop, top, bottom, left, right, cv2.BORDER_REPLICATE
            )
        return crop
