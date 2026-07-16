"""Real-time loop, front half: sample ~N frames/sec and preprocess them (Issue #53).

The webcam pushes 30 frames a second, but the audit only needs **>=1 prediction per
second**, and detection + a CNN forward pass per frame wouldn't fit that budget anyway.
So we *sample*: emit roughly ``video.predictions_per_second`` frames each second, run
only those through ``FacePreprocessor`` (#52), and hand the model-ready tensors to the
predictor (#54). This is the clean seam between **capture** (``VideoSource``, #51) and
**prediction**: a generator of ``(timestamp, tensor)`` pairs.

**Why FPS-derived timestamps, not wall-clock.** Each frame's time is ``index / fps``, a
property of the *stream*, not of how fast we decode it. That matters because a recorded
file decodes far faster than real time: pacing by ``time.time()`` would rip through a
3-second clip in milliseconds and emit a single sample. Deriving the timestamp from the
frame index instead makes sampling identical for live and recorded input (§3) and
deterministic to test. For a live webcam, frames arrive at real time, so index/fps still
tracks the wall clock.

**No-face frames** are skipped (not emitted, never fatal): the loop samples ~1/sec, and
a sampled second with no visible face just yields no tensor while capture keeps rolling.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

from numpy.typing import NDArray

from src.emotion_detector.utils.logging import logger
from src.emotion_detector.video.preprocess import FacePreprocessor, NoFaceError

# A yielded sample: (timestamp in seconds from stream start, model-ready 48x48 tensor).
Sample = Tuple[float, NDArray]


def preprocessed_frames(
    source: Any,
    cfg: dict,
    preprocessor: Optional[FacePreprocessor] = None,
) -> Iterator[Sample]:
    """Yield ``(timestamp, tensor)`` ~``predictions_per_second`` times per second.

    Args:
        source: An open ``VideoSource`` (#51): any object exposing ``frames()`` and an
            ``fps`` property. Its lifecycle (release) belongs to the caller, typically
            via ``with VideoSource(cfg) as src: ...``.
        cfg: Loaded config. Reads ``video.predictions_per_second`` (the sampling rate)
            and, when no ``preprocessor`` is given, builds ``FacePreprocessor(cfg)``.
        preprocessor: Optional injected ``FacePreprocessor`` (tests); else from cfg.

    Yields:
        ``(timestamp_seconds, tensor)`` where ``tensor`` is the normalized 48×48 array
        from ``FacePreprocessor.to_model_input`` — exactly what the predictor (#54)
        consumes. Sampled frames with no detected face are skipped (nothing yielded).

    Raises:
        ValueError: if ``video.predictions_per_second`` is not > 0, or a frame is
            malformed (propagated from the detector).
    """
    pps = float(cfg["video"].get("predictions_per_second", 1))
    if pps <= 0:
        raise ValueError(f"video.predictions_per_second must be > 0, got {pps}.")
    interval = 1.0 / pps

    pre = preprocessor if preprocessor is not None else FacePreprocessor(cfg)
    fps = (
        float(source.fps) or 30.0
    )  # VideoSource already defaults; never divide by zero

    next_sample = 0.0  # earliest stream-time at which to take the next sample
    for index, frame in enumerate(source.frames()):
        timestamp = index / fps
        if timestamp + 1e-9 < next_sample:
            continue  # between sample boundaries — not this frame
        next_sample = (
            timestamp + interval
        )  # schedule the next one off the frame we took

        try:
            tensor = pre.to_model_input(frame)
        except NoFaceError:
            logger.debug(f"No face at t={timestamp:.2f}s — skipping this sample.")
            continue  # a face-less second yields nothing; the stream keeps rolling (§9)

        logger.info(f"Preprocessing ... (t={timestamp:.2f}s)")
        yield timestamp, tensor
