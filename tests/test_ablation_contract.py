"""§3 guard: every ablation strategy option dispatches to a class; unknowns fail loud.

The Ablation-Driven Architecture (CONTRIBUTING §3) promises that each ``config.yaml``
strategy value maps through a switch-case ``dispatch`` to a concrete class, and that an
unrecognized value raises **loudly** rather than silently no-op'ing. This consolidates
that contract across the strategy dimensions in one place, so a typo'd option (or a
documented option that was never wired) is caught here, not found as a silent bug at
train time.

Only the light (no-TensorFlow) dimensions are *built* here; the heavy ones (model
architecture, DNN detector) are covered for construction in ``test_builders`` /
``test_factory``, so this file asserts their **fail-loud** side — what §3 rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.base import BaseImagePreprocessor
from src.emotion_detector.data.imbalance import resolve_imbalance
from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.models.builders import build_model
from src.emotion_detector.video.factory import build_face_detector

# The documented options (mirrors the `# options:` comments in config.yaml).
NORMALIZATION_OPTIONS = ["none", "rescale", "standardize", "histogram_eq", "clahe"]
IMBALANCE_OPTIONS = ["none", "class_weight", "oversample", "undersample"]


# ---------------------------------------------------------------------------
# normalization (build_normalizer) — build every option
# ---------------------------------------------------------------------------


def _norm_cfg(option: str) -> dict:
    return {
        "stages": {"preprocessing": True},
        "preprocessing": {
            "normalization": option,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": 8,
        },
    }


@pytest.mark.parametrize("option", NORMALIZATION_OPTIONS)
def test_every_normalization_option_builds_a_preprocessor(option) -> None:
    assert isinstance(build_normalizer(_norm_cfg(option)), BaseImagePreprocessor)


def test_unknown_normalization_fails_loud() -> None:
    with pytest.raises(ValueError):
        build_normalizer(_norm_cfg("sharpen"))


# ---------------------------------------------------------------------------
# imbalance (resolve_imbalance) — dispatch every option
# ---------------------------------------------------------------------------


def _imb_cfg(option: str) -> dict:
    return {
        "global": {"seed": 42},
        "cleaning": {"handle_imbalance": True, "imbalance_strategy": option},
    }


@pytest.mark.parametrize("option", IMBALANCE_OPTIONS)
def test_every_imbalance_option_dispatches(option) -> None:
    X = np.zeros((6, 4, 4), np.float32)
    y = np.array([0, 0, 0, 1, 1, 2])  # imbalanced multiclass
    X_out, y_out, class_weight = resolve_imbalance(_imb_cfg(option), X, y)
    assert len(X_out) == len(y_out)  # a strategy ran and returned aligned data


def test_unknown_imbalance_fails_loud() -> None:
    X = np.zeros((3, 4, 4), np.float32)
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError):
        resolve_imbalance(_imb_cfg("smote"), X, y)


# ---------------------------------------------------------------------------
# heavy dimensions — assert the fail-loud side (dispatch raises before building)
# ---------------------------------------------------------------------------


def test_unknown_architecture_fails_loud() -> None:
    cfg = {
        "model": {"architecture": "transformer_xl", "num_classes": 7},
        "preprocessing": {"image_size": 48, "grayscale": True},
    }
    with pytest.raises(ValueError):
        build_model(cfg)


def test_unknown_face_detector_backend_fails_loud() -> None:
    with pytest.raises(ValueError):
        build_face_detector({"face_detector": {"backend": "teleport"}})
