"""Unit tests for the adversarial-setup logic (#57) — no TensorFlow, no model.

The source-selection logic and the report/target-index helpers are pure; the script's
``find_and_save`` is exercised with fake probabilities + tiny image arrays (matplotlib
writes to a headless Agg backend), so the whole part-1 flow is verified without a model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.emotion_detector.adversarial import (
    probabilities_report,
    select_target_image,
    target_index,
)
from src.emotion_detector.models.labels import FER_EMOTIONS

_HAPPY = FER_EMOTIONS.index("Happy")  # 3
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "adversarial.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("adversarial_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# target_index
# ---------------------------------------------------------------------------


def test_target_index_maps_happy_to_three() -> None:
    assert target_index("Happy") == 3


def test_target_index_unknown_fails_loud() -> None:
    with pytest.raises(ValueError):
        target_index("Ecstatic")


# ---------------------------------------------------------------------------
# select_target_image
# ---------------------------------------------------------------------------


def _probs(*rows) -> np.ndarray:
    return np.array(rows, dtype=float)


def test_selects_highest_confidence_matching_image() -> None:
    # Rows 1 and 3 are predicted Happy above 0.9; row 3 is more confident.
    probs = _probs(
        [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1],  # 0: Happy but only 0.4
        [0.0, 0.0, 0.02, 0.93, 0.03, 0.02, 0.0],  # 1: Happy 0.93
        [0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0],  # 2: Sad
        [0.0, 0.0, 0.01, 0.97, 0.02, 0.0, 0.0],  # 3: Happy 0.97  ← winner
    )
    idx, confidence = select_target_image(probs, _HAPPY, 0.9)
    assert idx == 3
    assert confidence == pytest.approx(0.97)


def test_threshold_is_strict_and_class_must_be_argmax() -> None:
    probs = _probs(
        [0.0, 0.0, 0.0, 0.90, 0.10, 0.0, 0.0],  # Happy exactly 0.90 (not > 0.90)
        [0.0, 0.0, 0.0, 0.55, 0.45, 0.0, 0.0],  # Happy is argmax but < threshold
    )
    with pytest.raises(ValueError):
        select_target_image(probs, _HAPPY, 0.9)


def test_no_matching_image_fails_loud() -> None:
    probs = _probs([0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0])  # predicted Angry
    with pytest.raises(ValueError):
        select_target_image(probs, _HAPPY, 0.9)


def test_requires_2d_probabilities() -> None:
    with pytest.raises(ValueError):
        select_target_image(np.zeros(7), _HAPPY, 0.9)  # a single vector, not (N, C)


# ---------------------------------------------------------------------------
# probabilities_report
# ---------------------------------------------------------------------------


def test_probabilities_report_maps_labels() -> None:
    vector = [0.01, 0.02, 0.02, 0.9, 0.03, 0.01, 0.01]
    report = probabilities_report(vector)
    assert set(report) == set(FER_EMOTIONS)
    assert report["Happy"] == pytest.approx(0.9)


def test_probabilities_report_wrong_length_fails_loud() -> None:
    with pytest.raises(ValueError):
        probabilities_report([0.5, 0.5])


# ---------------------------------------------------------------------------
# find_and_save (script core, fake probs — no TF)
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, threshold=0.9) -> dict:
    return {
        "adversarial": {
            "target_class": "Happy",
            "source_confidence_threshold": threshold,
            "epsilon": 0.01,
            "scan_split": "test",
            "source_image_path": str(tmp_path / "source_image.png"),
            "source_array_path": str(tmp_path / "source_array.npy"),
            "source_probs_path": str(tmp_path / "source_probabilities.json"),
        }
    }


def test_find_and_save_writes_three_artifacts(tmp_path: Path) -> None:
    script = _load_script()
    probs = _probs(
        [0.1, 0.1, 0.1, 0.2, 0.5, 0.0, 0.0],  # 0: Sad
        [0.0, 0.0, 0.01, 0.95, 0.04, 0.0, 0.0],  # 1: Happy 0.95 ← winner
    )
    images_uint8 = np.stack(
        [np.full((48, 48), 30, np.uint8), np.full((48, 48), 200, np.uint8)]
    )
    images_norm = (images_uint8[..., np.newaxis] / 255.0).astype(np.float32)

    summary = script.find_and_save(_cfg(tmp_path), probs, images_uint8, images_norm)

    assert summary["index"] == 1
    assert summary["confidence"] == pytest.approx(0.95)
    assert summary["label"] == "Happy"
    # 1. viewable PNG
    assert (tmp_path / "source_image.png").exists()
    # 2. normalized model array for #58 — the chosen row, (48, 48, 1)
    saved = np.load(tmp_path / "source_array.npy")
    assert saved.shape == (48, 48, 1)
    np.testing.assert_allclose(saved, images_norm[1])
    # 3. full probability report
    report = json.loads((tmp_path / "source_probabilities.json").read_text())
    assert set(report) == set(FER_EMOTIONS)
    assert report["Happy"] == pytest.approx(0.95)


def test_find_and_save_no_confident_source_raises(tmp_path: Path) -> None:
    script = _load_script()
    probs = _probs([0.2, 0.1, 0.1, 0.3, 0.3, 0.0, 0.0])  # Happy only 0.3
    images_uint8 = np.full((1, 48, 48), 100, np.uint8)
    images_norm = (images_uint8[..., np.newaxis] / 255.0).astype(np.float32)
    with pytest.raises(ValueError):
        script.find_and_save(_cfg(tmp_path), probs, images_uint8, images_norm)
