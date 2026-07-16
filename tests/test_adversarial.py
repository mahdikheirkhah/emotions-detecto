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
    AttackResult,
    fgsm_perturbation,
    probabilities_report,
    run_attack,
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


# ---------------------------------------------------------------------------
# fgsm_perturbation
# ---------------------------------------------------------------------------


def test_fgsm_perturbation_is_signed_epsilon_step() -> None:
    grad = np.array([[-2.0, 0.5], [0.0, 3.0]])
    step = fgsm_perturbation(grad, 0.1)
    np.testing.assert_allclose(step, [[-0.1, 0.1], [0.0, 0.1]])
    assert np.abs(step).max() <= 0.1  # L-infinity == epsilon


# ---------------------------------------------------------------------------
# run_attack — loop logic with fake grad/predict hooks (no TensorFlow)
# ---------------------------------------------------------------------------

_SAD = FER_EMOTIONS.index("Sad")  # 4


def _const_grad(value):
    return lambda x, class_index: np.full_like(np.asarray(x, float), float(value))


def _fixed_predict(argmax_index):
    def predict(x):
        v = np.full(7, 0.1)
        v[argmax_index] = 0.4
        return v

    return predict


class _FlipAfter:
    """Predicts *other* for the first *k* calls, then *target* (for BIM early-stop)."""

    def __init__(self, target: int, other: int, k: int) -> None:
        self.target, self.other, self.k, self.calls = target, other, k, 0

    def __call__(self, x):
        self.calls += 1
        idx = self.target if self.calls > self.k else self.other
        v = np.full(7, 0.05)
        v[idx] = 0.6
        return v


def test_fgsm_steps_against_the_gradient_and_stays_bounded() -> None:
    x = np.full((2, 2, 1), 0.5)
    # Targeted: raise P(target) → step is x - eps*sign(grad); grad > 0 → move down.
    result = run_attack(
        x, _SAD, _const_grad(1.0), _fixed_predict(_SAD), epsilon=0.1, attack_type="fgsm"
    )
    np.testing.assert_allclose(result.adversarial, np.full((2, 2, 1), 0.4))
    np.testing.assert_allclose(result.perturbation, np.full((2, 2, 1), -0.1))
    assert np.abs(result.perturbation).max() <= 0.1 + 1e-9  # within budget
    assert result.iterations == 1
    assert result.success is True  # predict says argmax == Sad


def test_adversarial_is_clipped_to_valid_pixel_range() -> None:
    x = np.full((2, 2, 1), 0.05)  # near 0; a 0.1 step would go negative
    result = run_attack(
        x, _SAD, _const_grad(1.0), _fixed_predict(_SAD), epsilon=0.1, attack_type="fgsm"
    )
    assert result.adversarial.min() >= 0.0  # clipped to [0, 1]
    np.testing.assert_allclose(result.adversarial, 0.0)


def test_bim_stops_early_when_target_reached() -> None:
    x = np.full((3, 3, 1), 0.5)
    # calls: 1=Happy, 2=Happy, 3=Happy, 4=Sad -> flips on iter 3, loop stops there.
    predict = _FlipAfter(target=_SAD, other=FER_EMOTIONS.index("Happy"), k=3)
    result = run_attack(
        x,
        _SAD,
        _const_grad(1.0),
        predict,
        epsilon=0.1,
        attack_type="bim",
        iterations=20,
        step_size=0.01,
    )
    assert result.iterations == 3  # stopped as soon as it flipped, not all 20
    assert result.success is True


def test_bim_perturbation_stays_within_epsilon_ball() -> None:
    x = np.full((3, 3, 1), 0.5)
    predict = _fixed_predict(
        FER_EMOTIONS.index("Happy")
    )  # never flips → runs all steps
    result = run_attack(
        x,
        _SAD,
        _const_grad(1.0),
        predict,
        epsilon=0.05,
        attack_type="bim",
        iterations=50,  # 50 * 0.01 = 0.5 >> 0.05, so clipping must cap it
        step_size=0.01,
    )
    assert np.abs(result.perturbation).max() <= 0.05 + 1e-9  # L-inf ball holds
    assert result.iterations == 50
    assert result.success is False


@pytest.mark.parametrize("bad_epsilon", [0.0, -0.1])
def test_run_attack_non_positive_epsilon_fails_loud(bad_epsilon) -> None:
    with pytest.raises(ValueError):
        run_attack(
            np.zeros((2, 2, 1)),
            _SAD,
            _const_grad(1.0),
            _fixed_predict(_SAD),
            bad_epsilon,
        )


def test_run_attack_unknown_type_fails_loud() -> None:
    with pytest.raises(ValueError):
        run_attack(
            np.zeros((2, 2, 1)),
            _SAD,
            _const_grad(1.0),
            _fixed_predict(_SAD),
            epsilon=0.1,
            attack_type="teleport",
        )


# ---------------------------------------------------------------------------
# keras_attack_functions + run_attack on a tiny real model (TensorFlow)
# ---------------------------------------------------------------------------


def test_attack_raises_target_probability_on_a_real_model() -> None:
    tf = pytest.importorskip("tensorflow")
    from src.emotion_detector.adversarial import keras_attack_functions

    tf.random.set_seed(0)
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input((48, 48, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(7, activation="softmax"),
        ]
    )
    x = np.random.default_rng(0).random((48, 48, 1)).astype(np.float32)
    grad_fn, predict_fn = keras_attack_functions(model)

    result = run_attack(
        x,
        _SAD,
        grad_fn,
        predict_fn,
        epsilon=0.1,
        attack_type="bim",
        iterations=25,
        step_size=0.02,
    )
    # A targeted attack must raise the target-class probability, in budget and in range.
    assert result.adversarial_probs[_SAD] > result.original_probs[_SAD]
    assert np.abs(result.perturbation).max() <= 0.1 + 1e-5
    assert 0.0 <= result.adversarial.min() and result.adversarial.max() <= 1.0


# ---------------------------------------------------------------------------
# save_attack_artifacts (script part 2, fake result — no TF)
# ---------------------------------------------------------------------------


def test_save_attack_artifacts_writes_figure_and_array(tmp_path: Path) -> None:
    script = _load_script()
    adv_img = np.full((48, 48, 1), 0.6, np.float32)
    pert = np.full((48, 48, 1), 0.04, np.float32)
    original_probs = np.zeros(7)
    original_probs[FER_EMOTIONS.index("Happy")] = 0.95
    adversarial_probs = np.zeros(7)
    adversarial_probs[_SAD] = 0.80
    result = AttackResult(
        adversarial=adv_img,
        perturbation=pert,
        iterations=7,
        success=True,
        original_probs=original_probs,
        adversarial_probs=adversarial_probs,
    )
    cfg = {
        "adversarial": {
            "target_class": "Happy",
            "attack_target_class": "Sad",
            "comparison_path": str(tmp_path / "attack_comparison.png"),
            "adversarial_array_path": str(tmp_path / "adversarial_array.npy"),
        }
    }
    saved = script.save_attack_artifacts(cfg, result)
    assert Path(saved["comparison_path"]).exists()  # side-by-side figure
    loaded = np.load(saved["adversarial_array_path"])
    assert loaded.shape == (48, 48, 1)  # the adversarial model input, for the record
    np.testing.assert_allclose(loaded, adv_img)
