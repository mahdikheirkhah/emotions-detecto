"""Unit tests for optional post-training quantization (Issue #47).

Needs TensorFlow (TFLite is part of it). Uses a tiny model + random data so each
conversion is fast; the key guarantee is that a quantized model still predicts a
valid class in every mode.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.emotion_detector.models.optimize import (
    TFLitePredictor,
    optimize_model,
    polynomial_decay_sparsity,
    prune,
    prune_and_report,
    quantize,
    save_tflite,
    sparsity,
)


def _model():
    from tensorflow import keras
    from tensorflow.keras import layers

    m = keras.Sequential(
        [
            keras.Input((48, 48, 1)),
            layers.Conv2D(8, 3, activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(7, activation="softmax"),
        ]
    )
    m.compile("adam", "categorical_crossentropy")
    return m


def _cfg(mode="float16") -> dict:
    return {
        "optimization": {
            "quantization": mode,
            "max_accuracy_drop": 0.05,
            "representative_samples": 10,
        },
        "model": {"num_classes": 7},
    }


def _data(n=12):
    rng = np.random.default_rng(0)
    return rng.random((n, 48, 48, 1)).astype("float32"), rng.integers(0, 7, n)


# ---------------------------------------------------------------------------
# every mode still predicts a valid class (the core requirement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "dynamic", "float16", "int8"])
def test_quantized_model_predicts_valid_class(mode) -> None:
    X, _ = _data()
    tflite = quantize(_model(), _cfg(mode), representative_data=X[:10])
    cls = TFLitePredictor(tflite).predict_classes(X[:5])
    assert cls.shape == (5,)
    assert cls.min() >= 0 and cls.max() < 7  # valid class indices


def test_predict_returns_probability_shape() -> None:
    X, _ = _data()
    tflite = quantize(_model(), _cfg("float16"))
    probs = TFLitePredictor(tflite).predict(X[:4])
    assert probs.shape == (4, 7)


# ---------------------------------------------------------------------------
# fail-loud
# ---------------------------------------------------------------------------


def test_unknown_mode_fails_loud() -> None:
    with pytest.raises(ValueError):
        quantize(_model(), _cfg("bogus"))


def test_int8_without_representative_fails_loud() -> None:
    with pytest.raises(ValueError):
        quantize(_model(), _cfg("int8"), representative_data=None)


# ---------------------------------------------------------------------------
# the before/after measurement report
# ---------------------------------------------------------------------------


def test_optimize_model_reports_size_accuracy_latency() -> None:
    X, y = _data(16)
    report = optimize_model(_model(), _cfg("float16"), X, y)
    for key in (
        "keras_size_bytes",
        "tflite_size_bytes",
        "keras_accuracy",
        "tflite_accuracy",
        "accuracy_drop",
        "keras_latency_ms",
        "tflite_latency_ms",
        "speedup",
        "passed",
    ):
        assert key in report
    assert isinstance(report["passed"], bool)
    assert 0.0 <= report["tflite_accuracy"] <= 1.0


def test_save_tflite_writes_file(tmp_path) -> None:
    tflite = quantize(_model(), _cfg("none"))
    path = tmp_path / "m.tflite"
    save_tflite(tflite, str(path))
    assert path.exists() and path.stat().st_size == len(tflite)


# ---------------------------------------------------------------------------
# weight pruning (#48)
# ---------------------------------------------------------------------------


def _prune_cfg(target=0.5, epochs=1) -> dict:
    return {
        "optimization": {
            "max_accuracy_drop": 0.5,
            "pruning": {
                "target_sparsity": target,
                "fine_tune_epochs": epochs,
                "frequency": 5,
            },
        },
        "model": {"num_classes": 7, "batch_size": 16},
    }


def test_polynomial_decay_ramps_initial_to_final() -> None:
    assert polynomial_decay_sparsity(0, 100, 0.0, 0.6) == 0.0  # starts at initial
    assert polynomial_decay_sparsity(100, 100, 0.0, 0.6) == pytest.approx(
        0.6
    )  # ends final
    mid = polynomial_decay_sparsity(50, 100, 0.0, 0.6)
    assert 0.0 < mid < 0.6  # monotonic in between


def test_one_shot_prune_reaches_target_sparsity() -> None:
    model = _model()
    prune(model, _prune_cfg(target=0.5))  # no fine-tune data → one-shot
    assert sparsity(model) == pytest.approx(0.5, abs=0.02)


def test_pruned_model_still_predicts_valid_class() -> None:
    X, _ = _data()
    model = _model()
    prune(model, _prune_cfg(target=0.6))
    cls = np.argmax(model.predict(X[:5], verbose=0), axis=1)
    assert cls.shape == (5,)
    assert cls.min() >= 0 and cls.max() < 7


def test_prune_with_fine_tune_recovers_and_stays_sparse() -> None:
    from tensorflow import keras

    X, y = _data(32)
    y_oh = keras.utils.to_categorical(y, 7)
    model = _model()
    prune(model, _prune_cfg(target=0.6, epochs=1), X, y_oh)  # prune → fine-tune
    assert sparsity(model) == pytest.approx(
        0.6, abs=0.02
    )  # still sparse after training


def test_invalid_target_sparsity_fails_loud() -> None:
    with pytest.raises(ValueError):
        prune(_model(), _prune_cfg(target=1.5))


def test_prune_and_report_has_sparsity_and_accuracy_deltas() -> None:
    X, y = _data(24)
    report = prune_and_report(_model(), _prune_cfg(target=0.5), X, y)
    for key in (
        "target_sparsity",
        "achieved_sparsity",
        "dense_gzip_bytes",
        "pruned_gzip_bytes",
        "gzip_reduction",
        "dense_accuracy",
        "pruned_accuracy",
        "accuracy_drop",
        "passed",
    ):
        assert key in report
    assert report["achieved_sparsity"] == pytest.approx(0.5, abs=0.02)
    assert isinstance(report["passed"], bool)
