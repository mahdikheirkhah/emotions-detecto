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
    quantize,
    save_tflite,
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
