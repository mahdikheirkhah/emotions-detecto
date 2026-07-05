"""Unit tests for config-driven compilation (skipped without TensorFlow)."""
from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras

from src.emotion_detector.models.compile import (
    build_loss,
    build_metrics,
    build_optimizer,
    compile_model,
)


def _cfg(optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy"):
    return {
        "model": {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "loss": loss,
        }
    }


def _tiny_model():
    inp = keras.Input((4,))
    out = keras.layers.Dense(3, activation="softmax")(inp)
    return keras.Model(inp, out)


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------

def test_adam_optimizer_type_and_lr() -> None:
    opt = build_optimizer(_cfg("adam", 0.001))
    assert isinstance(opt, keras.optimizers.Adam)
    assert float(opt.learning_rate) == pytest.approx(0.001)


def test_sgd_optimizer_type_and_lr() -> None:
    opt = build_optimizer(_cfg("sgd", 0.01))
    assert isinstance(opt, keras.optimizers.SGD)
    assert float(opt.learning_rate) == pytest.approx(0.01)


def test_sgd_has_momentum() -> None:
    opt = build_optimizer(_cfg("sgd", 0.01))
    assert float(opt.momentum) == pytest.approx(0.9)


def test_rmsprop_optimizer_type() -> None:
    opt = build_optimizer(_cfg("rmsprop", 0.0001))
    assert isinstance(opt, keras.optimizers.RMSprop)
    assert float(opt.learning_rate) == pytest.approx(0.0001)


def test_unknown_optimizer_raises() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        build_optimizer(_cfg("nadam"))


# ---------------------------------------------------------------------------
# build_loss / build_metrics
# ---------------------------------------------------------------------------

def test_loss_categorical_crossentropy() -> None:
    assert build_loss(_cfg()) == "categorical_crossentropy"


def test_loss_focal_not_yet_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported model.loss"):
        build_loss(_cfg(loss="focal_loss"))


def test_metrics_include_accuracy() -> None:
    assert "accuracy" in build_metrics(_cfg())


# ---------------------------------------------------------------------------
# compile_model
# ---------------------------------------------------------------------------

def test_compile_model_wires_optimizer_and_loss() -> None:
    model = compile_model(_tiny_model(), _cfg("adam", 0.005))
    assert isinstance(model.optimizer, keras.optimizers.Adam)
    assert float(model.optimizer.learning_rate) == pytest.approx(0.005)


def test_compile_model_returns_same_model() -> None:
    model = _tiny_model()
    assert compile_model(model, _cfg()) is model  # compiled in place


def test_compiled_model_can_train_one_step() -> None:
    import numpy as np

    model = compile_model(_tiny_model(), _cfg())
    X = np.random.default_rng(0).random((8, 4)).astype("float32")
    Y = keras.utils.to_categorical(np.arange(8) % 3, 3)
    history = model.fit(X, Y, epochs=1, verbose=0)
    assert "loss" in history.history  # a real gradient step ran
