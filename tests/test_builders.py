"""Unit tests for the config-driven model builders (skipped without TensorFlow)."""
from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")

from src.emotion_detector.models.base import BaseModelBuilder
from src.emotion_detector.models.builders import (
    SimpleCnnBuilder,
    VggSmallBuilder,
    build_model,
)


def _cfg(architecture="vgg_small", **model_over) -> dict:
    model = {
        "architecture": architecture,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "dropout_rate": 0.5,
        "num_conv_blocks": 3,
        "filters_start": 32,
        "kernel_size": 3,
        "convs_per_block": 2,
        "loss": "categorical_crossentropy",
        "num_classes": 7,
    }
    model.update(model_over)
    return {"model": model, "preprocessing": {"image_size": 48, "grayscale": True}}


# ---------------------------------------------------------------------------
# base is abstract; builders are subclasses
# ---------------------------------------------------------------------------

def test_base_model_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseModelBuilder()  # type: ignore[abstract]


def test_builders_are_base_subclasses() -> None:
    assert issubclass(SimpleCnnBuilder, BaseModelBuilder)
    assert issubclass(VggSmallBuilder, BaseModelBuilder)


# ---------------------------------------------------------------------------
# output shape / compilation
# ---------------------------------------------------------------------------

def test_vgg_small_output_shape() -> None:
    model = build_model(_cfg("vgg_small"), summary=False)
    assert model.output_shape == (None, 7)


def test_simple_cnn_output_shape() -> None:
    model = build_model(_cfg("simple_cnn"), summary=False)
    assert model.output_shape == (None, 7)


def test_model_is_compiled() -> None:
    model = build_model(_cfg(), summary=False)
    assert model.optimizer is not None  # compiled → ready for .fit()


def test_input_shape_from_config_grayscale() -> None:
    model = build_model(_cfg(), summary=False)
    assert model.input_shape == (None, 48, 48, 1)


def test_input_shape_rgb_when_not_grayscale() -> None:
    cfg = _cfg()
    cfg["preprocessing"]["grayscale"] = False
    model = build_model(cfg, summary=False)
    assert model.input_shape == (None, 48, 48, 3)


def test_num_classes_controls_output_width() -> None:
    model = build_model(_cfg(num_classes=10), summary=False)
    assert model.output_shape == (None, 10)


def test_num_conv_blocks_changes_depth() -> None:
    shallow = build_model(_cfg(num_conv_blocks=2), summary=False)
    deep = build_model(_cfg(num_conv_blocks=4), summary=False)
    n_conv = lambda m: sum(l.__class__.__name__ == "Conv2D" for l in m.layers)
    assert n_conv(deep) > n_conv(shallow)


# ---------------------------------------------------------------------------
# dispatch / error paths
# ---------------------------------------------------------------------------

def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        build_model(_cfg("resnet_giant"), summary=False)


def test_unknown_optimizer_raises() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        build_model(_cfg(optimizer="nadam"), summary=False)


def test_unsupported_loss_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model.loss"):
        build_model(_cfg(loss="focal_loss"), summary=False)
