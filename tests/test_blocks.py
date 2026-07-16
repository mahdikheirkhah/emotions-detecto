"""Unit tests for the reusable conv-block builder (skipped without TensorFlow)."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras

from src.emotion_detector.models.blocks import conv_block


def _model(input_shape=(48, 48, 1), **kwargs):
    inp = keras.Input(input_shape)
    out = conv_block(inp, **kwargs)
    return keras.Model(inp, out)


# ---------------------------------------------------------------------------
# shape behavior
# ---------------------------------------------------------------------------


def test_block_halves_spatial_dims() -> None:
    m = _model(filters=32)
    assert m.output_shape == (None, 24, 24, 32)  # 48 -> 24 after MaxPool


def test_block_channel_count_matches_filters() -> None:
    m = _model(filters=64)
    assert m.output_shape[-1] == 64


def test_blocks_stack_cleanly() -> None:
    inp = keras.Input((48, 48, 1))
    x = conv_block(inp, filters=32, name="b1")
    x = conv_block(x, filters=64, name="b2")
    m = keras.Model(inp, x)
    assert m.output_shape == (None, 12, 12, 64)  # halved twice, 64 channels


def test_odd_input_uses_floor_division() -> None:
    m = _model(input_shape=(28, 28, 1), filters=16)
    assert m.output_shape == (None, 14, 14, 16)


# ---------------------------------------------------------------------------
# parameterization
# ---------------------------------------------------------------------------


def test_n_convs_controls_conv_count() -> None:
    for n in (1, 2, 3):
        m = _model(filters=16, n_convs=n)
        n_conv_layers = sum(l.__class__.__name__ == "Conv2D" for l in m.layers)
        assert n_conv_layers == n


def test_dropout_zero_omits_dropout_layer() -> None:
    m = _model(filters=16, dropout=0.0)
    assert not any(l.__class__.__name__ == "Dropout" for l in m.layers)


def test_dropout_positive_adds_dropout_layer() -> None:
    m = _model(filters=16, dropout=0.3)
    assert any(l.__class__.__name__ == "Dropout" for l in m.layers)


def test_conv_param_count_is_kernel2_x_in_x_out() -> None:
    # first conv: 3x3 kernel, 1 in-channel, 32 out, no bias (BN handles shift)
    m = _model(filters=32, kernel_size=3, n_convs=1)
    conv = next(l for l in m.layers if l.__class__.__name__ == "Conv2D")
    assert conv.count_params() == 3 * 3 * 1 * 32  # 288


def test_kernel_size_changes_param_count() -> None:
    m = _model(filters=32, kernel_size=5, n_convs=1)
    conv = next(l for l in m.layers if l.__class__.__name__ == "Conv2D")
    assert conv.count_params() == 5 * 5 * 1 * 32  # 800


def test_block_has_batchnorm_before_relu() -> None:
    m = _model(filters=16, n_convs=1)
    names = [l.__class__.__name__ for l in m.layers]
    # Conv2D, then BatchNormalization, then ReLU, in that order
    conv_i = names.index("Conv2D")
    assert names[conv_i + 1] == "BatchNormalization"
    assert names[conv_i + 2] == "ReLU"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_invalid_filters_raises() -> None:
    with pytest.raises(ValueError, match="filters must be"):
        conv_block(keras.Input((48, 48, 1)), filters=0)


def test_invalid_n_convs_raises() -> None:
    with pytest.raises(ValueError, match="n_convs must be"):
        conv_block(keras.Input((48, 48, 1)), filters=16, n_convs=0)


def test_invalid_dropout_raises() -> None:
    with pytest.raises(ValueError, match="dropout must be"):
        conv_block(keras.Input((48, 48, 1)), filters=16, dropout=1.5)
