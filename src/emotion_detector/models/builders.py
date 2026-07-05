"""Config-driven model builders — assemble a full CNN behind ``BaseModelBuilder``.

Each architecture (``simple_cnn``, ``vgg_small``, …) is a ``BaseModelBuilder``
subclass selected by dispatch from ``model.architecture`` (the Strategy pattern),
so swapping architectures is a one-line config change. ``build`` returns a
**compiled** ``keras.Model`` (per the base contract) ready for ``.fit()``.

TensorFlow is imported lazily so importing this module is cheap.
"""
from __future__ import annotations

from typing import Any, Tuple

from src.emotion_detector.models.base import BaseModelBuilder
from src.emotion_detector.models.blocks import conv_block
from src.emotion_detector.models.compile import compile_model
from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.logging import logger


class SimpleCnnBuilder(BaseModelBuilder):
    """Baseline CNN: two plain Conv→ReLU→MaxPool layers + a dense head.

    Deliberately simple — **no BatchNorm** — so it is the in-project floor the
    deeper nets must beat (the MNIST logreg-vs-CNN habit, applied to architectures).
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._dropout = cfg["model"]["dropout_rate"]

    def build(self, input_shape: Tuple[int, int, int], num_classes: int) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers

        inp = keras.Input(input_shape)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self._dropout)(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inp, out, name="simple_cnn")
        return compile_model(model, self._cfg)


class VggSmallBuilder(BaseModelBuilder):
    """VGG-style net: stacked conv blocks (#33) + a dense classifier head.

    Reads ``num_conv_blocks``, ``filters_start`` (channels double per block),
    ``kernel_size``, ``convs_per_block``, and ``dropout_rate`` from config
    (see results/model/final_emotion_model_arch.txt).
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        m = cfg["model"]
        self._num_blocks = m["num_conv_blocks"]
        self._filters_start = m["filters_start"]
        self._kernel = m["kernel_size"]
        self._convs = m["convs_per_block"]
        self._dropout = m["dropout_rate"]

    def build(self, input_shape: Tuple[int, int, int], num_classes: int) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers

        inp = keras.Input(input_shape)
        x = inp
        filters = self._filters_start
        for b in range(self._num_blocks):
            x = conv_block(
                x,
                filters=filters,
                kernel_size=self._kernel,
                dropout=self._dropout * 0.5,  # lighter dropout in the conv stack
                n_convs=self._convs,
                name=f"block{b + 1}",
            )
            filters *= 2

        x = layers.Flatten()(x)
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self._dropout)(x)  # heavier dropout before the classifier
        out = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inp, out, name="vgg_small")
        return compile_model(model, self._cfg)


def build_model(cfg: dict, summary: bool = True) -> Any:
    """Build (and compile) the model selected by ``model.architecture`` (dispatch).

    Input shape is derived from config: ``(image_size, image_size, 1 or 3)``.

    Args:
        cfg:     Loaded config dict.
        summary: If True, log ``model.summary()``.

    Returns:
        A compiled ``keras.Model`` ready for ``.fit()``.

    Raises:
        ValueError: if ``model.architecture`` is not a known builder.
        KeyError:   if a required config key is missing.
    """
    m = cfg["model"]
    size = cfg["preprocessing"]["image_size"]
    channels = 1 if cfg["preprocessing"]["grayscale"] else 3
    input_shape = (size, size, channels)
    num_classes = m["num_classes"]

    registry = {
        "simple_cnn": lambda: SimpleCnnBuilder(cfg),
        "vgg_small": lambda: VggSmallBuilder(cfg),
        # resnet_mini / transfer_vgg16 land later — add options, never delete.
    }
    builder = dispatch(m["architecture"], registry)
    model = builder.build(input_shape, num_classes)

    logger.info(
        f"Built '{m['architecture']}' — input {input_shape}, "
        f"{model.count_params():,} params."
    )
    if summary:
        model.summary(print_fn=logger.info)
    return model
