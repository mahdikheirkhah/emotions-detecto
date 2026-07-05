"""Transfer-learning builder — reuse an ImageNet backbone (optional path).

**Transfer learning** reuses features a network already learned on a huge dataset
(ImageNet, ~1.2M photos) instead of learning everything from scratch on our small
FER-2013. The early conv layers encode generic vision primitives (edges, textures,
simple shapes) that transfer to faces; only the classifier head is task-specific.

Two regimes, chosen by ``transfer.trainable_layers``:
  * **Feature extraction** (``0``): freeze the whole backbone, train only the new head.
    Fast, few parameters, best when the new dataset is small — the safe default.
  * **Fine-tuning** (``N``): unfreeze the top ``N`` backbone layers and continue with a
    **low** learning rate so the pretrained weights adapt gently (a high LR would wreck
    them). BatchNorm layers stay frozen — updating their running stats on a tiny batch
    destabilizes fine-tuning.

**Input adaptation.** ImageNet backbones expect 3-channel images at a minimum size,
but FER-2013 is 48x48 grayscale. The model bakes the adaptation in so it still accepts
the pipeline's ``(48, 48, 1)`` tensors in ``[0, 1]``: resize -> replicate the single
channel to RGB -> rescale to ``[0, 255]`` -> the backbone's own ``preprocess_input``.

This is just another ``model.architecture`` option behind the same dispatch, so it is
compared to the from-scratch CNN by a one-line config change (Ablation §3). TensorFlow
is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Any, Tuple

from src.emotion_detector.models.base import BaseModelBuilder
from src.emotion_detector.models.compile import compile_model
from src.emotion_detector.utils.logging import logger

_BACKBONES = ("vgg16", "resnet50")


class TransferModelBuilder(BaseModelBuilder):
    """Wrap a pretrained VGG16 / ResNet50 backbone + a fresh classifier head."""

    def __init__(self, cfg: dict, backbone: str = "vgg16") -> None:
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unknown transfer backbone '{backbone}'. "
                f"Valid options: {', '.join(_BACKBONES)}."
            )
        self._cfg = cfg
        self._backbone = backbone
        t = cfg["transfer"]
        # "none" (str) -> None so Keras uses random init (offline / tests).
        w = t.get("weights", "imagenet")
        self._weights = None if str(w).lower() == "none" else w
        self._trainable_layers = int(t.get("trainable_layers", 0))
        self._input_size = int(t.get("input_size", 96))
        self._dropout = cfg["model"]["dropout_rate"]

    def _load_backbone(self, input_shape: Tuple[int, int, int]):
        """Return (backbone_model, preprocess_input_fn) for the selected backbone."""
        from tensorflow.keras import applications

        if self._backbone == "vgg16":
            model = applications.VGG16(
                include_top=False, weights=self._weights, input_shape=input_shape
            )
            return model, applications.vgg16.preprocess_input
        model = applications.ResNet50(
            include_top=False, weights=self._weights, input_shape=input_shape
        )
        return model, applications.resnet50.preprocess_input

    def _set_trainable(self, backbone: Any) -> None:
        """Freeze the backbone (feature extraction) or unfreeze its top N layers."""
        from tensorflow.keras.layers import BatchNormalization

        if self._trainable_layers <= 0:
            backbone.trainable = False  # pure feature extraction
            return

        backbone.trainable = True
        cutoff = len(backbone.layers) - self._trainable_layers
        for i, layer in enumerate(backbone.layers):
            # Unfreeze only the top N, and keep BatchNorm frozen even there.
            layer.trainable = i >= cutoff and not isinstance(layer, BatchNormalization)

    def build(self, input_shape: Tuple[int, int, int], num_classes: int) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers

        norm = self._cfg.get("preprocessing", {}).get("normalization")
        if norm not in (None, "rescale"):
            logger.warning(
                f"transfer expects inputs in [0, 1] (preprocessing.normalization="
                f"'rescale'); got '{norm}' — the backbone's preprocess_input may be off."
            )

        size = self._input_size
        backbone, preprocess_input = self._load_backbone((size, size, 3))
        self._set_trainable(backbone)

        inp = keras.Input(input_shape)  # (48, 48, 1), values in [0, 1]
        x = layers.Resizing(size, size)(inp)  # -> backbone's spatial size
        x = layers.Concatenate(axis=-1)([x, x, x])  # grayscale -> 3-channel RGB
        x = layers.Rescaling(255.0)(x)  # [0, 1] -> [0, 255] for ImageNet preprocessing
        x = preprocess_input(x)  # backbone-specific mean-subtraction / channel order
        x = backbone(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self._dropout)(x)
        out = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inp, out, name=f"transfer_{self._backbone}")
        mode = (
            "feature-extraction"
            if self._trainable_layers <= 0
            else f"fine-tune-top-{self._trainable_layers}"
        )
        logger.info(
            f"Built transfer_{self._backbone} ({mode}, weights={self._weights}, "
            f"input {size}x{size}) — {model.count_params():,} params."
        )
        return compile_model(model, self._cfg)
