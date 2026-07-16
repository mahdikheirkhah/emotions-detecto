"""Reusable VGG-style convolutional block (Keras functional API).

One block is ``[Conv2D -> BatchNorm -> ReLU] x n_convs -> MaxPool2D -> Dropout``.
The full network (#34) is just a stack of these — expressing the net as repeated,
parameterized blocks keeps it readable and lets every architectural knob
(channels, kernel, dropout, #convs) be ablated from ``config.yaml``.

Canonical ordering: **Conv -> BatchNorm -> activation**. BN standardizes the conv
output *before* the non-linearity so ReLU sees a stable, zero-centered
distribution each step (see results/model/final_emotion_model_arch.txt §5).

TensorFlow is imported lazily so importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, Optional


def conv_block(
    x: Any,
    filters: int,
    kernel_size: int = 3,
    dropout: float = 0.25,
    n_convs: int = 2,
    name: Optional[str] = None,
) -> Any:
    """Apply one VGG-style conv block to a tensor and return the result.

    Structure: ``n_convs`` x ``(Conv2D(filters, kernel_size, same) -> BN -> ReLU)``
    then ``MaxPool2D(2)`` (halves H and W) then ``Dropout(dropout)``.

    Args:
        x:           Input tensor (Keras functional API), shape ``(N, H, W, C)``.
        filters:     Number of output channels for every conv in the block.
        kernel_size: Square conv kernel side (3 for VGG-style 3x3).
        dropout:     Dropout rate after pooling; ``0`` disables the Dropout layer.
        n_convs:     How many Conv->BN->ReLU sub-layers before the pool (>= 1).
        name:        Optional prefix for the layer names (nice in ``model.summary``).

    Returns:
        The block's output tensor, with H and W halved and ``filters`` channels.

    Raises:
        ValueError: if *filters* or *n_convs* is < 1, or *dropout* not in [0, 1].
    """
    from tensorflow.keras import layers

    if filters < 1:
        raise ValueError(f"filters must be >= 1, got {filters}")
    if n_convs < 1:
        raise ValueError(f"n_convs must be >= 1, got {n_convs}")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")

    prefix = f"{name}_" if name else ""
    for i in range(n_convs):
        x = layers.Conv2D(
            filters,
            kernel_size,
            padding="same",  # keep H,W until the pool, so RF math is clean
            use_bias=False,  # BN's beta replaces the conv bias
            name=f"{prefix}conv{i + 1}" if name else None,
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}bn{i + 1}" if name else None)(x)
        x = layers.ReLU(name=f"{prefix}relu{i + 1}" if name else None)(x)

    x = layers.MaxPooling2D(2, name=f"{prefix}pool" if name else None)(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{prefix}dropout" if name else None)(x)
    return x
