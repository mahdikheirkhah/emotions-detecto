"""Config-driven, train-only data augmentation (feature engineering C).

``build_augmenter(cfg)`` returns a callable augmentation pipeline built from Keras
preprocessing layers (random horizontal flip, small rotation, zoom, shift). It is
gated by the ``stages.augmentation`` toggle — off (or all params zero) returns an
identity no-op. Augmentation applies to **training batches only**; validation/test
are never augmented (CONTRIBUTING §8).

Only **label-preserving** transforms are used: a horizontally flipped or slightly
rotated face is still the same emotion. Vertical flips or large rotations would
create unnatural faces and are deliberately excluded.

TensorFlow is imported lazily so the stage-off / zero-param paths work without it.
"""

from __future__ import annotations

from typing import Any

from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.stages import is_stage_on


class IdentityAugmenter:
    """No-op augmenter — returns inputs unchanged.

    Used when the augmentation stage is off or no transform is enabled, so a
    disabled stage and a raw-batch baseline behave identically. The ``training``
    kwarg mirrors the Keras layer call signature.
    """

    def __call__(self, X: Any, training: bool = False) -> Any:
        return X


def build_augmenter(cfg: dict) -> Any:
    """Return the configured augmentation pipeline (train-only).

    When ``stages.augmentation`` is off, or every augmentation parameter is
    disabled/zero, returns :class:`IdentityAugmenter`. Otherwise returns a
    ``tf.keras.Sequential`` of seeded preprocessing layers.

    Reads ``augmentation.{horizontal_flip, rotation_range, zoom_range,
    width_shift_range, height_shift_range}`` and seeds from ``global.seed``.

    Args:
        cfg: Loaded config dict.

    Returns:
        A callable ``augmenter(X, training=True)`` — apply to **training**
        batches of shape ``(N, H, W, C)`` only.

    Raises:
        KeyError: if a required ``augmentation`` config key is missing.
    """
    if not is_stage_on(cfg, "augmentation"):
        return IdentityAugmenter()

    try:
        a = cfg["augmentation"]
        flip = a["horizontal_flip"]
        rotation_range = a["rotation_range"]
        zoom_range = a["zoom_range"]
        width_shift = a["width_shift_range"]
        height_shift = a["height_shift_range"]
    except KeyError as exc:
        raise KeyError(
            f"Missing augmentation config key: {exc}. "
            "Check the 'augmentation:' section in config.yaml."
        ) from exc

    if not (flip or rotation_range or zoom_range or width_shift or height_shift):
        logger.info("Augmentation stage on but all params zero — identity no-op.")
        return IdentityAugmenter()

    import tensorflow as tf  # lazy: only needed when actually augmenting

    seed = cfg.get("global", {}).get("seed", 42)
    layers = []
    if flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    if rotation_range:
        # config rotation is in degrees; Keras factor is a fraction of a full turn.
        layers.append(tf.keras.layers.RandomRotation(rotation_range / 360.0, seed=seed))
    if zoom_range:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=zoom_range, width_factor=zoom_range, seed=seed
            )
        )
    if width_shift or height_shift:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=height_shift, width_factor=width_shift, seed=seed
            )
        )

    logger.info(
        f"Augmenter built — flip={bool(flip)}, rotation={rotation_range}°, "
        f"zoom={zoom_range}, shift=({width_shift}, {height_shift}), seed={seed}"
    )
    return tf.keras.Sequential(layers, name="augmenter")
