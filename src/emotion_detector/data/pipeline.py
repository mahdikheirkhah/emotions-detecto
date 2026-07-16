"""Turn split arrays into Keras-ready ``tf.data`` pipelines.

Reshapes images to ``(N, 48, 48, 1)`` (Conv2D needs an explicit channel axis,
``C=1`` for grayscale), one-hot encodes labels to ``(N, 7)`` (to pair with
``categorical_crossentropy``), and wraps them in ``tf.data.Dataset`` pipelines
(cache -> shuffle -> batch -> augment -> prefetch). Shuffle and augmentation
apply to the **training** dataset only; augmentation runs *after batching* and
respects its stage toggle (#22).

TensorFlow is imported lazily so ``to_tensors`` can be used without building a
dataset.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.data.augmentation import build_augmenter
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.stages import is_stage_on


def to_tensors(X: NDArray, y: NDArray, num_classes: int = 7) -> Tuple[NDArray, NDArray]:
    """Reshape images to ``(N, H, W, 1)`` and one-hot encode labels to ``(N, C)``.

    Args:
        X:           Image array ``(N, H, W)`` or already ``(N, H, W, 1)``.
        y:           Integer label array ``(N,)``.
        num_classes: Number of classes for the one-hot encoding (7 for FER-2013).

    Returns:
        ``(images, onehot)`` — *images* float32 ``(N, H, W, 1)``; *onehot* float32
        ``(N, num_classes)`` where each row sums to 1.

    Raises:
        ValueError: if *X* is not 3-D or 4-D, or lengths mismatch.
    """
    from tensorflow.keras.utils import to_categorical

    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

    if X.ndim == 3:
        images = X[..., np.newaxis]  # (N, H, W) → (N, H, W, 1)
    elif X.ndim == 4:
        images = X
    else:
        raise ValueError(f"Expected 3-D or 4-D image array, got shape {X.shape}")

    images = images.astype(np.float32)
    onehot = to_categorical(y, num_classes=num_classes).astype(np.float32)
    return images, onehot


def make_dataset(X: NDArray, y: NDArray, cfg: dict, training: bool) -> Any:
    """Build a batched, prefetched ``tf.data.Dataset`` for one split.

    Pipeline order: ``from_tensor_slices → [cache] → [shuffle if training] →
    batch → [augment if training] → prefetch``. Only the **training** dataset is
    shuffled and augmented; augmentation is applied *after batching* and only when
    the ``augmentation`` stage is on.

    Args:
        X:        Image array for the split.
        y:        Integer label array for the split.
        cfg:      Loaded config dict (reads ``model.batch_size``,
                  ``model.num_classes``, ``pipeline.*``, ``global.seed``).
        training: Whether this is the training split (enables shuffle + augment).

    Returns:
        A ``tf.data.Dataset`` yielding ``(images (B, H, W, 1), labels (B, C))``.
    """
    import tensorflow as tf

    num_classes = cfg["model"]["num_classes"]
    batch_size = cfg["model"]["batch_size"]
    shuffle_buffer = cfg["pipeline"]["shuffle_buffer"]
    use_cache = cfg["pipeline"]["cache"]
    seed = cfg["global"]["seed"]

    images, labels = to_tensors(X, y, num_classes=num_classes)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if use_cache:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if training and is_stage_on(cfg, "augmentation"):
        augmenter = build_augmenter(cfg)
        ds = ds.map(
            lambda batch_x, batch_y: (augmenter(batch_x, training=True), batch_y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    logger.info(
        f"Built {'train' if training else 'eval'} dataset — "
        f"{len(images):,} samples, batch_size={batch_size}, "
        f"shuffle={training}, augment={training and is_stage_on(cfg, 'augmentation')}"
    )
    return ds
