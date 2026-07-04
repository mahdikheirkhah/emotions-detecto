"""Config-driven dataset cleaning strategies (duplicates, corrupt/constant images).

Implements the cleaning decisions documented in data.md §3 behind the
Ablation-Driven dispatch. ``clean_dataset`` is gated by the ``stages.cleaning``
toggle — when off, the data passes through untouched.
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.emotion_detector.data.base import BaseCleaner
from src.emotion_detector.utils.logging import logger
from src.emotion_detector.utils.stages import is_stage_on


class DuplicateRemover(BaseCleaner):
    """Removes exact-duplicate images, keeping the first occurrence (P2.2 / P2.3).

    Each image is fingerprinted with MD5 of its raw pixel bytes; pandas'
    ``duplicated`` then flags every repeat in near-linear time (it hashes each
    row once into a hash table rather than comparing all pairs). Idempotent:
    once duplicates are gone, a second pass removes nothing.
    """

    def clean(self, images: NDArray, labels: NDArray) -> Tuple[NDArray, NDArray]:
        if len(images) == 0:
            return images, labels
        hashes = [
            hashlib.md5(images[i].tobytes()).hexdigest() for i in range(len(images))
        ]
        keep = ~pd.Series(hashes).duplicated(keep="first").to_numpy()
        removed = int((~keep).sum())
        logger.info(
            f"DuplicateRemover: removed {removed:,} duplicate image(s), "
            f"kept {int(keep.sum()):,}."
        )
        return images[keep], labels[keep]


class CorruptImageRemover(BaseCleaner):
    """Removes constant / near-blank images that carry no signal (P2.5).

    A constant image (``std == 0``) is a single flat intensity — it cannot
    depict any expression and can destabilise batch statistics. Optionally also
    drops near-blank images whose intensity std falls below ``min_contrast``.

    Args:
        drop_constant: Drop images with ``std == 0``.
        min_contrast:  If > 0, also drop images with intensity std below this
            (subsumes ``drop_constant``, since 0 < min_contrast).
    """

    def __init__(self, drop_constant: bool = True, min_contrast: float = 0.0) -> None:
        self._drop_constant = bool(drop_constant)
        self._min_contrast = float(min_contrast)

    def clean(self, images: NDArray, labels: NDArray) -> Tuple[NDArray, NDArray]:
        if len(images) == 0:
            return images, labels
        std = images.reshape(len(images), -1).std(axis=1)
        if self._min_contrast > 0:
            keep = std >= self._min_contrast
        elif self._drop_constant:
            keep = std > 0
        else:
            keep = np.ones(len(images), dtype=bool)
        removed = int((~keep).sum())
        logger.info(
            f"CorruptImageRemover: removed {removed:,} constant/low-contrast "
            f"image(s), kept {int(keep.sum()):,}."
        )
        return images[keep], labels[keep]


def build_cleaners(cfg: dict) -> List[BaseCleaner]:
    """Assemble the enabled cleaners from config (the dispatch step).

    Reads the ``cleaning:`` block and returns only the cleaners whose switches
    are on, in a fixed order (dedup first, then corrupt-image removal).

    Args:
        cfg: Loaded config dict.

    Returns:
        A list of ``BaseCleaner`` instances (possibly empty).

    Raises:
        KeyError: if a required ``cleaning`` config key is missing.
    """
    try:
        cc = cfg["cleaning"]
        remove_duplicates = cc["remove_duplicates"]
        drop_constant = cc["drop_constant_images"]
        min_contrast = cc["min_contrast"]
    except KeyError as exc:
        raise KeyError(
            f"Missing cleaning config key: {exc}. "
            "Check the 'cleaning:' section in config.yaml."
        ) from exc

    cleaners: List[BaseCleaner] = []
    if remove_duplicates:
        cleaners.append(DuplicateRemover())
    if drop_constant or min_contrast > 0:
        cleaners.append(
            CorruptImageRemover(drop_constant=drop_constant, min_contrast=min_contrast)
        )
    return cleaners


def clean_dataset(
    cfg: dict, images: NDArray, labels: NDArray
) -> Tuple[NDArray, NDArray]:
    """Run the enabled cleaners in sequence, gated by the cleaning stage toggle.

    When ``stages.cleaning`` is off, returns ``(images, labels)`` unchanged so
    the stage can be ablated. When on, chains every cleaner from
    ``build_cleaners`` and logs the running row count.

    Args:
        cfg:    Loaded config dict.
        images: Image array of shape ``(N, H, W)`` or ``(N, H, W, C)``.
        labels: Integer label array of shape ``(N,)``.

    Returns:
        The cleaned ``(images, labels)`` pair.

    Raises:
        ValueError: if *images* and *labels* have mismatched lengths.
        KeyError:   if the ``stages`` or ``cleaning`` config is malformed.
    """
    if len(images) != len(labels):
        raise ValueError(
            f"images/labels length mismatch: {len(images)} vs {len(labels)}"
        )

    if not is_stage_on(cfg, "cleaning"):
        return images, labels

    start = len(images)
    for cleaner in build_cleaners(cfg):
        images, labels = cleaner.clean(images, labels)
    logger.info(
        f"Cleaning complete — {len(images):,} of {start:,} images remain "
        f"({start - len(images):,} removed)."
    )
    return images, labels
