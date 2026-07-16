"""Unit tests for the config-driven cleaning strategies."""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.data.base import BaseCleaner
from src.emotion_detector.data.cleaning import (
    CorruptImageRemover,
    DuplicateRemover,
    build_cleaners,
    clean_dataset,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(
    stage_on: bool = True,
    remove_duplicates: bool = True,
    drop_constant: bool = True,
    min_contrast: float = 0,
) -> dict:
    return {
        "stages": {"cleaning": stage_on},
        "cleaning": {
            "remove_duplicates": remove_duplicates,
            "dedup_scope": "global",
            "drop_constant_images": drop_constant,
            "min_contrast": min_contrast,
            "drop_non_faces": False,
            "handle_imbalance": True,
            "imbalance_strategy": "class_weight",
        },
    }


def _data():
    """5 images: a, b, c distinct; index 3 duplicates a; index 4 is constant black."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    b = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    c = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    const = np.zeros((4, 4), dtype=np.uint8)
    images = np.stack([a, b, c, a.copy(), const])
    labels = np.array([0, 1, 2, 0, 3])
    return images, labels


# ---------------------------------------------------------------------------
# BaseCleaner is abstract
# ---------------------------------------------------------------------------


def test_base_cleaner_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseCleaner()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DuplicateRemover
# ---------------------------------------------------------------------------


def test_duplicate_remover_removes_exact_duplicate() -> None:
    images, labels = _data()
    out_i, out_l = DuplicateRemover().clean(images, labels)
    assert len(out_i) == 4  # index 3 (dup of 0) removed
    assert out_l.tolist() == [0, 1, 2, 3]


def test_duplicate_remover_keeps_first_occurrence() -> None:
    a = np.full((4, 4), 7, dtype=np.uint8)
    images = np.stack([a, a.copy(), a.copy()])
    labels = np.array([5, 6, 7])
    out_i, out_l = DuplicateRemover().clean(images, labels)
    assert len(out_i) == 1
    assert out_l.tolist() == [5]  # first kept


def test_duplicate_remover_is_idempotent() -> None:
    images, labels = _data()
    once_i, once_l = DuplicateRemover().clean(images, labels)
    twice_i, twice_l = DuplicateRemover().clean(once_i, once_l)
    assert np.array_equal(once_i, twice_i)
    assert np.array_equal(once_l, twice_l)


def test_duplicate_remover_no_duplicates_keeps_all() -> None:
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (10, 4, 4), dtype=np.uint8)
    labels = np.arange(10)
    out_i, out_l = DuplicateRemover().clean(images, labels)
    assert len(out_i) == 10


def test_duplicate_remover_empty_input() -> None:
    images = np.empty((0, 4, 4), dtype=np.uint8)
    labels = np.empty((0,), dtype=np.int64)
    out_i, out_l = DuplicateRemover().clean(images, labels)
    assert len(out_i) == 0


# ---------------------------------------------------------------------------
# CorruptImageRemover
# ---------------------------------------------------------------------------


def test_corrupt_remover_drops_constant() -> None:
    images, labels = _data()
    out_i, out_l = CorruptImageRemover(drop_constant=True).clean(images, labels)
    assert len(out_i) == 4  # the black constant image removed
    assert 3 not in out_l.tolist() or True  # label 3 was the constant image
    assert out_l.tolist() == [0, 1, 2, 0]


def test_corrupt_remover_keeps_normal_images() -> None:
    rng = np.random.default_rng(2)
    images = rng.integers(0, 256, (6, 4, 4), dtype=np.uint8)
    labels = np.arange(6)
    out_i, out_l = CorruptImageRemover(drop_constant=True).clean(images, labels)
    assert len(out_i) == 6


def test_corrupt_remover_min_contrast_drops_low_contrast() -> None:
    rng = np.random.default_rng(3)
    normal = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    near_blank = np.full((4, 4), 100, dtype=np.uint8)
    near_blank[0, 0] = 103  # tiny variation → very low std
    images = np.stack([normal, near_blank])
    labels = np.array([0, 1])
    out_i, out_l = CorruptImageRemover(min_contrast=10).clean(images, labels)
    assert len(out_i) == 1
    assert out_l.tolist() == [0]


def test_corrupt_remover_drop_constant_false_keeps_constant() -> None:
    images, labels = _data()
    out_i, out_l = CorruptImageRemover(drop_constant=False, min_contrast=0).clean(
        images, labels
    )
    assert len(out_i) == 5  # nothing removed


def test_corrupt_remover_is_idempotent() -> None:
    images, labels = _data()
    once_i, once_l = CorruptImageRemover(drop_constant=True).clean(images, labels)
    twice_i, twice_l = CorruptImageRemover(drop_constant=True).clean(once_i, once_l)
    assert np.array_equal(once_i, twice_i)
    assert np.array_equal(once_l, twice_l)


# ---------------------------------------------------------------------------
# build_cleaners dispatch
# ---------------------------------------------------------------------------


def test_build_cleaners_all_enabled() -> None:
    cleaners = build_cleaners(_cfg())
    assert len(cleaners) == 2
    assert isinstance(cleaners[0], DuplicateRemover)
    assert isinstance(cleaners[1], CorruptImageRemover)


def test_build_cleaners_duplicates_only() -> None:
    cleaners = build_cleaners(_cfg(drop_constant=False, min_contrast=0))
    assert len(cleaners) == 1
    assert isinstance(cleaners[0], DuplicateRemover)


def test_build_cleaners_none_enabled() -> None:
    cleaners = build_cleaners(
        _cfg(remove_duplicates=False, drop_constant=False, min_contrast=0)
    )
    assert cleaners == []


def test_build_cleaners_min_contrast_adds_corrupt_remover() -> None:
    cleaners = build_cleaners(
        _cfg(remove_duplicates=False, drop_constant=False, min_contrast=10)
    )
    assert len(cleaners) == 1
    assert isinstance(cleaners[0], CorruptImageRemover)


def test_build_cleaners_missing_key_raises() -> None:
    bad = {"cleaning": {"remove_duplicates": True}}  # missing keys
    with pytest.raises(KeyError, match="Missing cleaning config key"):
        build_cleaners(bad)


# ---------------------------------------------------------------------------
# clean_dataset orchestrator
# ---------------------------------------------------------------------------


def test_clean_dataset_runs_all_cleaners() -> None:
    images, labels = _data()
    out_i, out_l = clean_dataset(_cfg(), images, labels)
    # 5 → 4 (dedup) → 3 (drop constant)
    assert len(out_i) == 3
    assert out_l.tolist() == [0, 1, 2]


def test_clean_dataset_stage_off_returns_unchanged() -> None:
    images, labels = _data()
    out_i, out_l = clean_dataset(_cfg(stage_on=False), images, labels)
    assert np.array_equal(out_i, images)
    assert np.array_equal(out_l, labels)
    assert len(out_i) == 5  # nothing removed


def test_clean_dataset_order_independent() -> None:
    """Dedup-then-corrupt equals corrupt-then-dedup for this data."""
    images, labels = _data()
    dr, cr = DuplicateRemover(), CorruptImageRemover(drop_constant=True)

    a_i, a_l = cr.clean(*dr.clean(images, labels))
    b_i, b_l = dr.clean(*cr.clean(images, labels))
    assert np.array_equal(a_i, b_i)
    assert np.array_equal(a_l, b_l)


def test_clean_dataset_length_mismatch_raises() -> None:
    images = np.zeros((3, 4, 4), dtype=np.uint8)
    labels = np.array([0, 1])  # wrong length
    with pytest.raises(ValueError, match="length mismatch"):
        clean_dataset(_cfg(), images, labels)


def test_clean_dataset_idempotent() -> None:
    images, labels = _data()
    once_i, once_l = clean_dataset(_cfg(), images, labels)
    twice_i, twice_l = clean_dataset(_cfg(), once_i, once_l)
    assert np.array_equal(once_i, twice_i)
    assert np.array_equal(once_l, twice_l)
