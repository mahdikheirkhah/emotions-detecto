"""Unit tests for Fer2013Fetcher — all I/O is backed by tmp_path fake CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.emotion_detector.data.fer2013 import Fer2013Fetcher

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(data_dir: Path) -> dict:
    return {
        "paths": {"data_dir": str(data_dir)},
        "data": {"primary_csv": "train.csv"},
        "preprocessing": {"image_size": 48},
    }


def _pixels(value: int = 128) -> str:
    """Return 2304 space-separated copies of *value*."""
    return " ".join([str(value)] * 2304)


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "train.csv"
    lines = ["emotion,pixels,Usage"]
    for r in rows:
        lines.append(f"{r['emotion']},{r['pixels']},{r['usage']}")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_fetch_training_returns_correct_shape(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 0, "pixels": _pixels(100), "usage": "Training"},
            {"emotion": 3, "pixels": _pixels(200), "usage": "Training"},
            {"emotion": 6, "pixels": _pixels(50), "usage": "PublicTest"},
        ],
    )
    images, labels = Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")
    assert images.shape == (2, 48, 48)
    assert labels.shape == (2,)
    assert images.dtype == np.uint8
    assert labels.tolist() == [0, 3]


def test_fetch_public_test_split_isolates_rows(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 2, "pixels": _pixels(10), "usage": "Training"},
            {"emotion": 5, "pixels": _pixels(20), "usage": "PublicTest"},
        ],
    )
    images, labels = Fer2013Fetcher(_cfg(tmp_path)).fetch("PublicTest")
    assert images.shape == (1, 48, 48)
    assert labels[0] == 5


def test_fetch_private_test_split(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 1, "pixels": _pixels(77), "usage": "PrivateTest"},
        ],
    )
    images, labels = Fer2013Fetcher(_cfg(tmp_path)).fetch("PrivateTest")
    assert images.shape == (1, 48, 48)
    assert labels[0] == 1


def test_fetch_pixel_values_are_correct(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 0, "pixels": _pixels(255), "usage": "Training"},
        ],
    )
    images, _ = Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")
    assert np.all(images == 255)


def test_parse_pixels_returns_correct_shape_and_dtype(tmp_path: Path) -> None:
    fetcher = Fer2013Fetcher(_cfg(tmp_path))
    arr = fetcher._parse_pixels(_pixels(123))
    assert arr.shape == (48, 48)
    assert arr.dtype == np.uint8
    assert np.all(arr == 123)


def test_labels_dtype_is_int64(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 4, "pixels": _pixels(), "usage": "Training"},
        ],
    )
    _, labels = Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")
    assert labels.dtype == np.int64


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_fetch_raises_file_not_found_when_csv_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")


def test_fetch_raises_on_invalid_split(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 0, "pixels": _pixels(), "usage": "Training"},
        ],
    )
    with pytest.raises(ValueError, match="Unknown split"):
        Fer2013Fetcher(_cfg(tmp_path)).fetch("Validation")


def test_fetch_raises_on_empty_split(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 0, "pixels": _pixels(), "usage": "Training"},
        ],
    )
    with pytest.raises(ValueError, match="No rows found"):
        Fer2013Fetcher(_cfg(tmp_path)).fetch("PrivateTest")


def test_fetch_raises_on_malformed_pixels(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "emotion,pixels,Usage\n0,128 255 64,Training\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pixel values"):
        Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")


def test_fetch_raises_on_label_out_of_range(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        [
            {"emotion": 7, "pixels": _pixels(), "usage": "Training"},
        ],
    )
    with pytest.raises(ValueError, match="out of range"):
        Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")


def test_fetch_raises_on_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("emotion,Usage\n0,Training\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        Fer2013Fetcher(_cfg(tmp_path)).fetch("Training")


def test_parse_pixels_raises_on_wrong_count(tmp_path: Path) -> None:
    fetcher = Fer2013Fetcher(_cfg(tmp_path))
    with pytest.raises(ValueError, match="pixel values"):
        fetcher._parse_pixels("1 2 3")
