"""Unit tests for Fer2013Downloader — all heavy I/O is mocked."""
from __future__ import annotations

import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.emotion_detector.data.download import Fer2013Downloader


def _cfg(expected: list | None = None) -> dict:
    return {
        "data": {
            "url": "https://example.com/data.zip",
            "zip_name": "data.zip",
            "expected_files": expected or ["train.csv", "test.csv"],
        }
    }


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------

def test_fetch_skips_download_when_files_exist(tmp_path: Path) -> None:
    cfg = _cfg()
    (tmp_path / "train.csv").write_text("emotion,pixels\n")
    (tmp_path / "test.csv").write_text("emotion,pixels\n")

    with patch("urllib.request.urlretrieve") as mock_dl:
        Fer2013Downloader(cfg).fetch(tmp_path)
        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# download path
# ---------------------------------------------------------------------------

def test_fetch_calls_urlretrieve_when_files_missing(tmp_path: Path) -> None:
    cfg = _cfg()
    zip_path = tmp_path / "data.zip"

    def fake_urlretrieve(url, dest, reporthook=None):
        # write a valid zip with expected CSVs
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("train.csv", "emotion,pixels\n1,0 1 2\n")
            zf.writestr("test.csv", "emotion,pixels\n0,3 4 5\n")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        Fer2013Downloader(cfg).fetch(tmp_path)

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_fetch_raises_on_url_error(tmp_path: Path) -> None:
    cfg = _cfg()
    with patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("timeout")):
        with pytest.raises(urllib.error.URLError, match="Failed to download"):
            Fer2013Downloader(cfg).fetch(tmp_path)


def test_fetch_raises_on_bad_zip(tmp_path: Path) -> None:
    cfg = _cfg()
    zip_path = tmp_path / "data.zip"
    zip_path.write_bytes(b"not a zip")

    with patch("urllib.request.urlretrieve", lambda u, d, reporthook=None: None):
        with pytest.raises(zipfile.BadZipFile):
            Fer2013Downloader(cfg).fetch(tmp_path)


def test_fetch_raises_when_expected_files_missing_after_extract(tmp_path: Path) -> None:
    cfg = _cfg()

    def fake_urlretrieve(url, dest, reporthook=None):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("only_train.csv", "data")  # test.csv missing

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        with pytest.raises(FileNotFoundError, match="missing after extraction"):
            Fer2013Downloader(cfg).fetch(tmp_path)


# ---------------------------------------------------------------------------
# zip already cached
# ---------------------------------------------------------------------------

def test_fetch_skips_urlretrieve_when_zip_already_cached(tmp_path: Path) -> None:
    cfg = _cfg()
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train.csv", "emotion,pixels\n1,0 1\n")
        zf.writestr("test.csv", "emotion,pixels\n0,2 3\n")

    with patch("urllib.request.urlretrieve") as mock_dl:
        Fer2013Downloader(cfg).fetch(tmp_path)
        mock_dl.assert_not_called()
