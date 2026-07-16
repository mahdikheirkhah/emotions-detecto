"""FER-2013 dataset downloader and extractor."""

from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from src.emotion_detector.data.base import BaseDatasetFetcher
from src.emotion_detector.utils.logging import logger


class Fer2013Downloader(BaseDatasetFetcher):
    """Downloads the emotions-detector.zip and extracts the FER-2013 CSVs.

    Idempotent: if the expected CSV files already exist in *data_dir* the
    download and extraction steps are skipped entirely.

    Config keys consumed:
        data.url             — source URL for the zip archive
        data.zip_name        — filename to save the zip as
        data.expected_files  — list of CSVs that must exist after extraction
    """

    def __init__(self, cfg: dict) -> None:
        self._url: str = cfg["data"]["url"]
        self._zip_name: str = cfg["data"]["zip_name"]
        self._expected: List[str] = cfg["data"]["expected_files"]

    # ------------------------------------------------------------------
    # BaseDatasetFetcher contract
    # ------------------------------------------------------------------

    def fetch(self, data_dir: Path):  # type: ignore[override]
        """Download + extract the dataset into *data_dir* (idempotent).

        Args:
            data_dir: Directory where the zip and extracted CSVs are stored.

        Returns:
            *data_dir* Path after successful extraction.

        Raises:
            urllib.error.URLError: if the download fails (network / firewall).
            zipfile.BadZipFile: if the downloaded archive is corrupt.
            FileNotFoundError: if expected CSVs are missing after extraction.
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if self._already_extracted(data_dir):
            logger.info("Dataset already present — skipping download.")
            self._log_row_counts(data_dir)
            return data_dir

        zip_path = self._download(data_dir)
        self._extract(zip_path, data_dir)
        self._verify(data_dir)
        self._log_row_counts(data_dir)
        return data_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _already_extracted(self, data_dir: Path) -> bool:
        return all((data_dir / f).exists() for f in self._expected)

    def _download(self, data_dir: Path) -> Path:
        zip_path = data_dir / self._zip_name
        if zip_path.exists():
            logger.info(f"Zip already cached at {zip_path} — skipping download.")
            return zip_path

        logger.info(f"Downloading dataset from {self._url} …")
        try:
            urllib.request.urlretrieve(self._url, zip_path, reporthook=self._progress)
        except urllib.error.URLError as exc:
            zip_path.unlink(missing_ok=True)
            raise urllib.error.URLError(
                f"Failed to download dataset: {exc.reason}. "
                "Check network access or download the zip manually and place it "
                f"at {zip_path}."
            ) from exc
        logger.info(
            f"Download complete: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)"
        )
        return zip_path

    def _extract(self, zip_path: Path, data_dir: Path) -> None:
        logger.info(f"Extracting {zip_path} → {data_dir} …")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
        except zipfile.BadZipFile as exc:
            raise zipfile.BadZipFile(
                f"Archive is corrupt: {zip_path}. Delete it and re-run to re-download."
            ) from exc
        logger.info("Extraction complete.")

    def _verify(self, data_dir: Path) -> None:
        missing = [f for f in self._expected if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Expected files missing after extraction: {missing}. "
                "The zip structure may differ — check data_dir manually."
            )
        logger.info(f"Verified: {self._expected} all present in {data_dir}.")

    def _log_row_counts(self, data_dir: Path) -> None:
        for fname in self._expected:
            path = data_dir / fname
            if not path.exists():
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    # subtract 1 for header row
                    rows = sum(1 for _ in f) - 1
                logger.info(f"{fname}: {rows:,} rows")
            except OSError as exc:
                logger.warning(f"Could not read {fname}: {exc}")

    @staticmethod
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = downloaded / total_size * 100
        if block_num % 500 == 0 or downloaded >= total_size:
            logger.debug(f"Download progress: {pct:.1f}% ({downloaded / 1e6:.1f} MB)")
