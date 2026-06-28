"""FER-2013 CSV loader — parses the pixels column into (N, H, W) arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.emotion_detector.data.base import BaseDatasetFetcher
from src.emotion_detector.utils.logging import logger

_VALID_SPLITS = frozenset({"Training", "PublicTest", "PrivateTest"})
_REQUIRED_COLS = frozenset({"emotion", "pixels", "Usage"})
# icml_face_data.csv ships with lowercase 'usage'; we normalise after load.


class Fer2013Fetcher(BaseDatasetFetcher):
    """Loads a FER-2013 CSV and returns parsed image / label arrays for one split.

    The FER-2013 CSV has three columns:
    - ``emotion``: integer 0–6
    - ``pixels``: space-separated string of ``image_size²`` uint8 values
    - ``Usage``: ``"Training"``, ``"PublicTest"``, or ``"PrivateTest"``

    Config keys consumed:
        paths.data_dir          — directory containing the CSV
        data.primary_csv        — CSV with a Usage column (``"icml_face_data.csv"``)
        preprocessing.image_size — side length in pixels (default 48)
    """

    def __init__(self, cfg: dict) -> None:
        self._data_dir = Path(cfg["paths"]["data_dir"])
        self._primary_csv: str = cfg["data"]["primary_csv"]
        self._image_size: int = cfg["preprocessing"]["image_size"]

    # ------------------------------------------------------------------
    # BaseDatasetFetcher contract
    # ------------------------------------------------------------------

    def fetch(self, split: str) -> Tuple[NDArray, NDArray]:  # type: ignore[override]
        """Load one split of the FER-2013 CSV as NumPy arrays.

        Args:
            split: One of ``"Training"``, ``"PublicTest"``, ``"PrivateTest"``.

        Returns:
            ``(images, labels)`` where *images* has shape
            ``(N, image_size, image_size)`` dtype ``uint8`` and *labels* has
            shape ``(N,)`` dtype ``int64``.

        Raises:
            FileNotFoundError: if the CSV is not found in *data_dir*.
            ValueError: if *split* is invalid, required columns are missing,
                pixels are malformed, labels are out of range 0–6, or the
                split contains no rows.
        """
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. Valid options: {sorted(_VALID_SPLITS)}"
            )

        csv_path = self._data_dir / self._primary_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. Run the download stage first."
            )

        df = pd.read_csv(csv_path)

        # Normalise the split column: real data ships as lowercase 'usage'
        usage_col = next((c for c in df.columns if c.lower() == "usage"), None)
        if usage_col and usage_col != "Usage":
            df = df.rename(columns={usage_col: "Usage"})

        missing_cols = _REQUIRED_COLS - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

        subset = df[df["Usage"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"No rows found for split '{split}' in {csv_path}.")

        logger.info(f"Loading split '{split}': {len(subset):,} rows from {csv_path.name}")

        if subset["pixels"].isna().any():
            raise ValueError(f"NaN found in pixels column for split '{split}'.")

        images = np.stack(subset["pixels"].apply(self._parse_pixels).to_numpy())

        labels = subset["emotion"].to_numpy(dtype=np.int64)
        invalid_mask = (labels < 0) | (labels > 6)
        if invalid_mask.any():
            bad = np.unique(labels[invalid_mask]).tolist()
            raise ValueError(
                f"Labels out of range 0–6 in split '{split}': {bad}"
            )

        logger.info(
            f"Split '{split}' ready — images {images.shape} uint8, labels {labels.shape}"
        )
        return images, labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_pixels(self, s: str) -> NDArray:
        """Parse a space-separated pixel string into a square uint8 array.

        Args:
            s: Space-separated string of ``image_size²`` integers (0–255).

        Returns:
            ``np.ndarray`` of shape ``(image_size, image_size)`` dtype ``uint8``.

        Raises:
            ValueError: if *s* does not yield exactly ``image_size²`` values.
        """
        n = self._image_size
        expected = n * n
        arr = np.fromstring(s, sep=" ", dtype=np.uint8)
        if arr.size != expected:
            raise ValueError(
                f"Expected {expected} pixel values, got {arr.size}. "
                f"Snippet: '{s[:50]}...'"
            )
        return arr.reshape(n, n)
