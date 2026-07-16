"""Unit tests for setup_logging."""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from src.emotion_detector.utils.logging import setup_logging


def _minimal_cfg(tmp_path: Path) -> dict:
    return {
        "global": {"log_level": "DEBUG"},
        "paths": {"logs_dir": str(tmp_path / "logs")},
    }


def test_setup_logging_runs_without_error(tmp_path: Path) -> None:
    setup_logging(_minimal_cfg(tmp_path))


def test_setup_logging_creates_logs_dir(tmp_path: Path) -> None:
    cfg = _minimal_cfg(tmp_path)
    setup_logging(cfg)
    assert Path(cfg["paths"]["logs_dir"]).is_dir()


def test_setup_logging_writes_log_file(tmp_path: Path) -> None:
    cfg = _minimal_cfg(tmp_path)
    setup_logging(cfg)
    logger.info("test log line")
    log_files = list(Path(cfg["paths"]["logs_dir"]).glob("*.log"))
    assert len(log_files) >= 1
    assert "test log line" in log_files[0].read_text(encoding="utf-8")


def test_setup_logging_missing_key_raises(tmp_path: Path) -> None:
    bad_cfg: dict = {"global": {}}  # missing log_level and paths
    with pytest.raises(KeyError):
        setup_logging(bad_cfg)
