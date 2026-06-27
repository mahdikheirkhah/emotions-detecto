"""Project-wide Loguru logger configuration.

Every module imports the logger from here:

    from emotion_detector.utils.logging import logger

Call ``setup_logging(cfg)`` once at the top of each script entrypoint
before any pipeline work begins.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger  # re-exported for use by all modules


def setup_logging(cfg: dict) -> None:
    """Configure console + rotating file sinks for the project logger.

    Removes Loguru's default sink, then adds:
    - A ``stderr`` sink at the level set in ``cfg["global"]["log_level"]``.
    - A rotating file sink under ``cfg["paths"]["logs_dir"]`` that rolls
      over at 10 MB and keeps the last 7 files.

    Args:
        cfg: Loaded config dict (from ``load_config``).

    Raises:
        KeyError: if expected config keys are missing.
    """
    level: str = cfg["global"]["log_level"]
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.remove()  # drop the default stderr sink

    logger.add(
        sys.stderr,
        level=level,
        format=log_format,
        colorize=True,
    )

    logger.add(
        logs_dir / "run_{time:YYYY-MM-DD}.log",
        level=level,
        format=log_format,
        rotation="10 MB",
        retention=7,
        encoding="utf-8",
    )

    logger.info(f"Logging initialised — level={level}, log_dir={logs_dir}")
