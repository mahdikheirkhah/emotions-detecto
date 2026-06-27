"""Stage-toggle helper for the Ablation-Driven Architecture."""
from __future__ import annotations

from loguru import logger


def is_stage_on(cfg: dict, stage: str) -> bool:
    """Return True if the named pipeline stage is enabled in config.

    When a stage is off, the caller should pass data through unchanged and
    log the fact so ablation runs are traceable.

    Args:
        cfg:   Loaded config dict (from load_config).
        stage: Key under cfg["stages"], e.g. "cleaning" or "augmentation".

    Returns:
        True if the stage toggle is set to true, False otherwise.

    Raises:
        KeyError: if *stage* is not listed under cfg["stages"].
    """
    try:
        value = cfg["stages"][stage]
    except KeyError:
        raise KeyError(f"Unknown stage '{stage}'. Check the 'stages:' section in config.yaml.")

    if not value:
        logger.info(f"Stage OFF: {stage} — passing data through unchanged.")
    return bool(value)
