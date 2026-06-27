"""Config loader and dotted-key accessor for the Ablation-Driven Architecture."""
from __future__ import annotations

import functools
import operator
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict:
    """Load config.yaml and return its contents as a nested dict.

    Raises:
        FileNotFoundError: if *path* does not exist on disk.
        yaml.YAMLError: if the file is not valid YAML.
    """
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(text)


def cfg_get(cfg: dict, dotted_key: str) -> Any:
    """Return the value at a dotted path inside a nested config dict.

    Example::

        cfg_get(cfg, "model.optimizer")   # -> "adam"
        cfg_get(cfg, "stages.cleaning")   # -> True

    Raises:
        KeyError: if any segment of the dotted path is absent.
    """
    keys = dotted_key.split(".")
    try:
        return functools.reduce(operator.getitem, keys, cfg)
    except (KeyError, TypeError):
        raise KeyError(f"Config key not found: '{dotted_key}'")
