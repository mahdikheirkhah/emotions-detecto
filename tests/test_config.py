"""Unit tests for the config loader and dotted-key accessor."""
from __future__ import annotations

import pytest

from src.emotion_detector.utils.config import cfg_get, load_config

CONFIG_PATH = "config.yaml"


def test_load_config_returns_dict() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, dict)


def test_load_config_has_required_sections() -> None:
    cfg = load_config(CONFIG_PATH)
    for section in ("global", "paths", "stages", "cleaning", "preprocessing",
                    "model", "face_detector", "video"):
        assert section in cfg, f"Missing config section: '{section}'"


def test_cfg_get_nested_key() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg_get(cfg, "model.optimizer") == "adam"


def test_cfg_get_top_level_via_dotted_path() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg_get(cfg, "global.seed") == 42


def test_cfg_get_missing_key_raises_keyerror() -> None:
    cfg = load_config(CONFIG_PATH)
    with pytest.raises(KeyError):
        cfg_get(cfg, "nonexistent.key")


def test_cfg_get_partial_missing_key_raises_keyerror() -> None:
    cfg = load_config(CONFIG_PATH)
    with pytest.raises(KeyError):
        cfg_get(cfg, "model.nonexistent_param")


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_stages_are_all_booleans() -> None:
    cfg = load_config(CONFIG_PATH)
    for stage, value in cfg["stages"].items():
        assert isinstance(value, bool), f"stages.{stage} must be bool, got {type(value)}"


def test_model_num_classes_is_seven() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg_get(cfg, "model.num_classes") == 7
