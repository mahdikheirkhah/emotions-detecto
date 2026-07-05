"""Unit tests for promoting winning values back into config.yaml (Issue #44).

Pure text manipulation — no TensorFlow / keras_tuner needed. The key guarantees:
comments survive, only the named block is touched (``tuning.search_space`` arrays
that share key names are left alone), and the result is still valid YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.emotion_detector.utils.config_writer import promote_values

_SAMPLE = """\
# ===== Model =====
model:
  architecture: "vgg_small"        # options: simple_cnn | vgg_small
  optimizer: "adam"                # options: adam | sgd | rmsprop
  learning_rate: 0.001             # options: 0.01 | 0.001 | 0.0001
  batch_size: 64                   # options: 32 | 64 | 128
  dropout_rate: 0.5                # options: 0.3 | 0.4 | 0.5
  filters_start: 32                # options: 32 | 64
  num_classes: 7                   # fixed

# ===== Tuning =====
tuning:
  strategy: "random"               # options: grid | random | bayesian
  search_space:
    learning_rate: [0.01, 0.001, 0.0001]
    batch_size: [32, 64, 128]
    dropout_rate: [0.3, 0.4, 0.5]
    filters_start: [32, 64]
    optimizer: ["adam", "sgd", "rmsprop"]
"""

_BEST = {
    "learning_rate": 0.0001,
    "batch_size": 128,
    "dropout_rate": 0.3,
    "filters_start": 64,
    "optimizer": "sgd",
}


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(_SAMPLE, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# values are promoted and the file stays valid YAML
# ---------------------------------------------------------------------------


def test_model_defaults_are_overwritten(tmp_path: Path) -> None:
    p = _write(tmp_path)
    updated = promote_values(str(p), "model", _BEST)
    assert set(updated) == set(_BEST)

    cfg = yaml.safe_load(p.read_text())
    for key, value in _BEST.items():
        assert cfg["model"][key] == value


def test_search_space_arrays_are_untouched(tmp_path: Path) -> None:
    p = _write(tmp_path)
    promote_values(str(p), "model", _BEST)
    cfg = yaml.safe_load(p.read_text())
    # identically-named keys under tuning.search_space must NOT be promoted
    assert cfg["tuning"]["search_space"]["learning_rate"] == [0.01, 0.001, 0.0001]
    assert cfg["tuning"]["search_space"]["optimizer"] == ["adam", "sgd", "rmsprop"]


def test_inline_comments_survive(tmp_path: Path) -> None:
    p = _write(tmp_path)
    promote_values(str(p), "model", _BEST)
    text = p.read_text()
    # the # options menus (the ablation record) are preserved on promoted lines
    assert "# options: 0.01 | 0.001 | 0.0001" in text
    assert "# options: adam | sgd | rmsprop" in text
    assert "# fixed" in text  # untouched line kept verbatim


def test_string_value_is_quoted_and_numbers_are_bare(tmp_path: Path) -> None:
    p = _write(tmp_path)
    promote_values(str(p), "model", _BEST)
    text = p.read_text()
    assert 'optimizer: "sgd"' in text  # strings quoted
    assert "learning_rate: 0.0001" in text  # floats bare
    assert "batch_size: 128" in text  # ints bare


def test_untouched_keys_keep_their_value(tmp_path: Path) -> None:
    p = _write(tmp_path)
    promote_values(str(p), "model", {"learning_rate": 0.0001})
    cfg = yaml.safe_load(p.read_text())
    assert cfg["model"]["batch_size"] == 64  # not in the update → unchanged
    assert cfg["model"]["num_classes"] == 7


# ---------------------------------------------------------------------------
# fail-loud
# ---------------------------------------------------------------------------


def test_missing_key_raises(tmp_path: Path) -> None:
    p = _write(tmp_path)
    with pytest.raises(KeyError):
        promote_values(str(p), "model", {"nonexistent": 1})


def test_missing_block_raises(tmp_path: Path) -> None:
    p = _write(tmp_path)
    with pytest.raises(KeyError):
        promote_values(str(p), "no_such_block", {"learning_rate": 0.1})
