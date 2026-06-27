"""Unit tests for dispatch, is_stage_on, and set_global_seed."""
from __future__ import annotations

import pytest

from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.stages import is_stage_on


# ---------------------------------------------------------------------------
# dispatch()
# ---------------------------------------------------------------------------

class _FakeA:
    pass

class _FakeB:
    pass

REGISTRY: dict = {"option_a": _FakeA, "option_b": _FakeB}


def test_dispatch_known_name_returns_correct_type() -> None:
    result = dispatch("option_a", REGISTRY)
    assert isinstance(result, _FakeA)


def test_dispatch_second_known_name() -> None:
    result = dispatch("option_b", REGISTRY)
    assert isinstance(result, _FakeB)


def test_dispatch_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        dispatch("nonexistent", REGISTRY)


def test_dispatch_error_message_lists_valid_options() -> None:
    with pytest.raises(ValueError, match="option_a"):
        dispatch("bad_name", REGISTRY)


def test_dispatch_empty_registry_raises() -> None:
    with pytest.raises(ValueError):
        dispatch("anything", {})


# ---------------------------------------------------------------------------
# is_stage_on()
# ---------------------------------------------------------------------------

def _cfg(stage_value: bool) -> dict:
    return {"stages": {"cleaning": stage_value, "augmentation": True}}


def test_is_stage_on_returns_true_when_enabled() -> None:
    assert is_stage_on(_cfg(True), "cleaning") is True


def test_is_stage_on_returns_false_when_disabled() -> None:
    assert is_stage_on(_cfg(False), "cleaning") is False


def test_is_stage_on_unknown_stage_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="unknown_stage"):
        is_stage_on(_cfg(True), "unknown_stage")
