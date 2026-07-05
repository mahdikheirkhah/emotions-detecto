"""Unit tests for the model-finalization helpers (Issue #45).

Cover the TF-free deliverable helpers — content hashing (incl. the Git-LFS pointer
case), config snapshot, and manifest assembly. The full sanity-load + accuracy
re-check needs the real .keras + FER data (a VM step), so it is not unit-tested here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_FINALIZE_PY = Path(__file__).resolve().parent.parent / "scripts" / "finalize_model.py"


def _load_finalize():
    spec = importlib.util.spec_from_file_location("finalize_model", _FINALIZE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:da06b36cf9ae342bac55b95105d344c7cad85f678ab1b2fa555b25c8a92ed26b\n"
    "size 17759479\n"
)


def test_content_hash_reads_lfs_pointer_oid(tmp_path: Path) -> None:
    # A committed .keras tracked by LFS is a text pointer; its oid IS the real hash.
    f = tmp_path / "final.keras"
    f.write_text(_LFS_POINTER)
    fin = _load_finalize()
    assert (
        fin.model_content_hash(str(f))
        == "da06b36cf9ae342bac55b95105d344c7cad85f678ab1b2fa555b25c8a92ed26b"
    )


def test_content_hash_of_a_plain_file(tmp_path: Path) -> None:
    import hashlib

    f = tmp_path / "weights.bin"
    f.write_bytes(b"not a pointer, real bytes")
    fin = _load_finalize()
    assert fin.model_content_hash(str(f)) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_snapshot_config_copies_verbatim(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("model:\n  architecture: vgg_small  # keep this comment\n")
    dst = tmp_path / "model" / "snap.yaml"
    fin = _load_finalize()
    fin.snapshot_config(str(src), str(dst))
    assert dst.read_text() == src.read_text()  # byte-for-byte, comments intact


def test_build_manifest_includes_hash_and_best_epoch(tmp_path: Path) -> None:
    model = tmp_path / "final.keras"
    model.write_text(_LFS_POINTER)
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps({"val_loss": [1.5, 0.9, 1.0], "val_accuracy": [0.4, 0.64, 0.6]})
    )
    cfg = {
        "model": {"architecture": "vgg_small", "num_classes": 7},
        "global": {"seed": 42},
    }

    fin = _load_finalize()
    manifest = fin.build_manifest(cfg, str(model), str(history))

    assert manifest["architecture"] == "vgg_small"
    assert manifest["content_sha256"].startswith("da06b36c")
    assert manifest["best_epoch"] == 2  # argmin(val_loss) is index 1 -> epoch 2
    assert manifest["best_val_loss"] == 0.9
    assert manifest["best_val_accuracy"] == 0.64


def test_write_manifest_is_valid_json(tmp_path: Path) -> None:
    fin = _load_finalize()
    out = tmp_path / "manifest.json"
    fin.write_manifest({"a": 1, "b": "x"}, str(out))
    assert json.loads(out.read_text()) == {"a": 1, "b": "x"}
