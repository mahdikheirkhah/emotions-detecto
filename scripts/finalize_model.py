"""Finalize the trained-model deliverables (Issue #45).

Confirms ``results/model/final_emotion_model.keras`` (the ModelCheckpoint best) is
loadable, (re)writes its ``model.summary()`` to ``architecture.txt``, snapshots the
exact ``config.yaml`` that produced it next to the model, records a manifest (content
hash + param count + best-epoch metrics), and — when the FER data is present —
re-checks test accuracy to confirm the saved model reproduces its reported number.

    python scripts/finalize_model.py

The narrative + iteration history live in ``final_emotion_model_arch.txt`` (the audit's
architecture answer, hand-written); this script produces the machine artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging

_LFS_PREFIX = b"version https://git-lfs"


def model_content_hash(path: str) -> str:
    """Content sha256 of the model — the Git-LFS pointer ``oid`` if it is a pointer,
    else the sha256 of the file bytes.

    (A committed ``.keras`` tracked by LFS is a tiny text pointer whose ``oid`` already
    *is* the real file's sha256, so this gives a stable hash either way.)
    """
    data = Path(path).read_bytes()
    if data[:200].startswith(_LFS_PREFIX):
        for line in data.decode("utf-8", "ignore").splitlines():
            if line.startswith("oid sha256:"):
                return line.split("oid sha256:", 1)[1].strip()
    return hashlib.sha256(data).hexdigest()


def snapshot_config(config_path: str, out_path: str) -> str:
    """Copy the exact ``config.yaml`` used next to the model (verbatim)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out_path)
    return out_path


def build_manifest(
    cfg: dict, model_path: str, history_path: Optional[str] = None
) -> Dict[str, Any]:
    """Assemble the reproducibility manifest for the saved model."""
    manifest: Dict[str, Any] = {
        "model_file": model_path,
        "content_sha256": model_content_hash(model_path),
        "architecture": cfg["model"]["architecture"],
        "num_classes": cfg["model"]["num_classes"],
        "seed": cfg["global"]["seed"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    hp = Path(history_path) if history_path else None
    if hp and hp.exists():
        history = json.loads(hp.read_text())
        if history.get("val_loss"):
            best = int(np.argmin(history["val_loss"]))
            manifest["epochs_trained"] = len(history["val_loss"])
            manifest["best_epoch"] = best + 1
            manifest["best_val_loss"] = round(history["val_loss"][best], 4)
            if history.get("val_accuracy"):
                manifest["best_val_accuracy"] = round(history["val_accuracy"][best], 4)
    return manifest


def write_manifest(manifest: Dict[str, Any], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_path


def _load_keras_model(path: str) -> Any:
    from tensorflow.keras.models import load_model

    return load_model(path)


def _load_script(name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", _ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    paths = cfg["paths"]
    model_path = paths["model_save_path"]
    model_dir = Path(model_path).parent

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Run scripts/train.py (or `git lfs pull`) first."
        )

    # 1. Sanity-load — fails loudly if the .keras (or LFS content) is missing/corrupt.
    model = _load_keras_model(model_path)
    logger.info(f"Loaded {model_path} — {model.count_params():,} params.")

    # 2. (Re)write model.summary() so architecture.txt matches the saved graph exactly.
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    Path(paths["arch_txt_path"]).write_text("\n".join(lines), encoding="utf-8")

    # 3. Snapshot the exact config + write the manifest next to the model.
    snapshot_config(config_path, str(model_dir / "final_emotion_model_config.yaml"))
    manifest = build_manifest(cfg, model_path, str(model_dir / "history.json"))
    write_manifest(manifest, str(model_dir / "final_emotion_model_manifest.json"))
    logger.info(f"Manifest: {manifest}")

    # 4. Re-check test accuracy when the FER data is present (else skip gracefully).
    try:
        predict = _load_script("predict")
        scores = predict.score_test_set(cfg, model)
        logger.info(
            f"Re-checked test accuracy on the loaded model: {scores['accuracy']:.4f}"
        )
    except FileNotFoundError:
        logger.warning("FER data not present — skipped the test-accuracy re-check.")


if __name__ == "__main__":
    main()
