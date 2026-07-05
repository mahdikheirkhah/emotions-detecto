"""Unit tests for the optional transfer-learning builder (Issue #46).

Backbone builds need TensorFlow (guarded); constructor logic (backbone validation,
``weights: none`` -> ``None``) runs without it. Always uses ``weights="none"`` so no
ImageNet download happens, and a small ``input_size`` to stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.emotion_detector.models.transfer import TransferModelBuilder


def _cfg(backbone_weights="none", trainable_layers=0, input_size=32) -> dict:
    return {
        "preprocessing": {
            "image_size": 48,
            "grayscale": True,
            "normalization": "rescale",
        },
        "model": {
            "architecture": "transfer_vgg16",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "loss": "categorical_crossentropy",
            "num_classes": 7,
            "dropout_rate": 0.5,
        },
        "transfer": {
            "weights": backbone_weights,
            "trainable_layers": trainable_layers,
            "input_size": input_size,
        },
    }


# ---------------------------------------------------------------------------
# constructor logic (no TensorFlow needed)
# ---------------------------------------------------------------------------


def test_unknown_backbone_fails_loud() -> None:
    with pytest.raises(ValueError):
        TransferModelBuilder(_cfg(), backbone="alexnet")


def test_weights_none_string_maps_to_none() -> None:
    b = TransferModelBuilder(_cfg(backbone_weights="none"), backbone="vgg16")
    assert b._weights is None  # so Keras uses random init (offline)


def test_weights_imagenet_kept() -> None:
    b = TransferModelBuilder(_cfg(backbone_weights="imagenet"), backbone="vgg16")
    assert b._weights == "imagenet"


# ---------------------------------------------------------------------------
# backbone builds (need TensorFlow)
# ---------------------------------------------------------------------------


def test_accepts_grayscale_input_and_outputs_num_classes() -> None:
    pytest.importorskip("tensorflow")
    model = TransferModelBuilder(_cfg(), backbone="vgg16").build((48, 48, 1), 7)
    assert model.input_shape == (None, 48, 48, 1)  # plugs into the (48,48,1) pipeline
    assert model.output_shape == (None, 7)
    # a real forward pass on [0,1] grayscale exercises resize + RGB + preprocess
    preds = model.predict(np.random.rand(2, 48, 48, 1).astype("float32"), verbose=0)
    assert preds.shape == (2, 7)


def test_feature_extraction_freezes_backbone() -> None:
    pytest.importorskip("tensorflow")
    model = TransferModelBuilder(_cfg(trainable_layers=0), backbone="vgg16").build(
        (48, 48, 1), 7
    )
    # frozen backbone -> only the Dense head trains: exactly kernel + bias
    assert len(model.trainable_weights) == 2


def test_fine_tune_unfreezes_top_layers() -> None:
    pytest.importorskip("tensorflow")
    model = TransferModelBuilder(_cfg(trainable_layers=6), backbone="vgg16").build(
        (48, 48, 1), 7
    )
    assert len(model.trainable_weights) > 2  # head + some unfrozen backbone layers


def test_resnet50_backbone_builds() -> None:
    pytest.importorskip("tensorflow")
    model = TransferModelBuilder(_cfg(), backbone="resnet50").build((48, 48, 1), 7)
    assert model.name == "transfer_resnet50"
    assert model.output_shape == (None, 7)


# ---------------------------------------------------------------------------
# dispatch integration + train.py artifact routing
# ---------------------------------------------------------------------------


def test_dispatch_builds_transfer_architectures() -> None:
    pytest.importorskip("tensorflow")
    from src.emotion_detector.models.builders import build_model

    for arch in ("transfer_vgg16", "transfer_resnet50"):
        cfg = _cfg()
        cfg["model"]["architecture"] = arch
        model = build_model(cfg, summary=False)
        assert model.name == arch


def test_train_routes_transfer_artifacts_separately(tmp_path) -> None:
    pytest.importorskip("tensorflow")
    import importlib.util
    from pathlib import Path

    train_py = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_routing", train_py)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    paths = {
        "model_save_path": "results/model/final_emotion_model.keras",
        "arch_txt_path": "results/model/architecture.txt",
        "pretrained_model_save_path": "results/model/pre_trained_model.keras",
        "pretrained_arch_txt_path": "results/model/pre_trained_model_architecture.txt",
    }
    # scratch architecture -> default paths
    scratch = {"model": {"architecture": "vgg_small"}, "paths": paths}
    assert train._artifact_paths(scratch)[0] == paths["model_save_path"]
    # transfer architecture -> pretrained paths (never overwrites the scratch model)
    transfer = {"model": {"architecture": "transfer_vgg16"}, "paths": paths}
    model_path, arch_path = train._artifact_paths(transfer)
    assert model_path == paths["pretrained_model_save_path"]
    assert arch_path == paths["pretrained_arch_txt_path"]
