"""Verify that abstract base classes cannot be instantiated directly."""
from __future__ import annotations

import pytest

from src.emotion_detector.data.base import BaseDatasetFetcher, BaseImagePreprocessor
from src.emotion_detector.models.base import BaseEmotionClassifier, BaseModelBuilder
from src.emotion_detector.video.base import BaseFaceDetector


def test_base_dataset_fetcher_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseDatasetFetcher()  # type: ignore[abstract]


def test_base_image_preprocessor_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseImagePreprocessor()  # type: ignore[abstract]


def test_base_face_detector_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseFaceDetector()  # type: ignore[abstract]


def test_base_model_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseModelBuilder()  # type: ignore[abstract]


def test_base_emotion_classifier_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseEmotionClassifier()  # type: ignore[abstract]


def test_concrete_subclass_without_all_methods_is_abstract() -> None:
    """A subclass that only implements some abstractmethods is still abstract."""
    class PartialPreprocessor(BaseImagePreprocessor):
        def fit(self, X):  # type: ignore[override]
            return self
        # transform not implemented

    with pytest.raises(TypeError):
        PartialPreprocessor()  # type: ignore[abstract]


def test_concrete_subclass_with_all_methods_can_be_instantiated() -> None:
    class ConcretePreprocessor(BaseImagePreprocessor):
        def fit(self, X):  # type: ignore[override]
            return self
        def transform(self, X):  # type: ignore[override]
            return X

    p = ConcretePreprocessor()
    assert isinstance(p, BaseImagePreprocessor)
