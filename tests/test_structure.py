"""Verify the project directory layout required by Issue #1."""
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_data_dir_exists() -> None:
    assert (ROOT / "data").is_dir()


def test_notebooks_dir_exists() -> None:
    assert (ROOT / "notebooks").is_dir()


def test_results_model_dir_exists() -> None:
    assert (ROOT / "results" / "model").is_dir()


def test_results_preprocessing_test_dir_exists() -> None:
    assert (ROOT / "results" / "preprocessing_test").is_dir()


def test_scripts_dir_exists() -> None:
    assert (ROOT / "scripts").is_dir()


def test_src_package_exists() -> None:
    assert (ROOT / "src" / "emotion_detector" / "__init__.py").is_file()


def test_src_submodules_exist() -> None:
    base = ROOT / "src" / "emotion_detector"
    for submodule in ("data", "models", "video", "utils"):
        assert (base / submodule / "__init__.py").is_file(), f"missing {submodule}/__init__.py"


def test_tests_dir_exists() -> None:
    assert (ROOT / "tests").is_dir()


def test_scripts_entrypoints_exist() -> None:
    scripts = ROOT / "scripts"
    for name in ("train.py", "predict.py", "predict_live_stream.py", "preprocess.py", "validation_loss_accuracy.py"):
        assert (scripts / name).is_file(), f"missing scripts/{name}"


def test_gitignore_exists() -> None:
    assert (ROOT / ".gitignore").is_file()


def test_readme_exists() -> None:
    assert (ROOT / "README.md").is_file()


def test_data_md_exists() -> None:
    assert (ROOT / "data.md").is_file()
