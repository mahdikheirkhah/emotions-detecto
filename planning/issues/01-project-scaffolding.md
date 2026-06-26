---
title: "[Phase 0] Project scaffolding & repository structure"
labels: ["phase-0-foundations", "setup"]
---
## 1. Description

Create the exact folder/file structure the subject requires so every later issue has a clear home, and so the audit question *"Is the structure of the project equivalent to the one described in the subject?"* passes. We set up the data, results, scripts, source-package, notebook, and test folders, plus placeholder versions of every deliverable file. Pure plumbing — but getting it right now means we never fight the layout later.

**Target layout**
```
emotions-detecto
├── config.yaml                 # Ablation-Driven single source of truth (built in #04)
├── data/                       # raw csv data (git-ignored)
├── notebooks/                  # EDA / validation notebooks
├── results/
│   ├── model/                  # final_emotion_model.keras, *_arch.txt, learning_curves.png, tensorboard.png
│   └── preprocessing_test/     # image0..imageN.png + input_video.mp4
├── scripts/                    # thin entrypoints
├── src/emotion_detector/       # reusable OOP package (data, models, video, utils)
├── tests/
├── requirements.txt
├── README.md
└── data.md
```

## 2. Learning Objective

- **Why structure matters in ML:** clean separation of *data* (inputs), *code* (logic), *results* (artifacts), and *config* (decisions) is what makes a project reproducible and auditable.
- **Source package vs scripts:** `src/emotion_detector/` holds reusable, testable OOP classes; `scripts/*.py` are thin entrypoints that just wire them together. Understand why logic does not live in scripts.
- **How the layout serves the Ablation-Driven Architecture:** one `config.yaml` at the root + each pipeline stage in its own module is what later lets us flip a stage on/off.
- **What each deliverable is** (the `.keras` model, architecture `.txt`, learning-curve plot, TensorBoard screenshot, preprocessing_test images) and why graders expect each.

## 3. To-Do list for coding

- [ ] Create dirs: `data/`, `notebooks/`, `results/model/`, `results/preprocessing_test/`, `scripts/`, `src/emotion_detector/{data,models,video,utils}/`, `tests/`
- [ ] Add `__init__.py` to the package and every submodule
- [ ] Create placeholder entrypoints with a `main()` stub + docstring: `scripts/train.py`, `predict.py`, `predict_live_stream.py`, `preprocess.py`, `validation_loss_accuracy.py`
- [ ] Add `.gitignore` (`.venv/`, `data/*.csv`, `*.keras`, `*.pkl`, `__pycache__/`, `logs/`, `.ipynb_checkpoints`)
- [ ] Add `.gitkeep` so empty tracked dirs survive
- [ ] Stub `README.md` and `data.md` with top-level headings only
- [ ] `tests/test_structure.py` asserting key folders exist

## 4. Code learning (packages & methods)

- **`pathlib`** — `Path(...)`, `Path.mkdir(parents=True, exist_ok=True)`, `Path.touch()`
- **`os`** — path/env helpers if needed

➡️ **After we implement:** you explain back what each directory is for and why logic lives in `src/` not `scripts/`. I'll explain how `.gitignore` glob patterns are matched and why large artifacts (data, `.keras`, logs) must never be committed.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — OOP with `abc` bases, Loguru logging (never `print`, except required script stdout), type hints, docstrings, granular try/except that re-raises, matching test under `tests/`.

> 🔀 **Note — Ablation-Driven Architecture:** This issue lays the ground: `config.yaml` lives at the root and each stage gets its own module so it can later be toggled on/off. See `CONTRIBUTING.md` §3.
