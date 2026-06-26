---
title: "[Phase 4] train.py — wire data + model + callbacks and train"
labels: ["phase-4-model", "model"]
---
## 1. Description

Write the `scripts/train.py` entrypoint that orchestrates the whole training run from `config.yaml`: set seeds → load/clean/preprocess/split data → build & compile model → fit with callbacks → save the best model and history. This is the thin conductor; all logic lives in `src/`.

## 2. Learning Objective

- **Orchestration vs logic:** a script wires components; it shouldn't contain algorithms.
- **The training loop end-to-end:** how epochs, batches, forward/backward passes, and validation fit together (conceptually).
- **Config-first runs:** one command reproduces an exact experiment because every choice is in the config.
- **Persisting outputs:** model file + serialized `history` for the learning-curve plot (#39).

## 3. To-Do list for coding

- [ ] `scripts/train.py::main()` — load config, `set_global_seed`, `setup_logging`
- [ ] Build datasets (Phase 1–3 components) honoring all stage toggles
- [ ] `build_model` → `compile_model` → `model.fit(train, validation_data=val, callbacks=...)`
- [ ] Pass `class_weight` if `cleaning.imbalance == "class_weight"`
- [ ] Save `history` (e.g. JSON) for #39; log final val metrics
- [ ] Make it runnable: `python scripts/train.py`

## 4. Code learning (packages & methods)

- **`tensorflow.keras`** — `model.fit(..., callbacks, class_weight)`, `History`
- **project modules** — config, seeding, fetchers, preprocessing, splits, builders, callbacks
- **`json`** — persist the history dict

➡️ **After we implement:** you trace one full run from config to saved model and explain each stage's hand-off. I'll explain what Keras's `fit` actually does per batch (forward → loss → backprop → optimizer step) and how `validation_data` is evaluated each epoch.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — thin script, Loguru logs, train-only fitting, reproducible seeds.

> 🔀 **Note — Ablation-Driven Architecture:** `train.py` reads the entire run from `config.yaml`; flipping any stage toggle and re-running is a complete ablation experiment. See `CONTRIBUTING.md` §3.
