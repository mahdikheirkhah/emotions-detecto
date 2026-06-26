---
title: "[Phase 5] Hyperparameter tuning-a — define config grids + tuner setup"
labels: ["phase-5-eval", "model"]
---
## 1. Description

Set up systematic hyperparameter search. Define the search space as **arrays of candidate values in `config.yaml`** (learning rate, batch size, dropout, #filters, optimizer) and wire a tuner (Keras Tuner for the CNN; `GridSearchCV` is the concept for lighter models). This issue is the scaffolding; the run is #44.

## 2. Learning Objective

- **Hyperparameters vs parameters:** weights are learned; hyperparameters are chosen.
- **Search strategies:** grid vs random vs Bayesian — coverage vs cost trade-offs.
- **Why search against validation, never test:** keeping the final number honest.
- **Defining a sane space:** which knobs matter most (LR first) and realistic ranges.

## 3. To-Do list for coding

- [ ] Add a `tuning:` block to `config.yaml`: arrays per hyperparameter + `strategy` (`grid | random | bayesian`) + `max_trials`
- [ ] `models/tuning.py` → `build_hypermodel(hp, cfg)` mapping config arrays to Keras-Tuner search dimensions
- [ ] `make_tuner(cfg)` dispatch on `tuning.strategy`
- [ ] `tests/test_tuning.py`: the search space is built from config arrays correctly

## 4. Code learning (packages & methods)

- **`keras_tuner`** — `HyperParameters` (`hp.Choice`, `hp.Float`), `RandomSearch`/`Hyperband`/`BayesianOptimization`
- **`sklearn.model_selection`** — `GridSearchCV` / `ParameterGrid` (concept for non-deep models)

➡️ **After we implement:** you explain why we tune against validation and which knob you'd search first. I'll explain how Keras Tuner samples the space and prunes weak trials (e.g. Hyperband's successive halving).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — tune on validation only, seeded, documented.

> 🔀 **Note — Ablation-Driven Architecture:** The search space lives entirely in `config.yaml` as commented arrays — the purest expression of the ablation philosophy. See `CONTRIBUTING.md` §3.
