---
title: "[Phase 5] Hyperparameter tuning-b — run search, select best, log results"
labels: ["phase-5-eval", "model"]
---
## 1. Description

Run the search defined in #43, select the best hyperparameter set by validation performance, log the full results table, and write the winning values back into `config.yaml` as the new defaults (keeping the searched arrays as comments for reproducibility).

## 2. Learning Objective

- **Reading a trials table:** ranking configs and spotting which hyperparameter actually moved the metric.
- **Best-config selection:** picking by validation score, then confirming once on test.
- **Diminishing returns & overfitting the search:** why more trials isn't always better.
- **Promoting results to config:** closing the loop so the best run is reproducible by default.

## 3. To-Do list for coding

- [ ] `scripts/tune.py` → run the tuner from #43; capture per-trial metrics
- [ ] Log a sorted results table; persist it (CSV/JSON) under `results/`
- [ ] Update `config.yaml` defaults to the winning values (leave search arrays as comments)
- [ ] Retrain once with the best config; confirm test accuracy

## 4. Code learning (packages & methods)

- **`keras_tuner`** — `tuner.search(...)`, `get_best_hyperparameters`, `get_best_models`
- **`pandas`** — results table
- **project `train.py`** — retrain with the chosen config

➡️ **After we implement:** you interpret the trials table and justify the winning config. I'll explain how the tuner stores trial state and resumes/compares runs so a search is itself reproducible.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — log everything, select on validation, confirm once on test.

> 🔀 **Note — Ablation-Driven Architecture:** Winning values become `config.yaml` defaults; the searched arrays stay as comments so the experiment is never lost. See `CONTRIBUTING.md` §3.
