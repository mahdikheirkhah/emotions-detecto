---
title: "[Phase 3] Train / validation / test split (no leakage, stratified)"
labels: ["phase-3-features", "data"]
---
## 1. Description

Create the canonical data splits: respect FER-2013's `Usage` column for the test set, and carve a **stratified** validation set out of training. Fit every transform (standardization stats, PCA) on **train only**. This issue enforces the no-leakage rule that the whole project's credibility rests on.

## 2. Learning Objective

- **Why split before fitting anything:** computing stats on the full set leaks test information into training.
- **Train / validation / test roles:** train fits weights, validation guides early-stopping/tuning, test is touched once for the final number.
- **Stratified splitting:** preserving class ratios so the validation set reflects the (imbalanced) reality.
- **Honoring provided splits:** using `Usage` (PublicTest/PrivateTest) as the held-out test, matching the audit's `test_with_emotions.csv` expectation.

## 3. To-Do list for coding

- [ ] `data/splits.py` → `make_splits(cfg, X, y, usage) -> (X_train, y_train, X_val, y_val, X_test, y_test)`
- [ ] Use `Usage` for test; stratified train/val split with `val_size` + `seed` from config
- [ ] Assert no index overlap between splits; log split sizes + per-class balance
- [ ] `tests/test_splits.py`: splits are disjoint; class ratios preserved; deterministic under fixed seed

## 4. Code learning (packages & methods)

- **`sklearn.model_selection`** — `train_test_split(stratify=y, random_state=seed)`
- **`numpy` / `pandas`** — masking by `Usage`, overlap assertions

➡️ **After we implement:** you explain why fitting standardization before splitting would be leakage. I'll explain how `train_test_split` performs a stratified shuffle (per-class index partitioning) under the hood.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — split before fitting, seeded, stratified, no leakage.

> 🔀 **Note — Ablation-Driven Architecture:** `val_size` and `seed` come from `config.yaml`; the split is fixed so every ablation trains/evaluates on identical data. See `CONTRIBUTING.md` §3.
