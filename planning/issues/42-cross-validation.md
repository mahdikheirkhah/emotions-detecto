---
title: "[Phase 5] Cross-validation concept + optional k-fold support"
labels: ["phase-5-eval", "model"]
---
## 1. Description

Add **optional, config-toggleable** k-fold cross-validation for more robust performance estimates. For a deep CNN this is expensive, so it defaults off and is primarily for the lighter models (the MNIST baseline, smaller architectures) — but understanding it is essential.

## 2. Learning Objective

- **Why a single split can mislead:** the estimate depends on which samples happened to land in validation.
- **k-fold cross-validation:** rotating the validation fold k times and averaging — lower-variance estimates.
- **Stratified k-fold:** preserving class ratios in each fold (vital for imbalance).
- **The cost trade-off:** k× training time — why it's a toggle, not a default, for deep nets.

## 3. To-Do list for coding

- [ ] `models/cross_val.py` → `cross_validate(build_fn, X, y, cfg) -> list[dict]` using `StratifiedKFold`
- [ ] Gate on `evaluation.cross_validation` (bool) + `evaluation.k` from config
- [ ] Aggregate mean ± std of metrics across folds; log them
- [ ] `tests/test_cross_val.py`: produces k result entries on a tiny set; off → skipped

## 4. Code learning (packages & methods)

- **`sklearn.model_selection`** — `StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)`
- **`numpy`** — aggregate mean/std across folds

➡️ **After we implement:** you explain why stratified folds matter for FER-2013 and when k-fold is worth the cost. I'll explain how `StratifiedKFold` partitions indices per class to keep each fold representative.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — StratifiedKFold, report mean ± std, seeded.

> 🔀 **Note — Ablation-Driven Architecture:** `evaluation.cross_validation` is a config toggle (default off for the CNN); turn it on to get variance estimates for any model. See `CONTRIBUTING.md` §3.
