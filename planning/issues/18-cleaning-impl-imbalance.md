---
title: "[Phase 2] Cleaning impl-b — class-imbalance handling (class_weight / over / under)"
labels: ["phase-2-cleaning", "data"]
---
## 1. Description

Implement the imbalance remedy chosen in #16 behind the dispatch: support `none`, `class_weight`, `oversample`, and `undersample` as selectable strategies in `config.yaml`. `class_weight` returns a weight dict for training; over/under resample the training set.

## 2. Learning Objective

- **How each remedy works mathematically:** `class_weight` scales the per-sample loss; oversampling replicates/augments minority samples; undersampling discards majority samples.
- **Trade-offs revisited in code:** oversampling → overfitting risk on duplicated minorities; undersampling → information loss.
- **Train-only application:** resampling/weights apply to the **training split only** — never validation/test.
- **Why this is a prime ablation target:** the imbalance fix often moves macro-F1 noticeably.

## 3. To-Do list for coding

- [ ] `data/imbalance.py` → strategies: `NoResample`, `ClassWeightStrategy`, `Oversampler`, `Undersampler`
- [ ] `resolve_imbalance(cfg, X_train, y_train)` dispatch on `cleaning.imbalance`
- [ ] `class_weight` path returns a `{class: weight}` dict for `model.fit`
- [ ] Guard: only touch the training split; log resulting class counts
- [ ] `tests/test_imbalance.py`: each strategy yields expected counts/weights; unknown option raises

## 4. Code learning (packages & methods)

- **`sklearn.utils.class_weight`** — `compute_class_weight("balanced", ...)`
- **`numpy`** — index sampling (`np.random.choice`) for over/under
- **`pandas`** — group sizes for resampling

➡️ **After we implement:** you explain why weights/resampling must never touch the test set. I'll explain how `compute_class_weight("balanced")` derives `n_samples / (n_classes * count_c)` and how that rebalances the gradient.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — no leakage, seeded resampling, report per-class effects.

> 🔀 **Note — Ablation-Driven Architecture:** `cleaning.imbalance` is one of the most informative toggles — comment all four options in `config.yaml`. See `CONTRIBUTING.md` §3.
