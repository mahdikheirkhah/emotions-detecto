---
title: "[Phase 3.5] MNIST — logistic-regression baseline (scikit-learn)"
labels: ["phase-3.5-preliminary", "model"]
---
## 1. Description

Train a **logistic-regression** classifier on flattened MNIST pixels. This is the subject's suggested first model: it teaches how to turn images into a feature matrix and gives a non-deep baseline to later compare the CNN against — the start of our "does the deep model actually earn its complexity?" habit.

## 2. Learning Objective

- **Logistic regression as a linear classifier:** weighted sum of pixels → softmax → class probabilities.
- **Flattening images for a non-spatial model:** `28×28 → 784` features, and why this throws away spatial structure (foreshadowing why CNNs win).
- **Baselines matter:** you can't claim a CNN is good without a simpler reference point.
- **Reading accuracy & confusion on a clean set.**

## 3. To-Do list for coding

- [ ] Flatten MNIST to `(N, 784)`, rescale
- [ ] Train `LogisticRegression` (multinomial); record test accuracy
- [ ] Confusion matrix; note which digits are confused
- [ ] Save the baseline number to compare against #31

## 4. Code learning (packages & methods)

- **`sklearn.linear_model`** — `LogisticRegression(max_iter=..., multi_class="multinomial")`
- **`sklearn.metrics`** — `accuracy_score`, `confusion_matrix`, `classification_report`
- **`numpy`** — reshape/flatten

➡️ **After we implement:** you explain why flattening loses spatial info and how softmax turns scores into probabilities. I'll explain how scikit-learn fits logistic regression (the loss and the solver, e.g. L-BFGS, doing the optimization).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — baseline vs deep model comparison, seeded, report more than accuracy.

> 🔀 **Note — Ablation-Driven Architecture:** Model choice (`logreg | cnn`) and its hyperparameters come from `config.yaml`, so the baseline is just another dispatchable option. See `CONTRIBUTING.md` §3.
