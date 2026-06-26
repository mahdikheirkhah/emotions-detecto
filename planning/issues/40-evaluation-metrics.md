---
title: "[Phase 5] Evaluation metrics module (accuracy, macro-F1, confusion matrix, per-class report)"
labels: ["phase-5-eval", "model"]
---
## 1. Description

Build a reusable evaluation module that reports more than raw accuracy: overall accuracy, **macro-F1**, a **confusion matrix**, and a per-class precision/recall report. Which metrics to compute is config-driven. This is the honest scorecard for an imbalanced 7-class problem.

## 2. Learning Objective

- **Why accuracy misleads on imbalance:** a model can score "well" by favoring "Happy" and ignoring "Disgust".
- **Precision / recall / F1:** what each captures and why macro-averaging weights every emotion equally.
- **Confusion matrix literacy:** reading which emotions get mistaken for which (e.g. Fear↔Surprise) — and what that says about the data.
- **Choosing the metric to optimize:** aligning the reported metric with the clinical goal.

## 3. To-Do list for coding

- [ ] `models/evaluation.py` → `evaluate(model, X_test, y_test, cfg) -> dict`
- [ ] Compute accuracy, macro-F1, confusion matrix, classification report (per config `evaluation.metrics`)
- [ ] Plot + save the confusion matrix (`results/model/confusion_matrix.png`)
- [ ] `tests/test_evaluation.py`: on a tiny fake set, metrics match hand-computed values

## 4. Code learning (packages & methods)

- **`sklearn.metrics`** — `accuracy_score`, `f1_score(average="macro")`, `confusion_matrix`, `classification_report`
- **`numpy`** — `argmax` to turn softmax/one-hot into class indices
- **`matplotlib`/`seaborn`** — confusion-matrix heatmap

➡️ **After we implement:** you read the confusion matrix and explain which emotions confuse the model and why. I'll explain how F1 is the harmonic mean of precision and recall and why macro-averaging is the fair choice here.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — report macro-F1 alongside accuracy, per-class breakdown.

> 🔀 **Note — Ablation-Driven Architecture:** `evaluation.metrics` is config-driven; compute the same metrics across ablation runs to compare fairly. See `CONTRIBUTING.md` §3.
