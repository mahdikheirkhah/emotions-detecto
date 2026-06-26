---
title: "[Phase 1] EDA-b — class distribution & imbalance analysis"
labels: ["phase-1-data", "notebook"]
---
## 1. Description

Analyze how the 7 emotions are distributed across the training set and quantify the imbalance (FER-2013 is famously skewed — "Disgust" is tiny, "Happy" is large). Produce clear bar plots and per-class counts/percentages.

## 2. Learning Objective

- **Class imbalance:** what it is and why it biases a classifier toward majority classes.
- **Why accuracy alone misleads** on imbalanced data, motivating macro-F1 and per-class metrics later.
- **Reading a distribution plot:** counts vs proportions, and spotting the long tail (Disgust).
- **Foreshadowing remedies:** class weights, oversampling, augmentation — chosen in #18.

## 3. To-Do list for coding

- [ ] In the EDA notebook: per-class counts + percentages (`value_counts(normalize=True)`)
- [ ] Bar plot of class frequencies (with class names, not codes)
- [ ] Compute an imbalance ratio (max class / min class)
- [ ] Record the numbers for `data.md` §2

## 4. Code learning (packages & methods)

- **`pandas`** — `value_counts(normalize=...)`, `map` (code → emotion name)
- **`matplotlib` / `seaborn`** — `bar` / `countplot`, axis labels, titles

➡️ **After we implement:** you explain what the imbalance ratio implies for training. I'll explain how seaborn's `countplot` aggregates categories internally and how matplotlib renders a figure (Figure → Axes → Artists).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — imbalance-aware thinking; we will report macro-F1, not just accuracy.

> 🔀 **Note — Ablation-Driven Architecture:** The imbalance remedy is a `config.yaml` option (`cleaning.imbalance: none | class_weight | oversample | undersample`) we can ablate. See `CONTRIBUTING.md` §3.
