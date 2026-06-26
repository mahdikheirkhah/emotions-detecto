---
title: "[Phase 1] data.md §1 — Raw data documentation"
labels: ["phase-1-data", "documentation"]
---
## 1. Description

Write **Section 1** of `data.md`: a clear description of the **raw** dataset before any analysis — what it is, its structure, its relationships, and *why it is a good fit* for an emotion-classification CNN (and what it may lack). This is documentation, not code.

## 2. Learning Objective

- **Documenting data as a first-class artifact:** why every serious ML project keeps a living `data.md`.
- **Reading a dataset's "shape of meaning":** the `pixels → emotion` relationship, the 7-class label space, the train/test `Usage` split.
- **Fitness-for-purpose reasoning:** what makes FER-2013 suitable (large, labeled, face-centered, standard benchmark) and its known weaknesses (label noise, class imbalance, low resolution, some non-face images).

## 3. To-Do list for coding

- [ ] Create `data.md` Section 1 with numbered heading **`## 1. Raw data`**
- [ ] Cover: source/provenance, number of samples, image format (48×48 grayscale), the 7 emotion classes + their integer codes, the `Usage` column, the `pixels` encoding
- [ ] State the `pixels → emotion` relationship and how we'll use it (supervised image classification)
- [ ] State why it's a good choice and what it may lack (set up the EDA in §2)

## 4. Code learning (packages & methods)

- *No code* — this is a documentation issue. Optionally embed one sample image rendered in #13.

➡️ **After we implement:** you explain back, in your own words, why FER-2013 fits an emotion CNN and what risks you already anticipate. I'll add context on how facial-expression datasets are typically collected/labeled (and why that creates label noise).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — clear structure, numbered headings, honest about limitations.

> 🔀 **Note — Ablation-Driven Architecture:** `data.md` records the *why* behind each data decision; `config.yaml` records the *what*. Keep them in sync. See `CONTRIBUTING.md` §3.
