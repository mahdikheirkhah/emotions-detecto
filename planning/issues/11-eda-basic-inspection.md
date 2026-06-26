---
title: "[Phase 1] EDA-a — basic inspection (shape, dtypes, NaN, value ranges, Usage split)"
labels: ["phase-1-data", "notebook"]
---
## 1. Description

Start the EDA notebook with the fundamentals: load the data, inspect headers, dtypes, shape, missing values, pixel value ranges, label range, and the `Usage` split counts. This is the "know your data" pass before any plotting.

## 2. Learning Objective

- **What EDA is and why it precedes modeling:** you cannot clean or model what you haven't characterized.
- **Data types & ranges:** confirming `pixels` are 0–255 ints, labels are 0–6, and there are no NaNs/empties.
- **Splits & leakage awareness:** understanding `Usage` (Training / PublicTest / PrivateTest) so we never train on test data.
- **Sanity invariants:** every row must have exactly 2304 pixel values.

## 3. To-Do list for coding

- [ ] `notebooks/01_eda.ipynb` — load with `Fer2013Fetcher`
- [ ] `df.head()`, `df.info()`, `df.describe()`; column dtypes
- [ ] Count NaN/empty/whitespace rows; assert pixel-count == 2304 per row
- [ ] Min/max/mean of pixel intensities; unique label values; `Usage` value counts
- [ ] Note every anomaly found (feeds #15 and the cleaning issues)

## 4. Code learning (packages & methods)

- **`pandas`** — `read_csv`, `head`, `info`, `describe`, `isna`, `value_counts`, `nunique`
- **`numpy`** — `min`, `max`, `mean`, array validation

➡️ **After we implement:** you explain what each summary told us about data health. I'll explain how `df.describe()` computes its percentiles and how pandas represents missing values (`NaN` as a float sentinel).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — clear cells, log findings, reproducible (seeded) sampling.

> 🔀 **Note — Ablation-Driven Architecture:** Every anomaly found here becomes a *named, switchable* cleaning/FE option in `config.yaml` later. See `CONTRIBUTING.md` §3.
