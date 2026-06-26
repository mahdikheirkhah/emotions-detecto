---
title: "[Phase 1] data.md §2 — EDA results"
labels: ["phase-1-data", "documentation"]
---
## 1. Description

Write **Section 2** of `data.md`: a concise, evidence-backed summary of everything the EDA (#11–#14) uncovered — distributions, imbalance, brightness/contrast spread, duplicates, label noise — with the key plots embedded. This section defines the **problem list** that §3 (strategies) and §4 (cleaning) will answer.

## 2. Learning Objective

- **Turning exploration into a problem statement:** distilling notebooks into a crisp list of issues.
- **Evidence over assertion:** every claimed problem is backed by a number or a figure.
- **Traceability:** numbering problems so later sections can reference them (e.g. "Problem 2.3").

## 3. To-Do list for coding

- [ ] Add `data.md` Section **`## 2. Exploratory Data Analysis`**
- [ ] Subsections for: class distribution/imbalance, pixel-intensity/brightness/contrast, duplicates, label noise/non-faces, missing/malformed rows
- [ ] Embed the key plots (export PNGs from the notebook into `results/` or `docs/`)
- [ ] End with a **numbered problem list** that §3 and §4 will reference

## 4. Code learning (packages & methods)

- **`matplotlib`** — `savefig` to export the figures referenced here
- *Mostly documentation* — prose + embedded images

➡️ **After we implement:** you summarize the top 3 data problems in your own words. I'll help you rank them by likely impact on model accuracy.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — numbered headings, evidence-backed, honest about limitations.

> 🔀 **Note — Ablation-Driven Architecture:** Each numbered problem here maps to a switchable strategy in `config.yaml`, so we can later measure each fix's effect. See `CONTRIBUTING.md` §3.
