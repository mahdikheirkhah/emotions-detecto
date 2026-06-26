---
title: "[Phase 2] Clean validation (re-run EDA checks) → data.md §4"
labels: ["phase-2-cleaning", "documentation"]
---
## 1. Description

Go back to the notebook and re-run the EDA checks (#11–#14) on the **cleaned** data to confirm the cleaning actually worked: duplicates gone, distribution as intended, no malformed rows. Then write **Section 4** of `data.md` documenting the exact cleaning steps taken and their measured effect.

## 2. Learning Objective

- **Validating a transform by re-measuring:** never trust a cleaning step without before/after evidence.
- **Before/after comparison:** counts and distributions side-by-side prove the effect.
- **Closing the loop:** §2 (problems) → §3 (strategies) → §4 (what we did + result).
- **Guarding against over-cleaning:** confirm we didn't shrink or skew the data unacceptably.

## 3. To-Do list for coding

- [ ] Re-run inspection/duplicate/distribution cells on the cleaned dataset
- [ ] Produce before/after comparisons (row counts, class balance, dup count = 0)
- [ ] Add `data.md` Section **`## 4. Cleaning performed`** referencing the §2 problem numbers
- [ ] Record each step, the `config.yaml` option used, and the measured outcome

## 4. Code learning (packages & methods)

- **`pandas`** — `value_counts`, `shape`, comparison tables
- **`matplotlib` / `seaborn`** — before/after distribution plots

➡️ **After we implement:** you explain whether the cleaning met expectations and any surprises. I'll help interpret any distribution shift the cleaning introduced.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — evidence-backed, reproducible, numbered headings.

> 🔀 **Note — Ablation-Driven Architecture:** Validate with the `data_cleaning` stage **on** vs **off** to quantify cleaning's contribution. See `CONTRIBUTING.md` §3.
