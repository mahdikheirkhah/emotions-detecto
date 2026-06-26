---
title: "[Phase 2] Cleaning strategy discussion → data.md §3"
labels: ["phase-2-cleaning", "documentation"]
---
## 1. Description

Before writing any cleaning code, discuss the **options** for each problem found in §2 — for duplicates, imbalance, malformed/constant images, brightness outliers — weighing the advantages and disadvantages of each, then record the chosen approach (and *why*) in **Section 3** of `data.md`. This is a conversation + documentation issue.

## 2. Learning Objective

- **Strategy thinking:** every data problem has multiple valid fixes with trade-offs — there's rarely one "correct" answer.
- **Imbalance remedies compared:** `class_weight` (reweight loss) vs **oversampling** (duplicate/augment minorities) vs **undersampling** (drop majorities) — cost, overfitting risk, information loss.
- **Duplicate handling:** drop vs keep, and the leakage angle.
- **Knowing when *not* to clean:** over-cleaning can erase signal or shrink the set below what a CNN needs.

## 3. To-Do list for coding

- [ ] Add `data.md` Section **`## 3. Cleaning strategies (options & decisions)`**
- [ ] For each numbered problem from §2: list candidate strategies, pros/cons, and the **chosen** one with justification
- [ ] Map each decision to the exact `config.yaml` key + option it will use
- [ ] Note which decisions are *ablation candidates* (worth toggling later)

## 4. Code learning (packages & methods)

- *No code* — this is a design + documentation issue that defines the `config.yaml` `cleaning` block.

➡️ **After we implement:** you argue for one imbalance strategy over the others for *this* dataset. I'll stress-test your reasoning and add where each strategy tends to fail in practice.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — imbalance-aware, no-leakage reasoning documented here.

> 🔀 **Note — Ablation-Driven Architecture:** This issue literally specifies the `cleaning` options in `config.yaml` (with comments) that #17–#18 implement behind a dispatch. See `CONTRIBUTING.md` §3.
