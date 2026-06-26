---
title: "[Phase 3] data.md §5 — Feature-engineering write-up"
labels: ["phase-3-features", "documentation"]
---
## 1. Description

Write **Section 5** of `data.md`: explain the feature-engineering decisions for this image dataset — normalization, contrast enhancement, and augmentation — what each does, why it suits FER-2013, and which `config.yaml` options control it. Document what we considered adding/removing and why.

## 2. Learning Objective

- **Feature engineering for images vs tabular data:** here we transform pixel intensities and synthesize variations rather than craft new columns.
- **Justifying each transform from evidence:** tie every choice back to a §2 finding (low contrast → equalization, small set + imbalance → augmentation).
- **Documenting the inference contract:** which transforms must also run live on the webcam frames.

## 3. To-Do list for coding

- [ ] Add `data.md` Section **`## 5. Feature engineering`**
- [ ] Subsections: normalization, histogram equalization/CLAHE, augmentation — each with rationale + the controlling `config.yaml` key
- [ ] State which transforms are train-only vs applied everywhere (incl. live inference)
- [ ] Note the ablation expectation (what we predict each toggle does to accuracy)

## 4. Code learning (packages & methods)

- *Documentation* — references the code from #20–#22; may embed augmented-sample images from #24.

➡️ **After we implement:** you explain which transform you expect to help most and why. I'll add notes on transforms that commonly look helpful but don't move the needle on 48×48 grayscale faces.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — numbered headings, evidence-linked decisions.

> 🔀 **Note — Ablation-Driven Architecture:** Section 5 is the *why* for the FE stage; `config.yaml` holds the *what*. Document the expected effect of each toggle. See `CONTRIBUTING.md` §3.
