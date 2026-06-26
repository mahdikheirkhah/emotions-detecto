---
title: "[Phase 1] EDA-d — duplicate detection & label-noise / non-face inspection"
labels: ["phase-1-data", "notebook"]
---
## 1. Description

Hunt for the subtler data problems: exact/near-duplicate images, mislabeled faces, and rows that aren't really faces (FER-2013 contains some). Quantify how many there are so we can decide what to do in cleaning.

## 2. Learning Objective

- **Label noise:** what it is, how it caps achievable accuracy, and why a "60% target" is partly a reflection of dataset noise.
- **Duplicates & leakage risk:** identical images appearing in both train and test inflate scores.
- **Detecting duplicates cheaply:** hashing image bytes vs near-duplicate similarity.
- **The limits of cleaning:** why we accept some noise rather than over-cleaning.

## 3. To-Do list for coding

- [ ] Exact-duplicate detection via per-image hash (e.g. hash of the pixel bytes); count duplicates within and across splits
- [ ] Spot-check: render a sample of each class and eyeball obvious mislabels / non-faces
- [ ] (Optional) near-duplicate check via a simple similarity metric on a subsample
- [ ] Tally counts for `data.md` §2 and the cleaning discussion (#16)

## 4. Code learning (packages & methods)

- **`hashlib`** — `md5`/`sha1` of `array.tobytes()` for exact-dup detection
- **`pandas`** — `duplicated`, `groupby`
- **`numpy`** — array hashing/equality
- **`matplotlib`** — render suspects for manual review

➡️ **After we implement:** you explain how hashing finds exact duplicates and why cross-split duplicates are dangerous. I'll explain how a cryptographic hash maps arbitrary bytes to a fixed-size fingerprint and why collisions are negligible here.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §7 — handle edge cases (empty/constant images) explicitly; log counts.

> 🔀 **Note — Ablation-Driven Architecture:** Duplicate handling becomes the toggleable `cleaning.duplicates: drop | keep` option. See `CONTRIBUTING.md` §3.
