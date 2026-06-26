---
title: "[Phase 5] Save & document final artifacts (final_emotion_model.keras + _arch.txt)"
labels: ["phase-5-eval", "documentation"]
---
## 1. Description

Finalize and document the model deliverables: ensure `results/model/final_emotion_model.keras` is the best model, and write `results/model/final_emotion_model_arch.txt` with the full `model.summary()` **plus** the narrative of *why* this architecture was chosen and what earlier iterations looked like. This directly answers the audit on architecture justification.

## 2. Learning Objective

- **Model serialization:** what the `.keras` format stores (architecture + weights + optimizer state) and why it's portable.
- **Communicating an architecture:** reading `model.summary()` and explaining design intent.
- **The value of an iteration log:** documenting failed/weaker attempts is part of real ML engineering.
- **Reproducibility of the artifact:** the config that produced it is recorded alongside.

## 3. To-Do list for coding

- [ ] Confirm `ModelCheckpoint` saved the best model to `final_emotion_model.keras`
- [ ] Write `final_emotion_model_arch.txt`: full `model.summary()` + rationale + **iteration history** (what was tried, what changed, why)
- [ ] Record the exact `config.yaml` used (snapshot or hash) next to the model
- [ ] Sanity-load the saved model and re-check test accuracy matches

## 4. Code learning (packages & methods)

- **`tensorflow.keras`** — `model.save`, `model.summary(print_fn=...)`, `load_model`
- **`io` / `pathlib`** — capture summary text to the `.txt`

➡️ **After we implement:** you narrate the architecture and its iteration history in your own words. I'll explain how the `.keras` zip archive lays out `config.json` + weight shards and how `load_model` rebuilds the exact graph.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — explain *why* the architecture was chosen and prior iterations (audit requirement).

> 🔀 **Note — Ablation-Driven Architecture:** The iteration history *is* your ablation log — record which config toggles led to each accuracy change. See `CONTRIBUTING.md` §3.
