---
title: "[Phase 5] predict.py — evaluate test set, print 'Accuracy on test set: X%'"
labels: ["phase-5-eval", "model"]
---
## 1. Description

Write the `scripts/predict.py` entrypoint required by the audit: load the saved `final_emotion_model.keras`, evaluate it on `test_with_emotions.csv`, and **print** exactly `Accuracy on test set: XX%`. Must run error-free and report **> 60%**.

## 2. Learning Objective

- **Inference vs training:** loading a frozen model and only doing forward passes.
- **The test set's one job:** a single, final, untouched measurement — no tuning against it.
- **Reproducible evaluation:** identical preprocessing as training, applied to test.
- **The print contract:** why the exact stdout format matters for automated grading.

## 3. To-Do list for coding

- [ ] `scripts/predict.py::main()` — load config + model (`keras.models.load_model`)
- [ ] Load & preprocess the test split with the **same** transforms used in training (no leakage, no re-fit)
- [ ] Compute accuracy; `print(f"Accuracy on test set: {acc:.0%}")`
- [ ] Optionally log macro-F1/confusion via #40 (but keep the required print exact)
- [ ] Verify `python ./scripts/predict.py` runs clean and prints > 60%

## 4. Code learning (packages & methods)

- **`tensorflow.keras.models`** — `load_model`
- **project modules** — fetcher + the *fitted* preprocessing (reused, not refit)
- **`numpy`** — `argmax`, accuracy computation

➡️ **After we implement:** you explain why the test preprocessing must reuse training-fitted stats and never refit. I'll explain how `load_model` reconstructs the architecture + weights from the `.keras` archive (config JSON + weight tensors).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §5 — this script is one of the two allowed to `print` its required stdout.

> 🔀 **Note — Ablation-Driven Architecture:** `predict.py` reads the model/test paths from `config.yaml`, so it scores whichever model an ablation run produced. See `CONTRIBUTING.md` §3.
