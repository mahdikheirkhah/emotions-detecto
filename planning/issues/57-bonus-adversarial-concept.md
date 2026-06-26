---
title: "[Phase 8] Bonus-a — pick a >90% 'Happy' image + the FGSM adversarial concept"
labels: ["phase-8-delivery", "bonus", "optional"]
---
## 1. Description

**Bonus setup:** from the dataset, find an image the trained model classifies as **Happy with > 90%** confidence — the starting point for the "hack the CNN" attack (#58). Alongside, study the adversarial-example concept so the attack in #58 is understood, not copy-pasted.

## 2. Learning Objective

- **What an adversarial example is:** a tiny, often imperceptible input perturbation that flips the prediction.
- **Why neural nets are fooled:** local linearity in high dimensions — many small coordinated nudges sum to a big logit change.
- **Gradient w.r.t. the input (not the weights):** the key inversion — we freeze weights and ask "how should pixels change to raise the target class?"
- **FGSM intuition:** step every pixel by `epsilon * sign(gradient)` toward the target class.

## 3. To-Do list for coding

- [ ] `scripts/adversarial.py` (part 1) → scan the dataset, find images predicted **Happy > 90%**
- [ ] Save a chosen source image + its predicted probabilities
- [ ] Write a short note (in the script/docstring or `data.md`/README) explaining FGSM in your words
- [ ] Confirm the model's confidence on the chosen image

## 4. Code learning (packages & methods)

- **`tensorflow.keras`** — `model.predict` to score candidates
- **`numpy`** — filter by predicted class/probability
- **`matplotlib`** — view the chosen image

➡️ **After we implement:** you explain, in plain words, why imperceptible changes can fool the model. I'll explain the gradient-of-output-w.r.t.-input idea and how it differs from normal weight-gradient training.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — bonus, but still documented, seeded, reproducible.

> 🔀 **Note — Ablation-Driven Architecture:** Target class, source-confidence threshold, and `epsilon` are `config.yaml` values for the bonus block. See `CONTRIBUTING.md` §3.
