---
title: "[Phase 8] Bonus-b — gradient-sign attack → flip 'Happy' to 'Sad', verify slight change"
labels: ["phase-8-delivery", "bonus", "optional"]
---
## 1. Description

**Bonus payoff:** implement the adversarial attack. Starting from the >90%-Happy image (#57), compute the gradient of the loss toward the target class **Sad** w.r.t. the input pixels, and nudge the pixels (FGSM / iterative FGSM) until the model predicts **Sad** — while keeping the change **slight** so the image is still clearly the same face.

## 2. Learning Objective

- **Attacking inputs, not weights:** the model is frozen; we backprop to the pixels.
- **FGSM vs iterative FGSM (BIM):** one big signed step vs several tiny clipped steps for a subtler perturbation.
- **The imperceptibility constraint:** bounding the perturbation (`epsilon`, L∞) so the audit's "still recognizable" check passes.
- **Why this matters:** adversarial robustness is a real safety concern for a clinical model.

## 3. To-Do list for coding

- [ ] `scripts/adversarial.py` (part 2) → `tf.GradientTape` on the input; loss toward "Sad"
- [ ] FGSM step `x_adv = x + eps * sign(grad)` (+ optional iterative loop with clipping to an `epsilon` ball)
- [ ] Stop when predicted class == "Sad"; keep perturbation small
- [ ] Save original vs adversarial side-by-side + the perturbation; log both predictions
- [ ] Confirm: original → Happy, adversarial → Sad, images look the same

## 4. Code learning (packages & methods)

- **`tensorflow`** — `tf.GradientTape`, `tape.gradient(loss, input)`, `tf.sign`, `tf.clip_by_value`
- **`tensorflow.keras.losses`** — `CategoricalCrossentropy` toward the target class
- **`numpy`/`matplotlib`** — visualize original / perturbation / adversarial

➡️ **After we implement:** you explain why stepping along `sign(grad)` flips the class with minimal visible change. I'll explain how `GradientTape` records the forward ops to compute the input gradient by reverse-mode autodiff.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — bonus, documented, reproducible; keep the perturbation bounded.

> 🔀 **Note — Ablation-Driven Architecture:** `epsilon`, iteration count, and attack type (`fgsm | bim`) are config-driven so the attack strength is a tunable knob. See `CONTRIBUTING.md` §3.
