---
title: "[Phase 6] Pre-trained CNN / transfer learning (VGG / ResNet) — optional"
labels: ["phase-6-optional", "optional", "model"]
---
## 1. Description

**Optional** (does not replace the from-scratch CNN): use a pre-trained backbone (VGG16 / ResNet50) via transfer learning to try to beat the scratch model's accuracy. Save it as `pre_trained_model` and document its architecture in `pre_trained_model_architecture.txt`.

## 2. Learning Objective

- **Transfer learning:** reusing features learned on a huge dataset (ImageNet) for a new, smaller task.
- **Feature extraction vs fine-tuning:** freezing the backbone vs unfreezing top layers with a low LR.
- **Input adaptation:** 48×48 grayscale → the backbone's expected size/3-channels (resize + channel replication).
- **Cost/benefit:** why pre-trained nets save compute but bring domain-mismatch and size overhead.

## 3. To-Do list for coding

- [ ] `models/transfer.py` → `TransferModelBuilder(BaseModelBuilder)` wrapping `VGG16`/`ResNet50` (`include_top=False`) + a new head
- [ ] Config: `model.architecture: "transfer_vgg16"`, `transfer.trainable_layers`, input resize params
- [ ] Train (feature-extract first, optional fine-tune); evaluate vs the scratch model
- [ ] Save model + write `results/model/pre_trained_model_architecture.txt`

## 4. Code learning (packages & methods)

- **`tensorflow.keras.applications`** — `VGG16`/`ResNet50(include_top=False, weights="imagenet")`, `preprocess_input`
- **`tensorflow.keras.layers`** — `Resizing`, new dense head; `layer.trainable = False`
- **`tf.image`** — `grayscale_to_rgb`, `resize`

➡️ **After we implement:** you explain when to freeze vs fine-tune and why grayscale must be expanded to 3 channels. I'll explain what ImageNet features the early conv layers encode and why they transfer to faces.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — optional, but still OOP (`BaseModelBuilder`), documented, no leakage.

> 🔀 **Note — Ablation-Driven Architecture:** Transfer model is just another `model.architecture` option behind the same dispatch — compare it to scratch via config. See `CONTRIBUTING.md` §3.
