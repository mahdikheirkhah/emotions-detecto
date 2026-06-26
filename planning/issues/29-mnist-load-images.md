---
title: "[Phase 3.5] MNIST — load & handle images in Python"
labels: ["phase-3.5-preliminary", "notebook"]
---
## 1. Description

The subject explicitly requires a preliminary warm-up: handle the MNIST digits dataset to learn how images work in Python *before* the harder emotion problem. Here we just load MNIST, inspect it, and visualize digits — the gentle on-ramp to everything in Phase 1.

## 2. Learning Objective

- **Images as arrays, again, on a clean dataset:** 28×28 grayscale digits, labels 0–9.
- **Why MNIST is the field's "hello world":** small, clean, balanced — perfect to isolate *technique* from *data mess*.
- **Train/test discipline from the start:** even on a toy set, never peek at test.
- **Connecting to the main task:** the same `pixels → reshape → visualize` muscle we used for FER-2013.

## 3. To-Do list for coding

- [ ] `notebooks/00_mnist.ipynb` — load MNIST (`keras.datasets.mnist.load_data`)
- [ ] Inspect shapes, dtypes, label distribution
- [ ] Visualize a grid of digits with labels
- [ ] Rescale to `[0,1]`; note the parallel to #20

## 4. Code learning (packages & methods)

- **`tensorflow.keras.datasets`** — `mnist.load_data()`
- **`numpy`** — shapes, `unique`, rescale
- **`matplotlib`** — `imshow`, grid of samples

➡️ **After we implement:** you explain how MNIST's structure mirrors FER-2013's. I'll explain how `mnist.load_data` fetches/caches the dataset and the IDX binary format it originally ships in.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — reproducible, clear cells.

> 🔀 **Note — Ablation-Driven Architecture:** Even the warm-up reads `seed`/paths from `config.yaml`; the MNIST fetcher is another dispatchable `BaseDatasetFetcher`. See `CONTRIBUTING.md` §3.
