---
title: "[Phase 0] Declare dependencies, resolve conflicts & export requirements.txt"
labels: ["phase-0-foundations", "setup"]
---
## 1. Description

Add every runtime and dev dependency, resolve the classic conflicts (TensorFlow ↔ NumPy ↔ OpenCV), lock them, and export a `requirements.txt` so the project also installs without Poetry. Directly satisfies the audit *"Does the environment contain all libraries used and their versions?"*

## 2. Learning Objective

- **Transitive dependencies & conflicts:** why two packages can demand incompatible versions of a third (e.g. NumPy), and how a resolver finds a compatible set.
- **ABI pinning:** why TensorFlow pins a narrow NumPy range (compiled C extensions are built against a specific NumPy ABI).
- **Prebuilt wheels:** why `opencv-python` ships a compiled binary so you don't build OpenCV from source.
- **Lockfile vs `requirements.txt`:** the lock is the source of truth; the exported file is a portable snapshot.

## 3. To-Do list for coding

- [ ] `poetry add tensorflow tf-keras opencv-python numpy pandas matplotlib seaborn scikit-learn pyyaml tensorboard streamlit loguru`
- [ ] `poetry add -G dev black pytest jupyter ipykernel keras-tuner`
- [ ] Resolve any conflict surfaced by the resolver (adjust constraints, not by force)
- [ ] `poetry lock` to freeze the resolved set
- [ ] `poetry export -f requirements.txt -o requirements.txt --without-hashes`
- [ ] Smoke-test: `poetry run python -c "import tensorflow, cv2, numpy, pandas, sklearn, yaml, loguru"`

## 4. Code learning (packages & methods)

- **Poetry CLI** — `add`, `add -G dev`, `lock`, `export`
- **`importlib` / plain `import`** — smoke-testing that everything loads
- **`pip`** (conceptually) — how `requirements.txt` is consumed downstream

➡️ **After we implement:** you explain why we pin versions and what a lockfile guarantees. I'll explain how dependency resolution conflicts arise and how compiled wheels embed platform-specific binaries.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §2 — Poetry + exported `requirements.txt`, documented versions.

> 🔀 **Note — Ablation-Driven Architecture:** Pinned, reproducible deps ensure ablation comparisons differ only by the toggled stage, never by a library version drift. See `CONTRIBUTING.md` §3.
