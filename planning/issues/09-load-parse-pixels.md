---
title: "[Phase 1] Load CSV & parse the pixels column into image arrays (Fer2013Fetcher)"
labels: ["phase-1-data", "data"]
---
## 1. Description

Implement the first concrete `BaseDatasetFetcher` — `Fer2013Fetcher` — that loads a CSV and converts the space-separated `pixels` string of each row into a proper `48×48` NumPy array, returning a clean DataFrame / array pair the rest of the pipeline can use.

## 2. Learning Objective

- **How images live as numbers:** a grayscale image is a 2-D matrix of intensities (0–255); the `pixels` string is just that matrix flattened to 2304 values.
- **Flatten ↔ reshape:** the round trip between a `(2304,)` vector and a `(48, 48)` image, and why row-major order matters.
- **dtype & memory:** why we store pixels as `uint8` (or `float32` after scaling) and what that costs in memory.
- **Vectorization:** why parsing all rows with NumPy beats a Python loop.

## 3. To-Do list for coding

- [ ] `data/fer2013.py` → `Fer2013Fetcher(BaseDatasetFetcher)`
  - [ ] `fetch(split: str) -> tuple[np.ndarray, np.ndarray]` returning `(images[N,48,48], labels[N])`
  - [ ] `_parse_pixels(s: str) -> np.ndarray` (split → `np.fromstring`/`np.array` → reshape `48×48`)
- [ ] Validate: correct row count, label range 0–6, no NaN
- [ ] `tests/test_fer2013.py`: a tiny fake CSV parses to the right shape; a malformed pixel row raises `ValueError`

## 4. Code learning (packages & methods)

- **`pandas`** — `read_csv`, column access, `Series.apply`
- **`numpy`** — `np.fromstring` / `np.array(..., dtype=np.uint8)`, `reshape`, `np.stack`
- **`abc`** — implements the `BaseDatasetFetcher` contract

➡️ **After we implement:** you explain the flatten↔reshape round trip and why we vectorize. I'll explain how `pandas.read_csv` streams and type-infers columns, and how NumPy stores an array as a contiguous buffer + shape/stride metadata.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §4/§6 — extends `BaseDatasetFetcher`, type hints, docstrings, try/except (`KeyError`, `ValueError`), tests on both paths.

> 🔀 **Note — Ablation-Driven Architecture:** Which fetcher to use (`fer2013` | `mnist`) is selected via dispatch from `config.yaml`. See `CONTRIBUTING.md` §3.
