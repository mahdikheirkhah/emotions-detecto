---
title: "[Phase 1] Download & unzip the FER-2013 dataset"
labels: ["phase-1-data", "data"]
---
## 1. Description

Fetch the project dataset (the `emotions-detector.zip` from the 01-edu assets, i.e. the FER-2013 Kaggle data), unzip it into `data/`, and verify the expected CSVs are present (`train.csv`, `test.csv` / `test_with_emotions.csv`). Keep the raw files out of git.

## 2. Learning Objective

- **Where CV datasets come from** and why FER-2013 is a standard benchmark for facial-expression recognition.
- **The shape of this dataset:** a CSV where each row is one 48×48 grayscale face encoded as a `pixels` string plus an `emotion` label (0–6) and a `Usage` split column.
- **Data provenance & integrity:** why we verify a download (size/row count) before trusting it.
- **Why raw data is git-ignored** (size, licensing, reproducibility via a fetch step instead).

## 3. To-Do list for coding

- [ ] `data/download.py` (or a `Fer2013Downloader` class) → `download(url, dest) -> Path` and `extract(zip_path, dest) -> Path`
- [ ] Read the dataset URL + target paths from `config.yaml` (`global.data_dir`, `data.url`)
- [ ] Verify after extraction: expected files exist, log row counts
- [ ] Skip re-download if files already present (idempotent)

## 4. Code learning (packages & methods)

- **`urllib.request`** — `urlretrieve` (or `requests.get`) to fetch the zip
- **`zipfile`** — `ZipFile`, `.extractall()`
- **`pathlib`** — path handling, existence checks
- **`hashlib`** (optional) — checksum verification

➡️ **After we implement:** you explain why the download step is idempotent and what `Usage` means. I'll explain how `zipfile` reads the central directory of a zip archive to extract members without unpacking the whole file into memory.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — try/except (`URLError`, `FileNotFoundError`, `BadZipFile`), Loguru progress logs, type hints, docstrings.

> 🔀 **Note — Ablation-Driven Architecture:** Dataset URL and paths live in `config.yaml`; nothing is hardcoded. See `CONTRIBUTING.md` §3.
