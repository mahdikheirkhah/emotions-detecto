# Data

This file is the **living data record** for the Emotions Detecto project.
It documents *why* every data decision was made; `config.yaml` records *what* value is currently active.
Keep both in sync whenever the pipeline or dataset understanding changes.

---

## 1. Raw data

### 1.1 Source and provenance

**Dataset:** FER-2013 (Facial Expression Recognition 2013)
**Origin:** Introduced at ICML 2013 by Goodfellow et al. Images were gathered automatically via the Google Images API using emotion-related search queries; face regions were then cropped and aligned using the OpenCV Haar cascade detector.
**Download:** `https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip`
(managed by `Fer2013Downloader`; see `config.yaml → data.url`)

The zip extracts to (verified in EDA §2.5):

| File | Columns | Role |
|---|---|---|
| `icml_face_data.csv` | `emotion`, `usage`, `pixels` | **Primary CSV** — all 35,887 rows *with* the `Usage` split column |
| `train.csv` | `emotion`, `pixels` | Flat training rows (28,709) — no split column |
| `test.csv` | `pixels` | Unlabelled test rows (no `emotion` column) — not used |
| `test_with_emotions.csv` | `emotion`, `pixels` | Labelled held-out test (7,178 = PublicTest + PrivateTest) |
| `fer2013.tar.gz` | → `fer2013/fer2013.csv` | Same schema as `icml_face_data.csv` |

This pipeline uses **`icml_face_data.csv`** as the primary file — it is the only top-level CSV carrying the `Usage` column, so it alone can separate Training / PublicTest / PrivateTest (see `config.yaml → data.primary_csv`). `test_with_emotions.csv` is reserved for the single final evaluation pass (`config.yaml → data.test_csv`).

---

### 1.2 Size and splits

`icml_face_data.csv` contains **35,887 labelled samples** distributed across three named splits via the `Usage` column (80 / 10 / 10):

| Split | `Usage` value | Count | Purpose |
|---|---|---|---|
| Training | `"Training"` | 28,709 | Model training |
| Validation | `"PublicTest"` | 3,589 | Hyperparameter tuning / early stopping |
| Test | `"PrivateTest"` | 3,589 | Held-out (not used during training) |

`test_with_emotions.csv` provides **7,178 labelled samples** (the PublicTest + PrivateTest rows combined) as an independent evaluation set.

---

### 1.3 Image format

Every sample is a **48 × 48 pixel grayscale face image** (2 304 intensity values per image, range 0–255).
Images were automatically aligned so the face region fills the frame; alignment quality varies.

The images are **not stored as files** — each row of the CSV encodes one image as a single `pixels` column: a space-separated string of 2 304 unsigned 8-bit integers in **row-major (C) order**:

```
pixels = "128 201 45 ... 99"   # 2304 values: row 0 left→right, then row 1, ...
```

Parsing:
```python
np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
```

This is the entire input signal. There are no bounding boxes, landmarks, or metadata beyond the label and split identifier.

---

### 1.4 Label space

The `emotion` column is an integer in **0–6** mapping to seven mutually exclusive classes:

| Code | Label | Notes |
|---|---|---|
| 0 | Angry | |
| 1 | Disgust | Smallest class (~1.5 % of training data) |
| 2 | Fear | |
| 3 | Happy | Largest class (~25 % of training data) |
| 4 | Sad | |
| 5 | Surprise | |
| 6 | Neutral | |

The mapping is fixed and does not depend on any config value (`config.yaml → model.num_classes: 7`).

---

### 1.5 The pixels → emotion relationship

Each row encodes one supervised learning example:

```
(pixels string)  →  (emotion integer)
   X (input)            y (label)
```

We frame this as **multi-class image classification**: given a 48 × 48 grayscale pixel grid, predict which of the 7 emotion categories the depicted face expresses. A CNN is the natural architecture because convolutions capture local spatial patterns (edges, textures, facial muscle configurations) that are invariant to minor shifts and scales — which is exactly what distinguishes a raised eyebrow (Surprise/Fear) from a pressed lip (Angry/Disgust).

---

### 1.6 Why FER-2013 is a good fit

| Property | Why it matters |
|---|---|
| **Scale** — ~35 k labelled images | Enough for a CNN to learn generalizable features without external data |
| **Face-centered crops** | Input is already in the domain the model must classify; no detection pre-step needed for training |
| **Standard benchmark** | Published results (≈65–75 % human accuracy, ≈63–72 % SotA CNN) give a concrete target and make comparisons meaningful |
| **7-class label space** | Covers the six Ekman basic emotions + Neutral, which is the standard clinical and HCI reference set |
| **Single-file CSV distribution** | No special I/O library required; `pandas.read_csv` + one `np.fromstring` call yields a ready tensor |
| **Grayscale** | Removes color as a confound (skin tone, lighting colour temperature); 1-channel input also reduces model size by 3× vs. RGB |

---

### 1.7 Known limitations and anticipated risks

These limitations motivate several pipeline choices documented in `config.yaml` and set up the EDA in §2:

| Limitation | Risk to the model | Mitigation (see config.yaml) |
|---|---|---|
| **Label noise** | Images were labelled by crowd workers; inter-rater agreement is low (~65 %). Some samples carry the wrong label, adding irreducible noise to the loss | No direct fix; EDA (§2) will surface the worst cases; early stopping prevents overfitting to noisy labels |
| **Class imbalance** | Happy (≈25 %) vs. Disgust (≈1.5 %): a naive always-predict-Happy model already scores ~25 % accuracy (quantified in §2.1) | `cleaning.imbalance_strategy: "class_weight"` in config.yaml passes inverse-frequency weights to `model.fit` |
| **Low resolution** | 48 × 48 px loses fine-grained muscle detail (e.g. subtle eye-corner movements) | Accepted; 48 px is the native resolution — upscaling would not add information |
| **Automatic collection** | Google Images search queries introduce domain bias (stock photos, actor headshots, cartoon-style images) and some non-face images pass the Haar filter | EDA (§2) will inspect per-class image grids for obvious outliers |
| **No landmark or pose metadata** | Head pose variation (profile, tilted) is confounded with emotion signal | `augmentation.horizontal_flip: true` and small rotation range in config.yaml add robustness |
| **Single label per image** | Emotions often co-occur or are ambiguous; forcing one label per image discards uncertainty | Accepted for now; softmax output and `predict_proba` expose the full distribution for downstream use |

---

## 2. Exploratory Data Analysis

This section distils the four EDA notebooks (`notebooks/01`–`04`) into an
evidence-backed problem statement. Every claim is tied to a number or a figure.
The numbered **Problem list** in §2.6 is the contract that §3 (strategies) and
§4 (cleaning) answer — later sections reference problems as *"Problem 2.x"*.

> **Figures** are generated into `results/eda/` by running notebooks `02`–`04`.
> Run them and commit the PNGs for the embeds below to render on GitHub.

---

### 2.1 Class distribution & imbalance

*(notebook `02_eda_class_distribution.ipynb`)*

The 7 emotions are strongly imbalanced across the full 35,887 samples:

| Code | Emotion | Count | % | 
|---|---|---|---|
| 3 | Happy | 8,989 | 25.0 % |
| 6 | Neutral | 6,198 | 17.3 % |
| 4 | Sad | 6,077 | 16.9 % |
| 2 | Fear | 5,121 | 14.3 % |
| 0 | Angry | 4,953 | 13.8 % |
| 5 | Surprise | 4,002 | 11.2 % |
| 1 | Disgust | 547 | 1.5 % |
| | **Total** | **35,887** | **100 %** |

- **Imbalance ratio ≈ 16.4×** overall (Happy / Disgust), and **16.5×** on the
  Training split alone (7,215 / 436). The skew is consistent across all three splits.
- **Naive baseline:** always predicting *Happy* already scores ≈ **25 %** accuracy —
  the floor any real model must clear. This is *why accuracy alone is misleading*
  and why we report **macro-F1** and **per-class recall** (CONTRIBUTING §8).

![Class distribution](results/eda/class_distribution_train.png)

---

### 2.2 Pixel intensity, brightness & contrast

*(notebooks `01_eda.ipynb` §7, `03_eda_image_grids_intensity.ipynb`)*

- **Global intensity:** mean **129.5**, median **134**, std **65** on 0–255.
  Mean < median ⇒ a **slight left-skew** (a tail toward dark pixels) — expected and
  healthy for face crops (hair / shadow / background).
- **Per-image lighting varies widely.** Flagging the Training split (28,709 images):

  | Flag | Threshold | Count | % |
  |---|---|---|---|
  | Dark | brightness < 40 | 93 | 0.32 % |
  | Bright | brightness > 215 | 84 | 0.29 % |
  | Low-contrast | std < 15 | 15 | 0.05 % |
  | Constant | std == 0 | 11 | 0.04 % |

- **Deep-dive triage** (§6b of notebook `03`, via Laplacian-variance *sharpness* and
  Sobel *edge density*) splits these extremes into three distinct kinds:
  - **degenerate** — totally black / constant frames (the 11 `std==0` images) → *drop*;
  - **suspected non-face** — bright, edge-dense crops that are **blurry text /
    watermarks**, not faces (a subset of the 84 bright images) → *note / drop*;
  - **low-quality** — genuine faces that are just too dark or washed-out → *fixable by
    normalization*, not dropping.

![Global intensity histogram](results/eda/intensity_histogram_global.png)
![Brightness & contrast](results/eda/brightness_contrast_hist.png)
![Brightest images, annotated](results/eda/flagged_brightest_annotated.png)

**Why it matters for a CNN:** convolution filters respond to *local intensity
gradients*. If the same expression appears at wildly different brightness levels,
the network wastes capacity learning lighting-invariance instead of expression.
This is the empirical case for `preprocessing.normalization`.

---

### 2.3 Duplicates & leakage

*(notebook `04_eda_duplicates_label_noise.ipynb`)*

- **Exact duplicates:** the full dataset has **34,034 unique** pixel strings out of
  35,887 → **≈ 1,853 duplicate rows** (identical images).
- **Cross-split duplicates = leakage risk.** An image present in *both* Training and
  a test split means the model is evaluated on data it trained on, inflating the
  metric (CONTRIBUTING §8). Notebook `04` §3 fingerprints every image with MD5 and
  reports the exact cross-split count and which split-pairs leak.
- **Conflicting-label duplicates** (same pixels, two different emotions) are an
  automatable slice of label noise — notebook `04` §4 renders them.

![Cross-split leakage examples](results/eda/leakage_examples.png)

---

### 2.4 Label noise & non-faces

*(notebooks `01` §1.7, `04` §4–§5, `03` §6b)*

- **Label noise is intrinsic.** FER-2013 was crowd-labelled from in-the-wild images;
  inter-rater agreement is only ~65 %, so some labels are simply wrong. This **caps
  achievable accuracy** — partly why the project target is *>60 %*, not *>95 %*.
- **Non-face images exist.** The automatic Google-Images + Haar-cascade collection
  passed some crops that are text/watermarks or not faces (see the "suspected
  non-face" triage in §2.2). We *note* these rather than exhaustively hand-cleaning.
- **Stance:** we **accept the residual noise** and fight it with early stopping (don't
  memorise wrong labels) and honest metrics, rather than over-cleaning — which risks
  discarding legitimately hard examples.

![Spot-check per class](results/eda/spotcheck_per_class.png)

---

### 2.5 Missing / malformed rows

*(notebook `01_eda.ipynb`)*

Structural health of `icml_face_data.csv` — all checks pass:

| Check | Result |
|---|---|
| NaN values | **0** |
| Empty / whitespace pixel strings | **0** |
| Pixel count per row = 2304 | ✓ all 35,887 rows |
| Pixel values in [0, 255] | ✓ (uint8) |
| Labels in [0, 6] | ✓ (0 out of range) |
| `Usage` values | only Training / PublicTest / PrivateTest |

Two data-hygiene notes:
- Column headers ship space-padded (`" Usage"`, `" pixels"`) and `usage` is lowercase —
  `Fer2013Fetcher` strips and normalises them.
- `test_with_emotions.csv` carries a phantom `Unnamed: 0` index column (load with
  `index_col=0`).

---

### 2.6 Problem list

Each problem is evidence-backed and maps to a **switchable `config.yaml` option**, so
§3/§4 can measure each fix's effect via ablation.

| # | Problem | Evidence | Config lever |
|---|---|---|---|
| **P2.1** | **Class imbalance** (16.4× overall) | §2.1 — Disgust 1.5 % vs Happy 25 % | `cleaning.imbalance_strategy` (`none \| class_weight \| oversample \| undersample`) + macro-F1 metric |
| **P2.2** | **Cross-split duplicates (leakage)** | §2.3 — MD5 cross-split matches | `cleaning.remove_duplicates` — **must drop** |
| **P2.3** | **Within-split duplicates** | §2.3 — ≈ 1,853 duplicate rows | `cleaning.remove_duplicates` |
| **P2.4** | **Lighting variance** (dark / bright / low-contrast) | §2.2 — 93 dark, 84 bright, 15 low-contrast | `preprocessing.normalization` (`rescale \| standardize \| histogram_eq`) |
| **P2.5** | **Degenerate images** (blank / constant) | §2.2 — 11 `std==0` images | cleaning: drop degenerate rows |
| **P2.6** | **Non-face images** (text / watermark) | §2.2 — "suspected non-face" triage | note / optional drop |
| **P2.7** | **Label noise** (wrong / conflicting labels) | §2.4 — ~65 % rater agreement; conflicting-label dupes | accept residual; early stopping + honest metrics |
| **P2.8** | **Low resolution** (48×48 grayscale) | §1.3 — native format | accepted — upscaling adds no information |

*§3 will propose and justify a strategy for each problem above; §4 documents the
concrete cleaning pipeline that implements the chosen switches.*
