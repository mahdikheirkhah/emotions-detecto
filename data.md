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

The zip extracts to:

| File | Role |
|---|---|
| `train.csv` | Primary split file — Training + PublicTest + PrivateTest rows, distinguished by the `Usage` column |
| `test.csv` | Unlabelled test rows (no `emotion` column) |
| `test_with_emotions.csv` | Labelled held-out test set used for final evaluation |
| `icml_face_data.csv` | Original ICML release; not used directly in this pipeline |
| `fer2013.tar.gz` | Nested archive containing the same data in an alternative layout |

This pipeline uses `train.csv` for training/validation and `test_with_emotions.csv` for the single final evaluation pass (see `config.yaml → data.primary_csv` and `data.test_csv`).

---

### 1.2 Size and splits

`train.csv` contains **35,887 labelled samples** distributed across three named splits via the `Usage` column:

| Split | `Usage` value | Approx. count | Purpose |
|---|---|---|---|
| Training | `"Training"` | 28,709 | Model training |
| Validation | `"PublicTest"` | 3,589 | Hyperparameter tuning / early stopping |
| Test | `"PrivateTest"` | 3,589 | Held-out (not used during training) |

`test_with_emotions.csv` provides **3,589 labelled samples** as the independent evaluation set.

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
| **Class imbalance** | Happy (≈25 %) vs. Disgust (≈1.5 %): a naive model can reach ~46 % accuracy by predicting Happy/Neutral only | `cleaning.imbalance_strategy: "class_weight"` in config.yaml passes inverse-frequency weights to `model.fit` |
| **Low resolution** | 48 × 48 px loses fine-grained muscle detail (e.g. subtle eye-corner movements) | Accepted; 48 px is the native resolution — upscaling would not add information |
| **Automatic collection** | Google Images search queries introduce domain bias (stock photos, actor headshots, cartoon-style images) and some non-face images pass the Haar filter | EDA (§2) will inspect per-class image grids for obvious outliers |
| **No landmark or pose metadata** | Head pose variation (profile, tilted) is confounded with emotion signal | `augmentation.horizontal_flip: true` and small rotation range in config.yaml add robustness |
| **Single label per image** | Emotions often co-occur or are ambiguous; forcing one label per image discards uncertainty | Accepted for now; softmax output and `predict_proba` expose the full distribution for downstream use |

---

*§2 will cover exploratory data analysis: class distribution, pixel-intensity statistics, sample image grids, and brightness/contrast distributions.*
