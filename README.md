# Emotions Detecto

> Real-time facial emotion detection — a CNN trained on FER-2013, OpenCV face
> detection, and a live webcam pipeline, all driven by a single `config.yaml`.

Point a webcam (or a recorded clip) at a face and the system detects the face, crops it to
the FER-2013 format, classifies the emotion with a trained CNN, and reports the result at
least once per second — on the terminal, an OpenCV overlay, or a Streamlit dashboard.

---

## Overview

The **global approach** is a straight line from pixels to a live prediction:

1. **Data** — FER-2013 (48×48 grayscale faces, 7 emotions) is downloaded, cleaned
   (deduplicated across splits to remove leakage, blank/constant frames dropped), and
   normalized. See [`data.md`](data.md) for the full EDA and every cleaning decision.
2. **CNN** — a from-scratch `vgg_small` convolutional network is trained on the cleaned
   data with class-weighting for imbalance, augmentation, early stopping, and TensorBoard
   logging. Evaluation reports **accuracy + macro-F1 + a confusion matrix** (an imbalanced
   7-class problem where accuracy alone lies).
3. **Face detection** — at inference a Haar cascade (or an SSD/ResNet DNN) locates the
   face in each frame; the crop is squared, centered, and resized to `48×48` grayscale so
   the live input matches the training format exactly (closing the train/inference gap).
4. **Real-time inference** — frames are sampled ~1/sec, classified, and reported live, with
   an automatic **recorded-video fallback** when no webcam is present.

Every one of those steps is a set of **`config.yaml` options** — you change one value and
re-run to measure what it contributes. That is the project's core paradigm (below).

### The 7 emotions

Class index → label (the FER-2013 `emotion` column order, used everywhere):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |

---

## Dataset

**FER-2013** — 48×48 grayscale face crops labelled with one of the 7 emotions above,
split into Training / PublicTest / PrivateTest. It is fetched automatically on the first
training run from the URL in `config.yaml` (`data.url`) and extracted into `data/`.

The dataset is imbalanced (Happy is ~7× Disgust) and carries some label noise — both are
handled deliberately, not ignored. The full analysis (class distribution, brightness /
contrast, duplicates & leakage, label noise, the numbered problem list, and the cleaning
strategy chosen for each) lives in **[`data.md`](data.md)**.

![FER-2013 class distribution](results/eda/class_distribution_train.png)

---

## Project structure

```
emotions-detecto/
├── config.yaml                 # THE single source of truth — every decision is a value here
├── data.md                     # dataset provenance, EDA, cleaning decisions
├── CONTRIBUTING.md             # conventions + the Ablation-Driven Architecture (§3)
├── src/emotion_detector/       # all the LOGIC (tested, imported by the scripts)
│   ├── data/                   # fetch, clean, split, preprocess, imbalance, tf.data pipeline
│   ├── models/                 # builders, callbacks, tuning, evaluation, optimize, classifier
│   ├── video/                  # capture, face detectors (haar/dnn), preprocess, stream, overlay
│   ├── dashboard.py            # Streamlit dashboard logic (analysis + history)
│   ├── adversarial.py          # FGSM/BIM attack + the adversarial concept note
│   └── utils/                  # config, logging (Loguru), seeding, dispatch, stage toggles
├── scripts/                    # thin entrypoints — they only orchestrate src/
├── tests/                      # pytest suite (fast, offline, fakes for heavy I/O + the model)
└── results/                    # models, learning curves, EDA figures, TensorBoard logs
```

**Design rule:** logic lives in `src/` and is unit-tested; `scripts/*.py` are thin
conductors that read `config.yaml` and call `src/`. Only `predict.py` and
`predict_live_stream.py` are allowed to `print` (their exact stdout is graded); everything
else logs through Loguru.

---

## Setup

Requires **Python 3.11+**. Use Poetry (preferred) or pip:

```bash
# Option A — Poetry (uses pyproject.toml)
poetry install
poetry shell

# Option B — pip (pinned export)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **Trained models (Git LFS) — required before `predict.py`, the live stream, or the
  dashboard.** The `.keras` models are stored with Git LFS, so a fresh clone only pulls
  small text *pointers*. Fetch the real files once:
  ```bash
  git lfs install
  git lfs pull            # downloads final_emotion_model.keras (+ the transfer model)
  ```
  Sanity check: `results/model/final_emotion_model.keras` should be **~18 MB**, not a few
  hundred bytes. (Its sha256 is recorded in `final_emotion_model_manifest.json`; a pointer
  stub or a half-trained file makes the model predict ~1/7 for every class.)
- **Dataset:** downloaded automatically on the first `train.py` / `predict.py` run.
- **DNN face model (optional):** only needed for `face_detector.backend: dnn`:
  ```bash
  python scripts/download_face_model.py     # fetches the SSD/ResNet caffemodel into models/
  ```
- **GPU (optional):** the code runs on CPU; a CUDA GPU just makes training faster. The
  pinned `tensorflow` wheel includes CUDA support.

---

## Running the pipeline

All scripts are run from the repo root and read `config.yaml`. Reproduce each deliverable:

### Train

```bash
python scripts/train.py                        # trains the from-scratch model per config.yaml
python scripts/train.py config_transfer.yaml   # or the transfer-learning (VGG16) variant
```
Writes `results/model/final_emotion_model.keras`, `history.json`, and the architecture
summary; logs scalars + the graph to TensorBoard.

### Predict (evaluate on the held-out test set)

```bash
python scripts/predict.py                        # from-scratch model → confusion_matrix.png
python scripts/predict.py config_transfer.yaml   # transfer model → pre_trained_confusion_matrix.png
```
Reproduces the exact training preprocessing on the test split and prints the graded line:
```
Accuracy on test set: XX%
```
(plus macro-F1 and a confusion matrix under `results/model/` — a separate file per model).

### Preprocess (the functional `preprocessing_test`)

```bash
python scripts/preprocess.py [path/to/face_video.mp4]
```
Samples ~1 frame/sec from a ≥20-second face video and saves ≥20 detected-and-cropped
`48×48` grayscale faces to `results/preprocessing_test/`.

### Predict — live stream (the headline deliverable)

```bash
python scripts/predict_live_stream.py               # print only
python scripts/predict_live_stream.py --display     # + OpenCV window (face box + label)
python scripts/predict_live_stream.py clip.mp4      # explicit recorded source
```
Prints, at least once per second:
```
Reading video stream ...
Preprocessing ...
00:00:00s : Happy , 73%
```
No webcam? It falls back to `video.fallback_path` automatically.

### Streamlit dashboard

```bash
streamlit run scripts/dashboard.py
```
Pick a **Source** in the sidebar — everything renders inside the browser (boxed face +
current emotion + a bar chart over all 7 probabilities + a rolling emotion timeline):

- **Upload a video** — process an `.mp4/.mov/.avi` frame by frame (most reliable; no camera permission).
- **Webcam · snapshot** — classify one photo from the *browser* camera (the reliable path on macOS).
- **Webcam · live** — continuous OpenCV webcam stream.

### Other tools

```bash
python scripts/validation_loss_accuracy.py                      # → learning_curves.png (scratch)
python scripts/validation_loss_accuracy.py config_transfer.yaml # → pre_trained_learning_curves.png (transfer)
python scripts/tune.py                        # hyperparameter search, persists the winner to config
python scripts/optimize.py                    # optional pruning / TFLite quantization (with accuracy check)
python scripts/finalize_model.py              # writes the final-model manifest + snapshot
python scripts/adversarial.py                 # bonus: pick a Happy>90% face, FGSM/BIM-flip it to Sad
tensorboard --logdir logs/tensorboard         # training curves + graph
```

---

## Ablation-Driven Architecture

The core paradigm (see **[`CONTRIBUTING.md` §3](CONTRIBUTING.md)**): **`config.yaml` is the
single source of truth — no decision is ever hardcoded in source.** Every value carries an
`# options:` comment listing its valid choices, and each choice is wired through a
switch-case `dispatch(...)` that fails loudly on an unknown option. To run an experiment you
change **one** value and re-run.

**Stage toggles** turn whole steps on/off so you can measure what each contributes:

```yaml
stages:
  cleaning: true          # false → keep duplicates/leakage and watch test accuracy collapse
  preprocessing: true     # false → raw-pixel baseline (no normalization)
  augmentation: true      # false → observe faster overfitting on the learning curves
  tuning: false           # true  → run the hyperparameter search before the final train
```

Other one-line ablations: `model.architecture` (`simple_cnn | vgg_small | resnet_mini |
transfer_vgg16 | transfer_resnet50`), `cleaning.imbalance_strategy` (`class_weight |
oversample | undersample | none`), `preprocessing.normalization` (`rescale | standardize |
histogram_eq | clahe`), `face_detector.backend` (`haar | dnn`), `model.loss`
(`categorical_crossentropy | focal_loss`). Flip one, re-run `train.py` + `predict.py`, and
compare — the same pipeline serves training, evaluation, and live/recorded inference
unchanged.

---

## Results

The graded model is the from-scratch **`vgg_small`** CNN (seed 42, early-stopped):

| metric | value |
|--------|-------|
| architecture | `vgg_small` (3 VGG-style conv blocks + dense head) |
| best epoch | 59 / 69 trained (early stopping) |
| best validation accuracy | **63%** (peak 64%) |
| best validation loss | 0.9789 |

(from `results/model/final_emotion_model_manifest.json`.) The exact **held-out test
accuracy** is printed by `python scripts/predict.py` (`Accuracy on test set: XX%`) — a
single, final measurement on data never used for tuning. For context, human agreement on
FER-2013 is ~65±5%, so a ~63% from-scratch CNN is competitive.

- **Learning curves:** `python scripts/validation_loss_accuracy.py` →
  `results/model/learning_curves.png` (train vs val loss/accuracy — the gap shows how much
  augmentation + early stopping tamed overfitting).
- **TensorBoard:** `tensorboard --logdir logs/tensorboard` for the scalars and the model graph.
- **Architecture:** the full `model.summary()` + iteration history is in
  [`results/model/final_emotion_model_arch.txt`](results/model/final_emotion_model_arch.txt);
  the exact config that produced the model is snapshotted alongside it.
- **Confusion matrix + per-class report:** written by `predict.py` under `results/model/`
  (Disgust/Fear are the hardest classes — expected, they are the rarest).
- **EDA figures:** every dataset figure referenced in [`data.md`](data.md) is under
  `results/eda/`.

---

## Audit — how to verify every requirement

> **Before anything:** `git lfs install && git lfs pull` (fetch the real `.keras`
> models), then `poetry install` (or `pip install -r requirements.txt`). The FER-2013
> dataset auto-downloads on the first `predict.py` run.

### Preliminary

| Requirement | Where / how to check |
|---|---|
| Structure matches the subject | `scripts/{train,predict,predict_live_stream,preprocess,validation_loss_accuracy}.py`, `results/model/`, `results/preprocessing_test/`, `data/` — see [Project structure](#project-structure) |
| README explains how to run + the global approach | This file — [Overview](#overview) + [Running the pipeline](#running-the-pipeline) |
| All libraries + versions present | `requirements.txt` (fully pinned) + `pyproject.toml` |
| Text files explain the architectures | `results/model/final_emotion_model_arch.txt` (scratch) · `results/model/pre_trained_model_architecture.txt` (transfer) |

### CNN emotion classifier

| Requirement | Where / how to check |
|---|---|
| Trained only on the training set | Test split is `Usage != Training` (PublicTest+PrivateTest), seeded; the normalizer is fit on the **train** split only (`scripts/train.py` / `scripts/predict.py`) |
| Test accuracy > 60% | `python scripts/predict.py` → `Accuracy on test set: 6X%` |
| Learning curves prove no overfitting | `results/model/learning_curves.png` (regen: `python scripts/validation_loss_accuracy.py`) — val loss flattens while train keeps dropping |
| Training stopped early enough | Same figure: early-stop line at **epoch 59/69** (min val_loss, `restore_best_weights`) |
| TensorBoard screenshot | `results/model/tensorboard.png` (live: `tensorboard --logdir logs/tensorboard`) |
| Doc explains architecture choice + prior iterations | `results/model/final_emotion_model_arch.txt` — §2 (design) + §6 (iteration/ablation log) |
| `python ./scripts/predict.py` runs, returns > 60% | `Accuracy on test set: 6X%` |

### Face detection on the video stream

| Requirement | Where / how to check |
|---|---|
| ≥20s video → ≥20 preprocessed images in a separate folder | `python scripts/preprocess.py your_face_video.mp4` → `results/preprocessing_test/image0.png …` |
| All images contain a face | A frame is saved only after a cv2 detection (no-face frames are skipped) |
| Reshaped + centered on the face | Square, centered crop → resize (`src/emotion_detector/video/preprocess.py`) |
| Face detector imported via cv2 | `cv2.CascadeClassifier` (Haar) / `cv2.dnn` (SSD) |
| Converted to 48×48 grayscale | `cv2.cvtColor(..., COLOR_BGR2GRAY)` + `cv2.resize(..., (48, 48))` |
| Webcam issue → recorded fallback | `VideoSource` auto-falls back to `video.fallback_path` |
| `python ./scripts/predict_live_stream.py` runs + right format | `Reading video stream …` / `Preprocessing …` / `HH:MM:SSs : Happy , 73%` |

### Bonus — adversarial attack (hack the CNN)

| Requirement | Where / how to check |
|---|---|
| Happy image > 90% chosen | `results/adversarial/source_image.png` + `source_probabilities.json` (run: `python scripts/adversarial.py`) |
| Pixels modified so the net predicts Sad | `results/adversarial/attack_comparison.png` (original · perturbation · adversarial) |
| Slightly changed & still recognizable | The perturbation is bounded by `adversarial.epsilon` (L∞ ≤ 0.05) — imperceptible; the two faces look identical in the comparison figure |

---

## Testing

```bash
pytest -q          # the full suite (fast: heavy I/O and the model are faked)
black .            # formatting (line length 88)
```

Tests are offline and deterministic; anything needing TensorFlow, a real video codec, or a
webcam is either faked or skipped so the suite stays fast.

---

## Documentation

- **[`data.md`](data.md)** — dataset provenance, EDA, and the cleaning decision behind every option.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — conventions, the Ablation-Driven Architecture (§3),
  reproducibility (§8), and testing (§9).
- **`config.yaml`** — the single source of truth; read the `# options:` comments to see every knob.
