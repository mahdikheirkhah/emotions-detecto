# Emotions Detector — Issue Backlog (Index)

This folder contains the full, fine-grained issue backlog for the project, broken into **small,
single-step** issues. Each file is one GitHub issue. They are intentionally tiny so each one can be
**generated → read → explained back → understood** before moving on (see `CONTRIBUTING.md` §10).

## How each issue is structured

1. **Description** — the purpose of the issue.
2. **Learning Objective** — the computer-vision / neural-network / ML concept behind it. We learn the
   *idea*, not just the code.
3. **To-Do list for coding** — the concrete classes/methods/files to create.
4. **Code learning (packages & methods)** — the main packages + the specific functions/methods used,
   followed by the *explain-back* loop: after the code exists, **you explain it to me**; where a
   package hides an algorithm, **I explain how it's implemented under the hood**.

Plus two standing notes on every issue: follow **`CONTRIBUTING.md`** and follow the
**Ablation-Driven Architecture** (everything config-driven, switch-case dispatch, stages toggleable).

## How to turn these into real GitHub issues

Write access for the current session is read-only, so these live as files for now. Once write access
is available, create them all with the included script:

```bash
export GITHUB_TOKEN=ghp_...           # a token with repo + issues:write
export GITHUB_REPO=mahdikheirkhah/emotions-detecto
python planning/create_github_issues.py        # add --dry-run to preview
```

The script reads the YAML front-matter (`title`, `labels`) from each `NN-*.md`, strips it, and posts
the body as a new issue, in filename order.

## The backlog (61 issues)

### Phase 0 · Foundations
- `01` Project scaffolding & repository structure
- `02` Poetry + virtual-environment setup
- `03` Declare dependencies, resolve conflicts & export `requirements.txt`
- `04` `config.yaml` schema + loader (the Ablation-Driven backbone)
- `05` Reproducibility seeding + stage-toggle + switch-case dispatch helpers
- `06` Loguru logging setup
- `07` Abstract base classes skeleton (the OOP contracts)

### Phase 1 · Data understanding & documentation
- `08` Download & unzip the FER-2013 dataset
- `09` Load CSV & parse the `pixels` column → image arrays (`Fer2013Fetcher`)
- `10` `data.md` §1 — Raw data documentation
- `11` EDA-a — basic inspection (shape, dtypes, NaN, value ranges, `Usage` split)
- `12` EDA-b — class distribution & imbalance analysis
- `13` EDA-c — image visualization grids + pixel-intensity / brightness / contrast distributions
- `14` EDA-d — duplicate detection & label-noise / non-face inspection
- `15` `data.md` §2 — EDA results

### Phase 2 · Cleaning
- `16` Cleaning strategy discussion → `data.md` §3
- `17` Cleaning impl-a — duplicates & corrupt/constant images (config-driven)
- `18` Cleaning impl-b — class-imbalance handling (class_weight / over / under)
- `19` Clean validation (re-run EDA checks) → `data.md` §4

### Phase 3 · Feature engineering & data prep
- `20` FE-a — normalization strategies (rescale / standardize) (toggleable)
- `21` FE-b — histogram equalization / CLAHE (toggleable)
- `22` FE-c — data augmentation (rotation / flip / zoom / shift) (toggleable)
- `23` `data.md` §5 — Feature-engineering write-up
- `24` Post-FE checks notebook (before/after distributions, augmented samples)
- `25` Data decomposition / PCA (toggleable) + `data.md` §6
- `26` Train / validation / test split (no leakage, stratified)
- `27` Reshape + one-hot encode + build `tf.data` pipeline
- `28` Final sanity-check EDA on model-ready tensors

### Phase 3.5 · Preliminary warm-up (MNIST — required by the subject)
- `29` MNIST — load & handle images in Python
- `30` MNIST — logistic-regression baseline (scikit-learn)
- `31` MNIST — first CNN (Keras) + compare to baseline

### Phase 4 · Model build & training
- `32` CNN architecture discussion (conv / pool / activation / BN / dropout, VGG-style)
- `33` Build-a — reusable conv-block builder
- `34` Build-b — assemble the full CNN (config-driven `ModelBuilder`)
- `35` Compile the model (loss / optimizer / metrics from config)
- `36` Callbacks — EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
- `37` TensorBoard integration + capture `tensorboard.png`
- `38` `train.py` — wire data + model + callbacks and train
- `39` Learning-curves plot (`learning_curves.png`) + `validation_loss_accuracy.py`

### Phase 5 · Evaluation & tuning
- `40` Evaluation metrics module (accuracy, macro-F1, confusion matrix, per-class report)
- `41` `predict.py` — evaluate test set, print `Accuracy on test set: X%`
- `42` Cross-validation concept + optional k-fold support
- `43` Hyperparameter tuning-a — define config grids + tuner setup
- `44` Hyperparameter tuning-b — run search, select best, log results
- `45` Save & document final artifacts (`final_emotion_model.keras` + `_arch.txt`)

### Phase 6 · Optional model work
- `46` Pre-trained CNN / transfer learning (VGG / ResNet) — optional
- `47` Model optimization — quantization (optional, toggleable)
- `48` Model optimization — pruning (optional, toggleable)

### Phase 7 · Face detection & video
- `49` Face detector-a — Haar cascade via cv2 (`HaarFaceDetector`)
- `50` Face detector-b — DNN face detector via cv2 (`DnnFaceDetector`) + dispatch
- `51` Video capture — webcam stream + recorded-video fallback
- `52` Preprocess — detect → crop → 48×48 grayscale → save images (`preprocess.py`, `preprocessing_test`)
- `53` Real-time-a — per-second frame sampling + preprocessing glue
- `54` Real-time-b — load model, predict, print `HH:MM:SSs : Emotion , XX%` (`predict_live_stream.py`)

### Phase 8 · Dashboard, bonus, docs, tests
- `55` Dashboard-a — OpenCV overlay window (face box + emotion label)
- `56` Dashboard-b — Streamlit dashboard (live feed, probabilities, history)
- `57` Bonus-a — pick a >90% "Happy" image + the FGSM adversarial concept
- `58` Bonus-b — gradient-sign attack → flip to "Sad", verify slight change
- `59` `README.md` — run instructions + global approach + results
- `60` Test suite — unit tests (config, data, model utils)
- `61` Functional `preprocessing_test` + CI workflow (GitHub Actions: black + pytest)
