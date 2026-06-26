# Contributing to the Emotions Detector Project

Thank you for contributing to our **real-time facial emotion detection** system!
This project trains a **Convolutional Neural Network (CNN)** with **Keras / TensorFlow** to
classify seven facial emotions — **Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral** —
from the **FER‑2013** dataset, then runs it live on a **webcam video stream** using **OpenCV**
for face detection. The system is meant to be reliable enough for clinical mental‑health
monitoring, so **high accuracy without overfitting** and **full reproducibility** are critical.

To keep the codebase clean, scalable, reproducible, and — above all — **learnable**, every
contributor must follow the guidelines below.

---

## 1. Development Workflow (Branching & CI/CD)

* **Branching Strategy:** Never commit or push directly to `main`. Always create a dedicated
  branch named after the work and (where relevant) the issue, e.g.
  `feature/3-config-system`, `feature/cnn-architecture`, `fix/face-detection-crop`,
  `experiment/vgg-vs-resnet`.
* **One issue per branch / PR:** Keep PRs scoped to a single issue. Each issue is intentionally
  small so it can be read, understood, and explained back before moving on.
* **CI/CD Checks:** Your branch must pass all automated checks (Black formatting, linting, and the
  `pytest` suite) before it can be merged.
* **Merging:** Once the pipeline is green, open a Pull Request to `main` and request a review.

---

## 2. Dependency Management & Formatting

* **Poetry:** We use **Poetry** for dependency resolution and environment management inside a
  dedicated **virtual environment**. Install with `poetry install`. Core dependencies include
  `tensorflow`, `tf-keras`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn`,
  `scikit-learn`, `pyyaml`, `tensorboard`, `streamlit`, and `loguru`.
* **Reproducible export:** A `requirements.txt` is exported from Poetry (`poetry export`) so the
  audit "does the environment contain all libraries and their versions" passes without Poetry.
* **Black Formatter:** We enforce a uniform code style (`line-length = 88`). Before committing, run:

```bash
poetry run black .
```

---

## 3. Ablation‑Driven Architecture (THE core paradigm)

This is the **most important rule in the project**. We build the whole pipeline so that **every
decision is data, not code** — and so that **any single stage can be switched off** to observe its
effect ("ablation").

### 3.1 Single source of truth: `config.yaml`

* **Nothing is hardcoded.** Every *decision*, *strategy*, *hyperparameter*, and *constant*
  (seeds, paths, learning rate, which normalization, which architecture, which face detector …)
  lives in a single `config.yaml`.
* **Every variable lists ALL its options as an inline comment**, so the menu of choices is visible
  at the point of decision:

```yaml
# ===== Global =====
seed: 42                       # any int — fixes numpy / tf / random for reproducibility

# ===== Pipeline stage toggles (ablation switches) =====
stages:
  data_cleaning:        true   # options: true | false
  feature_engineering:  true   # options: true | false  (off = raw‑pixel baseline)
  decomposition:        false  # options: true | false  (PCA; usually off for a CNN)
  augmentation:         true   # options: true | false

# ===== Data cleaning =====
cleaning:
  duplicates: "drop"           # options: "drop" | "keep"
  imbalance:  "class_weight"   # options: "none" | "class_weight" | "oversample" | "undersample"

# ===== Preprocessing / feature engineering =====
preprocessing:
  normalization: "rescale"     # options: "none" | "rescale" | "standardize" | "histogram_eq"

# ===== Model =====
model:
  architecture: "vgg_small"    # options: "simple_cnn" | "vgg_small" | "resnet_mini"
  optimizer:    "adam"         # options: "adam" | "sgd" | "rmsprop"
  learning_rate: 0.001         # tuning grid lives under `tuning:` below
  batch_size:   64
  epochs:       100

# ===== Face detection (video stage) =====
face_detector: "haar"          # options: "haar" | "dnn"
```

### 3.2 Switch‑case dispatch

Code **reads the config and dispatches** to the chosen implementation with an explicit
switch‑case (an `if/elif` ladder or a strategy‑registry `dict`). Each branch maps one config
option to one class/function:

```python
def build_normalizer(cfg: dict) -> BaseImagePreprocessor:
    strategy = cfg["preprocessing"]["normalization"]
    if strategy == "none":
        return IdentityPreprocessor()
    elif strategy == "rescale":
        return RescalePreprocessor()        # pixels / 255.0
    elif strategy == "standardize":
        return StandardizePreprocessor()    # (x - mean) / std
    elif strategy == "histogram_eq":
        return HistogramEqualizer()
    raise ValueError(f"Unknown normalization strategy: {strategy}")
```

### 3.3 Stage toggles (the ablation)

Each pipeline **stage** checks its toggle first and becomes a **no‑op pass‑through when off**:

```python
if cfg["stages"]["feature_engineering"]:
    df = feature_engineer(df, cfg)
else:
    logger.info("Stage OFF: feature_engineering — passing raw data through.")
```

This lets us answer questions like *"what does accuracy do if we remove augmentation / remove
histogram equalization / skip PCA?"* by flipping one boolean.

### 3.4 Rules

* Adding a new strategy = **add an option + comment in `config.yaml`** **and** **add a branch in
  the dispatcher**. Never bury a choice in code.
* **Never delete old options** — keep them so past ablations remain reproducible.
* An unknown/typo'd option must **raise** (fail loud), never silently default.
* Document *why* the current value was chosen (in `data.md`, the architecture `.txt` files, or the
  issue thread) — the config holds the *what*, the docs hold the *why*.

---

## 4. Architecture & Paradigm (OOP)

All code is structured using **Object‑Oriented Programming**. Encapsulate related logic in
well‑defined classes and prefer composition of small, single‑purpose objects. Define an abstract
base per capability and let concrete strategies implement it:

* `BaseDatasetFetcher` → `Fer2013Fetcher`, `MnistFetcher`
* `BaseImagePreprocessor` → `RescalePreprocessor`, `HistogramEqualizer`
* `BaseFaceDetector` → `HaarFaceDetector`, `DnnFaceDetector`
* `BaseModelBuilder` → `SimpleCnnBuilder`, `VggSmallBuilder`
* `BaseEmotionClassifier`

Every contribution must demonstrate the **four OOP principles**:

* **Abstraction:** an `abc.ABC` base with `@abstractmethod`s declaring *what* a component does.
* **Inheritance:** concrete classes inherit the contract and share reusable logic.
* **Encapsulation:** keep internal state/helpers private (`_`); expose a small public surface;
  validate inputs inside the object.
* **Polymorphism:** callers depend on the base type; the dispatcher (§3.2) returns a base type and
  the orchestrator never knows the concrete subtype.

When adding a capability, first ask: *"Which existing abstract base does this extend?"*

---

## 5. Coding Standards & Naming Conventions

* **Naming:** `snake_case` for variables/functions, `PascalCase` for classes. Use consistent shared
  names across the pipeline (`pixels`, `emotion`, `label`, `image`, `gray`, `face`).
* **Logging over Printing:** Use **Loguru** to record pipeline flow, metrics, and errors.

```python
from loguru import logger
logger.info("Loaded 28,709 training images from train.csv.")
logger.warning("Found 1,200 duplicate rows — dropping per cleaning.duplicates='drop'.")
logger.error("Haar cascade file not found. Check face_detector path in config.")
```

  **Exception:** the deliverable scripts `predict.py` and `predict_live_stream.py` must **print**
  their exact required stdout (e.g. `Accuracy on test set: 62%` and
  `11:11:11s : Happy , 73%`) because the audit checks that console output. Everything *else* is
  logged, not printed.

---

## 6. Function & Method Design

* **Single Responsibility:** each function/method does exactly one thing (load → clean → preprocess
  → split → build → train → evaluate → detect face → predict).
* **Type Hinting:** explicitly declare argument and return types everywhere.
* **Docstrings:** every function/method documents (1) goal/behavior, (2) parameters and their types,
  (3) the return type and meaning.

---

## 7. Error Handling

* **Mandatory Try/Except:** wrap the logic of each method in `try`/`except`, log the failure with
  Loguru, and **re‑raise** (never silently swallow).
* **Granular Exceptions:** catch the most specific exception, e.g. `FileNotFoundError` (missing
  dataset / cascade), `KeyError` (missing config key or DataFrame column), `ValueError`
  (bad/empty image, unknown strategy option), `cv2.error` (decode/resize failures).

---

## 8. Model Integrity & Reproducibility

* **No Data Leakage:** split into train/validation/test **before** fitting anything. The model is
  trained **only on the training set**; the test set (`test_with_emotions.csv`) is touched **once**
  for the final accuracy.
* **Avoid Overfitting:** use **early stopping**, dropout, batch norm, and (optionally) augmentation.
  Stop training **before** validation loss diverges from training loss; the `learning_curves.png`
  must visibly show this.
* **TensorBoard is mandatory:** every training run logs to TensorBoard; keep a `tensorboard.png`
  screenshot.
* **Target:** test‑set accuracy **> 60%**.
* **Reproducibility:** set and document seeds for every stochastic process (`np.random.seed`,
  `tf.random.set_seed`, `random.seed`, data shuffles/splits) — all read from `config.yaml`.

---

## 9. Testing

* **Test‑Driven Collaboration:** every new function/class/method gets a matching test under
  `tests/`.
* **Flow Coverage:** cover the main path **and** the `except` blocks (missing file, NaN/empty image,
  unknown config option). Inject fakes/mocks for heavy I/O and the model so the suite runs fast and
  offline.
* **Functional test:** the webcam preprocessing pipeline is validated by `preprocessing_test` —
  a ≥20‑second face video in must yield ≥20 images of 48×48 grayscale pixels, each centered on a
  face.

---

## 10. Learning Workflow (how we work these issues)

This project is also a **learning project**. For every issue:

1. The code is generated against the issue's **To‑Do list**.
2. **You read it and explain it back** — what each piece does and why.
3. We discuss **how the underlying package implements the algorithm** (e.g. how OpenCV's Haar
   cascade actually scans an image, how Keras computes a convolution, how PCA finds components).

Only move to the next issue once the current one is genuinely understood.
