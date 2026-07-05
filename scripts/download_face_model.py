"""Download OpenCV's SSD/ResNet face model into ``models/`` (for the DNN backend).

The caffemodel (~10 MB) isn't committed; fetch it so ``face_detector.backend: dnn``
works:

    python scripts/download_face_model.py

Idempotent — skips files already present. Respects the environment's HTTPS proxy.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# path -> URL. The prototxt (text) is small; the caffemodel is the ~10 MB weights.
_FILES = {
    _ROOT
    / "models"
    / "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    _ROOT
    / "models"
    / "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    ),
}
_MIN_BYTES = 1024  # anything smaller is a redirect stub, not the real file


def main() -> None:
    for path, url in _FILES.items():
        if path.exists() and path.stat().st_size > _MIN_BYTES:
            print(f"exists: {path.relative_to(_ROOT)} ({path.stat().st_size:,} bytes)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}\n        -> {path.relative_to(_ROOT)}")
        urllib.request.urlretrieve(url, path)
        size = path.stat().st_size
        if size <= _MIN_BYTES:
            print(
                f"ERROR: {path.name} is only {size} bytes — download failed.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"        {size:,} bytes")
    print("Face DNN model ready under models/.")


if __name__ == "__main__":
    main()
