from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog

IMAGE_SIZE = (64, 64)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

HOG_OPTIONS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True,
}


def label_from_path(path):
    """Return Cat=0, Dog=1, or None for an unknown filename."""
    path = Path(path)
    stem = path.stem.lower()

    # Kaggle competition filenames: cat.0.jpg, dog.123.jpg, etc.
    match = re.match(r"^(cat|dog)(?:[._-]\d+)?$", stem)
    if match:
        return 0 if match.group(1) == "cat" else 1

    # Also support datasets arranged in Cat/ and Dog/ folders.
    for part in reversed(path.parts[:-1]):
        folder = part.lower()
        if folder in {"cat", "cats"}:
            return 0
        if folder in {"dog", "dogs"}:
            return 1

    return None


def find_training_images(raw_dir):
    """Find labeled images recursively in train1 and train2."""
    raw_dir = Path(raw_dir)
    paths = []

    for folder_name in ("train1", "train2"):
        folder = raw_dir / folder_name

        if not folder.is_dir():
            continue

        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                label = label_from_path(path)
                if label is not None:
                    paths.append((path, label))

    return sorted(paths, key=lambda item: str(item[0]))


def feature_from_image(image):
    """Convert a PIL image into a HOG feature vector."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    pixels = np.asarray(image, dtype=np.float32) / 255.0

    features = hog(pixels, **HOG_OPTIONS)
    return features.astype(np.float32)


def feature_from_bytes(image_bytes):
    """Convert image bytes into a HOG feature vector."""
    with Image.open(BytesIO(image_bytes)) as image:
        return feature_from_image(image)