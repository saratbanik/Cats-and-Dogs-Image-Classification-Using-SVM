
from pathlib import Path
import argparse
import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from image_utils import find_training_images, feature_from_bytes

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "svm_cat_dog.joblib"


def select_images(records, max_images):
    """Select a balanced, reproducible sample of images."""
    by_class = {0: [], 1: []}

    for path, label in records:
        by_class[label].append(path)

    for label, paths in by_class.items():
        print(f"{'Cat' if label == 0 else 'Dog'} images found: {len(paths)}")

    if not by_class[0] or not by_class[1]:
        raise ValueError(
            "Both classes are required. Check train1/train2 and "
            "ensure filenames include cat or dog labels."
        )

    rng = random.Random(42)

    if max_images > 0:
        per_class = max(1, max_images // 2)

        for label in (0, 1):
            paths = by_class[label]
            if len(paths) > per_class:
                by_class[label] = rng.sample(paths, per_class)

    selected = (
        [(path, 0) for path in by_class[0]]
        + [(path, 1) for path in by_class[1]]
    )
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Train a HOG + Linear SVM cat/dog classifier."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=12000,
        help="Maximum total training images; use 0 for all images.",
    )
    args = parser.parse_args()

    if args.max_images < 0:
        parser.error("--max-images must be zero or greater.")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    records = find_training_images(RAW_DIR)
    if not records:
        raise FileNotFoundError(
            "No labeled images found. Expected cat/dog images inside "
            "data/raw/train1/ and data/raw/train2/."
        )

    selected = select_images(records, args.max_images)

    features = []
    labels = []
    unreadable = 0

    for index, (path, label) in enumerate(selected, start=1):
        try:
            features.append(feature_from_bytes(path.read_bytes()))
            labels.append(label)
        except Exception as exc:
            unreadable += 1
            if unreadable <= 5:
                print(f"Skipping unreadable image: {path.name} ({exc})")

        if index % 500 == 0:
            print(f"Processed {index}/{len(selected)} images")

    if len(labels) < 20 or len(set(labels)) != 2:
        raise RuntimeError(
            "Too few usable images. Check your dataset and image filenames."
        )

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    print(f"\nUsable images: {len(y)}")
    print(f"Unreadable images skipped: {unreadable}")
    print(f"HOG feature dimensions: {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Evaluate on images held aside from model fitting.
    validation_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=8000,
    )
    print("\nTraining validation SVM...")
    validation_model.fit(X_train, y_train)
    predictions = validation_model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    report = classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Cat", "Dog"],
        output_dict=True,
        zero_division=0,
    )

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Validation F1 score (Dog): {f1:.4f}")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Cat", "Dog"],
            zero_division=0,
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Cat", "Dog"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Cats vs Dogs — Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=160)
    plt.close()

    # Fit the final probability-calibrated model on all usable images.
    # Probability estimates are useful for the competition submission file.
    print("Training final calibrated SVM on all usable images...")
    final_model = CalibratedClassifierCV(
        estimator=LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=8000,
        ),
        method="sigmoid",
        cv=3,
    )
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "labels": {0: "Cat", 1: "Dog"},
        "image_size": [64, 64],
        "feature_method": "HOG",
        "classifier": "Calibrated LinearSVC",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "training_images": int(len(y)),
        "validation_images": int(len(y_val)),
        "unreadable_images": int(unreadable),
        "accuracy": float(accuracy),
        "f1_score_dog": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_method": "HOG",
        "classifier": "Calibrated LinearSVC",
        "image_size": "64x64 grayscale",
        "random_state": 42,
    }

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved trained model: {MODEL_PATH}")
    print(f"Saved metrics: {ARTIFACTS / 'metrics.json'}")
    print(f"Saved chart: {ARTIFACTS / 'confusion_matrix.png'}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()