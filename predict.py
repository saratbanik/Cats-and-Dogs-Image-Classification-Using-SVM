
from pathlib import Path
import argparse

import joblib
import numpy as np

from image_utils import feature_from_bytes

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "svm_cat_dog.joblib"


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an image contains a cat or dog."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a JPG, JPEG or PNG image.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model not found. Run python train_model.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    features = feature_from_bytes(image_path.read_bytes()).reshape(1, -1)

    model = bundle["model"]
    predicted_label = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    dog_index = list(model.classes_).index(1)
    cat_index = list(model.classes_).index(0)

    print(f"Image: {image_path.name}")
    print(f"Prediction: {bundle['labels'][predicted_label]}")
    print(f"Cat probability: {probabilities[cat_index]:.1%}")
    print(f"Dog probability: {probabilities[dog_index]:.1%}")


if __name__ == "__main__":
    main()