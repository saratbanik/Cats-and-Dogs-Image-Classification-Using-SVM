
from pathlib import Path
import zipfile

import joblib
import pandas as pd
import numpy as np

from image_utils import feature_from_bytes

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

TEST_ZIP = RAW_DIR / "test.zip"
SAMPLE_CSV = RAW_DIR / "sample_submission.csv"
MODEL_PATH = ARTIFACTS / "svm_cat_dog.joblib"


def normalize_id(value):
    """Normalize IDs such as 12, 12.0 and '12' to the same key."""
    value = str(value).strip()

    try:
        return str(int(float(value)))
    except ValueError:
        return value


def main():
    for path in (TEST_ZIP, SAMPLE_CSV, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    sample = pd.read_csv(SAMPLE_CSV)

    if "id" not in sample.columns or "label" not in sample.columns:
        raise ValueError(
            "sample_submission.csv must contain 'id' and 'label' columns."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    dog_index = list(model.classes_).index(1)

    # Process in batches to limit memory consumption.
    predictions = np.empty(len(sample), dtype=np.float64)
    missing_ids = []

    with zipfile.ZipFile(TEST_ZIP, "r") as archive:
        image_members = {}

        for info in archive.infolist():
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                continue

            key = normalize_id(Path(info.filename).stem)
            image_members[key] = info.filename

        for start in range(0, len(sample), 128):
            end = min(start + 128, len(sample))
            batch_features = []
            batch_positions = []

            for position in range(start, end):
                image_id = normalize_id(sample.iloc[position]["id"])
                member = image_members.get(image_id)

                if member is None:
                    missing_ids.append(image_id)
                    continue

                try:
                    image_bytes = archive.read(member)
                    batch_features.append(
                        feature_from_bytes(image_bytes)
                    )
                    batch_positions.append(position)
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot process test image {member}: {exc}"
                    ) from exc

            if batch_features:
                probabilities = model.predict_proba(
                    np.asarray(batch_features, dtype=np.float32)
                )
                predictions[batch_positions] = probabilities[:, dog_index]

            print(f"Processed test rows through {end}/{len(sample)}")

    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} sample IDs were not found in test.zip. "
            f"Example IDs: {missing_ids[:10]}"
        )

    result = sample.copy()
    result["label"] = predictions

    output_path = ARTIFACTS / "submission.csv"
    result.to_csv(output_path, index=False)

    print(f"\nCreated: {output_path}")
    print(f"Rows written: {len(result)}")
    print("The label column contains the predicted probability of Dog.")


if __name__ == "__main__":
    main()