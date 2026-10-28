
from pathlib import Path
import json

import joblib
import streamlit as st

from image_utils import feature_from_bytes

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "svm_cat_dog.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"

st.set_page_config(
    page_title="Cats vs Dogs Studio",
    page_icon="🐾",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 5% 0%, #172554, transparent 35%),
          #080e1b;
    }
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid #344766;
        background: linear-gradient(120deg, #172554, #12343d);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: clamp(2rem, 5vw, 3.2rem);
    }
    .hero p {
        color: #cbd5e1;
        line-height: 1.7;
    }
    div[data-testid="stMetric"] {
        background: #111d31;
        border: 1px solid #2b3d59;
        padding: 1rem;
        border-radius: 16px;
    }
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #9aacc5;
        font-size: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.is_file():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


st.markdown(
    """
    <div class="hero">
      <p>IMAGE CLASSIFICATION · SUPPORT VECTOR MACHINE</p>
      <h1>Cats vs Dogs Studio</h1>
      <p>
        Upload an image to classify it as a cat or dog using
        Histogram of Oriented Gradients and a trained Linear SVM.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle = load_model()

if bundle is None:
    st.error(
        "The trained model is missing. Open the project folder in VS Code "
        "and run `python train_model.py` first."
    )
    st.stop()

metrics = load_metrics()

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Validation accuracy",
        f"{metrics['accuracy']:.1%}",
    )
    col2.metric(
        "Validation F1 score",
        f"{metrics['f1_score_dog']:.3f}",
    )
    col3.metric(
        "Training images",
        f"{metrics['training_images']:,}",
    )

st.divider()
st.subheader("Classify an image")

uploaded = st.file_uploader(
    "Choose an image of a cat or dog",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image containing one main animal.",
)

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    features = feature_from_bytes(image_bytes).reshape(1, -1)

    model = bundle["model"]
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    class_indices = list(model.classes_)
    cat_probability = float(
        probabilities[class_indices.index(0)]
    )
    dog_probability = float(
        probabilities[class_indices.index(1)]
    )
    predicted_name = bundle["labels"][prediction]
    confidence = max(cat_probability, dog_probability)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.image(
            image_bytes,
            caption=uploaded.name,
            use_container_width=True,
        )

    with right:
        st.markdown("### Prediction result")

        if predicted_name == "Cat":
            st.success("Predicted class: Cat")
        else:
            st.success("Predicted class: Dog")

        st.metric("Probability for predicted class", f"{confidence:.1%}")
        st.write(f"**Cat:** {cat_probability:.1%}")
        st.progress(cat_probability)
        st.write(f"**Dog:** {dog_probability:.1%}")
        st.progress(dog_probability)

        st.caption(
            "These are model probability estimates, not guarantees. "
            "Results depend on the training data and input image quality."
        )

st.divider()
st.markdown(
    """
    <div class="footer">
      CATS VS DOGS · HOG FEATURE EXTRACTION · LINEAR SVM
      <br><br>
      Developed by
      <a href="https://www.linkedin.com/in/saratbanik"
         target="_blank"
         rel="noopener noreferrer"
         style="color:#6ee7dc;font-weight:700;text-decoration:none;">
        SARAT BANIK
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)