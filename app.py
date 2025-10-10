# app.py
# ♻️ SmartWasteAI – Waste Classification System
# An AI-powered smart waste bin recommendation tool.
#
# This Streamlit app:
# - Loads a trained TensorFlow/Keras model (waste_classifier.h5) once using st.cache_resource
# - Accepts JPG/JPEG/PNG uploads
# - Preprocesses to (224, 224), normalizes by 255.0, and adds batch dimension
# - Predicts one of: [Biodegradable, Recyclable, Hazardous]
# - Displays predicted category, confidence, and a recommended bin (Green/Blue/Red) with emoji
# - Shows a class probability breakdown with a simple bar chart
# - Handles errors gracefully and uses only relative paths
#
# Place waste_classifier.h5 in the same directory as this file before running.
# To run locally: `streamlit run app.py`

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model


# ------------------------------
# App/page configuration
# ------------------------------
st.set_page_config(
    page_title="SmartWasteAI",
    page_icon="♻️",
    layout="centered",
)

# Sidebar information
with st.sidebar:
    st.header("Project Info")
    st.markdown(
        "- AI in Action – Waste Classification Project\n"
        "- Powered by TensorFlow + Streamlit\n"
        "- Upload an image to get started"
    )


# ------------------------------
# Constants and mappings
# ------------------------------
# Model class order must match training: [Biodegradable, Recyclable, Hazardous]
CLASS_NAMES = ["Biodegradable", "Recyclable", "Hazardous"]

# Bin mapping for recommendations
BIN_MAPPING: Dict[str, Tuple[str, str]] = {
    # category: (Recommended Bin Label, Emoji)
    "Biodegradable": ("Green Bin", "🌿"),
    "Recyclable": ("Blue Bin", "♻️"),
    "Hazardous": ("Red Bin", "🔴"),
}

# Color mapping for probability bar chart
BAR_COLORS = {
    "Biodegradable": "#00A650",  # green
    "Recyclable": "#1677FF",     # blue
    "Hazardous": "#D40000",      # red
}


# ------------------------------
# Cached model loader
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_cached_model(model_path: Path):
    """
    Load the Keras model once and cache it across reruns.
    Uses compile=False to avoid loading training heads not needed for inference.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            "Place waste_classifier.h5 in the same directory as app.py."
        )
    # compile=False as we only need forward-pass inference
    model = load_model(model_path.as_posix(), compile=False)
    return model


# ------------------------------
# Image preprocessing
# ------------------------------
def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Convert PIL image to a normalized tensor suitable for the model:
    - Ensure RGB
    - Resize to (224, 224)
    - Convert to float32 and normalize by 255.0
    - Expand to shape (1, 224, 224, 3)
    """
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, preprocessed: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Run prediction and return:
    - predicted class name
    - confidence (0..1)
    - probabilities array of shape (3,)
    """
    preds = model.predict(preprocessed, verbose=0)
    # Ensure probabilities via softmax regardless of model final activation
    probs = tf.nn.softmax(preds[0]).numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


# ------------------------------
# UI
# ------------------------------
st.title("♻️ SmartWasteAI – Waste Classification System")
st.caption("An AI-powered smart waste bin recommendation tool.")

# File uploader: accept JPEG/PNG
uploaded_file = st.file_uploader(
    "Upload a waste image (JPG/JPEG/PNG):",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

# If no file uploaded yet, prompt the user
if uploaded_file is None:
    st.info("Upload an image to get started.")
else:
    # Display preview
    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("Unable to read image. Please upload a valid JPG, JPEG, or PNG file.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the image: {e}")
        st.stop()

    st.subheader("Image Preview")
    st.image(image, use_container_width=True)

    # Predict immediately after upload (no extra button needed)
    with st.spinner("Loading model and analyzing image..."):
        try:
            model_path = Path(__file__).with_name("waste_classifier_v2.h5")
            model = load_cached_model(model_path)
        except FileNotFoundError as fnf:
            st.error(str(fnf))
            st.stop()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        try:
            tensor = preprocess_image(image, target_size=(224, 224))
            predicted_label, confidence, probs = predict(model, tensor)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Map to recommended bin
    bin_label, bin_emoji = BIN_MAPPING.get(predicted_label, ("Unknown Bin", "🗑️"))

    # Display results
    st.markdown(f"**✅ Waste Category:** {predicted_label}")
    st.markdown(f"**🗑️ Recommended Bin:** {bin_label} {bin_emoji}")
    st.markdown(f"**🔍 Confidence:** {confidence * 100:.2f}%")

    # Probability breakdown (bar chart)
    st.subheader("Class Probabilities")
    try:
        fig, ax = plt.subplots(figsize=(5, 3))
        classes = CLASS_NAMES
        values = (probs * 100.0).tolist()
        colors = [BAR_COLORS[c] for c in classes]
        ax.bar(classes, values, color=colors)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        # Fallback textual list if plotting fails
        st.write({c: f"{p*100:.2f}%" for c, p in zip(CLASS_NAMES, probs)})

    # Final success message
    st.success("✅ Prediction complete! Bin recommendation generated successfully.")


# ------------------------------
# 🧭 Deployment Instructions:
# 1. Push app.py, waste_classifier.h5, requirements.txt to a GitHub repo.
# 2. Go to https://streamlit.io/cloud
# 3. Sign in with GitHub → “New App” → select your repo → choose app.py
# 4. Wait for build → your app will launch with a public link.
# ------------------------------
