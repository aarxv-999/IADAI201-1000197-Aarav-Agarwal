#!/usr/bin/env python3
"""
bin_recommendation.py

Step 4: Integration & Bin Recommendation Logic

This script:
- Loads a trained TensorFlow/Keras image classification model (waste_classifier.h5).
- Accepts a single image path (CLI argument or interactive prompt).
- Preprocesses the image to 224x224, normalizes to [0,1], and adds a batch dimension.
- Predicts the waste category using the model with a softmax output over 3 classes in this order:
  [Biodegradable, Recyclable, Hazardous]
- Maps the predicted class to a color-coded bin recommendation:
    Biodegradable -> Green Bin 🌿
    Recyclable   -> Blue Bin ♻️
    Hazardous    -> Red Bin 🔴
- Prints a friendly summary including the predicted category, confidence, and recommended bin.
- Handles common errors (missing files, unreadable images) gracefully.

Usage examples:
  python scripts/bin_recommendation.py --image /path/to/image.jpg --model waste_classifier.h5
  python scripts/bin_recommendation.py --image ./sample.png
  python scripts/bin_recommendation.py   # will prompt for image path
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# Class order MUST follow the model's softmax order:
# [Biodegradable, Recyclable, Hazardous]
CLASS_NAMES = ["Biodegradable", "Recyclable", "Hazardous"]

# Mapping from class label to color-coded bin with emoji
BIN_MAPPING = {
    "Biodegradable": "Green Bin 🌿",
    "Recyclable": "Blue Bin ♻️",
    "Hazardous": "Red Bin 🔴",
}

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model and image path."""
    parser = argparse.ArgumentParser(
        description="Recommend the proper disposal bin from a waste image using a trained model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="waste_classifier.h5",
        help="Path to the trained Keras model (.h5). Default: waste_classifier.h5",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the input image (jpg/png/etc.). If omitted, you will be prompted.",
    )
    return parser.parse_args()

def load_keras_model(model_path: Path) -> tf.keras.Model:
    """
    Load a Keras model from the provided .h5 path.
    Raises a RuntimeError with a friendly message if loading fails.
    """
    if not model_path.exists():
        raise RuntimeError(f"❌ Model file not found: {model_path}")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model from '{model_path}': {e}")

def preprocess_image(image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image with PIL, apply EXIF orientation correction, convert to RGB,
    resize to target_size, scale to [0,1], and add a batch dimension.

    Returns:
        A numpy array of shape (1, H, W, 3) with dtype float32.
    Raises:
        RuntimeError with a friendly message on failure.
    """
    if not image_path.exists():
        raise RuntimeError(f"❌ Image file not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            # Handle EXIF orientation and ensure RGB
            img = ImageOps.exif_transpose(img).convert("RGB")
            # Resize to model's expected input size
            img = img.resize(target_size, resample=Image.Resampling.BILINEAR)
    except Exception as e:
        raise RuntimeError(f"❌ Unable to open or process image '{image_path}': {e}")

    # Convert to numpy array and normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_category(model: tf.keras.Model, input_tensor: np.ndarray) -> Tuple[str, float]:
    """
    Run model prediction on the input tensor and return:
    - predicted class label (string)
    - confidence score (float in [0,1])

    Assumes model outputs softmax probabilities in the order of CLASS_NAMES.
    """
    # Perform prediction; expect shape (1, 3) with softmax probabilities
    probs = model.predict(input_tensor, verbose=0)
    if probs.ndim != 2 or probs.shape[0] != 1 or probs.shape[1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"❌ Unexpected model output shape {probs.shape}; expected (1, {len(CLASS_NAMES)})."
        )
    probs = probs[0]  # shape: (3,)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    return label, confidence

def main() -> int:
    args = parse_args()

    # If image path is not provided as an argument, prompt the user interactively.
    if not args.image:
        try:
            args.image = input("Please enter the image file path: ").strip()
        except EOFError:
            print("❌ No image path provided and no input available.", file=sys.stderr)
            return 1

    model_path = Path(args.model)
    image_path = Path(args.image)

    # Load model with clear error messaging
    try:
        model = load_keras_model(model_path)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    # Preprocess image safely
    try:
        tensor = preprocess_image(image_path, target_size=(224, 224))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    # Run prediction and map to bin recommendation
    try:
        label, confidence = predict_category(model, tensor)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Failed to run prediction: {e}", file=sys.stderr)
        return 1

    # Determine the recommended bin
    recommended_bin = BIN_MAPPING.get(label, "Unknown Bin")

    # Friendly, color-coded summary with emojis
    print("")
    print(f"✅ Waste Category: {label}")
    print(f"🗑️ Recommended Bin: {recommended_bin}")
    print(f"🔍 Confidence: {confidence * 100:.2f}%")
    print("")
    print("✅ Bin recommendation complete.")

    return 0

if __name__ == "__main__":
    # Exit code 0 on success, non-zero on error
    sys.exit(main())
