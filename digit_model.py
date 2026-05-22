"""Shared model loading and image preprocessing for digit recognition."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mnist_cnn.h5"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import tensorflow as tf

_model = None


def get_model():
    """Load the TensorFlow model once and reuse it for later predictions."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run python train.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert a drawn or uploaded image into the 28x28 MNIST input format."""
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0

    if arr.max() > 0:
        rows = np.any(arr > 0.1, axis=1)
        cols = np.any(arr > 0.1, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin : rmax + 1, cmin : cmax + 1]

        h, w = arr.shape
        size = max(h, w)
        padded = np.zeros((size, size), dtype="float32")
        row_offset = (size - h) // 2
        col_offset = (size - w) // 2
        padded[row_offset : row_offset + h, col_offset : col_offset + w] = arr

        centered = Image.fromarray((padded * 255).astype("uint8"))
        centered = centered.resize((20, 20), Image.Resampling.LANCZOS)
        final = np.zeros((28, 28), dtype="float32")
        final[4:24, 4:24] = np.array(centered, dtype="float32") / 255.0
        arr = final

    return arr[np.newaxis, ..., np.newaxis]


def predict_digit(img: Image.Image) -> dict:
    """Return the predicted digit, confidence, and all class probabilities."""
    arr = preprocess(img)
    probs = get_model().predict(arr, verbose=0)[0]
    digit = int(np.argmax(probs))
    probabilities = [float(prob) for prob in probs]
    return {
        "digit": digit,
        "confidence": probabilities[digit],
        "probabilities": probabilities,
    }
