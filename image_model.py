"""Shared model loading and preprocessing for broad image classification."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)


_model = None


def get_model():
    global _model
    if _model is None:
        _model = MobileNetV2(weights="imagenet")
    return _model


def preprocess(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr[None, ...]
    return preprocess_input(arr)


def classify_image(img: Image.Image) -> dict:
    arr = preprocess(img)
    probs = get_model().predict(arr, verbose=0)[0]
    decoded = decode_predictions(probs[tf.newaxis, ...], top=10)[0]
    probabilities = [
        {"label": label.replace("_", " "), "probability": float(probability)}
        for _, label, probability in decoded
    ]
    winner = probabilities[0]
    return {
        "label": winner["label"],
        "confidence": winner["probability"],
        "probabilities": probabilities,
    }
