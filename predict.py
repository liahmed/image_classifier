"""Single-image prediction CLI using the trained CIFAR-10 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from config import CLASS_NAMES, MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 class from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    return parser.parse_args()


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load image with PIL/OpenCV, resize, normalize, and batch."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_np = np.array(img)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0

    return np.expand_dims(img_norm, axis=0)


def main() -> None:
    args = parse_args()
    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Run train.py first."
        )

    model = load_model(model_file)
    image_batch = preprocess_image(Path(args.image))

    probs = model.predict(image_batch, verbose=0)[0]
    top3_indices = np.argsort(probs)[-3:][::-1]

    print("Top-3 predictions:")
    for rank, idx in enumerate(top3_indices, start=1):
        confidence = probs[idx] * 100
        print(f"{rank}. {CLASS_NAMES[idx]} — {confidence:.1f}%")


if __name__ == "__main__":
    main()

