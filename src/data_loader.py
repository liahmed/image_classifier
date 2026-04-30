"""Data loading and augmentation utilities for CIFAR-10."""

from __future__ import annotations

import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from config import (
    CLASS_NAMES,
    DATA_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)


def load_cifar10_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10, normalize images, and one-hot encode labels."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(DATA_DIR.resolve())

    # Dataset cache is redirected to project-local data/ through KERAS_HOME.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test


def build_train_datagen() -> ImageDataGenerator:
    """Return data generator with requested augmentation policy."""
    return ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=VALIDATION_SPLIT,
    )


def build_eval_datagen() -> ImageDataGenerator:
    """Return non-augmented generator for validation/testing."""
    return ImageDataGenerator(validation_split=VALIDATION_SPLIT)


def create_train_val_generators(
    x_train: np.ndarray, y_train: np.ndarray, batch_size: int
):
    """Create train and validation iterators from the training split."""
    aug_datagen = build_train_datagen()
    val_datagen = build_eval_datagen()

    train_generator = aug_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        subset="training",
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_generator = val_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        subset="validation",
        shuffle=False,
        seed=RANDOM_SEED,
    )
    return train_generator, val_generator


def get_class_names() -> list[str]:
    """Return ordered class names for labels and plotting."""
    return CLASS_NAMES

