"""Training helpers and callbacks."""

from __future__ import annotations

from typing import Any

from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import EPOCHS, MODEL_PATH, SAVED_MODEL_DIR


def get_callbacks():
    """Create standard callback list."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, verbose=1),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(
    model: Any,
    train_generator: Any,
    val_generator: Any,
    epochs: int = EPOCHS,
) -> dict[str, list[float]]:
    """Train model with callbacks and tqdm progress display."""
    callbacks = get_callbacks()
    fit_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[TqdmCallback(verbose=1)] + callbacks,
        verbose=0,
    )
    return fit_history.history

