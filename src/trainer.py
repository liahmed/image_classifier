"""Training helpers and callbacks."""

from __future__ import annotations

from typing import Any

from tqdm.auto import trange
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import EPOCHS, MODEL_PATH, SAVED_MODEL_DIR


def get_callbacks():
    """Create standard callback list."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
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
    """
    Train model epoch-by-epoch with tqdm progress and return a Keras-like history dict.
    """
    callbacks = get_callbacks()
    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in trange(epochs, desc="Training epochs", unit="epoch"):
        fit_history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epoch + 1,
            initial_epoch=epoch,
            callbacks=callbacks,
            verbose=0,
        ).history

        for key in history:
            if key in fit_history and fit_history[key]:
                history[key].append(float(fit_history[key][-1]))

        if model.stop_training:
            break

    return history

