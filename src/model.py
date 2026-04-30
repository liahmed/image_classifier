"""Model architecture definition for CIFAR-10 classification."""

from tensorflow.keras import layers, losses, models, optimizers, regularizers

from config import (
    INPUT_SHAPE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    NUM_CLASSES,
    WEIGHT_DECAY,
)


def build_cnn_model() -> models.Model:
    """Create and compile a tuned custom CNN model."""
    reg = regularizers.l2(WEIGHT_DECAY)
    model = models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(256, activation="relu", kernel_regularizer=reg),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu", kernel_regularizer=reg),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model

