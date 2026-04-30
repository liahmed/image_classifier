"""Model architecture definition for CIFAR-10 classification."""

from tensorflow.keras import layers, models, optimizers

from config import INPUT_SHAPE, LEARNING_RATE, NUM_CLASSES


def build_cnn_model() -> models.Model:
    """Create and compile the custom CNN model."""
    model = models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

