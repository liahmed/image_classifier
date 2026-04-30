"""Training entry-point for CIFAR-10 custom CNN."""

import random

import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, EPOCHS, RANDOM_SEED
from src.data_loader import create_train_val_generators, load_cifar10_data
from src.evaluator import evaluate_and_report, plot_training_curves
from src.model import build_cnn_model
from src.trainer import train_model


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    set_seed()

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10_data()
    train_gen, val_gen = create_train_val_generators(x_train, y_train, BATCH_SIZE)

    print("Building model...")
    model = build_cnn_model()
    model.summary()

    print("Training model...")
    history = train_model(model, train_gen, val_gen, epochs=EPOCHS)

    print("Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_training_curves(history)
    report = evaluate_and_report(model, x_test, y_test)
    print("\nClassification report:\n")
    print(report)

    print("\nTraining complete. Artifacts saved in output/ and saved_model/.")


if __name__ == "__main__":
    main()

