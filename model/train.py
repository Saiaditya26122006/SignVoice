"""
SignVoice - training pipeline smoke test with dummy data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import build_model


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
CURVES_PATH = os.path.join(MODEL_DIR, "training_curves.png")

NUM_SAMPLES = 500
NUM_CLASSES = 36
FEATURES = 126
EPOCHS = 5
BATCH_SIZE = 32
SEED = 42


def make_dummy_data():
    rng = np.random.default_rng(SEED)
    X = rng.random((NUM_SAMPLES, FEATURES)).astype(np.float32)
    labels = rng.integers(0, NUM_CLASSES, size=NUM_SAMPLES)
    y = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    return X, y


def plot_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, acc, label="train")
    axes[0].plot(epochs_range, val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs_range, loss, label="train")
    axes[1].plot(epochs_range, val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(CURVES_PATH)
    plt.close(fig)


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_model(input_shape=(FEATURES,), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    X, y = make_dummy_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    plot_curves(history)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
