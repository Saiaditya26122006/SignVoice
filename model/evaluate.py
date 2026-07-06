"""
SignVoice - evaluate the trained custom model on a test set.

Also reserves a section for pretrained-model results so the two can be
compared side-by-side later.
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from model import AttentionLayer


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

NUM_TEST_SAMPLES = 100
NUM_CLASSES = 100
TIME_STEPS = 30
FEATURES = 126
SEED = 123


def make_dummy_test_data():
    rng = np.random.default_rng(SEED)
    X = rng.random((NUM_TEST_SAMPLES, TIME_STEPS, FEATURES)).astype(np.float32)
    labels = rng.integers(0, NUM_CLASSES, size=NUM_TEST_SAMPLES)
    y = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    return X, y, labels


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_metrics_table(title, metrics):
    print()
    print(title)
    print("+---------------+-----------+")
    print("| Metric        | Score     |")
    print("+---------------+-----------+")
    print(f"| Accuracy      | {metrics['accuracy']:.4f}    |")
    print(f"| Precision     | {metrics['precision']:.4f}    |")
    print(f"| Recall        | {metrics['recall']:.4f}    |")
    print(f"| F1 Score      | {metrics['f1']:.4f}    |")
    print("+---------------+-----------+")


def main():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. "
            "Run model/train.py first."
        )

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
    )

    X_test, y_test, y_true = make_dummy_test_data()

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    custom_metrics = compute_metrics(y_true, y_pred)
    print_metrics_table("Custom Model (1D-CNN + Attention)", custom_metrics)

    # ------------------------------------------------------------------
    # PLACEHOLDER: pretrained model comparison
    # ------------------------------------------------------------------
    # Load a pretrained ISL model here, run it on the same X_test / y_true,
    # compute its metrics, then print a side-by-side comparison table.
    #
    # pretrained_model = tf.keras.models.load_model("path/to/pretrained.h5")
    # pretrained_probs = pretrained_model.predict(X_test, verbose=0)
    # pretrained_pred = np.argmax(pretrained_probs, axis=1)
    # pretrained_metrics = compute_metrics(y_true, pretrained_pred)
    # print_metrics_table("Pretrained Model", pretrained_metrics)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
