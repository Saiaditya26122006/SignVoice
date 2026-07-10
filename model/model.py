"""
SignVoice - Dense classifier for Indian Sign Language recognition.

Input:  (batch, 126) single-frame landmark vector
Output: (batch, 36) softmax over 0-9 and A-Z
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(126,), num_classes=36):
    """Build the Dense classifier over a single-frame landmark vector."""
    inputs = layers.Input(shape=input_shape, name="landmark_vector")

    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SignVoice_CNN_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_model(input_shape=(126,), num_classes=36)
    model.summary()

    dummy_input = np.random.rand(8, 126).astype(np.float32)
    output = model(dummy_input, training=False)
    print("Dummy input shape :", dummy_input.shape)
    print("Model output shape:", output.shape)
