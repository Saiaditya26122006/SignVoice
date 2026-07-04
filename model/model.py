"""
SignVoice - 1D-CNN with Attention model for Indian Sign Language recognition.

Input:  (batch, 30 frames, 126 landmarks)
Output: (batch, 100 classes) softmax
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class AttentionLayer(layers.Layer):
    """Simple additive (Bahdanau-style) attention over the time axis."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name="att_W",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_b",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="att_u",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_logits = tf.tensordot(score, self.u, axes=1)  # (batch, time, 1)
        attention_weights = tf.nn.softmax(attention_logits, axis=1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, features)
        return context


def build_model(input_shape=(30, 126), num_classes=100):
    """Build the 1D-CNN + Attention classifier."""
    inputs = layers.Input(shape=input_shape, name="landmark_sequence")

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = AttentionLayer(name="attention")(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SignVoice_CNN_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_model(input_shape=(30, 126), num_classes=100)
    model.summary()

    dummy_input = np.random.rand(8, 30, 126).astype(np.float32)
    output = model(dummy_input, training=False)
    print("Dummy input shape :", dummy_input.shape)
    print("Model output shape:", output.shape)
