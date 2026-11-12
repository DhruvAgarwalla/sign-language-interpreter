"""Model architectures.

Keypoint models (the main approach):
    mlp     - alphabet, input (63,)
    lstm    - words, input (T, F)
    gru     - words, input (T, F)
    cnn1d   - words, temporal convolution, input (T, F)

Pixel baselines (for the comparison experiment):
    smallcnn  - Prototype 4's CNN, fixed, trained from scratch on images
    mobilenet - MobileNetV2 (ImageNet weights) fine-tuned on images
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def _compile(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# Keypoint models
# --------------------------------------------------------------------------- #
def build_mlp(n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input((n_features,), name="landmarks")
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="alphabet_mlp"))


def build_lstm(seq_len: int, n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input((seq_len, n_features), name="sequence")
    x = layers.LayerNormalization()(inp)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="words_lstm"))


def build_gru(seq_len: int, n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input((seq_len, n_features), name="sequence")
    x = layers.LayerNormalization()(inp)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(128)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="words_gru"))


def build_cnn1d(seq_len: int, n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input((seq_len, n_features), name="sequence")
    x = layers.LayerNormalization()(inp)
    for filters in (64, 128):
        x = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="words_cnn1d"))


# --------------------------------------------------------------------------- #
# Pixel baselines
# --------------------------------------------------------------------------- #
def build_smallcnn(img_size: int, n_classes: int) -> keras.Model:
    """Prototype 4's architecture (grayscale input), with a single, correct rescaling."""
    inp = keras.Input((img_size, img_size, 1))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="pixel_smallcnn"))


def build_mobilenet(img_size: int, n_classes: int, trainable_layers: int = 30) -> keras.Model:
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = True
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    inp = keras.Input((img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inp, out, name="pixel_mobilenetv2"), lr=1e-4)


SEQUENCE_MODELS = {"lstm": build_lstm, "gru": build_gru, "cnn1d": build_cnn1d}


def compile_predictor(model: keras.Model):
    """Graph-compiled single-sample predictor (about 20x faster than eager calls on CPU).

    Used for real-time inference and for the latency numbers in the results table.
    """
    spec = tf.TensorSpec((1, *model.input_shape[1:]), tf.float32)

    @tf.function(input_signature=[spec])
    def predict(x):
        return model(x, training=False)

    return lambda x: predict(tf.convert_to_tensor(x, tf.float32)).numpy()
