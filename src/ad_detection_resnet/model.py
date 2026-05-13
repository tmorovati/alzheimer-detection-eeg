"""Model definitions for AD EEG time-series classification."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Activation,
    AveragePooling1D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
    SeparableConv1D,
)
from tensorflow.keras.models import Model

from .config import DEFAULT_INPUT_SHAPE, DEFAULT_NUM_CLASSES

SUPPORTED_MODEL_NAMES = ("eegnet", "inceptiontime")


def normalize_model_name(model_name):
    normalized = str(model_name).lower().replace("-", "").replace("_", "")

    if normalized in {"eegnet", "eeg"}:
        return "eegnet"
    if normalized in {"inceptiontime", "inception"}:
        return "inceptiontime"

    supported = ", ".join(SUPPORTED_MODEL_NAMES)
    raise ValueError(f"Unknown model '{model_name}'. Expected one of: {supported}.")


def _binary_output(x, num_classes):
    return Dense(num_classes, activation="sigmoid")(x)


def build_eegnet(input_shape=DEFAULT_INPUT_SHAPE, num_classes=DEFAULT_NUM_CLASSES):
    """Build an EEGNet-style 1D model for exported single-channel EEG segments."""
    inputs = Input(shape=input_shape)

    x = Conv1D(8, kernel_size=64, padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(16, kernel_size=1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling1D(pool_size=4)(x)
    x = Dropout(0.5)(x)

    x = SeparableConv1D(16, kernel_size=16, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling1D(pool_size=8)(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling1D()(x)
    outputs = _binary_output(x, num_classes)

    return Model(inputs=inputs, outputs=outputs, name="EEGNet")


def inception_module(x, filters=32, kernel_sizes=(40, 20, 10), bottleneck_size=32):
    bottleneck = Conv1D(bottleneck_size, kernel_size=1, padding="same", use_bias=False)(x)

    branches = [
        Conv1D(filters, kernel_size=kernel_size, padding="same", use_bias=False)(bottleneck)
        for kernel_size in kernel_sizes
    ]

    pooled = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    pooled = Conv1D(filters, kernel_size=1, padding="same", use_bias=False)(pooled)

    x = Concatenate()(branches + [pooled])
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def shortcut_layer(shortcut, x):
    shortcut = Conv1D(int(x.shape[-1]), kernel_size=1, padding="same", use_bias=False)(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x


def build_inceptiontime(
    input_shape=DEFAULT_INPUT_SHAPE,
    num_classes=DEFAULT_NUM_CLASSES,
    depth=6,
    filters=32,
):
    inputs = Input(shape=input_shape)
    x = inputs
    residual = inputs

    for module_idx in range(depth):
        x = inception_module(x, filters=filters)

        if module_idx % 3 == 2:
            x = shortcut_layer(residual, x)
            residual = x

    x = GlobalAveragePooling1D()(x)
    outputs = _binary_output(x, num_classes)

    return Model(inputs=inputs, outputs=outputs, name="InceptionTime")


def build_compiled_model(
    model_name="eegnet",
    input_shape=DEFAULT_INPUT_SHAPE,
    num_classes=DEFAULT_NUM_CLASSES,
    learning_rate=0.001,
):
    model_name = normalize_model_name(model_name)

    if model_name == "eegnet":
        model = build_eegnet(input_shape=input_shape, num_classes=num_classes)
    elif model_name == "inceptiontime":
        model = build_inceptiontime(input_shape=input_shape, num_classes=num_classes)
    else:
        raise AssertionError(f"Unsupported normalized model name: {model_name}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model
