"""ResNet model definition from the notebook."""

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Multiply,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model

from .config import DEFAULT_INPUT_SHAPE, DEFAULT_NUM_CLASSES


def se_block(input_tensor, reduction_ratio=8):
    """Squeeze-and-Excitation block to add attention."""
    filters = input_tensor.shape[-1]
    se_shape = (1, filters)

    se = GlobalAveragePooling1D()(input_tensor)
    se = Reshape(se_shape)(se)

    se = Dense(filters // reduction_ratio, activation="relu", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", use_bias=False)(se)

    x = Multiply()([input_tensor, se])
    return x


def residual_block(x, filters, kernel_size=3, stride=1, dilation_rate=1):
    """A residual block with SE block and dilated convolution for attention."""
    shortcut = x

    x = Conv1D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = ReLU()(x)

    x = Conv1D(
        filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    if x.shape[-1] != shortcut.shape[-1] or stride != 1:
        shortcut = Conv1D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            kernel_initializer="he_normal",
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)

    x = se_block(x)

    return x


def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, strides=1, padding="same", kernel_initializer=HeNormal())(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    x = ReLU()(x)

    x = residual_block(x, 64, stride=1, dilation_rate=1)
    x = residual_block(x, 128, stride=1, dilation_rate=2)
    x = residual_block(x, 256, stride=1, dilation_rate=4)
    x = residual_block(x, 512, stride=1, dilation_rate=8)
    x = residual_block(x, 1024, stride=1, dilation_rate=16)

    x = GlobalAveragePooling1D()(x)
    x = Dense(1024, activation="relu", kernel_initializer=HeNormal())(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation="relu", kernel_initializer=HeNormal())(x)
    x = Dropout(0.1)(x)

    outputs = Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def build_compiled_resnet(input_shape=DEFAULT_INPUT_SHAPE, num_classes=DEFAULT_NUM_CLASSES):
    model = build_resnet(input_shape, num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model
