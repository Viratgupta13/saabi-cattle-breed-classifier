import tensorflow as tf
from tensorflow.keras import layers

from . import config

BASE_NAME = "mobilenetv2_backbone"


def build_model(num_classes: int, dropout: float = 0.3) -> tf.keras.Model:
    """MobileNetV2 classifier that accepts raw RGB pixels in 0..255.

    Augmentation (training only) and ImageNet scaling live inside the graph, so
    training, evaluation and serving share one preprocessing path.
    """
    size = config.IMG_SIZE
    inputs = layers.Input((size, size, 3), name="rgb_image")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.06)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomContrast(0.15)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="imagenet_scaling")(x)

    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(size, size, 3), name=BASE_NAME
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="breed")(x)
    return tf.keras.Model(inputs, outputs, name="saabi")


def unfreeze_top(model: tf.keras.Model, n_layers: int = 40) -> None:
    """Fine-tune the last n backbone layers; BatchNorm stays frozen for stability."""
    base = model.get_layer(BASE_NAME)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
