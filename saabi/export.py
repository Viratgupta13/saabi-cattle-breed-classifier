"""Export the trained Keras model to TFLite and verify it matches.

TFLite lets the web server run without TensorFlow (about a fifth of the memory).

    python -m saabi.export
"""
import tempfile

import numpy as np
import tensorflow as tf

from . import config, data
from .model import BASE_NAME
from .scoring import cosine_to_centroids, softmax_t


def inference_model(model: tf.keras.Model) -> tf.keras.Model:
    """Re-wire the trained layers without the training-only augmentation ops.
    Returns (pooled features, class probabilities)."""
    L = tf.keras.layers
    inputs = tf.keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    x = model.get_layer("imagenet_scaling")(inputs)
    x = model.get_layer(BASE_NAME)(x, training=False)
    feats = next(l for l in model.layers if isinstance(l, L.GlobalAveragePooling2D))(x)
    drop = next(l for l in model.layers if isinstance(l, L.Dropout))
    probs = model.get_layer("breed")(drop(feats, training=False))
    return tf.keras.Model(inputs, [feats, probs])


def main():
    model = tf.keras.models.load_model(config.MODEL_PATH)
    extractor = inference_model(model)
    # Keras 3 models must go through a SavedModel to reach the TFLite converter (TF 2.16).
    with tempfile.TemporaryDirectory() as tmp:
        extractor.export(tmp, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        config.TFLITE_PATH.write_bytes(converter.convert())
    keras_mb = config.MODEL_PATH.stat().st_size / 1e6
    tflite_mb = config.TFLITE_PATH.stat().st_size / 1e6
    print(f"keras {keras_mb:.1f} MB -> tflite {tflite_mb:.1f} MB")

    # Parity check on the held-out test split
    paths, labels, _, _ = data.discover()
    _, _, (p_te, y_te) = data.split(paths, labels)
    x = np.concatenate([b.numpy() for b, _ in data.make_dataset(p_te, y_te, training=False)])

    k_feats, k_probs = extractor.predict(x, batch_size=32, verbose=0)
    interp = tf.lite.Interpreter(model_path=str(config.TFLITE_PATH))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    t_feats, t_probs = [], []
    for img in x:
        interp.set_tensor(inp["index"], img[None])
        interp.invoke()
        vals = {tuple(o["shape"])[-1]: interp.get_tensor(o["index"])[0] for o in outs}
        t_feats.append(vals[k_feats.shape[-1]]); t_probs.append(vals[k_probs.shape[-1]])
    t_feats, t_probs = np.array(t_feats), np.array(t_probs)

    agree = float((k_probs.argmax(1) == t_probs.argmax(1)).mean())
    print(f"max |prob diff| = {np.abs(k_probs - t_probs).max():.2e}")
    print(f"prediction agreement keras vs tflite = {agree:.4f}")
    print(f"tflite test accuracy = {(t_probs.argmax(1) == y_te).mean():.4f}")


if __name__ == "__main__":
    main()
