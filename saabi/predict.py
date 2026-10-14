"""Inference wrapper and CLI.

    python -m saabi.predict photo1.jpg photo2.jpg
"""
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from . import config, data
from .calibrate import cosine_to_centroids, feature_extractor, softmax_t


class Classifier:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.extractor = feature_extractor(self.model)
        self.class_names = data.load_class_names()
        cal = json.loads(config.CALIBRATION_PATH.read_text()) if config.CALIBRATION_PATH.exists() else {}
        self.temperature = cal.get("temperature", 1.0)
        self.threshold = cal.get("threshold", 0.0)
        self.ood_threshold = cal.get("ood_threshold", -1.0)
        self.centroids = np.load(config.OOD_PATH) if config.OOD_PATH.exists() else None

    def predict(self, img: Image.Image, top_k: int = 3):
        """Return {'predictions': [{'breed','confidence'}], 'status': ...}.

        Confidences are temperature-calibrated. status is one of:
          'ok'           confident, looks like a supported breed
          'uncertain'    looks like cattle but confidence is below the validation threshold
          'unsupported'  features are far from every known breed (probably not one of the 8)
        """
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
        batch = np.asarray(img, dtype="float32")[None]
        feats, raw = self.extractor(batch, training=False)
        probs = softmax_t(raw.numpy(), self.temperature)[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"breed": self.class_names[i], "confidence": float(probs[i])} for i in order]

        status = "ok"
        if self.centroids is not None and \
                cosine_to_centroids(feats.numpy(), self.centroids)[0] < self.ood_threshold:
            status = "unsupported"
        elif top[0]["confidence"] < self.threshold:
            status = "uncertain"
        return {"predictions": top, "status": status}


def main(argv):
    clf = Classifier()
    for path in argv:
        with Image.open(path) as img:
            result = clf.predict(img)
        print(f"{path}   [{result['status']}]")
        for r in result["predictions"]:
            print(f"  {r['breed']:<20} {r['confidence'] * 100:5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
