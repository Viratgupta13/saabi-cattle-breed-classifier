"""Inference wrapper and CLI. Runs on the TFLite model, so TensorFlow is not needed.

    python -m saabi.predict photo1.jpg photo2.jpg
"""
import json
import sys
import threading

import numpy as np
from PIL import Image, ImageOps

from . import config
from .scoring import cosine_to_centroids, softmax_t

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # dev machines that only have full TensorFlow
    from tensorflow.lite import Interpreter


class Classifier:
    def __init__(self, model_path=config.TFLITE_PATH):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]["index"]
        self._outputs = self.interpreter.get_output_details()
        self._lock = threading.Lock()  # a TFLite interpreter is not thread-safe

        self.class_names = json.loads(config.CLASSES_PATH.read_text())
        cal = json.loads(config.CALIBRATION_PATH.read_text()) if config.CALIBRATION_PATH.exists() else {}
        self.temperature = cal.get("temperature", 1.0)
        self.threshold = cal.get("threshold", 0.0)
        self.ood_threshold = cal.get("ood_threshold", -1.0)
        self.centroids = np.load(config.OOD_PATH) if config.OOD_PATH.exists() else None

    def _run(self, batch: np.ndarray):
        """Return (features, probabilities); the two outputs are told apart by width."""
        with self._lock:
            self.interpreter.set_tensor(self._input, batch)
            self.interpreter.invoke()
            outs = {o["shape"][-1]: self.interpreter.get_tensor(o["index"]).copy() for o in self._outputs}
        n = len(self.class_names)
        return next(v for k, v in outs.items() if k != n), outs[n]

    def predict(self, img: Image.Image, top_k: int = 3):
        """Return {'predictions': [{'breed','confidence'}], 'status': ...}.

        Confidences are temperature-calibrated. status is one of:
          'ok'           confident, looks like a supported breed
          'uncertain'    looks like cattle but confidence is below the validation threshold
          'unsupported'  features are far from every known breed (probably not one of the 8)
        """
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
        feats, raw = self._run(np.asarray(img, dtype="float32")[None])
        probs = softmax_t(raw, self.temperature)[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"breed": self.class_names[i], "confidence": float(probs[i])} for i in order]

        status = "ok"
        if self.centroids is not None and \
                cosine_to_centroids(feats, self.centroids)[0] < self.ood_threshold:
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
