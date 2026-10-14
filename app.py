"""SAABI web demo and JSON API.

    python app.py            # http://127.0.0.1:5001 (override with PORT=...)
"""
import io
import json
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from saabi import config
from saabi.breeds import BREED_INFO

MAX_UPLOAD_MB = 10


def create_app(classifier=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    if classifier is None:
        from saabi.predict import Classifier
        classifier = Classifier()

    metrics = {}
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        metrics = {"accuracy": m["accuracy"], "top3": m["top3_accuracy"], "test_images": m["test_images"]}

    @app.get("/")
    def home():
        breeds = [{"name": b, "label": b.replace("_", " "), **BREED_INFO.get(b, {})}
                  for b in classifier.class_names]
        return render_template("index.html", breeds=breeds, metrics=metrics,
                               github_url=os.environ.get("GITHUB_URL"))

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", classes=classifier.class_names)

    @app.post("/api/predict")
    def predict():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify(error="No image uploaded (form field 'image')."), 400
        if not file.filename.lower().endswith(tuple(config.IMAGE_EXTENSIONS)):
            return jsonify(error="Unsupported file type. Use JPG, PNG or BMP."), 400
        try:
            img = Image.open(io.BytesIO(file.read()))
            img.load()
        except (UnidentifiedImageError, OSError):
            return jsonify(error="Could not read that file as an image."), 400

        result = classifier.predict(img, top_k=3)
        best = result["predictions"][0]["breed"]
        return jsonify(**result, info=BREED_INFO.get(best))

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"File too large (max {MAX_UPLOAD_MB} MB)."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)))
