FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY saabi ./saabi
COPY templates ./templates
COPY app.py .
# Only what inference needs (the full Keras model stays out of the image)
COPY models/saabi.tflite models/class_names.json models/calibration.json \
     models/ood_centroids.npy models/metrics.json ./models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"

# One worker keeps memory low; threads handle concurrency (the classifier is lock-guarded).
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} "app:create_app()"
