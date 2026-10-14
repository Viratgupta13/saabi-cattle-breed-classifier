FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY saabi ./saabi
COPY templates ./templates
COPY models ./models
COPY app.py .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"

# One worker: TensorFlow holds the model in memory, so scale with threads, not processes.
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} "app:create_app()"
