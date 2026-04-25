# MLOps concept:
# Containerization solves the "works on my machine" problem.
# A Docker image bundles the application code, its dependencies, and the runtime
# into a single, reproducible unit. Anyone with Docker can run this API with a
# single command, regardless of their OS or Python version — the same image runs
# on a laptop, a CI server, or a cloud platform.
#
# Key decisions illustrated here:
#   1. python:3.11-slim — a minimal Debian image with Python pre-installed.
#      "slim" strips docs and test suites, cutting image size significantly.
#   2. Dependencies installed before copying source — Docker caches each layer.
#      If only src/ changes, the pip install layer is reused and the rebuild is fast.
#   3. mlruns/ is copied at build time so the registry and champion model are
#      embedded in the image. In production you would mount a shared volume or
#      point MLFLOW_TRACKING_URI at a remote store instead.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached independently of code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and the MLflow registry
COPY src/ ./src/
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
