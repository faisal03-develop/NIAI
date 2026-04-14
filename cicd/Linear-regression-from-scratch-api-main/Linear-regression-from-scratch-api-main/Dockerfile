# ====================================================================
# Dockerfile
# ====================================================================
# A Dockerfile is a recipe for building a Docker container image.
#
# WHY DOCKER?
#   "It works on my machine" is the most famous problem in software.
#   Docker packages your app AND its entire environment (Python version,
#   libraries, OS settings) into a single portable "container".
#   The same container runs identically on:
#     - Your laptop
#     - GitHub Actions CI server
#     - AWS / Google Cloud / Render production server
#
# HOW IT WORKS:
#   Each line creates a "layer". Docker caches layers, so rebuilds
#   are fast when only your code changes (not the dependencies).
#
# BUILD:  docker build -t house-price-api .
# RUN:    docker run -p 5000:5000 house-price-api
# TEST:   curl http://localhost:5000/health
# ====================================================================

# ── Base Image ───────────────────────────────────────────────────────
# We start from an official Python image (slim = smaller size)
FROM python:3.10-slim

# ── Metadata ─────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="House Price Prediction API — Linear Regression from scratch"
LABEL version="1.0.0"

# ── Set working directory inside the container ───────────────────────
WORKDIR /app

# ── Install system dependencies ──────────────────────────────────────
# We need these for numpy to compile correctly
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*   # clean up to keep image small

# ── Copy requirements FIRST (Docker layer caching trick) ─────────────
# If requirements.txt hasn't changed, Docker reuses the cached layer
# and skips re-installing packages — saves ~2 minutes on every rebuild.
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ─────────────────────────────────────────────────
# Copy everything AFTER installing packages (so code changes don't
# invalidate the package installation layer)
COPY src/     ./src/
COPY data/    ./data/
COPY app.py   .

# ── Generate data and train model ────────────────────────────────────
# In production you'd typically copy pre-trained models,
# but this shows the full pipeline in one container.
RUN python data/generate_data.py && \
    python src/train.py

# ── Expose port ──────────────────────────────────────────────────────
# This tells Docker that the container listens on port 5000
EXPOSE 5000

# ── Health check ─────────────────────────────────────────────────────
# Docker (and Kubernetes) periodically run this command to check
# if the container is healthy. If it fails, the container is restarted.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ── Non-root user (security best practice) ───────────────────────────
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Start command ────────────────────────────────────────────────────
# gunicorn is a production-grade WSGI server (better than Flask's dev server)
# -w 2          → 2 worker processes
# -b 0.0.0.0    → listen on all interfaces
# app:app       → the 'app' object inside app.py
CMD ["python", "-m", "gunicorn", "--workers", "2", \
     "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
