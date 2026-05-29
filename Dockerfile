# syntax=docker/dockerfile:1.7
# =============================================================================
# Dockerfile — FastAPI inference backend for HuggingFace Spaces (Docker SDK).
# -----------------------------------------------------------------------------
# Target:    HF Spaces, hardware = cpu-basic (2 vCPU / 16 GB RAM).
# Port:      7860 (HF Spaces convention).
# User:      UID 1000 named "user" (HF Spaces requirement).
# Workdir:   /home/user/app (HF Spaces convention).
# Worker:    uvicorn single worker — keeps the TF model loaded once in RAM.
# =============================================================================

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HOME=/home/user/.cache/huggingface

# libgomp1 is required by tensorflow-cpu (OpenMP runtime).
# curl is used by HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with UID 1000 named "user".
RUN useradd --create-home --uid 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# --- Dependency layer (cached across code changes) ---------------------------
COPY --chown=user:user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Application source ------------------------------------------------------
# Copy only what the runtime needs. Build context is pruned by .dockerignore.
COPY --chown=user:user pyproject.toml README.md ./
COPY --chown=user:user src/ ./src/
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user configs/ ./configs/
COPY --chown=user:user models/ ./models/

# Install the local captioning package without re-resolving deps.
RUN pip install --user --no-cache-dir --no-deps -e .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:7860/healthz || exit 1

CMD ["uvicorn", "app.main:app", \
     "--app-dir", "backend", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
