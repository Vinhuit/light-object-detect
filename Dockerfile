# Stage 1: Build stage
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install compilation tools needed for Cython & InsightFace build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY Pipfile Pipfile.lock ./

RUN python -m pip install --upgrade pip setuptools wheel cython \
    && python -m pip install \
        fastapi \
        uvicorn \
        python-multipart \
        pydantic \
        pydantic-settings \
        pillow \
        exceptiongroup \
        numpy \
        onnxruntime \
        opencv-python-headless \
        scipy \
        shapely \
        insightface \
    # Uninstall build-only packages from virtualenv to save space
    && python -m pip uninstall -y cython

# Optional: install tensorflow for tflite backend support
ARG INSTALL_TENSORFLOW=0
RUN if [ "$INSTALL_TENSORFLOW" = "1" ]; then python -m pip install tensorflow; fi

# Optional: install moondream (torch + transformers) for VLM backend
ARG INSTALL_MOONDREAM=0
RUN if [ "$INSTALL_MOONDREAM" = "1" ]; then \
    python -m pip install \
        "transformers>=4.51.1,<5.0" \
        "torch>=2.7.0" \
        "accelerate>=1.10.0"; \
    fi

COPY . .

# Optionally bake default models in build time
ARG DOWNLOAD_DEFAULT_MODEL=0
RUN if [ "$DOWNLOAD_DEFAULT_MODEL" = "1" ]; then python scripts/download_model.py; fi


# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Minimal runtime dependencies for headless OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment containing all pre-compiled packages
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
