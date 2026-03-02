# # Multi-stage Dockerfile for NLLB Translation Service

# # Stage 1: Base image with Python and dependencies
# FROM python:3.10-slim as base

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Stage 2: Dependencies
# FROM base as dependencies

# # Copy requirements
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# # Stage 3: Application
# FROM dependencies as application

# # Copy application code
# COPY ./app ./app
# COPY ./training ./training

# # Create necessary directories
# RUN mkdir -p /app/models/cache /app/models/custom-nllb /app/data

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8000/api/v1/health || exit 1

# # Run with gunicorn
# CMD ["gunicorn", "app.main:app", \
#      "--workers", "4", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--bind", "0.0.0.0:8000", \
#      "--timeout", "120", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-"]

# # Stage 4: GPU-enabled image (optional)
# FROM application as gpu

# # Install CUDA-compatible PyTorch (if needed)
# RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# # Use same command as application
# CMD ["gunicorn", "app.main:app", \
#      "--workers", "2", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--bind", "0.0.0.0:8000", \
#      "--timeout", "120", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-"]


# Multi-stage Dockerfile for NLLB Translation Service

FROM python:3.10-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base as dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM dependencies as application

COPY ./app ./app

RUN mkdir -p /app/models/cache /app/models/custom-nllb/latest

# ⬇️ Download latest model at build time (if MODEL_URI is set)
ARG MODEL_URI
ENV MODEL_URI=${MODEL_URI}

RUN if [ -n "$MODEL_URI" ]; then \
      echo "Downloading model from $MODEL_URI" && \
      curl -L $MODEL_URI -o /app/models/model.tar.gz && \
      tar -xzf /app/models/model.tar.gz -C /app/models/custom-nllb/latest && \
      rm /app/models/model.tar.gz ; \
    else \
      echo "MODEL_URI not set, skipping model download" ; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["gunicorn", "app.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]

