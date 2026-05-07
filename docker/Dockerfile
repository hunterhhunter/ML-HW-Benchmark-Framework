# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS frontend-build

WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
ENV VITE_API_BASE=/api
RUN npm run build


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    YOLO_AUTOINSTALL=false \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        python3.12 \
        python3.12-venv \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel uv

WORKDIR /app

COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY framework/requirements.txt framework/constraints.txt /tmp/

RUN UV_NO_CACHE=1 uv pip install --python /opt/venv/bin/python -r /tmp/backend-requirements.txt \
    && UV_NO_CACHE=1 uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt -c /tmp/constraints.txt \
    && rm -rf /root/.cache/pip /root/.cache/uv

COPY backend ./backend
COPY framework ./framework
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

RUN mkdir -p /app/framework/models /app/framework/datasets /app/framework/results /root/.cache/huggingface

ENV FRAMEWORK_DIR=/app/framework \
    FRAMEWORK_PYTHON=/opt/venv/bin/python \
    FRONTEND_DIST_DIR=/app/frontend/dist \
    CORS_ORIGINS=http://localhost:5173,http://localhost:8000

WORKDIR /app/backend
EXPOSE 8000

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
