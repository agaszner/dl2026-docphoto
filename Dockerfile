# Base image already carries Python 3.11, PyTorch 2.4.1 and CUDA 12.4, so we
# never reinstall torch: matching a CUDA build to a driver by hand is the most
# common way to lose an afternoon.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# libgl1 / libglib2.0-0 are OpenCV's runtime dependencies.
# git is here so W&B can record the commit each run was made from.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are installed before the code is mounted, so editing a source
# file does not invalidate the layer and trigger a reinstall.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Run as a normal user, not root. Without this, every file the container writes
# into the mounted project is owned by root on the host. If your host UID is not
# 1000, rebuild with:  docker compose build --build-arg UID=$(id -u)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev 2>/dev/null || true && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash dev 2>/dev/null || true && \
    mkdir -p /workspace && chown -R ${UID}:${GID} /workspace

# PYTHONPATH makes `from src... import ...` work from anywhere in the tree
# without an install step.
ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

WORKDIR /workspace
USER ${UID}:${GID}
