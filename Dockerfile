# RunPod-friendly image. NVIDIA's CUDA 12.4 runtime + Python 3.11 + Procgen
# build deps. Build:  docker build -t pcpg .
# Run on RunPod by selecting a template that points to this image, or push
# to your registry and use it as a custom template.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git cmake build-essential \
        libgl1 libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /workspace/PCPG

# Copy pyproject first for layer caching on dep changes only.
COPY pyproject.toml requirements-gpu.txt ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e . \
    && pip install -r requirements-gpu.txt

COPY . .

# Default to dropping into a shell; the actual training command is supplied
# by the user (or RunPod's "container start command" field).
CMD ["bash"]
