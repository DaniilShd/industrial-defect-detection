# analysis/Dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/utils:/app/scripts

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential git libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    opencv-python-headless==4.10.0.84 \
    tqdm==4.66.5 \
    scipy==1.14.0 \
    matplotlib==3.8.0 \
    seaborn \
    scikit-learn \
    scikit-image \
    pyyaml==6.0.1

RUN mkdir -p /app/{scripts,utils,data,reports}

WORKDIR /app

# Исправлено: analysis, не generation
COPY analysis/original/config.yaml /app/config.yaml
COPY analysis/original/utils/ /app/utils/
COPY analysis/original/scripts/ /app/scripts/

VOLUME ["/app/data", "/app/reports"]

CMD ["tail", "-f", "/dev/null"]