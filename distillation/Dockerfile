FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/distillation/scripts

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential git libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir \
    pandas numpy opencv-python-headless tqdm scipy matplotlib \
    seaborn scikit-learn pyyaml Pillow mlflow
RUN pip install --no-cache-dir lightly lightly_train
RUN pip install --no-cache-dir ultralytics

RUN mkdir -p /app
WORKDIR /app
CMD ["tail", "-f", "/dev/null"]