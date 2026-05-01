FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/utils:/app/scripts

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential git libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Виртуальное окружение
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel

# Основные библиотеки для анализа
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
    pyyaml==6.0.1 \
    PyWavelets

# Рабочие директории
RUN mkdir -p /app/{scripts,utils,data,reports}

WORKDIR /app

# Копируем скрипты и утилиты
COPY generation/config.yaml /app/config.yaml
COPY generation/utils/ /app/utils/
COPY generation/scripts/ /app/scripts/

# Данные монтируются через volumes
VOLUME ["/app/data", "/app/reports"]

CMD ["tail", "-f", "/dev/null"]