#!/usr/bin/env python3
"""Замер FPS на CPU"""

import time
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def measure_fps(
    model,
    test_images: Path,
    img_size: tuple = (640, 640),
    warmup: int = 50,
    iterations: int = 500,
    device: str = "cpu",
) -> dict:
    """Замеряет FPS модели на CPU."""
    model.to(device)
    model.eval()

    # Загружаем одно изображение как тестовый вход
    image_files = sorted(test_images.glob("*"))
    if not image_files:
        return {'fps': 0.0, 'latency_ms': 0.0}

    img = Image.open(image_files[0]).convert("RGB").resize(img_size)

    dummy_input = torch.randn(1, 3, *img_size).to(device)

    # Разогрев
    for _ in range(warmup):
        with torch.no_grad():
            try:
                model.predict(img, threshold=0.25)
            except AttributeError:
                model(dummy_input)

    # Замер
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            try:
                model.predict(img, threshold=0.25)
            except AttributeError:
                model(dummy_input)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000

    logger.info(f"FPS: {fps:.1f}, Latency: {latency_ms:.2f} ms")
    return {'fps': round(fps, 1), 'latency_ms': round(latency_ms, 2)}


def count_parameters(model) -> dict:
    """Подсчитывает количество параметров и размер модели."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Размер в MB
    torch.save(model.state_dict(), "/tmp/model_size_temp.pth")
    size_mb = Path("/tmp/model_size_temp.pth").stat().st_size / (1024 * 1024)
    Path("/tmp/model_size_temp.pth").unlink(missing_ok=True)

    return {
        'params_total': total,
        'params_trainable': trainable,
        'params_M': round(total / 1e6, 1),
        'size_MB': round(size_mb, 1),
    }