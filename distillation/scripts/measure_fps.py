#!/usr/bin/env python3
"""Замер FPS для разных типов моделей"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def measure_fps(model_wrapper, model_type: str, test_images: Path,
                img_size: tuple = (640, 640), warmup: int = 50,
                iterations: int = 200, device: str = "cpu") -> dict:
    """Замеряет FPS модели."""
    image_files = sorted(test_images.glob("*"))
    if not image_files:
        return {'fps': 0.0, 'latency_ms': 0.0}
    
    img = Image.open(image_files[0]).convert("RGB").resize(img_size)
    img.save("/tmp/test_img.jpg")
    img_path = "/tmp/test_img.jpg"
    
    for _ in range(warmup):
        try:
            model_wrapper.predict(img_path, conf_threshold=0.25)
        except Exception:
            pass
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            model_wrapper.predict(img_path, conf_threshold=0.25)
        except Exception:
            pass
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) if times else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    
    logger.info(f"{model_type}: FPS={fps:.1f}, Latency={avg_time*1000:.2f}ms")
    return {'fps': round(fps, 1), 'latency_ms': round(avg_time * 1000, 2)}


def count_parameters(model) -> dict:
    """Подсчитывает параметры модели."""
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        torch.save(model.state_dict(), "/tmp/model_size_temp.pth")
        size_mb = Path("/tmp/model_size_temp.pth").stat().st_size / (1024 * 1024)
        Path("/tmp/model_size_temp.pth").unlink(missing_ok=True)
        
        return {'params_total': total, 'params_trainable': trainable,
                'params_M': round(total / 1e6, 1), 'size_MB': round(size_mb, 1)}
    except Exception as e:
        logger.warning(f"Cannot count parameters: {e}")
        return {'params_total': 0, 'params_trainable': 0, 'params_M': 0.0, 'size_MB': 0.0}