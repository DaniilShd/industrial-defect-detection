"""Цветовой анализ: LAB, HSV, гистограммы."""
import cv2
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_color_stats(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Цветовая статистика в LAB и HSV.
    Возвращает средние и std для каждого канала.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    features = {}
    
    for name, space in [('lab', lab), ('hsv', hsv)]:
        for i, channel in enumerate(['ch0', 'ch1', 'ch2']):
            features[f'{name}_{channel}_mean'] = float(space[:, :, i].mean())
            features[f'{name}_{channel}_std'] = float(space[:, :, i].std())
    
    return features


def compute_histogram(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """Нормализованная гистограмма."""
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    return hist.flatten() / hist.sum()


def analyze_channel(channel: np.ndarray) -> Dict[str, float]:
    """Статистика одного канала."""
    return {
        'mean': float(channel.mean()),
        'std': float(channel.std()),
        'skewness': float(((channel - channel.mean()) ** 3).mean() / (channel.std() + 1e-8) ** 3),
        'kurtosis': float(((channel - channel.mean()) ** 4).mean() / (channel.std() + 1e-8) ** 4),
        'p5': float(np.percentile(channel, 5)),
        'p95': float(np.percentile(channel, 95))
    }