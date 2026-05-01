"""Утилиты для чистых патчей."""
import numpy as np


def has_black_pixels(patch: np.ndarray, black_threshold: int = 10) -> bool:
    """Проверяет наличие чёрных пикселей в патче."""
    is_black = np.all(patch < 30, axis=2)
    return np.sum(is_black) > black_threshold


def compute_clean_ratio(mask: np.ndarray) -> float:
    """Вычисляет долю чистых пикселей в маске."""
    total = mask.shape[0] * mask.shape[1]
    defect = np.sum(mask)
    return 1.0 - (defect / total)