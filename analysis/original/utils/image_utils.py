"""Загрузка и базовые операции с изображениями."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_image(path: Path) -> Optional[np.ndarray]:
    """Загрузка изображения RGB."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Не удалось загрузить: {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images_batch(paths: List[Path], max_images: Optional[int] = None) -> List[np.ndarray]:
    """Пакетная загрузка изображений."""
    from tqdm import tqdm
    
    if max_images and max_images < len(paths):
        import random
        paths = random.sample(paths, max_images)
    
    images = []
    for p in tqdm(paths, desc="Загрузка"):
        img = load_image(p)
        if img is not None:
            images.append(img)
    
    logger.info(f"Загружено {len(images)} изображений")
    return images


def compute_basic_stats(img: np.ndarray) -> dict:
    """Базовая статистика изображения: среднее, std, min, max по каналам."""
    return {
        'mean': img.mean(axis=(0, 1)),
        'std': img.std(axis=(0, 1)),
        'min': img.min(axis=(0, 1)),
        'max': img.max(axis=(0, 1)),
        'shape': img.shape
    }


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Вычисление магнитуды градиента (Sobel)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)