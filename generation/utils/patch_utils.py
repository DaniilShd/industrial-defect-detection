import numpy as np
import cv2
import re
from typing import List, Tuple, Dict


def has_black_background(img: np.ndarray, threshold: int = 30, max_ratio: float = 0.05) -> bool:
    """Много ли чёрного фона."""
    gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
    ratio = np.sum(gray < threshold) / (img.shape[0] * img.shape[1])
    return ratio > max_ratio


def resize_with_padding(img: np.ndarray, boxes: List[List[float]], target_size: int = 256) -> Tuple[np.ndarray, List[List[float]]]:
    """Ресайз с паддингом + адаптация bbox."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    
    new_boxes = []
    for box in boxes:
        cls, xc, yc, bw, bh = box
        xc_px = xc * w * scale + x_off
        yc_px = yc * h * scale + y_off
        bw_px = bw * w * scale
        bh_px = bh * h * scale
        
        new_boxes.append([
            cls,
            np.clip(xc_px / target_size, 0.0, 1.0),
            np.clip(yc_px / target_size, 0.0, 1.0),
            np.clip(bw_px / target_size, 1e-6, 1.0),
            np.clip(bh_px / target_size, 1e-6, 1.0)
        ])
    
    return canvas, new_boxes


def extract_offset(filename: str) -> tuple:
    """Смещение и ширина патча из имени файла."""
    m = re.search(r'_x(\d+)_w(\d+)', filename)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 256)