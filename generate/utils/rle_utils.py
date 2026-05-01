#!/usr/bin/env python3
"""RLE декодирование патча (маска уже 256x256, C-order)"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rle_to_mask_simple(rle_string: Optional[str], size: int = 256) -> np.ndarray:
    """
    Декодирование RLE патча (без смещения — маска уже для патча 256×256).
    RLE упакован в C-order (row-major) — строка за строкой.
    """
    if rle_string is None or pd.isna(rle_string):
        return np.zeros((size, size), dtype=np.uint8)
    
    rle_str = str(rle_string).strip()
    if rle_str == '' or rle_str.lower() == 'nan':
        return np.zeros((size, size), dtype=np.uint8)
    
    try:
        numbers = list(map(int, rle_str.split()))
        if len(numbers) % 2 != 0:
            logger.warning(f"Нечётное количество чисел в RLE")
            return np.zeros((size, size), dtype=np.uint8)
        
        starts = np.array(numbers[0::2]) - 1
        lengths = np.array(numbers[1::2])
        total_pixels = size * size
        
        mask_flat = np.zeros(total_pixels, dtype=np.uint8)
        for start, length in zip(starts, lengths):
            if start < 0:
                continue
            end = min(start + length, total_pixels)
            if start < total_pixels:
                mask_flat[start:end] = 1
        
        # C-order (row-major) — строка за строкой, совместимо с np.flatten()
        return mask_flat.reshape((size, size))
    
    except Exception as e:
        logger.error(f"Ошибка декодирования RLE: {e}")
        return np.zeros((size, size), dtype=np.uint8)


def rle_to_defect_bboxes(rle_string: Optional[str], class_id: int) -> List[Dict]:
    """Извлечение bbox из RLE патча (маска уже 256×256, C-order)"""
    mask = rle_to_mask_simple(rle_string)
    
    if mask.sum() == 0:
        return []
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []
    kernel = np.ones((3, 3), np.uint8)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area < 16 or w < 4 or h < 4:
            continue
        
        component_mask = (labels == i).astype(np.uint8)
        dilated_once = cv2.dilate(component_mask, kernel, iterations=1)
        boundary_mask = dilated_once - component_mask
        
        bboxes.append({
            'class': class_id,
            'x_center': (x + w/2) / 256,
            'y_center': (y + h/2) / 256,
            'width': w / 256,
            'height': h / 256,
            'component_mask': component_mask,
            'boundary_mask': boundary_mask,
            'eroded_mask': cv2.erode(component_mask, kernel, iterations=1),
            'x': x, 'y': y, 'w': w, 'h': h,
            'centroid': (int(centroids[i, 0]), int(centroids[i, 1]))
        })
    
    return bboxes