#!/usr/bin/env python3
"""Многослойное Poisson-смешивание"""

import cv2
import numpy as np


def apply_multiscale_blend(defect: np.ndarray, background: np.ndarray,
                          component_mask: np.ndarray) -> np.ndarray:
    mask_float = component_mask.astype(np.float32)
    
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core_mask = cv2.erode(mask_float, kernel_core, iterations=2)
    core_mask = cv2.GaussianBlur(core_mask, (5, 5), 2)
    
    main_mask = cv2.GaussianBlur(mask_float, (7, 7), 3)
    
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    outer_mask = cv2.dilate(mask_float, kernel_outer)
    outer_mask = cv2.GaussianBlur(outer_mask, (15, 15), 7)
    
    core_mask_3ch = np.stack([core_mask] * 3, axis=-1)
    main_mask_3ch = np.stack([main_mask] * 3, axis=-1)
    outer_mask_3ch = np.stack([outer_mask] * 3, axis=-1)
    
    result_core = defect * core_mask_3ch + background * (1 - core_mask_3ch)
    
    blend_mid = defect * 0.85 + background * 0.15
    mid_weight = np.clip(main_mask_3ch - core_mask_3ch, 0, 1)
    
    blend_outer = defect * 0.4 + background * 0.6
    outer_weight = np.clip(outer_mask_3ch - main_mask_3ch, 0, 1)
    
    result = background.copy()
    result = result * (1 - outer_mask_3ch) + blend_outer * outer_mask_3ch
    result = result * (1 - mid_weight) + blend_mid * mid_weight
    result = result * (1 - core_mask_3ch) + result_core * core_mask_3ch
    
    transition_zone = np.clip(outer_mask_3ch - main_mask_3ch, 0, 1)
    
    if transition_zone.max() > 0:
        bg_blur = cv2.GaussianBlur(background, (0, 0), sigmaX=5)
        bg_details = background.astype(np.float32) - bg_blur.astype(np.float32)
        result = result + bg_details * transition_zone * 0.3
    
    return np.clip(result, 0, 255).astype(np.uint8)