#!/usr/bin/env python3
"""Цветокоррекция: color_transfer_lab, adaptive_color_correction"""

import cv2
import numpy as np


def color_transfer_lab(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    
    result_lab = target_lab.copy()
    
    for channel in range(3):
        src_mean, src_std = source_lab[:, :, channel].mean(), source_lab[:, :, channel].std()
        tgt_mean, tgt_std = target_lab[:, :, channel].mean(), target_lab[:, :, channel].std()
        
        if tgt_std > 0:
            result_lab[:, :, channel] = ((target_lab[:, :, channel] - tgt_mean) * (src_std / tgt_std)) + src_mean
    
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    
    return result_rgb.astype(np.float32)


def adaptive_color_correction(sd_result: np.ndarray, original_bg: np.ndarray,
                             defect_mask: np.ndarray, strength: float = 0.85) -> np.ndarray:
    sd_result = sd_result.astype(np.float32)
    original_bg = original_bg.astype(np.float32)
    defect_mask = defect_mask.astype(np.float32)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_outer = cv2.dilate(defect_mask, kernel)
    mask_inner = cv2.erode(defect_mask, kernel)
    
    corrected_full = color_transfer_lab(original_bg, sd_result).astype(np.float32)
    corrected_defect = cv2.addWeighted(sd_result, 1.0 - strength * 0.3, corrected_full, strength * 0.3, 0)
    
    bg_weight_3ch = np.stack([mask_outer * strength] * 3, axis=-1).astype(np.float32)
    defect_weight_3ch = np.stack([mask_inner * (1.0 - strength * 0.3)] * 3, axis=-1).astype(np.float32)
    
    result = sd_result.copy()
    result = corrected_full * bg_weight_3ch + result * (1 - bg_weight_3ch)
    result = corrected_defect * defect_weight_3ch + result * (1 - defect_weight_3ch)
    
    return np.clip(result, 0, 255)