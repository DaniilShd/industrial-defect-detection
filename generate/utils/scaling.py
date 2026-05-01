#!/usr/bin/env python3
"""Масштабирование дефекта с обновлением bbox"""

from typing import Dict, Tuple

import cv2
import numpy as np


def scale_defect_and_mask(background_crop: np.ndarray, crop_comp_mask: np.ndarray,
                         scale_factor: tuple, orig_bbox: Dict, 
                         crop_offset_x: int, crop_offset_y: int,
                         img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    scale_x, scale_y = scale_factor
    
    if scale_x == 1.0 and scale_y == 1.0:
        return background_crop, crop_comp_mask, orig_bbox
    
    h, w = background_crop.shape[:2]
    
    moments = cv2.moments(crop_comp_mask)
    if moments['m00'] == 0:
        return background_crop, crop_comp_mask, orig_bbox
    
    cx_local = int(moments['m10'] / moments['m00'])
    cy_local = int(moments['m01'] / moments['m00'])
    
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    
    scaled_bg = cv2.resize(background_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    scaled_mask = cv2.resize(crop_comp_mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = (scaled_mask > 0.5).astype(np.uint8)
    
    new_cx = int(cx_local * scale_x)
    new_cy = int(cy_local * scale_y)
    
    x_offset = cx_local - new_cx
    y_offset = cy_local - new_cy
    
    output_bg = np.zeros_like(background_crop)
    output_mask = np.zeros_like(crop_comp_mask)
    
    src_x1 = max(0, -x_offset)
    src_y1 = max(0, -y_offset)
    src_x2 = min(new_w, w - x_offset)
    src_y2 = min(new_h, h - y_offset)
    
    dst_x1 = max(0, x_offset)
    dst_y1 = max(0, y_offset)
    
    copy_w = min(src_x2 - src_x1, w - dst_x1)
    copy_h = min(src_y2 - src_y1, h - dst_y1)
    
    if copy_w > 0 and copy_h > 0:
        output_bg[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
            scaled_bg[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
        output_mask[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
            scaled_mask[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
    
    empty_mask = (output_bg.sum(axis=2) == 0)
    if len(output_bg.shape) == 3:
        empty_mask_3ch = np.stack([empty_mask] * 3, axis=-1)
        output_bg[empty_mask_3ch] = background_crop[empty_mask_3ch]
    
    if output_mask.sum() > 0:
        ys, xs = np.where(output_mask > 0)
        new_x1_crop = xs.min()
        new_y1_crop = ys.min()
        new_x2_crop = xs.max()
        new_y2_crop = ys.max()
        
        new_w_bbox = new_x2_crop - new_x1_crop
        new_h_bbox = new_y2_crop - new_y1_crop
        
        new_x_global = crop_offset_x + new_x1_crop
        new_y_global = crop_offset_y + new_y1_crop
        
        updated_bbox = {
            'class': orig_bbox['class'],
            'x_center': (new_x_global + new_w_bbox / 2) / img_width,
            'y_center': (new_y_global + new_h_bbox / 2) / img_height,
            'width': new_w_bbox / img_width,
            'height': new_h_bbox / img_height,
            'x': new_x_global,
            'y': new_y_global,
            'w': new_w_bbox,
            'h': new_h_bbox,
        }
    else:
        updated_bbox = orig_bbox
    
    return output_bg, output_mask, updated_bbox