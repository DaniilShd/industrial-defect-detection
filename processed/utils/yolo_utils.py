import numpy as np
import cv2
from typing import Dict, List


def masks_to_yolo_boxes(masks: Dict[int, np.ndarray], min_area: int = 10) -> List[List[float]]:
    """Маски → YOLO bbox [class_id, xc, yc, w, h]."""
    if not masks:
        return []
    
    first = next(iter(masks.values()))
    h, w = first.shape
    boxes = []
    
    for class_id, mask in masks.items():
        if mask.sum() == 0:
            continue
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append([
                class_id,
                np.clip((x + bw / 2) / w, 0.0, 1.0),
                np.clip((y + bh / 2) / h, 0.0, 1.0),
                np.clip(bw / w, 1e-6, 1.0),
                np.clip(bh / h, 1e-6, 1.0)
            ])
    
    return boxes