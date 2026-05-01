#!/usr/bin/env python3
"""Оценка модели: COCO mAP"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)


def evaluate_model(model, test_path: Path, num_classes: int = 4, 
                   conf_threshold: float = 0.25) -> dict:
    """COCO-стандартные метрики"""
    
    images_dir = test_path / "images" if (test_path / "images").exists() else test_path
    labels_dir = test_path / "labels" if (test_path / "labels").exists() else test_path
    
    image_files = sorted([f for f in images_dir.glob('*')
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    all_preds, all_gts = [], []
    
    model.eval()
    for img_path in image_files:
        with torch.no_grad():
            results = model.predict(str(img_path))
        
        labels = results.get('labels', torch.tensor([]))
        bboxes = results.get('bboxes', torch.tensor([]))
        scores = results.get('scores', torch.tensor([]))
        
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(bboxes, torch.Tensor): bboxes = bboxes.cpu().numpy()
        if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
        
        for box, score, label in zip(bboxes, scores, labels):
            if score >= conf_threshold:
                all_preds.append({
                    'image_id': img_path.stem,
                    'bbox': box.tolist(),
                    'class': int(label),
                    'confidence': float(score)
                })
        
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with Image.open(img_path) as img:
                iw, ih = img.size
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0, (xc - w/2) * iw)
                    y1 = max(0, (yc - h/2) * ih)
                    x2 = min(iw, (xc + w/2) * iw)
                    y2 = min(ih, (yc + h/2) * ih)
                    if x2 > x1 and y2 > y1:
                        all_gts.append({
                            'image_id': img_path.stem,
                            'bbox': [x1, y1, x2, y2],
                            'class': cls
                        })
    
    if not all_preds or not all_gts:
        return {'mAP@50': 0.0, 'mAP@75': 0.0, 'mAP@50:95': 0.0}
    
    per_class_ap50, per_class_ap75 = [], []
    thresholds = np.linspace(0.5, 0.95, 10)
    map_per_threshold = []
    
    for thr in thresholds:
        aps = [_compute_ap(all_preds, all_gts, c, thr) for c in range(num_classes)]
        map_per_threshold.append(np.mean(aps))
    
    for c in range(num_classes):
        per_class_ap50.append(_compute_ap(all_preds, all_gts, c, 0.5))
        per_class_ap75.append(_compute_ap(all_preds, all_gts, c, 0.75))
    
    return {
        'mAP@50': float(np.mean(per_class_ap50)),
        'mAP@75': float(np.mean(per_class_ap75)),
        'mAP@50:95': float(np.mean(map_per_threshold)),
        'per_class_AP50': per_class_ap50,
    }


def _compute_ap(preds, gts, cls, iou_thr):
    cls_preds = sorted([p for p in preds if p['class'] == cls],
                      key=lambda x: x['confidence'], reverse=True)
    cls_gts = [g for g in gts if g['class'] == cls]
    
    if not cls_gts: return 0.0
    if not cls_preds: return 0.0
    
    tp = np.zeros(len(cls_preds))
    fp = np.zeros(len(cls_preds))
    matched = set()
    
    for i, pred in enumerate(cls_preds):
        img_gts = [(j, g) for j, g in enumerate(cls_gts)
                  if g['image_id'] == pred['image_id'] and j not in matched]
        if not img_gts:
            fp[i] = 1; continue
        
        pbox = torch.tensor([pred['bbox']])
        gboxes = torch.tensor([g[1]['bbox'] for g in img_gts])
        ious = box_iou(pbox, gboxes)[0]
        best_j = ious.argmax().item()
        
        if ious[best_j] >= iou_thr:
            tp[i] = 1
            matched.add(img_gts[best_j][0])
        else:
            fp[i] = 1
    
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(cls_gts)
    precs = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
    
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        ap += np.max(precs[recalls >= t]) / 101.0 if np.any(recalls >= t) else 0.0
    
    return float(ap)