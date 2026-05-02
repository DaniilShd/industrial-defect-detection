#!/usr/bin/env python3
"""Оценка модели: mAP + FPS + параметры"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    test_images: Path,
    test_labels: Path,
    num_classes: int = 4,
    conf_threshold: float = 0.25,
) -> dict:
    """COCO-стандартные метрики."""
    all_preds, all_gts = [], []

    image_files = sorted([
        f for f in test_images.glob("*")
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ])

    logger.info(f"Evaluating on {len(image_files)} test images")

    for img_path in image_files:
        with torch.no_grad():
            results = model.predict(str(img_path), threshold=conf_threshold)

        labels = results.get('labels', torch.tensor([]))
        bboxes = results.get('bboxes', torch.tensor([]))
        scores = results.get('scores', torch.tensor([]))

        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(bboxes, torch.Tensor): bboxes = bboxes.cpu().numpy()
        if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()

        for box, score, label in zip(bboxes, scores, labels):
            all_preds.append({
                'image_id': img_path.stem,
                'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
                'class': int(label),
                'confidence': float(score),
            })

        label_path = test_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            try:
                with Image.open(img_path) as img:
                    iw, ih = img.size
            except Exception:
                iw, ih = 640, 640

            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    try:
                        cls = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:5])
                        x1, y1 = max(0, (xc - w/2) * iw), max(0, (yc - h/2) * ih)
                        x2, y2 = min(iw, (xc + w/2) * iw), min(ih, (yc + h/2) * ih)
                        if x2 > x1 and y2 > y1:
                            all_gts.append({
                                'image_id': img_path.stem,
                                'bbox': [x1, y1, x2, y2],
                                'class': cls,
                            })
                    except (ValueError, IndexError):
                        continue

    if not all_preds or not all_gts:
        return {'mAP_50': 0.0, 'mAP_75': 0.0, 'mAP_50_95': 0.0,
                'num_predictions': len(all_preds), 'num_ground_truth': len(all_gts)}

    per_class_ap50 = {f'cls{c}_AP50': _compute_ap(all_preds, all_gts, c, 0.5) for c in range(num_classes)}
    map50 = float(np.mean(list(per_class_ap50.values())))
    map75 = float(np.mean([_compute_ap(all_preds, all_gts, c, 0.75) for c in range(num_classes)]))
    thresholds = np.linspace(0.5, 0.95, 10)
    map50_95 = float(np.mean([
        np.mean([_compute_ap(all_preds, all_gts, c, thr) for c in range(num_classes)])
        for thr in thresholds
    ]))

    return {
        'mAP_50': map50, 'mAP_75': map75, 'mAP_50_95': map50_95,
        **per_class_ap50,
        'num_predictions': len(all_preds), 'num_ground_truth': len(all_gts),
    }


def _compute_ap(preds, gts, cls, iou_thr):
    cls_preds = sorted([p for p in preds if p['class'] == cls],
                       key=lambda x: x['confidence'], reverse=True)
    cls_gts = [g for g in gts if g['class'] == cls]
    if not cls_gts or not cls_preds:
        return 0.0

    tp, fp, matched = np.zeros(len(cls_preds)), np.zeros(len(cls_preds)), set()
    for i, pred in enumerate(cls_preds):
        img_gts = [(j, g) for j, g in enumerate(cls_gts)
                   if g['image_id'] == pred['image_id'] and j not in matched]
        if not img_gts:
            fp[i] = 1; continue
        pbox = torch.tensor([pred['bbox']], dtype=torch.float32)
        gboxes = torch.tensor([g[1]['bbox'] for g in img_gts], dtype=torch.float32)
        ious = box_iou(pbox, gboxes)[0]
        best_j = ious.argmax().item()
        if ious[best_j] >= iou_thr:
            tp[i] = 1; matched.add(img_gts[best_j][0])
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / len(cls_gts)
    precs = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
    ap = sum(np.max(precs[recalls >= t]) / 101.0 if np.any(recalls >= t) else 0.0
             for t in np.linspace(0, 1, 101))
    return float(ap)