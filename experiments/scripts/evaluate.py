#!/usr/bin/env python3
"""Оценка модели LT-DETR: mAP, per-class AP, Precision, Recall, F1"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)


def evaluate_model(
    model_or_path,
    test_images: Path,
    test_labels: Path,
    num_classes: Optional[int] = None,
    conf_threshold: float = 0.001,
) -> dict:
    """
    Полная оценка модели.
    
    Поддерживает как путь к модели, так и загруженную модель.
    """
    import lightly_train

    # Загрузка модели
    if isinstance(model_or_path, (str, Path)):
        model = lightly_train.load_model(str(model_or_path))
    else:
        model = model_or_path

    # Автоопределение количества классов
    if num_classes is None:
        if hasattr(model, 'classes'):
            num_classes = len(model.classes)
        else:
            num_classes = getattr(model, 'num_classes', 4)

    logger.info(f"Model has {num_classes} classes")

    all_preds, all_gts = [], []

    image_files = sorted([
        f for f in test_images.glob("*")
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ])

    if not image_files:
        logger.error(f"No images found in {test_images}")
        return _empty_results(num_classes)

    logger.info(f"Evaluating on {len(image_files)} test images")

    for img_path in image_files:
        # Предсказания
        try:
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
                        'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
                        'class': int(label),
                        'confidence': float(score),
                    })
        except Exception as e:
            logger.warning(f"Prediction failed for {img_path.name}: {e}")

        # Ground truth
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
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        if cls >= num_classes:
                            continue
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = max(0, (xc - w / 2) * iw)
                        y1 = max(0, (yc - h / 2) * ih)
                        x2 = min(iw, (xc + w / 2) * iw)
                        y2 = min(ih, (yc + h / 2) * ih)
                        if x2 > x1 and y2 > y1:
                            all_gts.append({
                                'image_id': img_path.stem,
                                'bbox': [x1, y1, x2, y2],
                                'class': cls,
                            })
                    except (ValueError, IndexError):
                        continue

    num_preds, num_gts = len(all_preds), len(all_gts)
    logger.info(f"Predictions: {num_preds}, Ground truth: {num_gts}")

    if not all_preds or not all_gts:
        logger.warning(f"Insufficient data: preds={num_preds}, gts={num_gts}")
        return {
            **_empty_results(num_classes),
            'num_predictions': num_preds,
            'num_ground_truth': num_gts,
        }

    # === mAP метрики ===
    per_class_ap50 = {
        f'cls{c}_AP50': _compute_ap(all_preds, all_gts, c, 0.5)
        for c in range(num_classes)
    }
    map50 = float(np.mean(list(per_class_ap50.values())))
    map75 = float(np.mean([
        _compute_ap(all_preds, all_gts, c, 0.75)
        for c in range(num_classes)
    ]))

    thresholds = np.linspace(0.5, 0.95, 10)
    map50_95 = float(np.mean([
        np.mean([_compute_ap(all_preds, all_gts, c, thr)
                for c in range(num_classes)])
        for thr in thresholds
    ]))

    # === Precision, Recall, F1 ===
    precision, recall, f1 = _compute_pr_f1(all_preds, all_gts, iou_thr=0.5)

    # === Per-class Precision, Recall, F1 ===
    per_class_pr = {}
    for c in range(num_classes):
        p, r, f = _compute_pr_f1(all_preds, all_gts, iou_thr=0.5, specific_class=c)
        per_class_pr[f'cls{c}_Precision'] = p
        per_class_pr[f'cls{c}_Recall'] = r
        per_class_pr[f'cls{c}_F1'] = f

    return {
        'mAP_50': map50,
        'mAP_75': map75,
        'mAP_50_95': map50_95,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        **per_class_ap50,
        **per_class_pr,
        'num_predictions': num_preds,
        'num_ground_truth': num_gts,
    }


def _empty_results(num_classes: int) -> Dict:
    """Нулевые метрики при отсутствии данных."""
    results = {
        'mAP_50': 0.0, 'mAP_75': 0.0, 'mAP_50_95': 0.0,
        'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0,
    }
    for c in range(num_classes):
        results[f'cls{c}_AP50'] = 0.0
        results[f'cls{c}_Precision'] = 0.0
        results[f'cls{c}_Recall'] = 0.0
        results[f'cls{c}_F1'] = 0.0
    return results


def _compute_ap(preds: List, gts: List, cls: int, iou_thr: float) -> float:
    """Average Precision для одного класса."""
    cls_preds = sorted(
        [p for p in preds if p['class'] == cls],
        key=lambda x: x['confidence'], reverse=True,
    )
    cls_gts = [g for g in gts if g['class'] == cls]
    if not cls_gts or not cls_preds:
        return 0.0

    tp = np.zeros(len(cls_preds))
    fp = np.zeros(len(cls_preds))
    matched = set()

    for i, pred in enumerate(cls_preds):
        img_gts = [(j, g) for j, g in enumerate(cls_gts)
                   if g['image_id'] == pred['image_id'] and j not in matched]
        if not img_gts:
            fp[i] = 1
            continue
        pbox = torch.tensor([pred['bbox']], dtype=torch.float32)
        gboxes = torch.tensor([g[1]['bbox'] for g in img_gts], dtype=torch.float32)
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
        if np.any(recalls >= t):
            ap += np.max(precs[recalls >= t]) / 101.0
    return float(ap)


def _compute_pr_f1(
    preds: List, gts: List, iou_thr: float = 0.5,
    specific_class: Optional[int] = None,
) -> tuple:
    """Precision, Recall, F1-score."""
    if specific_class is not None:
        cls_preds = [p for p in preds if p['class'] == specific_class]
        cls_gts = [g for g in gts if g['class'] == specific_class]
    else:
        cls_preds = preds.copy()
        cls_gts = gts.copy()

    if not cls_gts or not cls_preds:
        return 0.0, 0.0, 0.0

    cls_preds_sorted = sorted(cls_preds, key=lambda x: x['confidence'], reverse=True)

    tp, fp = 0, 0
    matched_gts = set()

    for pred in cls_preds_sorted:
        img_gts = [(j, g) for j, g in enumerate(cls_gts)
                   if g['image_id'] == pred['image_id'] and j not in matched_gts]
        if not img_gts:
            fp += 1
            continue
        pbox = torch.tensor([pred['bbox']], dtype=torch.float32)
        gboxes = torch.tensor([g[1]['bbox'] for g in img_gts], dtype=torch.float32)
        ious = box_iou(pbox, gboxes)[0]
        best_idx = ious.argmax().item()
        if ious[best_idx] >= iou_thr:
            tp += 1
            matched_gts.add(img_gts[best_idx][0])
        else:
            fp += 1

    fn = len(cls_gts) - len(matched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1