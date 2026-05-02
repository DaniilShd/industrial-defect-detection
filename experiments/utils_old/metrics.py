#!/usr/bin/env python3
"""
COCO-стандартные метрики для оценки детекции.
Реализация: 101-point interpolated AP, IoU, per-class метрики.
"""

from typing import List, Dict, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Вычисляет Intersection over Union между двумя боксами [x1, y1, x2, y2].
    Используется как fallback если torchvision недоступен.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_coco_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    COCO-style 101-point interpolated Average Precision.
    
    Args:
        recall: recall values
        precision: precision values
    
    Returns:
        AP score
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Делаем precision монотонно убывающей
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Интегрируем площадь под PR-кривой
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return float(ap)


def compute_ap_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = None
) -> Dict:
    """
    Вычисляет AP для каждого класса при разных IoU порогах.
    
    Args:
        predictions: список предсказаний [{image_id, bbox, class, confidence}, ...]
        ground_truths: список GT [{image_id, bbox, class}, ...]
        num_classes: количество классов
        iou_thresholds: список IoU порогов (по умолчанию [0.5, 0.75])
    
    Returns:
        {
            'mAP@50': float,
            'mAP@75': float,
            'mAP@50:95': float,
            'per_class_AP50': [float, ...],
            'per_class_AP75': [float, ...]
        }
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    if not predictions or not ground_truths:
        return {
            'mAP@50': 0.0,
            'mAP@75': 0.0,
            'mAP@50:95': 0.0,
            'per_class_AP50': [0.0] * num_classes,
            'per_class_AP75': [0.0] * num_classes,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truths)
        }
    
    per_class_ap = {thr: [] for thr in iou_thresholds}
    
    for iou_thr in iou_thresholds:
        for cls in range(num_classes):
            ap = _compute_single_ap(predictions, ground_truths, cls, iou_thr)
            per_class_ap[iou_thr].append(ap)
    
    # mAP@50:95 (COCO standard)
    coco_thresholds = np.linspace(0.5, 0.95, 10)
    map_per_threshold = []
    for thr in coco_thresholds:
        aps = [_compute_single_ap(predictions, ground_truths, c, thr) 
               for c in range(num_classes)]
        map_per_threshold.append(np.mean(aps))
    
    return {
        'mAP@50': float(np.mean(per_class_ap[0.5])),
        'mAP@75': float(np.mean(per_class_ap[0.75])),
        'mAP@50:95': float(np.mean(map_per_threshold)),
        'per_class_AP50': [float(ap) for ap in per_class_ap[0.5]],
        'per_class_AP75': [float(ap) for ap in per_class_ap[0.75]],
        'num_predictions': len(predictions),
        'num_ground_truth': len(ground_truths)
    }


def _compute_single_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    target_class: int,
    iou_threshold: float
) -> float:
    """
    Вычисляет AP для одного класса при заданном IoU пороге.
    
    Использует torchvision box_iou для быстрых вычислений.
    """
    cls_preds = sorted(
        [p for p in predictions if p['class'] == target_class],
        key=lambda x: x['confidence'],
        reverse=True
    )
    cls_gts = [g for g in ground_truths if g['class'] == target_class]
    
    if not cls_gts:
        return 0.0
    if not cls_preds:
        return 0.0
    
    tp = np.zeros(len(cls_preds))
    fp = np.zeros(len(cls_preds))
    matched_gt = set()  # индексы уже сопоставленных GT
    
    for pred_idx, pred in enumerate(cls_preds):
        # Ищем GT с тем же image_id, ещё не сопоставленные
        img_gts = [
            (gt_idx, gt) for gt_idx, gt in enumerate(cls_gts)
            if gt['image_id'] == pred['image_id'] and gt_idx not in matched_gt
        ]
        
        if not img_gts:
            fp[pred_idx] = 1
            continue
        
        # Вычисляем IoU с помощью torchvision
        pred_box = torch.tensor([pred['bbox']], dtype=torch.float32)
        gt_boxes = torch.tensor([gt[1]['bbox'] for gt in img_gts], dtype=torch.float32)
        ious = box_iou(pred_box, gt_boxes)[0]
        
        best_idx = ious.argmax().item()
        best_iou = ious[best_idx].item()
        
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            matched_gt.add(img_gts[best_idx][0])
        else:
            fp[pred_idx] = 1
    
    # Кумулятивные суммы
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(cls_gts)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    
    return compute_coco_ap(recalls, precisions)


def compute_confusion_matrix(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> np.ndarray:
    """
    Вычисляет матрицу ошибок (confusion matrix) для детекции.
    
    Returns:
        np.ndarray размером (num_classes + 1, num_classes + 1)
        Последний ряд/столбец = фон (false positives / false negatives)
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    
    # Матчим предсказания к GT
    matched_gt = set()
    matched_pred = set()
    
    cls_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred_idx, pred in enumerate(cls_preds):
        if pred['confidence'] < conf_threshold:
            continue
        
        img_gts = [
            (gt_idx, gt) for gt_idx, gt in enumerate(ground_truths)
            if gt['image_id'] == pred['image_id'] and gt_idx not in matched_gt
        ]
        
        if not img_gts:
            # False positive
            matrix[pred['class'], num_classes] += 1
            continue
        
        pred_box = torch.tensor([pred['bbox']], dtype=torch.float32)
        gt_boxes = torch.tensor([gt[1]['bbox'] for gt in img_gts], dtype=torch.float32)
        ious = box_iou(pred_box, gt_boxes)[0]
        
        best_idx = ious.argmax().item()
        best_iou = ious[best_idx].item()
        
        if best_iou >= iou_threshold:
            gt_class = img_gts[best_idx][1]['class']
            matrix[pred['class'], gt_class] += 1
            matched_gt.add(img_gts[best_idx][0])
            matched_pred.add(pred_idx)
        else:
            matrix[pred['class'], num_classes] += 1
    
    # False negatives (не найденные GT)
    for gt_idx, gt in enumerate(ground_truths):
        if gt_idx not in matched_gt:
            matrix[num_classes, gt['class']] += 1
    
    return matrix


def compute_per_image_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> Dict[str, float]:
    """
    Вычисляет per-image precision, recall, F1-score.
    """
    image_ids = set(p['image_id'] for p in predictions) | set(g['image_id'] for g in ground_truths)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for img_id in image_ids:
        img_preds = [p for p in predictions if p['image_id'] == img_id and p['confidence'] >= conf_threshold]
        img_gts = [g for g in ground_truths if g['image_id'] == img_id]
        
        matched_gt = set()
        tp, fp = 0, 0
        
        for pred in sorted(img_preds, key=lambda x: x['confidence'], reverse=True):
            unmatched_gts = [(i, gt) for i, gt in enumerate(img_gts) if i not in matched_gt]
            
            if not unmatched_gts:
                fp += 1
                continue
            
            pred_box = torch.tensor([pred['bbox']], dtype=torch.float32)
            gt_boxes = torch.tensor([gt[1]['bbox'] for gt in unmatched_gts], dtype=torch.float32)
            ious = box_iou(pred_box, gt_boxes)[0]
            
            best_idx = ious.argmax().item()
            if ious[best_idx] >= iou_threshold:
                tp += 1
                matched_gt.add(unmatched_gts[best_idx][0])
            else:
                fp += 1
        
        fn = len(img_gts) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }