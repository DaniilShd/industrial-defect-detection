#!/usr/bin/env python3
"""Оценка модели LT-DETR: mAP_50, mAP_75, mAP_50_95, per-class AP50, Precision, Recall, F1"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    test_images: Path,
    test_labels: Path,
    num_classes: Optional[int] = None,
    conf_threshold: float = 0.001
) -> Dict:
    """Оценка модели на тестовом наборе с полными метриками."""
    import lightly_train
    
    model = lightly_train.load_model(model_path)
    
    # Автоопределение количества классов из модели
    if num_classes is None:
        if hasattr(model, 'classes'):
            num_classes = len(model.classes)
        else:
            # Пробуем определить из количества классов в модели
            num_classes = getattr(model, 'num_classes', 4)
    
    logger.info(f"Model has {num_classes} classes")
    
    all_preds = []
    all_gts = []
    
    image_files = sorted([f for f in test_images.glob("*")
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
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
            
            # Конвертируем в numpy/cpu
            if isinstance(labels, torch.Tensor): 
                labels = labels.cpu().numpy()
            if isinstance(bboxes, torch.Tensor): 
                bboxes = bboxes.cpu().numpy()
            if isinstance(scores, torch.Tensor): 
                scores = scores.cpu().numpy()
            
            for box, score, label in zip(bboxes, scores, labels):
                if score >= conf_threshold:
                    all_preds.append({
                        'image_id': img_path.stem,
                        'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
                        'class': int(label),
                        'confidence': float(score)
                    })
        except Exception as e:
            logger.warning(f"Prediction failed for {img_path.name}: {e}")
        
        # Ground truth
        label_path = test_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            try:
                with Image.open(img_path) as img:
                    iw, ih = img.size
                
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        cls = int(float(parts[0]))
                        # Проверяем, что класс в допустимом диапазоне
                        if cls >= num_classes:
                            continue
                        
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
            except Exception as e:
                logger.warning(f"Failed to read labels for {img_path.name}: {e}")
    
    num_preds = len(all_preds)
    num_gts = len(all_gts)
    logger.info(f"Predictions: {num_preds}, Ground truth: {num_gts}")
    
    if not all_preds or not all_gts:
        logger.warning(f"Insufficient data: preds={num_preds}, gts={num_gts}")
        return {
            **_empty_results(num_classes),
            'num_predictions': num_preds,
            'num_ground_truth': num_gts
        }
    
    # Вычисляем метрики
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
    
    # Precision, Recall, F1
    precision, recall, f1 = _compute_pr_f1(all_preds, all_gts, iou_thr=0.5)
    
    # Per-class метрики
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
        'num_ground_truth': num_gts
    }


def _empty_results(num_classes: int) -> Dict:
    """Возвращает словарь с нулевыми метриками."""
    results = {
        'mAP_50': 0.0, 
        'mAP_75': 0.0, 
        'mAP_50_95': 0.0,
        'Precision': 0.0, 
        'Recall': 0.0, 
        'F1': 0.0,
    }
    results.update({f'cls{c}_AP50': 0.0 for c in range(num_classes)})
    results.update({f'cls{c}_Precision': 0.0 for c in range(num_classes)})
    results.update({f'cls{c}_Recall': 0.0 for c in range(num_classes)})
    results.update({f'cls{c}_F1': 0.0 for c in range(num_classes)})
    return results


def _compute_ap(preds: List, gts: List, cls: int, iou_thr: float) -> float:
    """Вычисление Average Precision для одного класса."""
    cls_preds = sorted(
        [p for p in preds if p['class'] == cls],
        key=lambda x: x['confidence'], 
        reverse=True
    )
    cls_gts = [g for g in gts if g['class'] == cls]
    
    if not cls_gts:
        return 0.0
    
    if not cls_preds:
        return 0.0
    
    tp = np.zeros(len(cls_preds))
    fp = np.zeros(len(cls_preds))
    matched = set()
    
    for i, pred in enumerate(cls_preds):
        img_gts = [
            (j, g) for j, g in enumerate(cls_gts)
            if g['image_id'] == pred['image_id'] and j not in matched
        ]
        
        if not img_gts:
            fp[i] = 1
            continue
        
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
    
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precs[mask])
    ap /= 101.0
    
    return float(ap)


def _compute_pr_f1(
    preds: List, 
    gts: List, 
    iou_thr: float = 0.5, 
    specific_class: Optional[int] = None
) -> tuple:
    """Вычисление Precision, Recall, F1-score."""
    if specific_class is not None:
        cls_preds = [p for p in preds if p['class'] == specific_class]
        cls_gts = [g for g in gts if g['class'] == specific_class]
    else:
        cls_preds = preds.copy()
        cls_gts = gts.copy()
    
    if not cls_gts:
        return 0.0, 0.0, 0.0
    
    if not cls_preds:
        return 0.0, 0.0, 0.0
    
    cls_preds_sorted = sorted(cls_preds, key=lambda x: x['confidence'], reverse=True)
    
    tp = 0
    fp = 0
    matched_gts = set()
    
    for pred in cls_preds_sorted:
        img_gts = [
            (j, g) for j, g in enumerate(cls_gts)
            if g['image_id'] == pred['image_id'] and j not in matched_gts
        ]
        
        if not img_gts:
            fp += 1
            continue
        
        pbox = torch.tensor([pred['bbox']])
        gboxes = torch.tensor([g[1]['bbox'] for g in img_gts])
        ious = box_iou(pbox, gboxes)[0]
        best_iou = ious.max().item()
        best_idx = ious.argmax().item()
        
        if best_iou >= iou_thr:
            tp += 1
            matched_gts.add(img_gts[best_idx][0])
        else:
            fp += 1
    
    fn = len(cls_gts) - len(matched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def print_full_results(results: Dict):
    """Красивый вывод всех метрик."""
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print("=" * 60)
    
    print("\n🏆 ОСНОВНЫЕ МЕТРИКИ:")
    print(f"   mAP@0.5:     {results.get('mAP_50', 0):.4f}")
    print(f"   mAP@0.75:    {results.get('mAP_75', 0):.4f}")
    print(f"   mAP@0.5:0.95: {results.get('mAP_50_95', 0):.4f}")
    
    print("\n🎯 ТОЧНОСТЬ ОБНАРУЖЕНИЯ (IoU=0.5):")
    print(f"   Precision:   {results.get('Precision', 0):.4f}")
    print(f"   Recall:      {results.get('Recall', 0):.4f}")
    print(f"   F1-score:    {results.get('F1', 0):.4f}")
    
    print("\n📈 PER-CLASS AP@0.5:")
    ap50_keys = sorted([k for k in results if k.startswith('cls') and 'AP50' in k])
    for key in ap50_keys:
        cls_id = key.split('_')[0].replace('cls', '')
        print(f"   Class {cls_id}: {results[key]:.4f}")
    
    print("\n📊 PER-CLASS PRECISION/RECALL/F1 (IoU=0.5):")
    for c in sorted(set(
        int(k.split('_')[0].replace('cls', '')) 
        for k in results 
        if k.startswith('cls') and ('Precision' in k)
    )):
        p = results.get(f'cls{c}_Precision', 0)
        r = results.get(f'cls{c}_Recall', 0)
        f = results.get(f'cls{c}_F1', 0)
        print(f"   Class {c}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")
    
    print(f"\n📦 Статистика:")
    print(f"   Предсказаний: {results.get('num_predictions', 0)}")
    print(f"   GT объектов: {results.get('num_ground_truth', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    results = evaluate_model(
        model_path="experiments/models/exp1_frozen/real_baseline_frozen_seed42/exported_models/exported_best.pt",
        test_images=Path("/app/data/processed/balanced_defect_patches_v2/test/images"),
        test_labels=Path("/app/data/processed/balanced_defect_patches_v2/test/labels"),
    )
    
    print_full_results(results)