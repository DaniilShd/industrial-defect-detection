#!/usr/bin/env python3
"""Оценка best_model.pth на тестовой выборке"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import yaml
from torchvision.models import resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou

class DefectTestDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=(640, 640), num_classes=4):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.num_classes = num_classes
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = sorted([f for f in self.images_dir.glob("*") 
                                   if f.suffix.lower() in extensions])
        print(f"  Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            image = image.resize(self.img_size, Image.BILINEAR)
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except:
            img_tensor = torch.zeros(3, *self.img_size)
            orig_w, orig_h = self.img_size
        
        boxes, labels = self._load_labels(img_path, orig_w, orig_h)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
        return img_tensor, target
    
    def _load_labels(self, img_path, orig_w, orig_h):
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        
        if not label_path.exists():
            return boxes, labels
        
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls = int(float(parts[0]))
                        if 0 <= cls < self.num_classes:
                            xc, yc, w, h = map(float, parts[1:5])
                            x1 = max(0, (xc - w/2) * orig_w * scale_x)
                            y1 = max(0, (yc - h/2) * orig_h * scale_y)
                            x2 = min(self.img_size[1], (xc + w/2) * orig_w * scale_x)
                            y2 = min(self.img_size[0], (yc + h/2) * orig_h * scale_y)
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(cls + 1)
                    except:
                        continue
        return boxes, labels

def compute_map(predictions, ground_truths, num_classes, iou_thr=0.5):
    """Простое вычисление mAP"""
    aps = []
    for cls_id in range(num_classes):
        detections = []
        num_gt = 0
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            gt_mask = gt['labels'] == cls_id
            num_gt += gt_mask.sum().item()
            
            pred_mask = pred['labels'] == cls_id
            for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                detections.append({'image_id': img_idx, 'score': score.item(), 'bbox': box})
        
        if num_gt == 0:
            continue
        
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        gt_matched = {}
        for img_idx, gt in enumerate(ground_truths):
            gt_mask = gt['labels'] == cls_id
            gt_matched[img_idx] = [False] * gt_mask.sum().item()
        
        tp, fp = np.zeros(len(detections)), np.zeros(len(detections))
        
        for i, det in enumerate(detections):
            img_idx = det['image_id']
            gt_mask = ground_truths[img_idx]['labels'] == cls_id
            gt_boxes = ground_truths[img_idx]['boxes'][gt_mask]
            
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            ious = box_iou(det['bbox'].unsqueeze(0), gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            
            if best_iou >= iou_thr and not gt_matched[img_idx][best_idx.item()]:
                tp[i] = 1
                gt_matched[img_idx][best_idx.item()] = True
            else:
                fp[i] = 1
        
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cum / max(num_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
        
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                ap += np.max(precisions[recalls >= t]) / 11.0
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

def create_model(ssl_path, num_classes=4):
    """Создает Faster R-CNN с SSL бэкбоном"""
    base_model = resnet18(weights=None)
    checkpoint = torch.load(ssl_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        weights = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        weights = checkpoint['model_state_dict']
    else:
        weights = checkpoint
    
    cleaned_weights = {}
    for k, v in weights.items():
        clean_k = k
        for prefix in ['backbone.', 'model.', 'module.', 'encoder.']:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
        cleaned_weights[clean_k] = v
    
    base_model.load_state_dict(cleaned_weights, strict=False)
    backbone = _resnet_fpn_extractor(base_model, trainable_layers=5)
    model = FasterRCNN(backbone, num_classes=num_classes + 1)
    return model

def main():
    # Пути
    model_path = "thesis_experiments/results/ssl_distilled/best_model.pth"
    ssl_backbone_path = "thesis_experiments/ssl_pretraining/resnet18_distilled/exported_models/exported_last.pt"
    test_images = "../../data/experiment_v3/datasets/mixed_full/test/images"
    test_labels = "../../data/experiment_v3/datasets/mixed_full/test/labels"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Загрузка модели
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Best epoch: {checkpoint['epoch']}, mAP@50: {checkpoint['metrics']['mAP@50']:.4f}")
    
    print(f"Loading SSL backbone from {ssl_backbone_path}")
    model = create_model(ssl_backbone_path, num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Тестовый датасет
    print(f"\nLoading test dataset...")
    test_dataset = DefectTestDataset(test_images, test_labels, img_size=(640, 640), num_classes=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    
    # Inference
    predictions, ground_truths = [], []
    print(f"\nRunning inference...")
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, total=len(test_dataset)):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                keep = output['scores'] > 0.25
                predictions.append({
                    'boxes': output['boxes'][keep].cpu(),
                    'scores': output['scores'][keep].cpu(),
                    'labels': (output['labels'][keep] - 1).cpu(),
                })
                ground_truths.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': (target['labels'] - 1).cpu(),
                })
    
    # Вычисление метрик
    print(f"\n{'='*50}")
    print("COMPUTING METRICS")
    print(f"{'='*50}")
    
    map50 = compute_map(predictions, ground_truths, num_classes=4, iou_thr=0.5)
    map75 = compute_map(predictions, ground_truths, num_classes=4, iou_thr=0.75)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  mAP@50: {map50:.4f}")
    print(f"  mAP@75: {map75:.4f}")
    
    # Сохранение
    result = {
        'model_path': str(model_path),
        'test_mAP@50': float(map50),
        'test_mAP@75': float(map75),
        'train_best_epoch': checkpoint['epoch'],
        'train_best_mAP@50': checkpoint['metrics']['mAP@50'],
    }
    
    output_path = Path(model_path).parent / 'test_results.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()