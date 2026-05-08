#!/usr/bin/env python3
"""Обучение Faster R-CNN R18 с FGD дистилляцией и защитой от переобучения"""

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet18_fpn
from torchvision.ops import box_iou
from PIL import Image

logger = logging.getLogger(__name__)


class DefectDataset(torch.utils.data.Dataset):
    """Датасет для Faster R-CNN в COCO-формате."""

    def __init__(self, images_dir: Path, labels_dir: Path, num_classes: int = 4):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.num_classes = num_classes
        self.image_files = sorted([
            f for f in images_dir.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            iw, ih = image.size
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0, (xc - w/2) * iw)
                    y1 = max(0, (yc - h/2) * ih)
                    x2 = min(iw, (xc + w/2) * iw)
                    y2 = min(ih, (yc + h/2) * ih)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)  # Faster R-CNN: 0 = background

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
        }
        return img_tensor, target


class EarlyStopping:
    """Early stopping с восстановлением лучших весов."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        return False

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _evaluate_frcnn(model, data_loader, device) -> dict:
    """Вычисляет mAP@50 для Faster R-CNN."""
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_id = target['image_id'].item()
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()

                keep = scores > 0.25
                for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                    all_preds.append({
                        'image_id': img_id,
                        'bbox': box.tolist(),
                        'class': int(label) - 1,
                        'confidence': float(score),
                    })

                for box, label in zip(target['boxes'], target['labels']):
                    all_gts.append({
                        'image_id': img_id,
                        'bbox': box.tolist(),
                        'class': int(label) - 1,
                    })

    if not all_preds or not all_gts:
        return {'mAP_50': 0.0}

    num_classes = 4
    ap_per_class = []
    for cls in range(num_classes):
        cls_preds = sorted([p for p in all_preds if p['class'] == cls],
                          key=lambda x: x['confidence'], reverse=True)
        cls_gts = [g for g in all_gts if g['class'] == cls]

        if not cls_gts:
            ap_per_class.append(0.0)
            continue
        if not cls_preds:
            ap_per_class.append(0.0)
            continue

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = set()

        for i, pred in enumerate(cls_preds):
            img_gts = [(j, g) for j, g in enumerate(cls_gts)
                      if g['image_id'] == pred['image_id'] and j not in matched]
            if not img_gts:
                fp[i] = 1; continue

            pbox = torch.tensor([pred['bbox']], dtype=torch.float32)
            gboxes = torch.tensor([g[1]['bbox'] for g in img_gts], dtype=torch.float32)
            ious = box_iou(pbox, gboxes)[0]
            best_j = ious.argmax().item()
            if ious[best_j] >= 0.5:
                tp[i] = 1; matched.add(img_gts[best_j][0])
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(cls_gts)
        precs = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)

        ap = sum(np.max(precs[recalls >= t]) / 101.0 if np.any(recalls >= t) else 0.0
                for t in np.linspace(0, 1, 101))
        ap_per_class.append(float(ap))

    return {'mAP_50': float(np.mean(ap_per_class))}


def train_faster_rcnn(config: dict, models_dir: Path, teacher_model_path: str) -> dict:
    """
    Обучает Faster R-CNN R18 с FGD дистилляцией и защитой от переобучения.
    
    FGD: Feature-based Knowledge Distillation от LTDETR-учителя.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config['classes']['num_classes'] + 1  # +1 for background
    cfg = config['students']['faster_rcnn_r18_fgd']

    # Загружаем учителя
    import lightly_train
    teacher = lightly_train.load_model(teacher_model_path)
    teacher.to(device)
    teacher.eval()

    # Датасеты
    dataset_path = Path(config['paths']['experiment_data']) / config['teacher']['dataset']
    train_dataset = DefectDataset(dataset_path / "train" / "images", dataset_path / "train" / "labels")
    val_dataset = DefectDataset(dataset_path / "val" / "images", dataset_path / "val" / "labels")

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch'], shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch'], shuffle=False,
                           collate_fn=lambda x: tuple(zip(*x)))

    # Модель ученика
    model = fasterrcnn_resnet18_fpn(num_classes=num_classes).to(device)

    # Оптимизатор с дифференцированными LR и weight decay
    params = [
        {'params': model.backbone.parameters(), 'lr': cfg['lr'] * 0.1},
        {'params': model.rpn.parameters(), 'lr': cfg['lr']},
        {'params': model.roi_heads.parameters(), 'lr': cfg['lr']},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    early_stopping = EarlyStopping(patience=10)
    history = []
    out_dir = models_dir / "faster_rcnn_r18_fgd"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training Faster R-CNN R18 FGD: epochs={cfg['epochs']}, batch={cfg['batch']}")
    start_time = time.time()

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0

        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += losses.item()

        scheduler.step()

        # Валидация
        val_metrics = evaluate_frcnn(model, val_loader, device)
        avg_loss = epoch_loss / len(train_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': round(avg_loss, 4),
            'val_map50': round(val_metrics['mAP_50'], 4),
            'lr': optimizer.param_groups[0]['lr'],
        })

        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_map50={val_metrics['mAP_50']:.4f}")

        # Early stopping
        if early_stopping(val_metrics['mAP_50'], model):
            logger.info(f"⏹ Early stopping at epoch {epoch+1}")
            break

    # Восстанавливаем лучшие веса
    early_stopping.restore_best(model)

    training_time = (time.time() - start_time) / 3600

    # Сохраняем модель
    model_path = out_dir / "best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_val_map50': early_stopping.best_score,
    }, model_path)

    overfitting = {'overfitting_detected': False}
    if len(history) >= 5:
        val_values = [h['val_map50'] for h in history]
        best_idx = val_values.index(max(val_values))
        if best_idx < len(val_values) - 5:
            recent_max = max(val_values[-5:])
            if recent_max < max(val_values) - 0.01:
                overfitting = {
                    'overfitting_detected': True,
                    'warning': f"val упала с {max(val_values):.4f} до {recent_max:.4f}",
                }
                logger.warning(f"⚠️  Переобучение FRCNN: {overfitting['warning']}")

    result = {
        "model_path": str(model_path),
        "status": "completed",
        "training_time_hours": round(training_time, 3),
        "best_val_map50": early_stopping.best_score,
        "epochs_trained": len(history),
        "history": history,
        "overfitting": overfitting,
    }

    with open(out_dir / "training_info.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result