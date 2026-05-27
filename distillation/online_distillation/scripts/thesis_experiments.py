#!/usr/bin/env python3
"""
Distillation Pipeline для магистерской работы
DINOv3 LTDETR → Faster R-CNN

СТРУКТУРА ЭКСПЕРИМЕНТА:
1. Baseline: Faster R-CNN + ResNet-18 (ImageNet)
2. SSL Distillation: Faster R-CNN + distilled ResNet-18
3. SSL + Pseudo-labels: + псевдоразметка от учителя
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou, nms
from PIL import Image
from tqdm import tqdm

import lightly_train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ
# ============================================================================

EXPERIMENTS = {
    'baseline': {
        'use_ssl_backbone': False,
        'use_pseudo_labels': False,
        'description': 'Faster R-CNN + ResNet-18 (ImageNet pretrained)'
    },
    'ssl_distilled': {
        'use_ssl_backbone': True,
        'use_pseudo_labels': False,
        'description': 'Faster R-CNN + SSL-distilled ResNet-18'
    },
    'ssl_pseudo': {
        'use_ssl_backbone': True,
        'use_pseudo_labels': True,
        'description': 'SSL-distilled + pseudo-labels from LTDETR'
    }
}


# ============================================================================
# STAGE 1: SSL DISTILLATION (один раз для всех экспериментов)
# ============================================================================

def pretrain_backbone_ssl(config: dict) -> Path:
    """
    Дистилляция ResNet-18 бэкбона через LightlyTrain.
    DINOv3 ViT-S (учитель) → ResNet-18 (ученик).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"SSL DISTILLATION: DINOv3 → ResNet-18")
    logger.info(f"{'='*60}")
    
    ssl_cfg = config['ssl_pretraining']
    output_dir = Path(config['paths']['ssl_output'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Используем все изображения без разметки
    data_path = Path(config['data']['path'])
    train_images = data_path / "train" / "images"
    val_images = data_path / "val" / "images"
    
    logger.info(f"Data: {train_images}, {val_images}")
    logger.info(f"Method: {ssl_cfg['method']}")
    logger.info(f"Teacher: {ssl_cfg['teacher']}")
    logger.info(f"Epochs: {ssl_cfg['epochs']}")
    
    lightly_train.pretrain(
        out=str(output_dir / "resnet18_distilled"),
        data=[str(train_images), str(val_images)],
        model=ssl_cfg['model'],
        method=ssl_cfg['method'],
        method_args={"teacher": ssl_cfg['teacher']},
        epochs=ssl_cfg['epochs'],
        batch_size=ssl_cfg['batch_size'],
        precision='16-mixed',
        seed=config['seed'],
        overwrite=True,
    )
    
    backbone_path = output_dir / "resnet18_distilled" / "exported_models" / "exported_last.pt"
    
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")
    
    logger.info(f"✅ Distilled backbone: {backbone_path}")
    return backbone_path


# ============================================================================
# STAGE 2: PSEUDO-LABEL GENERATION (SAHI для мелких дефектов)
# ============================================================================

def generate_pseudo_labels(config: dict) -> Path:
    """
    Генерация псевдоразметки учителем LTDETR.
    Использует SAHI для улучшения детекции мелких объектов.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PSEUDO-LABEL GENERATION (SAHI)")
    logger.info(f"{'='*60}")
    
    teacher_cfg = config['teacher']
    pseudo_cfg = config['pseudo_labels']
    
    # Загружаем учителя
    logger.info(f"Loading teacher: {teacher_cfg['model']}")
    teacher = lightly_train.load_model(teacher_cfg['weights'])
    
    # Входные изображения
    data_path = Path(config['data']['path'])
    train_images_dir = data_path / "train" / "images"
    
    # Выходная директория
    pseudo_dir = Path(config['paths']['pseudo_labels_output'])
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    pseudo_labels_dir = pseudo_dir / "labels"
    pseudo_labels_dir.mkdir(exist_ok=True)
    
    image_files = sorted(
        list(train_images_dir.glob("*.jpg")) + 
        list(train_images_dir.glob("*.png"))
    )
    
    logger.info(f"Images: {len(image_files)}")
    logger.info(f"Confidence threshold: {pseudo_cfg['confidence_threshold']}")
    
    pseudo_count = 0
    total_boxes = 0
    
    for img_path in tqdm(image_files, desc="Generating pseudo-labels"):
        try:
            # SAHI predict для мелких дефектов
            results = teacher.predict_sahi(
                image=str(img_path),
                overlap=pseudo_cfg['sahi_overlap'],
                threshold=pseudo_cfg['confidence_threshold'],
            )
            
            img = Image.open(img_path)
            img_w, img_h = img.size
            
            yolo_lines = []
            
            for box, label, score in zip(
                results['bboxes'], results['labels'], results['scores']
            ):
                score_val = score.item()
                
                if score_val >= pseudo_cfg['confidence_threshold']:
                    x1, y1, x2, y2 = box.tolist()
                    cls_id = label.item()
                    
                    # Конвертация в YOLO формат
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    
                    yolo_lines.append(
                        f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score_val:.4f}"
                    )
                    total_boxes += 1
            
            if yolo_lines:
                label_path = pseudo_labels_dir / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                pseudo_count += 1
                
        except Exception as e:
            logger.warning(f"Failed {img_path.name}: {e}")
    
    logger.info(f"✅ Generated: {pseudo_count} images, {total_boxes} boxes")
    return pseudo_dir


# ============================================================================
# ДАТАСЕТ (с поддержкой псевдоразметки)
# ============================================================================

class DefectDataset(Dataset):
    """
    Датасет детекции дефектов.
    Поддерживает GT разметку и опциональную псевдоразметку.
    """
    
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        pseudo_dir: Optional[Path] = None,
        num_classes: int = 4,
        img_size: Tuple[int, int] = (640, 640),
        pseudo_conf_threshold: float = 0.6,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.pseudo_dir = Path(pseudo_dir) if pseudo_dir else None
        self.num_classes = num_classes
        self.img_size = img_size
        self.pseudo_conf_threshold = pseudo_conf_threshold
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = sorted([
            f for f in self.images_dir.glob("*")
            if f.suffix.lower() in extensions
        ])
        
        logger.info(f"  {len(self.image_files)} images")
        if pseudo_dir:
            logger.info(f"  + pseudo-labels (conf ≥ {pseudo_conf_threshold})")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Загрузка изображения
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            image = image.resize(self.img_size, Image.BILINEAR)
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except Exception as e:
            logger.error(f"Failed to load {img_path}: {e}")
            img_tensor = torch.zeros(3, *self.img_size)
            orig_w, orig_h = self.img_size
        
        # Загрузка аннотаций
        boxes, labels = self._load_annotations(img_path, orig_w, orig_h)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(
                [(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32
            ) if boxes else torch.zeros(0, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes) if boxes else 0, dtype=torch.int64),
        }
        
        return img_tensor, target
    
    def _load_annotations(self, img_path: Path, orig_w: int, orig_h: int):
        """Загружает GT + опционально псевдоразметку с фильтрацией."""
        all_boxes = []
        all_labels = []
        
        # 1. Загружаем GT разметку
        gt_boxes, gt_labels = self._parse_yolo(
            self.labels_dir / f"{img_path.stem}.txt",
            orig_w, orig_h
        )
        all_boxes.extend(gt_boxes)
        all_labels.extend(gt_labels)
        
        # 2. Загружаем псевдоразметку (если есть)
        if self.pseudo_dir:
            pseudo_path = self.pseudo_dir / f"{img_path.stem}.txt"
            if pseudo_path.exists():
                pseudo_boxes, pseudo_labels = self._parse_yolo(
                    pseudo_path,
                    orig_w, orig_h,
                    is_pseudo=True
                )
                
                # Фильтрация: убираем псевдо-боксы, которые дублируют GT
                if len(gt_boxes) > 0 and len(pseudo_boxes) > 0:
                    gt_tensor = torch.tensor(gt_boxes)
                    pseudo_tensor = torch.tensor(pseudo_boxes)
                    
                    ious = box_iou(pseudo_tensor, gt_tensor)
                    max_ious, _ = ious.max(dim=1)
                    keep = max_ious < 0.5  # Порог дублирования
                    
                    pseudo_boxes = [b for i, b in enumerate(pseudo_boxes) if keep[i]]
                    pseudo_labels = [l for i, l in enumerate(pseudo_labels) if keep[i]]
                
                all_boxes.extend(pseudo_boxes)
                all_labels.extend(pseudo_labels)
        
        return all_boxes, all_labels
    
    def _parse_yolo(self, label_path: Path, orig_w: int, orig_h: int, is_pseudo: bool = False):
        """Парсит YOLO аннотации."""
        boxes, labels = [], []
        
        if not label_path.exists():
            return boxes, labels
        
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    cls = int(float(parts[0]))
                    if cls < 0 or cls >= self.num_classes:
                        continue
                    
                    # Фильтрация псевдоразметки по confidence
                    if is_pseudo and len(parts) >= 6:
                        conf = float(parts[5])
                        if conf < self.pseudo_conf_threshold:
                            continue
                    
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # Конвертация в пиксели
                    x1 = max(0, (xc - w / 2) * orig_w * scale_x)
                    y1 = max(0, (yc - h / 2) * orig_h * scale_y)
                    x2 = min(self.img_size[1], (xc + w / 2) * orig_w * scale_x)
                    y2 = min(self.img_size[0], (yc + h / 2) * orig_h * scale_y)
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)  # 0 = background
                        
                except (ValueError, IndexError):
                    continue
        
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================================

def create_model(backbone_type: str, num_classes: int, ssl_path: Optional[Path] = None) -> nn.Module:
    """
    Создаёт Faster R-CNN модель.
    
    Args:
        backbone_type: 'imagenet' или 'ssl_distilled'
        num_classes: количество классов (без background)
        ssl_path: путь к дистиллированному бэкбону (для 'ssl_distilled')
    """
    if backbone_type == 'ssl_distilled' and ssl_path and ssl_path.exists():
        # Загружаем дистиллированный бэкбон
        logger.info(f"Loading SSL-distilled backbone: {ssl_path}")
        
        # Создаём чистый ResNet-18
        base_model = resnet18(weights=None)
        
        # Загружаем дистиллированные веса
        checkpoint = torch.load(ssl_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            weights = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
        else:
            weights = checkpoint
        
        # Очищаем ключи от префиксов
        cleaned_weights = {}
        for k, v in weights.items():
            clean_k = k
            for prefix in ['backbone.', 'model.', 'module.', 'encoder.']:
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cleaned_weights[clean_k] = v
        
        # Загружаем с проверкой совместимости
        missing, unexpected = base_model.load_state_dict(cleaned_weights, strict=False)
        logger.info(f"  Loaded weights: {len(cleaned_weights) - len(missing)} matched, "
                   f"{len(missing)} missing, {len(unexpected)} unexpected")
        
        # Строим FPN поверх дистиллированного бэкбона
        backbone = _resnet_fpn_extractor(base_model, trainable_layers=5)
        
    else:
        # Стандартный ImageNet бэкбон
        logger.info("Using ImageNet pretrained backbone")
        base_model = resnet18(weights='DEFAULT')
        backbone = _resnet_fpn_extractor(base_model, trainable_layers=5)
    
    # Создаём Faster R-CNN
    model = FasterRCNN(backbone, num_classes=num_classes + 1)
    
    return model


# ============================================================================
# ВЫЧИСЛЕНИЕ МЕТРИК
# ============================================================================

def compute_map(predictions: List[Dict], ground_truths: List[Dict], num_classes: int,
                iou_thresholds: List[float] = [0.5, 0.75]) -> Dict[str, float]:
    """
    Вычисление mAP по VOC метрике (11-point interpolation).
    Для магистерской работы — приемлемо, но лучше использовать COCO метрику.
    """
    results = {}
    
    for iou_thr in iou_thresholds:
        aps = []
        
        for cls_id in range(num_classes):
            # Собираем детекции для класса
            detections = []
            num_gt = 0
            
            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Ground truth для класса
                gt_mask = gt['labels'] == cls_id
                num_gt += gt_mask.sum().item()
                
                # Предсказания для класса
                pred_mask = pred['labels'] == cls_id
                for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                    detections.append({
                        'image_id': img_idx,
                        'score': score.item(),
                        'bbox': box
                    })
            
            if num_gt == 0:
                continue
            
            # Сортируем по confidence
            detections.sort(key=lambda x: x['score'], reverse=True)
            
            # Метчинг с GT
            gt_matched = {}
            for img_idx, gt in enumerate(ground_truths):
                gt_mask = gt['labels'] == cls_id
                gt_matched[img_idx] = [False] * gt_mask.sum().item()
            
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            
            for i, det in enumerate(detections):
                img_idx = det['image_id']
                gt_mask = ground_truths[img_idx]['labels'] == cls_id
                gt_boxes = ground_truths[img_idx]['boxes'][gt_mask]
                
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                
                # IoU с GT боксами
                ious = box_iou(det['bbox'].unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(0)
                
                if best_iou >= iou_thr and not gt_matched[img_idx][best_idx.item()]:
                    tp[i] = 1
                    gt_matched[img_idx][best_idx.item()] = True
                else:
                    fp[i] = 1
            
            # 11-point interpolation
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            
            recalls = tp_cum / max(num_gt, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
            
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                if np.any(recalls >= t):
                    ap += np.max(precisions[recalls >= t]) / 11.0
            
            aps.append(ap)
        
        results[f'mAP@{int(iou_thr*100)}'] = float(np.mean(aps)) if aps else 0.0
    
    return results


# ============================================================================
# ОБУЧЕНИЕ ДЕТЕКТОРА
# ============================================================================

def train_detector(
    config: dict,
    experiment_name: str,
    backbone_type: str,
    ssl_path: Optional[Path] = None,
    pseudo_dir: Optional[Path] = None,
) -> Dict:
    """
    Обучение Faster R-CNN для одного эксперимента.
    
    Returns:
        dict с результатами эксперимента
    """
    exp_cfg = EXPERIMENTS[experiment_name]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"Description: {exp_cfg['description']}")
    logger.info(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_cfg = config['training']
    data_cfg = config['data']
    num_classes = data_cfg['num_classes']
    img_size = tuple(data_cfg['img_size'])
    
    # Создаём модель
    model = create_model(backbone_type, num_classes, ssl_path)
    model.to(device)
    
    # Даталоадеры
    data_path = Path(data_cfg['path'])
    
    train_dataset = DefectDataset(
        images_dir=data_path / "train" / "images",
        labels_dir=data_path / "train" / "labels",
        pseudo_dir=pseudo_dir / "labels" if pseudo_dir else None,
        num_classes=num_classes,
        img_size=img_size,
        pseudo_conf_threshold=config['pseudo_labels']['confidence_threshold'],
    )
    
    val_dataset = DefectDataset(
        images_dir=data_path / "val" / "images",
        labels_dir=data_path / "val" / "labels",
        pseudo_dir=None,  # Валидация только на GT
        num_classes=num_classes,
        img_size=img_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'],
        eta_min=1e-6,
    )
    
    # Сохранение результатов
    output_dir = Path(config['paths']['experiments_output']) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Обучение
    best_map50 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                for output, target in zip(outputs, targets):
                    # Фильтрация по confidence
                    keep = output['scores'] > 0.25
                    
                    predictions.append({
                        'boxes': output['boxes'][keep].cpu(),
                        'scores': output['scores'][keep].cpu(),
                        'labels': (output['labels'][keep] - 1).cpu(),  # Убираем background
                    })
                    
                    ground_truths.append({
                        'boxes': target['boxes'].cpu(),
                        'labels': (target['labels'] - 1).cpu(),
                    })
        
        # Вычисление метрик
        metrics = compute_map(predictions, ground_truths, num_classes)
        
        # Логирование
        marker = " ⭐" if metrics['mAP@50'] > best_map50 else ""
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Loss: {avg_loss:.4f} | "
            f"mAP@50: {metrics['mAP@50']:.4f} | "
            f"mAP@75: {metrics['mAP@75']:.4f}{marker}"
        )
        
        # Сохранение истории
        history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            **metrics,
        })
        
        # Сохранение лучшей модели
        if metrics['mAP@50'] > best_map50:
            best_map50 = metrics['mAP@50']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': {
                    'experiment': experiment_name,
                    'backbone_type': backbone_type,
                    'use_pseudo_labels': exp_cfg['use_pseudo_labels'],
                },
            }, output_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= train_cfg['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step()
    
    # Сохранение результатов
    training_time = round((time.time() - start_time) / 3600, 3)
    
    result = {
        'experiment': experiment_name,
        'description': exp_cfg['description'],
        'backbone_type': backbone_type,
        'use_ssl_backbone': exp_cfg['use_ssl_backbone'],
        'use_pseudo_labels': exp_cfg['use_pseudo_labels'],
        'best_map50': best_map50,
        'best_epoch': best_epoch,
        'epochs_trained': len(history),
        'training_time_hours': training_time,
        'history': history,
    }
    
    # Сохраняем JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✅ {experiment_name}: mAP@50 = {best_map50:.4f}")
    
    return result


# ============================================================================
# ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Distillation experiments for thesis')
    parser.add_argument('--config', type=str, default='config_thesis.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--experiments', nargs='+', 
                       default=['baseline', 'ssl_distilled', 'ssl_pseudo'],
                       help='Which experiments to run')
    
    args = parser.parse_args()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    # Загружаем конфиг
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"DISTILLATION EXPERIMENTS FOR THESIS")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"{'#'*60}")
    
    # Stage 1: SSL pretraining (если нужно)
    ssl_path = None
    if any(exp in args.experiments for exp in ['ssl_distilled', 'ssl_pseudo']):
        ssl_output = Path(config['paths']['ssl_output']) / "resnet18_distilled" / "exported_models" / "exported_last.pt"
        
        if ssl_output.exists():
            logger.info(f"SSL backbone already exists: {ssl_output}")
            ssl_path = ssl_output
        else:
            logger.info("Running SSL pretraining...")
            ssl_path = pretrain_backbone_ssl(config)
    
    # Stage 2: Pseudo-labels (если нужно)
    pseudo_dir = None
    if 'ssl_pseudo' in args.experiments:
        pseudo_output = Path(config['paths']['pseudo_labels_output']) / "labels"
        
        if pseudo_output.exists() and len(list(pseudo_output.glob("*.txt"))) > 0:
            logger.info(f"Pseudo-labels already exist: {pseudo_output}")
            pseudo_dir = Path(config['paths']['pseudo_labels_output'])
        else:
            logger.info("Generating pseudo-labels...")
            pseudo_dir = generate_pseudo_labels(config)
    
    # Stage 3: Run experiments
    all_results = []
    
    for exp_name in args.experiments:
        exp_cfg = EXPERIMENTS[exp_name]
        
        # Определяем тип бэкбона
        backbone_type = 'ssl_distilled' if exp_cfg['use_ssl_backbone'] else 'imagenet'
        
        # Определяем использование псевдоразметки
        use_pseudo = pseudo_dir if exp_cfg['use_pseudo_labels'] else None
        
        # Запускаем эксперимент
        result = train_detector(
            config=config,
            experiment_name=exp_name,
            backbone_type=backbone_type,
            ssl_path=ssl_path if exp_cfg['use_ssl_backbone'] else None,
            pseudo_dir=use_pseudo,
        )
        
        all_results.append(result)
    
    # Сохраняем сводную таблицу
    summary = {
        'experiments': all_results,
        'config': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    summary_path = Path(config['paths']['experiments_output']) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Выводим таблицу результатов
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"{'Experiment':<25} {'mAP@50':<10} {'mAP@75':<10} {'Epochs':<10} {'Time (h)':<10}")
    logger.info(f"{'-'*80}")
    
    for r in all_results:
        logger.info(
            f"{r['experiment']:<25} "
            f"{r['best_map50']:<10.4f} "
            f"{r['history'][r['best_epoch']-1]['mAP@75']:<10.4f} "
            f"{r['best_epoch']:<10} "
            f"{r['training_time_hours']:<10.2f}"
        )
    
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()