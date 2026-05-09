#!/usr/bin/env python3
"""
Обучение Faster R-CNN детекторов с разной инициализацией бэкбона

Экспериментальные группы:
  1. faster_rcnn_r18_scratch    - случайная инициализация (контроль)
  2. faster_rcnn_r18_imagenet   - предобучение ImageNet (baseline)
  3. faster_rcnn_r18_distilled  - дистилляция от LTDETR (предложенный метод)

Для каждой группы:
  - Фиксированная архитектура: Faster R-CNN + ResNet18-FPN
  - Одинаковые гиперпараметры обучения
  - Одинаковый датасет: real_plus_synthetic_aug
  - Различается только инициализация бэкбона
"""

import json
import logging
import sys
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet18_fpn,
    FasterRCNN_ResNet18_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('02_train_detectors.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class DefectDetectionDataset(Dataset):
    """
    Датасет для детекции поверхностных дефектов.
    
    Формат аннотаций: YOLO (нормированные координаты)
    Формат изображений: RGB, любой размер (ресайзится до img_size)
    """
    
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        num_classes: int = 4,
        img_size: Tuple[int, int] = (640, 640)
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Собираем все изображения
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.image_files = sorted([
            f for f in self.images_dir.glob("*")
            if f.suffix.lower() in extensions
        ])
        
        if not self.image_files:
            logger.warning(f"No images found in {images_dir}")
        else:
            logger.info(f"Dataset: {len(self.image_files)} images from {images_dir.name}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_files[idx]
        
        # Загрузка изображения
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            image = image.resize(self.img_size, Image.BILINEAR)
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
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
    
    def _load_annotations(
        self, img_path: Path, orig_w: int, orig_h: int
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Загружает аннотации в формате YOLO.
        
        Формат: class_id x_center y_center width height (нормированные)
        Конвертирует в: [x1, y1, x2, y2] (абсолютные, масштабированные к img_size)
        """
        boxes, labels = [], []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            return boxes, labels
        
        # Коэффициенты масштабирования
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        logger.debug(f"Line {line_num} in {label_path.name}: expected 5 values, got {len(parts)}")
                        continue
                    
                    try:
                        cls = int(float(parts[0]))
                        if cls < 0 or cls >= self.num_classes:
                            logger.debug(f"Class {cls} out of range [0, {self.num_classes})")
                            continue
                        
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        # Конвертация из нормированных YOLO в абсолютные пиксели
                        x1 = max(0, (xc - w / 2) * orig_w * scale_x)
                        y1 = max(0, (yc - h / 2) * orig_h * scale_y)
                        x2 = min(self.img_size[1], (xc + w / 2) * orig_w * scale_x)
                        y2 = min(self.img_size[0], (yc + h / 2) * orig_h * scale_y)
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls + 1)  # +1: 0 зарезервирован для фона в Faster R-CNN
                    
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Parse error line {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Failed to read {label_path}: {e}")
        
        return boxes, labels


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """Стандартная collate функция для Faster R-CNN."""
    return tuple(zip(*batch))


# ============================================================================
# МАППИНГ ВЕСОВ БЭКБОНА
# ============================================================================

class BackboneWeightMapper:
    """
    Маппинг весов из чистого ResNet18 → бэкбон Faster R-CNN FPN.
    
    Архитектура Faster R-CNN ResNet18 FPN:
    - model.backbone.body.conv1, .bn1, .relu, .maxpool
    - model.backbone.body.layer1[0,1].conv1, .bn1, .conv2, .bn2
    - model.backbone.body.layer2, .layer3, .layer4
    - model.backbone.fpn.inner_blocks, .layer_blocks
    
    Стандартный ResNet18 (torchvision):
    - conv1, bn1, relu, maxpool
    - layer1, layer2, layer3, layer4
    - fc (не используется в FPN)
    """
    
    @staticmethod
    def load_weights(weights_path: str) -> Dict[str, torch.Tensor]:
        """Загружает веса из различных форматов."""
        
        logger.info(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            # Проверяем известные ключи
            for key in ['model_state_dict', 'state_dict', 'model', 'student_state_dict']:
                if key in checkpoint:
                    logger.info(f"Extracting '{key}' from checkpoint")
                    return checkpoint[key]
            
            # Если ключи не найдены, фильтруем только тензоры
            state_dict = {}
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v
                elif isinstance(v, dict) and '_state_dict' in k:
                    # Рекурсивно извлекаем state_dict
                    for k2, v2 in v.items():
                        if isinstance(v2, torch.Tensor):
                            state_dict[f"{k}.{k2}"] = v2
            
            if state_dict:
                logger.info(f"Extracted {len(state_dict)} tensor keys")
                return state_dict
        
        return checkpoint if isinstance(checkpoint, dict) else {}
    
    @staticmethod
    def map_to_faster_rcnn(
        pretrained_weights: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> Tuple[nn.Module, int, int]:
        """
        Маппит веса ResNet18 в Faster R-CNN FPN.
        
        Returns:
            model: модель с загруженными весами
            mapped_count: количество успешно маппированных весов
            total_count: общее количество параметров в модели
        """
        
        # Очистка префиксов
        cleaned = {}
        for key, value in pretrained_weights.items():
            # Удаляем возможные префиксы
            new_key = key
            for prefix in ['backbone.', 'model.', 'module.', 'student.', 
                          'feature_extractor.', 'encoder.', 'trunk.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            cleaned[new_key] = value
        
        logger.info(f"Cleaned {len(cleaned)} weight keys")
        
        # Получаем state_dict модели
        model_state = model.state_dict()
        mapped_count = 0
        skipped_keys = []
        
        for model_key in model_state.keys():
            if 'backbone.body.' in model_key:
                # Извлекаем имя слоя ResNet18
                resnet_key = model_key.replace('backbone.body.', '')
                
                if resnet_key in cleaned:
                    src = cleaned[resnet_key]
                    dst = model_state[model_key]
                    
                    if src.shape == dst.shape:
                        model_state[model_key] = src
                        mapped_count += 1
                    else:
                        skipped_keys.append(
                            f"Shape mismatch: {model_key} "
                            f"(src: {list(src.shape)}, dst: {list(dst.shape)})"
                        )
                else:
                    skipped_keys.append(f"Key not found: {model_key} -> {resnet_key}")
        
        # Загружаем обновлённый state_dict
        model.load_state_dict(model_state)
        
        logger.info(f"Weight mapping results:")
        logger.info(f"  Mapped: {mapped_count}/{len(model_state)} weights")
        logger.info(f"  Skipped: {len(skipped_keys)} weights")
        
        if skipped_keys and len(skipped_keys) <= 20:
            for s in skipped_keys:
                logger.debug(f"    {s}")
        elif skipped_keys:
            logger.debug(f"  First 10 skipped keys:")
            for s in skipped_keys[:10]:
                logger.debug(f"    {s}")
        
        return model, mapped_count, len(model_state)


# ============================================================================
# ВЫЧИСЛЕНИЕ МЕТРИК
# ============================================================================

class MetricsCalculator:
    """Вычисляет метрики детекции: mAP, Precision, Recall."""
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
    
    def compute_map(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict[str, float]:
        """
        Вычисляет mAP@50 и mAP@75.
        
        Args:
            predictions: список словарей с ключами 'boxes', 'scores', 'labels'
            ground_truths: список словарей с ключами 'boxes', 'labels'
        
        Returns:
            dict с метриками
        """
        
        iou_thresholds = [0.5, 0.75]
        aps_per_threshold = {thr: [] for thr in iou_thresholds}
        
        for class_id in range(self.num_classes):
            for thr in iou_thresholds:
                ap = self._compute_ap_for_class(
                    predictions, ground_truths, class_id, thr
                )
                aps_per_threshold[thr].append(ap)
        
        map50 = float(np.mean(aps_per_threshold[0.5]))
        map75 = float(np.mean(aps_per_threshold[0.75]))
        
        return {
            'mAP_50': map50,
            'mAP_75': map75,
        }
    
    def _compute_ap_for_class(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_id: int,
        iou_threshold: float
    ) -> float:
        """Вычисляет Average Precision для одного класса."""
        
        # Собираем все предсказания для данного класса
        all_detections = []
        num_gt_total = 0
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Ground truth данного класса
            gt_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_mask].numpy()
            num_gt_total += len(gt_boxes)
            
            # Предсказания данного класса
            pred_mask = pred['labels'] == class_id
            pred_boxes = pred['boxes'][pred_mask].numpy()
            pred_scores = pred['scores'][pred_mask].numpy()
            
            for box, score in zip(pred_boxes, pred_scores):
                all_detections.append({
                    'image_id': img_idx,
                    'score': float(score),
                    'bbox': box.tolist(),
                    'matched': False
                })
        
        if num_gt_total == 0:
            return 0.0 if len(all_detections) > 0 else 0.0
        
        if len(all_detections) == 0:
            return 0.0
        
        # Сортируем по уверенности
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Для каждого изображения отслеживаем matched GT
        gt_matched = {}
        for img_idx, gt in enumerate(ground_truths):
            gt_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_mask].numpy()
            gt_matched[img_idx] = [False] * len(gt_boxes)
        
        tp = np.zeros(len(all_detections))
        fp = np.zeros(len(all_detections))
        
        for i, det in enumerate(all_detections):
            img_idx = det['image_id']
            det_box = torch.tensor([det['bbox']], dtype=torch.float32)
            
            # Ищем соответствующий GT
            gt_mask = ground_truths[img_idx]['labels'] == class_id
            gt_boxes = ground_truths[img_idx]['boxes'][gt_mask]
            
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            # Вычисляем IoU
            ious = box_iou(det_box, gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            
            if best_iou >= iou_threshold and not gt_matched[img_idx][best_idx.item()]:
                tp[i] = 1
                gt_matched[img_idx][best_idx.item()] = True
            else:
                fp[i] = 1
        
        # Вычисляем precision-recall кривую
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt_total
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-16)
        
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                ap += np.max(precisions[recalls >= t]) / 11.0
        
        return float(ap)


# ============================================================================
# ТРЕНЕР
# ============================================================================

class FasterRCNNTrainer:
    """
    Тренер для Faster R-CNN с различной инициализацией бэкбона.
    
    Поддерживает:
    - scratch: случайная инициализация
    - imagenet_pretrained: веса ImageNet
    - lightly_pretrained: дистилляция от LTDETR
    """
    
    def __init__(
        self,
        config: dict,
        student_name: str,
        student_config: dict,
        pretrained_backbone_path: Optional[str] = None
    ):
        self.config = config
        self.student_name = student_name
        self.student_config = student_config
        self.pretrained_backbone_path = pretrained_backbone_path
        
        # Устройство
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Создаём модель
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=student_config['lr'],
            weight_decay=student_config.get('weight_decay', 0.0005)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=student_config['epochs'],
            eta_min=1e-6
        )
        
        # Даталоадеры
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Early stopping
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience = student_config.get('patience', 15)
        self.patience_counter = 0
        
        # Метрики
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['detection']['num_classes']
        )
        
        # Выходная директория
        self.output_dir = Path(config['paths']['detection_output']) / student_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        """Создаёт Faster R-CNN с выбранной инициализацией."""
        
        init_type = self.student_config['type']
        backbone_name = self.student_config['backbone']
        num_classes = self.config['detection']['num_classes']
        
        logger.info(f"Creating Faster R-CNN: backbone={backbone_name}, init={init_type}")
        
        if backbone_name == 'resnet18':
            if init_type == 'imagenet_pretrained':
                model = fasterrcnn_resnet18_fpn(
                    weights=FasterRCNN_ResNet18_FPN_Weights.DEFAULT
                )
                logger.info("  Using ImageNet pretrained weights (torchvision)")
                
            elif init_type == 'lightly_pretrained':
                model = fasterrcnn_resnet18_fpn(weights=None)
                
                if self.pretrained_backbone_path and Path(self.pretrained_backbone_path).exists():
                    logger.info(f"  Loading distilled backbone weights...")
                    weights = BackboneWeightMapper.load_weights(self.pretrained_backbone_path)
                    model, mapped, total = BackboneWeightMapper.map_to_faster_rcnn(weights, model)
                    logger.info(f"  Mapped {mapped}/{total} backbone weights")
                else:
                    logger.warning("  ⚠️ No distilled weights found, using random init!")
                
            else:  # scratch
                model = fasterrcnn_resnet18_fpn(weights=None)
                logger.info("  Using random initialization (scratch)")
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Заменяем классификатор
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        # Статистика параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"  Classes: {num_classes} (+1 background)")
        
        return model
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Создаёт DataLoader'ы для обучения и валидации."""
        
        data_path = Path(self.config['detection']['data_path'])
        img_size = tuple(self.config['detection']['img_size'])
        num_classes = self.config['detection']['num_classes']
        
        # Проверяем структуру
        train_imgs = data_path / "train" / "images"
        train_lbls = data_path / "train" / "labels"
        val_imgs = data_path / "val" / "images"
        val_lbls = data_path / "val" / "labels"
        
        for dir_path, name in [
            (train_imgs, "train images"),
            (val_imgs, "val images")
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(f"{name} not found: {dir_path}")
        
        # Датасеты
        train_dataset = DefectDetectionDataset(
            train_imgs, train_lbls, num_classes, img_size
        )
        val_dataset = DefectDetectionDataset(
            val_imgs, val_lbls, num_classes, img_size
        )
        
        # DataLoader'ы
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.student_config['batch'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.student_config['batch'],
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        logger.info(f"DataLoaders created:")
        logger.info(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
        logger.info(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def train(self) -> Dict:
        """Основной цикл обучения."""
        
        epochs = self.student_config['epochs']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING: {self.student_name}")
        logger.info(f"Init:     {self.student_config['type']}")
        logger.info(f"Epochs:   {epochs}")
        logger.info(f"Batch:    {self.student_config['batch']}")
        logger.info(f"LR:       {self.student_config['lr']}")
        logger.info(f"Output:   {self.output_dir}")
        logger.info(f"{'='*60}\n")
        
        history = []
        start_time = time.time()
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            # Обучение
            train_loss = self._train_epoch(epoch)
            
            # Валидация
            val_metrics = self._validate()
            
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_map50': val_metrics['mAP_50'],
                'val_map75': val_metrics['mAP_75'],
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            history.append(epoch_info)
            
            # Логирование
            self._log_epoch(epoch_info)
            
            # Сохранение лучшей модели
            if val_metrics['mAP_50'] > self.best_map:
                self.best_map = val_metrics['mAP_50']
                self.best_epoch = epoch
                self.patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                self._save_model('best_model.pth')
                logger.info(f"  ✅ New best! mAP@50 = {self.best_map:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"  ⏹ Early stopping at epoch {epoch} (best: {self.best_epoch})")
                break
            
            self.scheduler.step()
        
        training_time = (time.time() - start_time) / 3600
        
        # Восстанавливаем лучшую модель
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Сохраняем финальную модель
        self._save_model('model_final.pth')
        
        result = {
            'model_name': self.student_name,
            'init_type': self.student_config['type'],
            'backbone': self.student_config['backbone'],
            'best_val_map50': self.best_map,
            'best_epoch': self.best_epoch,
            'epochs_trained': len(history),
            'training_time_hours': round(training_time, 3),
            'model_path': str(self.output_dir / 'model_final.pth'),
            'history': history,
        }
        
        # Сохраняем результаты
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETED: {self.student_name}")
        logger.info(f"Best mAP@50: {self.best_map:.4f} (epoch {self.best_epoch})")
        logger.info(f"Time: {training_time:.2f} hours")
        logger.info(f"Results: {results_path}")
        logger.info(f"{'='*60}\n")
        
        return result
    
    def _train_epoch(self, epoch: int) -> float:
        """Одна эпоха обучения."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += losses.item()
            
            # Прогресс каждые 50 батчей
            if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                logger.debug(
                    f"  Epoch {epoch:3d} [{batch_idx:4d}/{num_batches}] "
                    f"Loss: {losses.item():.4f} ({elapsed:.1f}s)"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _validate(self) -> Dict[str, float]:
        """Валидация модели."""
        
        self.model.eval()
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
                for output, target in zip(outputs, targets):
                    # Фильтруем по confidence
                    keep = output['scores'] > 0.25
                    
                    predictions.append({
                        'boxes': output['boxes'][keep].cpu(),
                        'scores': output['scores'][keep].cpu(),
                        'labels': (output['labels'][keep] - 1).cpu(),  # 0-indexed
                    })
                    
                    ground_truths.append({
                        'boxes': target['boxes'],
                        'labels': (target['labels'] - 1),  # 0-indexed
                    })
        
        # Вычисляем метрики
        metrics = self.metrics_calculator.compute_map(predictions, ground_truths)
        
        return metrics
    
    def _save_model(self, filename: str):
        """Сохраняет модель."""
        save_path = self.output_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
            'model_config': {
                'num_classes': self.config['detection']['num_classes'],
                'backbone': self.student_config['backbone'],
                'init_type': self.student_config['type'],
            }
        }, save_path)
    
    def _log_epoch(self, info: Dict):
        """Логирует информацию об эпохе."""
        
        marker = " ⭐" if info['val_map50'] == self.best_map and info['val_map50'] > 0 else ""
        
        logger.info(
            f"Epoch {info['epoch']:3d} | "
            f"Loss: {info['train_loss']:.4f} | "
            f"mAP@50: {info['val_map50']:.4f} | "
            f"mAP@75: {info['val_map75']:.4f} | "
            f"LR: {info['lr']:.6f}{marker}"
        )


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Запускает обучение всех трёх моделей."""
    
    config_path = Path(__file__).parent / "config_pretrain_comparison.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Students to train: {list(config['students'].keys())}")
    
    # Ищем путь к предобученному бэкбону
    pretrained_path = None
    cache_file = Path(config['paths']['pretrain_output']) / "pretrained_model_path.txt"
    
    if cache_file.exists():
        pretrained_path = cache_file.read_text().strip()
        logger.info(f"Found pretrained backbone path: {pretrained_path}")
    else:
        # Пробуем стандартный путь
        default_path = Path(config['paths']['pretrain_output']) / "resnet18_distilled" / "exported_models" / "exported_last.pt"
        if default_path.exists():
            pretrained_path = str(default_path)
            logger.info(f"Using default pretrained backbone: {pretrained_path}")
        else:
            logger.warning("⚠️ No pretrained backbone found!")
            logger.warning("Models with 'lightly_pretrained' init will use random weights!")
    
    all_results = []
    
    for student_name, student_cfg in config['students'].items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"Training: {student_name}")
        logger.info(f"Description: {student_cfg['description']}")
        logger.info(f"{'#'*60}")
        
        try:
            trainer = FasterRCNNTrainer(
                config=config,
                student_name=student_name,
                student_config=student_cfg,
                pretrained_backbone_path=pretrained_path
            )
            
            result = trainer.train()
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Failed to train {student_name}: {e}", exc_info=True)
    
    # Сохраняем сводные результаты
    summary_path = Path(config['paths']['detection_output']) / "all_results.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Выводим итоговую таблицу
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<35} {'Init':<22} {'mAP@50':<10} {'Epochs':<8} {'Hours':<8}")
    logger.info("-"*80)
    
    for r in sorted(all_results, key=lambda x: x['best_val_map50'], reverse=True):
        logger.info(
            f"{r['model_name']:<35} "
            f"{r['init_type']:<22} "
            f"{r['best_val_map50']:<10.4f} "
            f"{r['epochs_trained']:<8} "
            f"{r['training_time_hours']:<8.2f}"
        )
    
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {summary_path}")
    
    if len(all_results) == 3:
        scratch = next(r for r in all_results if r['init_type'] == 'scratch')
        imagenet = next(r for r in all_results if r['init_type'] == 'imagenet_pretrained')
        distilled = next(r for r in all_results if r['init_type'] == 'lightly_pretrained')
        
        logger.info(f"\n📊 Analysis:")
        logger.info(f"  ImageNet vs Scratch:     +{imagenet['best_val_map50'] - scratch['best_val_map50']:.4f} mAP")
        logger.info(f"  Distilled vs Scratch:    +{distilled['best_val_map50'] - scratch['best_val_map50']:.4f} mAP")
        logger.info(f"  Distilled vs ImageNet:   +{distilled['best_val_map50'] - imagenet['best_val_map50']:.4f} mAP")


if __name__ == "__main__":
    main()