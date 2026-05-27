#!/usr/bin/env python3
"""
Online Knowledge Distillation: DINOv3 LTDETR → Faster R-CNN
Гибридный подход: Feature Distillation + Detection-level Distillation

ВСЕ ИСПРАВЛЕНИЯ:
- Feature extraction из RT-DETR backbone (DINOv3 ViT)
- Detection distillation через forward pass (tuple: scores, boxes, labels)
- Правильная размерность каналов: teacher=224, student=256
- Нормализация bounding boxes для distillation
- KeyError fix: features['0'] вместо features[0]
"""

import json
import logging
import sys
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou
from PIL import Image
from tqdm import tqdm

import lightly_train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('distillation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class DefectDetectionDataset(Dataset):
    """Датасет для детекции поверхностных дефектов."""
    
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
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.image_files = sorted([
            f for f in self.images_dir.glob("*")
            if f.suffix.lower() in extensions
        ])
        
        logger.info(f"Dataset: {len(self.image_files)} images from {images_dir.name}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_files[idx]
        
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
        """Загружает аннотации в формате YOLO."""
        boxes, labels = [], []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            return boxes, labels
        
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
        try:
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
                        
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        x1 = max(0, (xc - w / 2) * orig_w * scale_x)
                        y1 = max(0, (yc - h / 2) * orig_h * scale_y)
                        x2 = min(self.img_size[1], (xc + w / 2) * orig_w * scale_x)
                        y2 = min(self.img_size[0], (yc + h / 2) * orig_h * scale_y)
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls + 1)  # +1 для background
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"Failed to read {label_path}: {e}")
        
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# МЕТРИКИ
# ============================================================================

class MetricsCalculator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def compute_map(self, predictions, ground_truths):
        iou_thresholds = [0.5, 0.75]
        aps = {thr: [] for thr in iou_thresholds}
        
        for class_id in range(self.num_classes):
            for thr in iou_thresholds:
                ap = self._compute_ap(predictions, ground_truths, class_id, thr)
                aps[thr].append(ap)
        
        return {
            'mAP_50': float(np.mean(aps[0.5])),
            'mAP_75': float(np.mean(aps[0.75])),
        }
    
    def _compute_ap(self, predictions, ground_truths, class_id, iou_threshold):
        detections = []
        num_gt = 0
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            gt_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_mask]
            num_gt += len(gt_boxes)
            
            pred_mask = pred['labels'] == class_id
            for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                detections.append({'image_id': img_idx, 'score': float(score), 'bbox': box})
        
        if num_gt == 0 or len(detections) == 0:
            return 0.0
        
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        gt_matched = {}
        for img_idx, gt in enumerate(ground_truths):
            gt_mask = gt['labels'] == class_id
            gt_matched[img_idx] = [False] * gt_mask.sum().item()
        
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        for i, det in enumerate(detections):
            img_idx = det['image_id']
            gt_mask = ground_truths[img_idx]['labels'] == class_id
            gt_boxes = ground_truths[img_idx]['boxes'][gt_mask]
            
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            det_box = det['bbox'].unsqueeze(0) if det['bbox'].dim() == 1 else det['bbox']
            ious = box_iou(det_box, gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            
            if best_iou >= iou_threshold and not gt_matched[img_idx][best_idx.item()]:
                tp[i] = 1
                gt_matched[img_idx][best_idx.item()] = True
            else:
                fp[i] = 1
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / num_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
        
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                ap += np.max(precisions[recalls >= t]) / 11.0
        return float(ap)


# ============================================================================
# DISTILLATION LOSSES
# ============================================================================

class FeatureDistillationLoss(nn.Module):
    """Дистилляция на уровне признаков."""
    
    def __init__(self, student_channels: int, teacher_channels: int, 
                 temperature: float = 3.0, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        # Конвертируем teacher_channels → student_channels
        self.adaptation = nn.Conv2d(teacher_channels, student_channels, kernel_size=1)
        
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        # Адаптируем учителя под размерность ученика
        teacher_adapted = self.adaptation(teacher_features)
        
        # Приводим к одинаковому пространственному размеру
        if student_features.shape[-2:] != teacher_adapted.shape[-2:]:
            student_features = F.interpolate(
                student_features,
                size=teacher_adapted.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=1)
            teacher_adapted = F.normalize(teacher_adapted, p=2, dim=1)
        
        loss = F.mse_loss(
            student_features / self.temperature,
            teacher_adapted / self.temperature,
        )
        
        return loss * (self.temperature ** 2)


class DetectionDistillationLoss(nn.Module):
    """Дистилляция на уровне детекций."""
    
    def __init__(self, temperature: float = 4.0, iou_threshold: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        
    def forward(self, student_predictions, teacher_predictions) -> torch.Tensor:
        total_loss = torch.tensor(0.0)
        
        for s_pred, t_pred in zip(student_predictions, teacher_predictions):
            if len(s_pred['boxes']) == 0 or len(t_pred['boxes']) == 0:
                continue
            
            iou_matrix = box_iou(s_pred['boxes'], t_pred['boxes'])
            best_iou, best_match = iou_matrix.max(dim=1)
            valid_matches = best_iou > self.iou_threshold
            
            if valid_matches.sum() == 0:
                continue
            
            valid_indices = valid_matches.nonzero().squeeze(1)
            
            box_loss = F.smooth_l1_loss(
                s_pred['boxes'][valid_indices],
                t_pred['boxes'][best_match[valid_indices]],
            )
            
            if 'scores' in s_pred and 'scores' in t_pred and len(s_pred['scores']) > 0:
                t_soft = F.softmax(
                    t_pred['scores'][best_match[valid_indices]] / self.temperature,
                    dim=-1
                )
                s_soft = F.log_softmax(
                    s_pred['scores'][valid_indices] / self.temperature,
                    dim=-1
                )
                cls_loss = F.kl_div(s_soft, t_soft, reduction='batchmean')
                total_loss = total_loss + box_loss + cls_loss * (self.temperature ** 2)
            else:
                total_loss = total_loss + box_loss
        
        return total_loss


# ============================================================================
# ONLINE DISTILLATION TRAINER
# ============================================================================

class OnlineDistillationTrainer:
    """Тренер для online дистилляции с RT-DETR учителем."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Загружаем учителя
        self.teacher = self._load_teacher()
        
        # Создаем ученика
        self.student = self._create_student()
        self.student.to(self.device)
        
        # Инициализируем distillation лоссы
        dist_cfg = config['online_distillation']
        self.feature_loss = FeatureDistillationLoss(
            student_channels=256,  # ResNet-18 FPN output channels
            teacher_channels=224,  # DINOv3 ViT-S features (ИСПРАВЛЕНО!)
            temperature=dist_cfg['feature_distill']['temperature'],
            normalize=dist_cfg['feature_distill']['normalize_features'],
        ).to(self.device)
        
        self.det_distill_loss = DetectionDistillationLoss(
            temperature=dist_cfg['detection_distill']['temperature'],
            iou_threshold=dist_cfg['detection_distill']['iou_threshold'],
        )
        
        # Оптимизатор и планировщик
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=dist_cfg['lr'],
            weight_decay=dist_cfg.get('weight_decay', 0.0005)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=dist_cfg['epochs'], eta_min=1e-6
        )
        
        # Даталоадеры
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Метрики
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['detection']['num_classes']
        )
        
        # Отслеживание лучшей модели
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience = dist_cfg.get('patience', 20)
        self.patience_counter = 0
        
        # Пути сохранения
        self.output_dir = Path(config['paths']['detection_output'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Счетчики
        self.feat_success = 0
        self.feat_errors = 0
        self.det_success = 0
        self.det_errors = 0
    
    def _load_teacher(self):
        """Загружает и замораживает учителя."""
        teacher_cfg = self.config['teacher']
        
        logger.info(f"Loading teacher: {teacher_cfg['model']}")
        teacher = lightly_train.load_model(teacher_cfg['weights'])
        teacher.to(self.device)
        
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        
        logger.info(f"Teacher structure:")
        logger.info(f"  backbone: {type(teacher.backbone).__name__}")
        logger.info(f"  encoder: {type(teacher.encoder).__name__}")
        logger.info(f"  decoder: {type(teacher.decoder).__name__}")
        
        return teacher
    
    def _create_student(self) -> nn.Module:
        """Создает Faster R-CNN с ResNet-18 FPN (ImageNet pretrained)."""
        num_classes = self.config['detection']['num_classes']
        
        # Используем pretrained=True для ImageNet весов
        backbone = resnet_fpn_backbone('resnet18', pretrained=True)
        model = FasterRCNN(backbone, num_classes=num_classes + 1)
        
        logger.info(f"Student: Faster R-CNN + ResNet-18 (ImageNet pretrained), {num_classes} classes")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    
    def _extract_teacher_features(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Извлекает признаки из бэкбона DINOv3 ViT."""
        with torch.no_grad():
            try:
                backbone_features = self.teacher.backbone(images)
                
                if isinstance(backbone_features, dict):
                    features = list(backbone_features.values())[-1]
                elif isinstance(backbone_features, (list, tuple)):
                    features = backbone_features[-1]
                else:
                    features = backbone_features
                
                # ViT возвращает [B, N, C] -> [B, C, H, W]
                if features.dim() == 3:
                    B, N, C = features.shape
                    if N == 1601:  # 40x40 + CLS token
                        features = features[:, 1:, :]
                        N = 1600
                    H = W = int(N ** 0.5)
                    features = features.permute(0, 2, 1).reshape(B, C, H, W)
                
                return features
                
            except Exception as e:
                logger.warning(f"Backbone feature extraction failed: {e}")
                return None
    
    def _extract_student_features(self, images: torch.Tensor) -> torch.Tensor:
        """Извлекает признаки из FPN студента."""
        features = self.student.backbone(images)
        
        if isinstance(features, dict):
            features = features['0']  # ИСПРАВЛЕНО: строковой ключ
        elif isinstance(features, (list, tuple)):
            features = features[0]
        
        return features
    
    def _create_dataloaders(self):
        """Создает даталоадеры."""
        data_path = Path(self.config['detection']['data_path'])
        img_size = tuple(self.config['detection']['img_size'])
        num_classes = self.config['detection']['num_classes']
        batch_size = self.config['online_distillation']['batch_size']
        
        train_imgs = data_path / "train" / "images"
        train_lbls = data_path / "train" / "labels"
        val_imgs = data_path / "val" / "images"
        val_lbls = data_path / "val" / "labels"
        
        train_dataset = DefectDetectionDataset(train_imgs, train_lbls, num_classes, img_size)
        val_dataset = DefectDetectionDataset(val_imgs, val_lbls, num_classes, img_size)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, collate_fn=collate_fn, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=2, collate_fn=collate_fn
        )
        
        logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
        return train_loader, val_loader
    
    def _precompute_teacher_predictions(self):
        """Предвычисляет все предсказания учителя один раз."""
        logger.info("Precomputing teacher predictions for all training images...")
        
        self.teacher_preds_cache = {}
        
        self.teacher.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self.train_loader.dataset)), desc="Teacher predictions"):
                img, _ = self.train_loader.dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                
                # Forward pass учителя
                output = self.teacher(img)
                
                if isinstance(output, tuple) and len(output) == 3:
                    scores, boxes, labels = output
                    scores = scores[0].cpu()
                    boxes = boxes[0].cpu()
                    labels = labels[0].cpu()
                    
                    # Фильтруем и сохраняем
                    keep = scores > 0.1
                    self.teacher_preds_cache[idx] = {
                        'boxes': boxes[keep],
                        'labels': labels[keep].long(),
                        'scores': scores[keep],
                    }
                else:
                    self.teacher_preds_cache[idx] = {
                        'boxes': torch.zeros(0, 4),
                        'labels': torch.zeros(0, dtype=torch.long),
                        'scores': torch.zeros(0),
                    }
        
        logger.info(f"✅ Precomputed predictions for {len(self.teacher_preds_cache)} images")
    
    def train(self) -> Dict:
        """Полный цикл обучения."""
        epochs = self.config['online_distillation']['epochs']
        teacher_epochs = self.config['online_distillation']['teacher_epochs']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ONLINE DISTILLATION TRAINING")
        logger.info(f"Epochs: {epochs} (teacher helps first {teacher_epochs})")
        logger.info(f"{'='*60}\n")
        
        # ПРЕДВЫЧИСЛЯЕМ ПРЕДСКАЗАНИЯ УЧИТЕЛЯ ОДИН РАЗ
        self._precompute_teacher_predictions()
        
        # ТЕСТ: Проверяем извлечение признаков
        logger.info("Testing teacher feature extraction...")
        test_images, _ = next(iter(self.train_loader))
        test_batch = torch.stack([img.to(self.device) for img in test_images[:2]])
        
        teacher_feats = self._extract_teacher_features(test_batch)
        if teacher_feats is not None:
            logger.info(f"✅ Teacher features: {teacher_feats.shape}")
        else:
            logger.error("❌ Cannot extract teacher features!")
        
        history = []
        start_time = time.time()
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            use_teacher = epoch <= teacher_epochs
            
            self.feat_success = 0
            self.feat_errors = 0
            self.det_success = 0
            self.det_errors = 0
            
            train_metrics = self._train_epoch(epoch, use_teacher)
            val_metrics = self._validate()
            
            epoch_info = {
                'epoch': epoch,
                **train_metrics,
                **val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
                'teacher_active': use_teacher,
                'feat_success_rate': f"{self.feat_success}/{self.feat_success + self.feat_errors}",
                'det_success_rate': f"{self.det_success}/{self.det_success + self.det_errors}",
            }
            history.append(epoch_info)
            
            marker = " ⭐" if val_metrics['mAP_50'] > self.best_map else ""
            logger.info(
                f"Epoch {epoch:3d} {'👨‍🏫' if use_teacher else '🎓'} | "
                f"Loss: {train_metrics['total_loss']:.4f} | "
                f"Feat: {train_metrics['feat_loss']:.4f} (✓{self.feat_success}) | "
                f"Dist: {train_metrics['dist_loss']:.4f} (✓{self.det_success}) | "
                f"mAP@50: {val_metrics['mAP_50']:.4f}{marker}"
            )
            
            if val_metrics['mAP_50'] > self.best_map:
                self.best_map = val_metrics['mAP_50']
                self.best_epoch = epoch
                self.patience_counter = 0
                best_model_state = copy.deepcopy(self.student.state_dict())
                self._save_model('best_model.pth', epoch, val_metrics)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            self.scheduler.step()
        
        if best_model_state:
            self.student.load_state_dict(best_model_state)
        self._save_model('model_final.pth', self.best_epoch, {'mAP_50': self.best_map})
        
        # Очищаем кэш
        del self.teacher_preds_cache
        
        # Сохраняем историю
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump({'history': history}, f, indent=2)
        
        training_time = round((time.time() - start_time) / 3600, 3)
        logger.info(f"\nBest mAP@50: {self.best_map:.4f} (epoch {self.best_epoch})")
        
        return {'best_val_map50': self.best_map, 'best_epoch': self.best_epoch,
                'training_time_hours': training_time}
    
    def _train_epoch(self, epoch: int, use_teacher: bool) -> Dict[str, float]:
        """Одна эпоха обучения (использует предвычисленные предсказания)."""
        self.student.train()
        
        cfg = self.config['online_distillation']
        img_size = self.config['detection']['img_size'][0]
        
        total_loss_sum = 0.0
        det_loss_sum = 0.0
        feat_loss_sum = 0.0
        dist_loss_sum = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 1. Detection loss (GT)
            loss_dict = self.student(images, targets)
            det_loss = sum(loss for loss in loss_dict.values())
            total_loss = det_loss * cfg['detection_weight']
            
            feat_loss = torch.tensor(0.0).to(self.device)
            distill_loss = torch.tensor(0.0).to(self.device)
            
            # 2. Дистилляция
            if use_teacher:
                # Feature distillation
                try:
                    images_tensor = torch.stack(images)
                    teacher_feats = self._extract_teacher_features(images_tensor)
                    
                    if teacher_feats is not None:
                        student_feats = self._extract_student_features(images_tensor)
                        feat_loss = self.feature_loss(student_feats, teacher_feats)
                        total_loss = total_loss + cfg['feature_distill_weight'] * feat_loss
                        self.feat_success += 1
                    else:
                        self.feat_errors += 1
                except Exception as e:
                    self.feat_errors += 1
                    if self.feat_errors == 1:
                        logger.warning(f"Feature distillation error: {e}")
                
                # Detection distillation (ИСПОЛЬЗУЕМ КЭШ)
                try:
                    # Получаем индексы изображений в батче
                    start_idx = batch_idx * cfg['batch_size']
                    
                    teacher_preds = []
                    for i in range(len(images)):
                        cache_idx = start_idx + i
                        if cache_idx in self.teacher_preds_cache:
                            # Копируем на GPU
                            cached = self.teacher_preds_cache[cache_idx]
                            teacher_preds.append({
                                'boxes': cached['boxes'].to(self.device),
                                'labels': cached['labels'].to(self.device),
                                'scores': cached['scores'].to(self.device),
                            })
                        else:
                            teacher_preds.append({
                                'boxes': torch.zeros(0, 4).to(self.device),
                                'labels': torch.zeros(0, dtype=torch.long).to(self.device),
                                'scores': torch.zeros(0).to(self.device),
                            })
                    
                    # Предсказания ученика
                    self.student.eval()
                    with torch.no_grad():
                        student_preds_raw = self.student(images)
                    self.student.train()
                    
                    student_preds = []
                    for pred in student_preds_raw:
                        student_preds.append({
                            'boxes': pred['boxes'],
                            'labels': pred['labels'] - 1,
                            'scores': pred['scores'],
                        })
                    
                    # Нормализация боксов
                    for tp in teacher_preds:
                        if len(tp['boxes']) > 0:
                            tp['boxes'] = tp['boxes'] / img_size
                    for sp in student_preds:
                        if len(sp['boxes']) > 0:
                            sp['boxes'] = sp['boxes'] / img_size
                    
                    distill_loss = self.det_distill_loss(student_preds, teacher_preds)
                    total_loss = total_loss + cfg['detection_distill_weight'] * distill_loss
                    self.det_success += 1
                    
                except Exception as e:
                    self.det_errors += 1
                    if self.det_errors == 1:
                        logger.warning(f"Detection distillation error: {e}")
            
            # 3. Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # 4. Аккумулируем лоссы
            total_loss_sum += total_loss.item()
            det_loss_sum += det_loss.item()
            feat_loss_sum += feat_loss.item() if torch.is_tensor(feat_loss) else feat_loss
            dist_loss_sum += distill_loss.item() if torch.is_tensor(distill_loss) else distill_loss
            
            pbar.set_postfix({
                'total': f'{total_loss.item():.3f}',
                'feat': f'{feat_loss.item() if torch.is_tensor(feat_loss) else feat_loss:.3f}',
                'dist': f'{distill_loss.item() if torch.is_tensor(distill_loss) else distill_loss:.3f}',
            })
        
        n = len(self.train_loader)
        return {
            'total_loss': total_loss_sum / n,
            'det_loss': det_loss_sum / n,
            'feat_loss': feat_loss_sum / n,
            'dist_loss': dist_loss_sum / n,
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Валидация."""
        self.student.eval()
        predictions = []
        ground_truths = []
        
        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            outputs = self.student(images)
            
            for output, target in zip(outputs, targets):
                keep = output['scores'] > 0.25
                predictions.append({
                    'boxes': output['boxes'][keep].cpu(),
                    'scores': output['scores'][keep].cpu(),
                    'labels': (output['labels'][keep] - 1).cpu(),
                })
                ground_truths.append({
                    'boxes': target['boxes'],
                    'labels': (target['labels'] - 1),
                })
        
        return self.metrics_calculator.compute_map(predictions, ground_truths)
    
    def _save_model(self, filename: str, epoch: int, metrics: Dict):
        """Сохраняет модель."""
        save_path = self.output_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'best_map': self.best_map,
            'metrics': metrics,
        }, save_path)
        logger.info(f"  Model saved: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_online_distillation.yaml')
    parser.add_argument('--device', type=str, default='0')
    
    args = parser.parse_args()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    trainer = OnlineDistillationTrainer(config)
    result = trainer.train()
    
    logger.info(f"\nFinal: mAP@50 = {result['best_val_map50']:.4f}")


if __name__ == "__main__":
    main()