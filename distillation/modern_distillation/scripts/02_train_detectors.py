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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet18
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
        
        if not self.image_files:
            logger.warning(f"No images found in {images_dir}")
        else:
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
                for line_num, line in enumerate(f, 1):
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
                            labels.append(cls + 1)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"Failed to read {label_path}: {e}")
        
        return boxes, labels


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    return tuple(zip(*batch))


# ============================================================================
# МЕТРИКИ
# ============================================================================

class MetricsCalculator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def compute_map(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
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
# ТРЕНЕР
# ============================================================================

class FasterRCNNTrainer:
    def __init__(self, config: dict, student_name: str, student_config: dict, pretrained_backbone_path: Optional[str] = None):
        self.config = config
        self.student_name = student_name
        self.student_config = student_config
        self.pretrained_backbone_path = pretrained_backbone_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = self._create_model()
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=student_config['lr'],
            weight_decay=student_config.get('weight_decay', 0.0005)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=student_config['epochs'], eta_min=1e-6
        )
        
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience = student_config.get('patience', 15)
        self.patience_counter = 0
        
        self.metrics_calculator = MetricsCalculator(num_classes=config['detection']['num_classes'])
        
        self.output_dir = Path(config['paths']['detection_output']) / student_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        init_type = self.student_config['type']
        backbone_name = self.student_config['backbone']
        num_classes = self.config['detection']['num_classes']
        
        logger.info(f"Creating Faster R-CNN: backbone={backbone_name}, init={init_type}")
        
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models.detection import FasterRCNN
        
        if backbone_name == 'resnet18':
            backbone = resnet_fpn_backbone('resnet18', pretrained=False)
            logger.info("  Backbone создан: ResNet18 + FPN")
            model = FasterRCNN(backbone, num_classes=num_classes + 1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        if init_type == 'imagenet_pretrained':
            pretrained_backbone = resnet_fpn_backbone('resnet18', pretrained=True)
            model.backbone.load_state_dict(pretrained_backbone.state_dict())
            logger.info("  Using ImageNet pretrained weights")
            
        elif init_type == 'lightly_pretrained':
            if self.pretrained_backbone_path and Path(self.pretrained_backbone_path).exists():
                logger.info(f"  Loading distilled backbone weights...")
                checkpoint = torch.load(self.pretrained_backbone_path, map_location='cpu', weights_only=False)
                
                # Извлекаем state_dict
                if 'state_dict' in checkpoint:
                    weights = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    weights = checkpoint['model_state_dict']
                else:
                    weights = checkpoint
                
                # КЛЮЧЕВОЕ: загружаем в backbone.body (а не в backbone)
                if hasattr(model.backbone, 'body'):
                    body_state = model.backbone.body.state_dict()
                    logger.info("  Loading into backbone.body")
                else:
                    body_state = model.backbone.state_dict()
                    logger.info("  Loading into backbone directly")
                
                mapped = 0
                for k, v in weights.items():
                    # Убираем все возможные префиксы
                    clean_k = k
                    for prefix in ['backbone.', 'model.', 'student_model.', 'module.', '_backbone.', 'body.']:
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                            break
                    
                    if clean_k in body_state and v.shape == body_state[clean_k].shape:
                        body_state[clean_k] = v
                        mapped += 1
                
                if hasattr(model.backbone, 'body'):
                    model.backbone.body.load_state_dict(body_state, strict=False)
                else:
                    model.backbone.load_state_dict(body_state, strict=False)
                
                logger.info(f"  Mapped {mapped} weights to backbone.body")
                
                if mapped == 0:
                    logger.warning("  ⚠️ No weights mapped! Debug info:")
                    logger.warning(f"     First 5 keys in weights: {list(weights.keys())[:5]}")
                    logger.warning(f"     First 5 keys in body_state: {list(body_state.keys())[:5]}")
        elif init_type == 'modern_distilled':
            if self.pretrained_backbone_path and Path(self.pretrained_backbone_path).exists():
                logger.info(f"  Loading multi-layer distilled backbone weights...")
                checkpoint = torch.load(self.pretrained_backbone_path, map_location='cpu', weights_only=False)
                
                if 'student_state_dict' in checkpoint:
                    weights = checkpoint['student_state_dict']
                elif 'state_dict' in checkpoint:
                    weights = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    weights = checkpoint['model_state_dict']
                else:
                    weights = checkpoint
                
                if hasattr(model.backbone, 'body'):
                    body_state = model.backbone.body.state_dict()
                    logger.info("  Loading into backbone.body")
                else:
                    body_state = model.backbone.state_dict()
                    logger.info("  Loading into backbone directly")
                
                mapped = 0
                for k, v in weights.items():
                    clean_k = k
                    for prefix in ['backbone.', 'model.', 'student_model.', 'module.', '_backbone.', 'body.']:
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                            break
                    
                    if clean_k in body_state and v.shape == body_state[clean_k].shape:
                        body_state[clean_k] = v
                        mapped += 1
                
                if hasattr(model.backbone, 'body'):
                    model.backbone.body.load_state_dict(body_state, strict=False)
                else:
                    model.backbone.load_state_dict(body_state, strict=False)
                
                logger.info(f"  Mapped {mapped} weights to backbone.body")
            else:
                logger.warning("  No multi-layer distilled weights found, using random init!")
        elif init_type == 'modern_distilled':
            if self.pretrained_backbone_path and Path(self.pretrained_backbone_path).exists():
                logger.info("  Loading modern distilled backbone weights...")
                checkpoint = torch.load(self.pretrained_backbone_path, map_location='cpu', weights_only=False)
                if 'student_state_dict' in checkpoint:
                    weights = checkpoint['student_state_dict']
                elif 'state_dict' in checkpoint:
                    weights = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    weights = checkpoint['model_state_dict']
                else:
                    weights = checkpoint
                if hasattr(model.backbone, 'body'):
                    body_state = model.backbone.body.state_dict()
                else:
                    body_state = model.backbone.state_dict()
                mapped = 0
                for k, v in weights.items():
                    clean_k = k
                    for prefix in ['backbone.', 'model.', 'student_model.', 'module.', '_backbone.', 'body.']:
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                            break
                    if clean_k in body_state and v.shape == body_state[clean_k].shape:
                        body_state[clean_k] = v
                        mapped += 1
                if hasattr(model.backbone, 'body'):
                    model.backbone.body.load_state_dict(body_state, strict=False)
                else:
                    model.backbone.load_state_dict(body_state, strict=False)
                logger.info(f"  Mapped {mapped} modern distilled weights to backbone")
            else:
                logger.warning("  No modern distilled weights found, using random init!")
        else:
            logger.info("  Using random initialization")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"  Classes: {num_classes} (+1 background)")
        
        return model
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_path = Path(self.config['detection']['data_path'])
        img_size = tuple(self.config['detection']['img_size'])
        num_classes = self.config['detection']['num_classes']
        
        train_imgs = data_path / "train" / "images"
        train_lbls = data_path / "train" / "labels"
        val_imgs = data_path / "val" / "images"
        val_lbls = data_path / "val" / "labels"
        
        train_dataset = DefectDetectionDataset(train_imgs, train_lbls, num_classes, img_size)
        val_dataset = DefectDetectionDataset(val_imgs, val_lbls, num_classes, img_size)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.student_config['batch'], shuffle=True,
            num_workers=4, collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.student_config['batch'], shuffle=False,
            num_workers=2, collate_fn=collate_fn
        )
        
        logger.info(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
        logger.info(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def train(self) -> Dict:
        epochs = self.student_config['epochs']
        logger.info(f"\n{'='*60}\nTRAINING: {self.student_name}\nInit: {self.student_config['type']}\n{'='*60}\n")
        
        history = []
        start_time = time.time()
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate()
            
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_map50': val_metrics['mAP_50'],
                'val_map75': val_metrics['mAP_75'],
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            history.append(epoch_info)
            
            marker = " ⭐" if val_metrics['mAP_50'] > self.best_map else ""
            logger.info(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | mAP@50: {val_metrics['mAP_50']:.4f} | mAP@75: {val_metrics['mAP_75']:.4f}{marker}")
            
            if val_metrics['mAP_50'] > self.best_map:
                self.best_map = val_metrics['mAP_50']
                self.best_epoch = epoch
                self.patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                self._save_model('best_model.pth')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            self.scheduler.step()
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        self._save_model('model_final.pth')
        
        # ===== ИСПРАВЛЕНИЕ: Сохраняем историю обучения =====
        history_path = self.output_dir / 'training_results.json'
        with open(history_path, 'w') as f:
            json.dump({'history': history}, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        return {
            'model_name': self.student_name,
            'init_type': self.student_config['type'],
            'best_val_map50': self.best_map,
            'best_epoch': self.best_epoch,
            'epochs_trained': len(history),
            'training_time_hours': round((time.time() - start_time) / 3600, 3),
            'history': history,
        }
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()
        
        logger.info(f"  Epoch {epoch}/{self.student_config['epochs']} - {num_batches} batches")
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += losses.item()
            
            # ИСПРАВЛЕНИЕ: реже логируем
            if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                logger.info(f"    Batch {batch_idx:4d}/{num_batches} | Loss: {losses.item():.4f} | {elapsed:.1f}s")
        
        avg_loss = total_loss / num_batches
        logger.info(f"  Epoch {epoch} avg_loss: {avg_loss:.4f}")
        return avg_loss
    
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
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
    
    def _save_model(self, filename: str):
        save_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
        }, save_path)
        logger.info(f"  Model saved: {filename}")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def is_model_already_trained(student_name: str, config: dict) -> bool:
    """Проверяет, обучена ли уже модель."""
    detection_output = Path(config['paths']['detection_output'])
    model_path = detection_output / student_name / 'model_final.pth'
    if model_path.exists():
        logger.info(f"✅ Model {student_name} already trained, skipping...")
        return True
    return False


def main():
    config_path = Path(__file__).parent / "../config_modern_distillation.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cache_file = Path(config["paths"]["pretrain_output"]) / "modern_model_path.txt"
    pretrained_path = None
    if cache_file.exists():
        pretrained_path = cache_file.read_text().strip()
        logger.info(f"Found pretrained backbone: {pretrained_path}")
    else:
        logger.warning("No pretrained backbone found")
        # Ищем напрямую
        pretrain_dir = Path(config['paths']['pretrain_output']) / "resnet18_distilled"
        exported = pretrain_dir / "exported_models" / "exported_last.pt"
        if exported.exists():
            pretrained_path = str(exported)
            logger.info(f"Found pretrained backbone directly: {pretrained_path}")
    
    all_results = []
    
    # Загружаем существующие результаты, если есть
    summary_path = Path(config['paths']['detection_output']) / "all_results.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                all_results = json.load(f)
            logger.info(f"Loaded existing results for {len(all_results)} models")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    
    for student_name, student_cfg in config['students'].items():
        # Проверяем, обучена ли уже модель
        if is_model_already_trained(student_name, config):
            existing_result = next((r for r in all_results if r.get('model_name') == student_name), None)
            if existing_result:
                logger.info(f"  Best mAP@50: {existing_result['best_val_map50']:.4f} (epoch {existing_result['best_epoch']})")
            continue
        
        logger.info(f"\n{'#'*60}\nTraining: {student_name}\n{'#'*60}")
        try:
            trainer = FasterRCNNTrainer(config, student_name, student_cfg, pretrained_path)
            result = trainer.train()
            all_results.append(result)
            
            # Сохраняем результаты после каждой модели
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved after {student_name}")
            
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
    
    # Финальное сохранение
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {summary_path}")
    
    # Выводим сводку
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        for r in all_results:
            logger.info(f"  {r['model_name']:<35} mAP@50: {r['best_val_map50']:.4f} (epoch {r['best_epoch']})")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()