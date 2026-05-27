#!/usr/bin/env python3
"""
Distillation Pipeline для магистерской работы
DINOv3 LTDETR → Faster R-CNN

Эксперименты:
1. baseline: Faster R-CNN + ResNet-18 (ImageNet)
2. ssl_distilled: Faster R-CNN + SSL-distilled ResNet-18
3. gkd_distilled: Global Knowledge Distillation (DINOv3 → ResNet-18)
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou, roi_align
from PIL import Image
from tqdm import tqdm

import lightly_train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/distillation/global/experiment.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ
# ============================================================================

EXPERIMENTS = {
    'baseline': {
        'use_ssl_backbone': False,
        'use_gkd': False,
        'description': 'Faster R-CNN + ResNet-18 (ImageNet pretrained)'
    },
    'ssl_distilled': {
        'use_ssl_backbone': True,
        'use_gkd': False,
        'description': 'Faster R-CNN + SSL-distilled ResNet-18 (DINOv3→R18)'
    },
    'gkd_distilled': {
        'use_ssl_backbone': False,
        'use_gkd': True,
        'description': 'Global Knowledge Distillation (DINOv3 teacher → R18 student)'
    }
}


# ============================================================================
# GLOBAL KNOWLEDGE DISTILLATION MODULE
# ============================================================================

class GlobalKnowledgeDistillation(nn.Module):
    """
    Global Knowledge Distillation из статьи:
    "Distilling Object Detectors with Global Knowledge" (Tang et al.)
    """
    
    def __init__(self, num_prototypes=10, num_classes=4, lambda_reg=10.0, 
                 img_h=640, img_w=640):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.img_h = img_h
        self.img_w = img_w
        
        self.prototypes = {}
        self.adaptation_layers = nn.ModuleDict()
    
    def _get_spatial_scale(self, feat_map):
        feat_h, feat_w = feat_map.shape[-2:]
        if self.img_h == self.img_w:
            return feat_h / self.img_h
        scale_h = feat_h / self.img_h
        scale_w = feat_w / self.img_w
        return (scale_h + scale_w) / 2
    
    def _extract_features_from_gt(self, t_feat_map, s_feat_map, targets, level_name):
        batch_size = t_feat_map.shape[0]
        device = t_feat_map.device
        spatial_scale = self._get_spatial_scale(t_feat_map)
        
        all_t_feats, all_s_feats, all_labels = [], [], []
        
        for i in range(batch_size):
            boxes = targets[i]['boxes']
            labels = targets[i]['labels']
            
            if len(boxes) == 0:
                continue
            
            valid_mask = labels > 0
            if not valid_mask.any():
                continue
            
            boxes = boxes[valid_mask]
            cls_labels = labels[valid_mask] - 1
            
            box_batch_indices = torch.full(
                (len(boxes), 1), i, dtype=torch.float32, device=device
            )
            rois = torch.cat([box_batch_indices, boxes], dim=1)
            
            t_roi_feats = roi_align(
                t_feat_map, rois, output_size=(1, 1),
                spatial_scale=spatial_scale
            ).squeeze(-1).squeeze(-1)
            
            s_roi_feats = roi_align(
                s_feat_map, rois, output_size=(1, 1),
                spatial_scale=spatial_scale
            ).squeeze(-1).squeeze(-1)
            
            all_t_feats.append(t_roi_feats)
            all_s_feats.append(s_roi_feats)
            all_labels.append(cls_labels)
        
        if len(all_t_feats) == 0:
            return (
                torch.zeros(0, t_feat_map.shape[1], device=device),
                torch.zeros(0, s_feat_map.shape[1], device=device),
                torch.zeros(0, dtype=torch.long, device=device)
            )
        
        return (
            torch.cat(all_t_feats, dim=0),
            torch.cat(all_s_feats, dim=0),
            torch.cat(all_labels, dim=0)
        )
    
    def _matching_pursuit(self, t_feats, s_feats, K=10):
        N = t_feats.shape[0]
        K = min(K, N)
        
        selected = []
        residuals_t = t_feats.clone()
        residuals_s = s_feats.clone()
        
        for n in range(K):
            best_idx = -1
            best_loss = float('inf')
            
            for k in range(N):
                if k in selected:
                    continue
                
                g_t = t_feats[k]
                g_s = s_feats[k]
                
                dot_t = torch.mv(residuals_t, g_t)
                dot_s = torch.mv(residuals_s, g_s)
                norm_t_sq = torch.dot(g_t, g_t)
                norm_s_sq = torch.dot(g_s, g_s)
                
                denom = ((self.lambda_reg + norm_t_sq) * 
                        (self.lambda_reg + norm_s_sq) - 
                        self.lambda_reg ** 2)
                
                w_t = (dot_t * (self.lambda_reg + norm_s_sq) + 
                       self.lambda_reg * dot_s) / denom
                w_s = (dot_s * (self.lambda_reg + norm_t_sq) + 
                       self.lambda_reg * dot_t) / denom
                
                new_res_t = residuals_t - w_t.unsqueeze(1) * g_t.unsqueeze(0)
                new_res_s = residuals_s - w_s.unsqueeze(1) * g_s.unsqueeze(0)
                
                loss = (torch.sum(new_res_t ** 2) + 
                       torch.sum(new_res_s ** 2) + 
                       self.lambda_reg * torch.sum((w_t - w_s) ** 2))
                
                if loss < best_loss:
                    best_loss = loss
                    best_idx = k
            
            selected.append(best_idx)
            
            g_t = t_feats[best_idx]
            g_s = s_feats[best_idx]
            
            dot_t = torch.mv(residuals_t, g_t)
            dot_s = torch.mv(residuals_s, g_s)
            norm_t_sq = torch.dot(g_t, g_t)
            norm_s_sq = torch.dot(g_s, g_s)
            
            denom = ((self.lambda_reg + norm_t_sq) * 
                    (self.lambda_reg + norm_s_sq) - 
                    self.lambda_reg ** 2)
            
            w_t = (dot_t * (self.lambda_reg + norm_s_sq) + 
                   self.lambda_reg * dot_s) / denom
            w_s = (dot_s * (self.lambda_reg + norm_t_sq) + 
                   self.lambda_reg * dot_t) / denom
            
            residuals_t = residuals_t - w_t.unsqueeze(1) * g_t.unsqueeze(0)
            residuals_s = residuals_s - w_s.unsqueeze(1) * g_s.unsqueeze(0)
        
        return selected
    
    def update_prototypes_accumulated(self, teacher_model, student_model,
                                      train_loader, device, num_batches=10):
        logger.info(f"Accumulating features from {num_batches} batches...")
        
        accumulated = {}
        teacher_model.eval()
        student_model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break
                
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                t_feats = extract_teacher_features(teacher_model, images)
                s_feats = extract_student_features(student_model, images)
                
                for level_name in t_feats.keys():
                    if level_name not in accumulated:
                        accumulated[level_name] = {
                            'teacher': [], 'student': [], 'labels': []
                        }
                    
                    t_feat, s_feat, labels = self._extract_features_from_gt(
                        t_feats[level_name], s_feats[level_name],
                        targets, level_name
                    )
                    
                    if len(t_feat) > 0:
                        accumulated[level_name]['teacher'].append(t_feat)
                        accumulated[level_name]['student'].append(s_feat)
                        accumulated[level_name]['labels'].append(labels)
        
        self.prototypes = {}
        total_prototypes = 0
        
        for level_name, data in accumulated.items():
            if not data['teacher']:
                continue
            
            all_t = torch.cat(data['teacher'], dim=0)
            all_s = torch.cat(data['student'], dim=0)
            all_labels = torch.cat(data['labels'], dim=0)
            
            self.prototypes[level_name] = {}
            
            unique_labels = torch.unique(all_labels)
            logger.info(f"Level {level_name} found classes: {unique_labels.tolist()}")
            
            for cls_id in unique_labels:
                cls_id = int(cls_id.item())
                cls_mask = all_labels == cls_id
                t_cls = all_t[cls_mask]
                s_cls = all_s[cls_mask]
                
                if len(t_cls) >= self.num_prototypes:
                    indices = self._matching_pursuit(t_cls, s_cls, self.num_prototypes)
                    self.prototypes[level_name][cls_id] = {
                        'teacher': t_cls[indices].detach(),
                        'student': s_cls[indices].detach()
                    }
                    total_prototypes += self.num_prototypes
                elif len(t_cls) > 0:
                    self.prototypes[level_name][cls_id] = {
                        'teacher': t_cls.detach(),
                        'student': s_cls.detach()
                    }
                    total_prototypes += len(t_cls)
        
        # Проверяем все ли классы (ВНУТРИ метода!)
        all_found = set()
        for lvl in self.prototypes:
            all_found.update(self.prototypes[lvl].keys())
        missing = set(range(self.num_classes)) - all_found
        if missing:
            logger.warning(f"MISSING CLASSES in prototypes: {missing}")
            logger.warning(f"Only found: {all_found}")
            if all_found:
                ref_cls = list(all_found)[0]
                for lvl in self.prototypes:
                    if ref_cls in self.prototypes[lvl]:
                        for mc in missing:
                            self.prototypes[lvl][mc] = {
                                'teacher': self.prototypes[lvl][ref_cls]['teacher'].clone(),
                                'student': self.prototypes[lvl][ref_cls]['student'].clone()
                            }
                logger.info(f"Copied prototypes from class {ref_cls} to {missing}")
        
        logger.info(f"Updated prototypes: {total_prototypes} total "
                   f"across {len(self.prototypes)} FPN levels")
    

    def initialize_adapters(self, student_features, teacher_features, fpn_levels, device):
        """Создаёт adaptation layers ДО оптимизатора"""
        for level_name in fpn_levels:
            if level_name in student_features and level_name in teacher_features:
                s_dim = student_features[level_name].shape[1]
                t_dim = teacher_features[level_name].shape[1]
                self.adaptation_layers[level_name] = nn.Linear(s_dim, t_dim).to(device)
        logger.info(f"Initialized {len(self.adaptation_layers)} adaptation layers")
    
    def _compute_projections(self, feature, prototypes):
        K = prototypes.shape[0]
        projections = torch.zeros(K, device=feature.device)
        residual = feature.clone()
        
        for k in range(K):
            proto = prototypes[k]
            dot_product = torch.dot(residual, proto)
            norm_sq = torch.dot(proto, proto)
            w_k = dot_product / (self.lambda_reg + norm_sq)
            projections[k] = w_k
            residual = residual - w_k * proto
        
        return projections
    
    def compute_global_knowledge_loss(self, student_features, teacher_features, 
                                       targets, level_names):
        total_loss = 0.0
        num_instances = 0
        
        for level_name in level_names:
            if (level_name not in student_features or 
                level_name not in self.prototypes):
                continue
            
            t_feats, s_feats, labels = self._extract_features_from_gt(
                teacher_features[level_name],
                student_features[level_name],
                targets, level_name
            )
            
            if len(t_feats) == 0:
                continue
            
            # Адаптируем признаки студента в пространство учителя
            s_adapted = self.adaptation_layers[level_name](s_feats)
            
            for i, (t_feat, s_adapt, label) in enumerate(zip(t_feats, s_adapted, labels)):
                cls_id = int(label.item())
                
                if cls_id not in self.prototypes[level_name]:
                    continue
                
                proto_t = self.prototypes[level_name][cls_id]['teacher']
                
                # ОБЕ проекции на teacher prototypes (единый базис)
                lambda_t = self._compute_projections(t_feat, proto_t)
                lambda_s = self._compute_projections(s_adapt, proto_t)
                
                diff = lambda_s - lambda_t
                actual_k = len(proto_t)
                instance_loss = torch.sum(diff ** 2) / actual_k
                
                # Eq. 8: sigma фильтрует шумные экземпляры
                discrepancy = torch.norm(diff, p=2) / max(actual_k ** 0.5, 1)
                sigma = 1.0 - torch.clamp(discrepancy, 0.0, 1.0)
                total_loss += sigma * instance_loss
                num_instances += 1
        
        if num_instances > 0:
            total_loss = total_loss / num_instances
        
        return total_loss
    
    def compute_local_feature_loss(self, student_features, teacher_features,
                                    targets, level_names):
        total_loss = 0.0
        num_instances = 0
        
        for level_name in level_names:
            if (level_name not in student_features or 
                level_name not in self.prototypes):
                continue
            
            if level_name not in self.adaptation_layers:
                # Fallback: создаём слой, но предупреждаем
                logger.warning(f"Creating adaptation layer for {level_name} AFTER optimizer! It won't be trained!")
                s_dim = student_features[level_name].shape[1]
                t_dim = teacher_features[level_name].shape[1]
                self.adaptation_layers[level_name] = nn.Linear(s_dim, t_dim).to(
                    student_features[level_name].device
                )
            
            t_feats, s_feats, labels = self._extract_features_from_gt(
                teacher_features[level_name],
                student_features[level_name],
                targets, level_name
            )
            
            if len(t_feats) == 0:
                continue
            
            s_adapted = self.adaptation_layers[level_name](s_feats)
            
            # Также адаптируем прототипы студента для согласованности
            if level_name in self.prototypes:
                adapted_protos = {}
                for cls_id in self.prototypes[level_name]:
                    adapted_protos[cls_id] = {
                        'teacher': self.prototypes[level_name][cls_id]['teacher'],
                        'student': self.adaptation_layers[level_name](
                            self.prototypes[level_name][cls_id]['student']
                        )
                    }
            else:
                adapted_protos = self.prototypes.get(level_name, {})
            
            for i, (t_feat, s_adapt, label) in enumerate(zip(t_feats, s_adapted, labels)):
                # MSE как в статье Eq. 7
                feat_loss = F.mse_loss(s_adapt, t_feat, reduction='mean')
                total_loss += feat_loss
                num_instances += 1
        
        if num_instances > 0:
            total_loss = total_loss / num_instances
        
        return total_loss


# ============================================================================
# SSL DISTILLATION
# ============================================================================

def pretrain_backbone_ssl(config: dict) -> Path:
    logger.info(f"\n{'='*60}")
    logger.info("SSL DISTILLATION: DINOv3 → ResNet-18")
    logger.info(f"{'='*60}")
    
    ssl_cfg = config['ssl_pretraining']
    output_dir = Path(config['paths']['ssl_output'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(config['data']['path'])
    train_images = data_path / "train" / "images"
    val_images = data_path / "val" / "images"
    
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
    
    logger.info(f"Distilled backbone: {backbone_path}")
    return backbone_path


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class DefectDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        num_classes: int = 4,
        img_size: Tuple[int, int] = (640, 640),
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = sorted([
            f for f in self.images_dir.glob("*")
            if f.suffix.lower() in extensions
        ])
        
        logger.info(f"  Dataset: {len(self.image_files)} images")
    
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
        except Exception as e:
            logger.error(f"Failed to load {img_path}: {e}")
            img_tensor = torch.zeros(3, *self.img_size)
            orig_w, orig_h = self.img_size
        
        boxes, labels = self._parse_yolo(
            self.labels_dir / f"{img_path.stem}.txt",
            orig_w, orig_h
        )
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(
                [(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32
            ) if boxes else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes) if boxes else 0, dtype=torch.int64),
        }
        
        return img_tensor, target
    
    def _parse_yolo(self, label_path: Path, orig_w: int, orig_h: int):
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
        
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================================

def create_model(backbone_type: str, num_classes: int, 
                 ssl_path: Optional[Path] = None) -> nn.Module:
    if backbone_type == 'ssl_distilled' and ssl_path and ssl_path.exists():
        logger.info(f"Loading SSL-distilled backbone: {ssl_path}")
        
        base_model = resnet18(weights=None)
        checkpoint = torch.load(ssl_path, map_location='cpu', weights_only=False)
        
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
        
        missing, unexpected = base_model.load_state_dict(cleaned_weights, strict=False)
        logger.info(f"  Loaded: {len(cleaned_weights) - len(missing)} matched, "
                   f"{len(missing)} missing, {len(unexpected)} unexpected")
        
        backbone = _resnet_fpn_extractor(base_model, trainable_layers=5)
    else:
        logger.info("Using ImageNet pretrained backbone")
        base_model = resnet18(weights='DEFAULT')
        backbone = _resnet_fpn_extractor(base_model, trainable_layers=5)
    
    model = FasterRCNN(backbone, num_classes=num_classes + 1)
    return model


# ============================================================================
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ============================================================================

def extract_teacher_features(teacher_model, images):
    """Извлекает multi-scale признаки из учителя (DINOv3 LTDETR)"""
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.stack(images)
        # DINOv3LTDETRObjectDetection.backbone возвращает tuple
        backbone_out = teacher_model.backbone(images)
        if isinstance(backbone_out, (tuple, list)):
            features = {str(i): feat for i, feat in enumerate(backbone_out)}
        elif isinstance(backbone_out, dict):
            features = backbone_out
        else:
            features = {'0': backbone_out}
    return features

def extract_student_features(model, images):
    if isinstance(images, list):
        images = torch.stack(images)
    features = model.backbone(images)
    return features


# ============================================================================
# МЕТРИКИ
# ============================================================================

def compute_map(predictions: List[Dict], ground_truths: List[Dict], 
                num_classes: int, iou_thresholds: List[float] = [0.5, 0.75]) -> Dict[str, float]:
    results = {}
    
    for iou_thr in iou_thresholds:
        aps = []
        
        for cls_id in range(num_classes):
            detections = []
            num_gt = 0
            
            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                gt_mask = gt['labels'] == cls_id
                num_gt += gt_mask.sum().item()
                
                pred_mask = pred['labels'] == cls_id
                for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                    detections.append({
                        'image_id': img_idx,
                        'score': score.item(),
                        'bbox': box
                    })
            
            if num_gt == 0:
                continue
            
            detections.sort(key=lambda x: x['score'], reverse=True)
            
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
                
                ious = box_iou(det['bbox'].unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(dim=0)
                
                if best_iou >= iou_thr and not gt_matched[img_idx][best_idx.item()]:
                    tp[i] = 1
                    gt_matched[img_idx][best_idx.item()] = True
                else:
                    fp[i] = 1
            
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
# ОБУЧЕНИЕ
# ============================================================================

def train_detector(
    config: dict,
    experiment_name: str,
    backbone_type: str,
    ssl_path: Optional[Path] = None,
    teacher_model=None,
) -> Dict:
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
    
    model = create_model(backbone_type, num_classes, ssl_path)
    model.to(device)
    
    data_path = Path(data_cfg['path'])
    
    train_dataset = DefectDataset(
        images_dir=data_path / "train" / "images",
        labels_dir=data_path / "train" / "labels",
        num_classes=num_classes,
        img_size=img_size,
    )
    
    val_dataset = DefectDataset(
        images_dir=data_path / "val" / "images",
        labels_dir=data_path / "val" / "labels",
        num_classes=num_classes,
        img_size=img_size,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'],
        shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=2, collate_fn=collate_fn,
    )
    
    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Автоопределение уровней FPN
    model.eval()
    with torch.no_grad():
        dummy = [torch.randn(3, *img_size).to(device)]
        stu_feats = extract_student_features(model, dummy)
        stu_levels = list(stu_feats.keys())
        logger.info(f"Student FPN levels: {stu_levels}")
    
    fpn_levels = stu_levels
    fpn_mapping = {}
    if teacher_model is not None:
        teacher_model.eval()
        with torch.no_grad():
            tch_feats = extract_teacher_features(teacher_model, dummy)
            tch_levels = list(tch_feats.keys())
            logger.info(f"Teacher feature levels: {tch_levels}")
        
        # Матчим уровни по пространственному размеру
        for s_name, s_feat in stu_feats.items():
            for t_name, t_feat in tch_feats.items():
                if s_feat.shape[-2:] == t_feat.shape[-2:]:
                    fpn_mapping[s_name] = t_name
                    break
        
        if fpn_mapping:
            fpn_levels = list(fpn_mapping.keys())
            logger.info(f"FPN mapping by spatial size: {fpn_mapping}")
        else:
            fpn_levels = [l for l in stu_levels if l in tch_levels]
            logger.info(f"Common GKD levels (fallback): {fpn_levels}")
    model.train()
    
    gkd_module = None
    if exp_cfg['use_gkd'] and teacher_model is not None:
        gkd_cfg = config.get('gkd', {})
        gkd_module = GlobalKnowledgeDistillation(
            num_prototypes=gkd_cfg.get('num_prototypes', 10),
            num_classes=num_classes,
            lambda_reg=gkd_cfg.get('lambda_reg', 10.0),
            img_h=img_size[0],
            img_w=img_size[1],
        ).to(device)
        
        gkd_module.update_prototypes_accumulated(
            teacher_model, model, train_loader, device,
            num_batches=gkd_cfg.get('prototype_accumulation_batches', 10)
        )
        
        # Инициализируем adaptation layers ДО оптимизатора
        gkd_module.initialize_adapters(stu_feats, tch_feats, fpn_levels, device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(gkd_module.parameters()) if gkd_module else []),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['epochs'], eta_min=1e-6,
    )
    
    output_dir = Path(config['paths']['experiments_output']) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_map50 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    
    start_time = time.time()
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        if (gkd_module is not None and 
            epoch % config['gkd'].get('prototype_update_epochs', 5) == 0 and 
            epoch > 1):
            logger.info(f"Updating GKD prototypes at epoch {epoch}...")
            gkd_module.update_prototypes_accumulated(
                teacher_model, model, train_loader, device,
                num_batches=config['gkd'].get('prototype_accumulation_batches', 10)
            )
            model.train()  # Возвращаем в training mode
            teacher_model.eval()
        
        model.train()
        epoch_loss = 0.0
        epoch_det_loss = 0.0
        epoch_gkd_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            det_loss = sum(loss_dict.values())
            total_loss = det_loss
            epoch_det_loss += det_loss.item()
            
            if gkd_module is not None and gkd_module.prototypes:
                with torch.no_grad():
                    t_feats_raw = extract_teacher_features(teacher_model, images)
                s_feats = extract_student_features(model, images)
                
                # Перепривязываем уровни учителя к уровням студента
                t_feats = {}
                for s_lvl in fpn_levels:
                    t_lvl = fpn_mapping.get(s_lvl, s_lvl)
                    if t_lvl in t_feats_raw:
                        t_feats[s_lvl] = t_feats_raw[t_lvl]
                
                # Диагностика первого батча
                if epoch == 1 and epoch_loss == 0:
                    for lvl in fpn_levels:
                        if lvl in gkd_module.prototypes and lvl in t_feats:
                            t_f, s_f, lbls = gkd_module._extract_features_from_gt(
                                t_feats[lvl], s_feats[lvl], targets, lvl
                            )
                            logger.info(f"Level {lvl}: {len(t_f)} instances, unique labels: {torch.unique(lbls).tolist() if len(lbls)>0 else 'none'}")
                            logger.info(f"  Proto classes: {list(gkd_module.prototypes[lvl].keys())}")
                
                global_loss = gkd_module.compute_global_knowledge_loss(
                    s_feats, t_feats, targets, fpn_levels
                )
                local_loss = gkd_module.compute_local_feature_loss(
                    s_feats, t_feats, targets, fpn_levels
                )
                if epoch == 1 and epoch_loss == 0:
                    logger.info(f"Global loss: {global_loss.item():.6f}, Local loss: {local_loss.item():.6f}")
                
                # KD warmup: плавно увеличиваем вес от 0 до 1
                kd_warmup = min(1.0, epoch / config['gkd'].get('kd_warmup_epochs', 10))
                gkd_loss = kd_warmup * (
                    config['gkd'].get('alpha_global', 0.1) * global_loss +
                    config['gkd'].get('alpha_local', 0.1) * local_loss
                )
                
                total_loss = det_loss + gkd_loss
                epoch_gkd_loss += gkd_loss.item()
                
                # Логируем соотношение GKD/det
                if epoch == 1 and epoch_loss == 0:
                    ratio = gkd_loss.item() / max(det_loss.item(), 1e-8)
                    logger.info(f"GKD/det ratio: {ratio:.4f} (warmup={kd_warmup:.2f})")
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            postfix = {'loss': f'{total_loss.item():.4f}'}
            if gkd_module is not None:
                postfix['gkd'] = f'{gkd_loss.item():.4f}'
            pbar.set_postfix(postfix)
        
        avg_loss = epoch_loss / len(train_loader)
        
        model.eval()
        predictions, ground_truths = [], []
        
        with torch.no_grad():
            for images, targets in val_loader:
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
        
        metrics = compute_map(predictions, ground_truths, num_classes)
        
        marker = " ⭐" if metrics['mAP@50'] > best_map50 else ""
        log_msg = (f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                   f"mAP@50: {metrics['mAP@50']:.4f} | "
                   f"mAP@75: {metrics['mAP@75']:.4f}{marker}")
        
        if gkd_module is not None:
            log_msg += f" | GKD: {epoch_gkd_loss/len(train_loader):.4f}"
        
        logger.info(log_msg)
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            **metrics,
        })
        
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
                    'use_gkd': exp_cfg['use_gkd'],
                },
            }, output_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= train_cfg['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step()
    
    training_time = round((time.time() - start_time) / 3600, 3)
    
    result = {
        'experiment': experiment_name,
        'description': exp_cfg['description'],
        'backbone_type': backbone_type,
        'use_ssl_backbone': exp_cfg['use_ssl_backbone'],
        'use_gkd': exp_cfg['use_gkd'],
        'best_map50': best_map50,
        'best_epoch': best_epoch,
        'epochs_trained': len(history),
        'training_time_hours': training_time,
        'history': history,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✅ {experiment_name}: mAP@50 = {best_map50:.4f} (epoch {best_epoch})")
    
    return result


# ============================================================================
# ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================================

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Distillation experiments for thesis')
    parser.add_argument('--config', type=str, default='config_thesis.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--experiments', nargs='+',
                       default=['baseline', 'ssl_distilled', 'gkd_distilled'],
                       help='Which experiments to run')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path('/app/distillation/global') / args.config
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"\n{'#'*60}")
    logger.info("DISTILLATION EXPERIMENTS FOR THESIS")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"{'#'*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ssl_path = None
    if 'ssl_distilled' in args.experiments:
        ssl_output = (Path(config['paths']['ssl_output']) / 
                     "resnet18_distilled" / "exported_models" / "exported_last.pt")
        
        if ssl_output.exists():
            logger.info(f"SSL backbone exists: {ssl_output}")
            ssl_path = ssl_output
        else:
            logger.info("Running SSL pretraining...")
            ssl_path = pretrain_backbone_ssl(config)
    
    teacher_model = None
    if 'gkd_distilled' in args.experiments:
        teacher_weights = config['teacher']['weights']
        
        if Path(teacher_weights).exists():
            logger.info(f"Loading teacher model: {teacher_weights}")
            teacher_model = lightly_train.load_model(teacher_weights)
            teacher_model.to(device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            logger.info("Teacher model loaded successfully (frozen)")
        else:
            logger.error(f"Teacher weights not found: {teacher_weights}")
            logger.error("Skipping GKD experiment!")
            args.experiments = [e for e in args.experiments if e != 'gkd_distilled']
    
    all_results = []
    
    for exp_name in args.experiments:
        exp_cfg = EXPERIMENTS[exp_name]
        backbone_type = 'ssl_distilled' if exp_cfg['use_ssl_backbone'] else 'imagenet'
        
        result = train_detector(
            config=config,
            experiment_name=exp_name,
            backbone_type=backbone_type,
            ssl_path=ssl_path if exp_cfg['use_ssl_backbone'] else None,
            teacher_model=teacher_model if exp_cfg['use_gkd'] else None,
        )
        
        all_results.append(result)
    
    summary = {
        'experiments': all_results,
        'config': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    summary_path = Path(config['paths']['experiments_output']) / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"{'Experiment':<25} {'mAP@50':<10} {'mAP@75':<10} {'Epochs':<10} {'Time(h)':<10}")
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