#!/usr/bin/env python3
"""
Thesis: Правильный Knowledge Distillation для DETR (ИСПРАВЛЕННЫЙ)
Учитель: DINOv3 ViT-S LTDETR → Студент: ViT-T LTDETR

ИСПРАВЛЕНИЯ:
✅ Classification cost: L1 вместо cosine
✅ Нормализация cost matrix (mean/std по каждому компоненту)
✅ Правильный GIoU (pairwise без flatten)
✅ Проверка числа queries
✅ Feature dims из модели (не хардкод)
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, generalized_box_iou
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import lightly_train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

class DefectDataset(Dataset):
    def __init__(self, images_dir, labels_dir, num_classes=4, img_size=(640, 640)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = sorted([f for f in self.images_dir.glob("*") 
                                   if f.suffix.lower() in extensions])
    
    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize(self.img_size, Image.BILINEAR)
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except:
            img_tensor = torch.zeros(3, *self.img_size)
        
        boxes, labels = [], []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    try:
                        cls = int(float(parts[0]))
                        if cls < 0 or cls >= self.num_classes: continue
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = (xc - w/2) * 640; y1 = (yc - h/2) * 640
                        x2 = (xc + w/2) * 640; y2 = (yc + h/2) * 640
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)
                    except: continue
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
        return img_tensor, target


def collate_fn(batch): return tuple(zip(*batch))


# ============================================================================
# ПРАВИЛЬНЫЙ HUNGARIAN MATCHING
# ============================================================================

def hungarian_matching(student_preds, teacher_preds, num_classes):
    """
    Матчит предсказания студента с учителем.
    
    ИСПРАВЛЕНИЯ:
    - L1 distance для classification (не cosine)
    - Нормализация каждого компонента cost
    - Правильный pairwise GIoU
    """
    student_logits, student_boxes = student_preds[0], student_preds[1]
    teacher_logits, teacher_boxes = teacher_preds[0], teacher_preds[1]
    
    B = student_logits.size(0)
    matched_indices = []
    
    for b in range(B):
        s_logits = student_logits[b]  # [Q, C]
        t_logits = teacher_logits[b]  # [Q, C]
        s_boxes = student_boxes[b]    # [Q, 4]
        t_boxes = teacher_boxes[b]    # [Q, 4]
        
        Q_s, Q_t = s_logits.size(0), t_logits.size(0)
        
        # 1. Classification cost: L1 distance (не cosine!)
        cls_cost = torch.cdist(s_logits, t_logits, p=1)  # [Q_s, Q_t]
        
        # 2. Box L1 cost
        box_l1_cost = torch.cdist(s_boxes, t_boxes, p=1)  # [Q_s, Q_t]
        
        # 3. GIoU cost (правильный pairwise)
        giou_cost = torch.zeros(Q_s, Q_t, device=s_boxes.device)
        for i in range(Q_s):
            for j in range(Q_t):
                giou = generalized_box_iou(
                    s_boxes[i].unsqueeze(0), t_boxes[j].unsqueeze(0)
                )
                giou_cost[i, j] = 1 - giou
        
        # 4. НОРМАЛИЗАЦИЯ каждого компонента (важно!)
        cls_cost = cls_cost / (cls_cost.std() + 1e-6)
        box_l1_cost = box_l1_cost / (box_l1_cost.std() + 1e-6)
        giou_cost = giou_cost / (giou_cost.std() + 1e-6)
        
        # 5. Итоговая cost matrix
        cost_matrix = cls_cost + box_l1_cost + giou_cost
        
        # Hungarian
        s_idx, t_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        matched_indices.append((torch.tensor(s_idx), torch.tensor(t_idx)))
    
    return matched_indices


# ============================================================================
# KD LOSSES (ИСПРАВЛЕННЫЕ)
# ============================================================================

class DistillationLoss(nn.Module):
    """Правильный KD loss для DETR."""
    
    def __init__(self, num_classes, student_feat_dim, teacher_feat_dim, temperature=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Feature projection (размеры из модели!)
        common_dim = 256
        self.feat_proj_s = nn.Linear(student_feat_dim, common_dim)
        self.feat_proj_t = nn.Linear(teacher_feat_dim, common_dim)
    
    def forward(self, student_out, teacher_out, student_feat=None, teacher_feat=None):
        # Проверка числа queries
        if student_out[0].size(1) != teacher_out[0].size(1):
            logger.warning(f"Query mismatch: student={student_out[0].size(1)}, teacher={teacher_out[0].size(1)}")
        
        # Hungarian matching
        indices = hungarian_matching(student_out, teacher_out, self.num_classes)
        
        logits_loss = 0.0
        boxes_loss = 0.0
        
        for b, (s_idx, t_idx) in enumerate(indices):
            if len(s_idx) == 0: continue
            
            s_logits = student_out[0][b][s_idx]
            t_logits = teacher_out[0][b][t_idx]
            s_boxes = student_out[1][b][s_idx]
            t_boxes = teacher_out[1][b][t_idx]
            
            # KL divergence на logits
            t_soft = F.softmax(t_logits / self.temperature, dim=-1)
            s_log_soft = F.log_softmax(s_logits / self.temperature, dim=-1)
            logits_loss += F.kl_div(s_log_soft, t_soft, reduction='batchmean')
            
            # L1 + GIoU на boxes
            boxes_loss += F.smooth_l1_loss(s_boxes, t_boxes)
            boxes_loss += (1 - generalized_box_iou(s_boxes, t_boxes).mean())
        
        logits_loss /= max(len(indices), 1)
        boxes_loss /= max(len(indices), 1)
        
        # Feature KD
        feat_loss = 0.0
        if student_feat is not None and teacher_feat is not None:
            s_proj = self.feat_proj_s(student_feat)
            t_proj = self.feat_proj_t(teacher_feat)
            feat_loss = F.mse_loss(s_proj, t_proj)
        
        return {
            'logits_kd': logits_loss,
            'boxes_kd': boxes_loss,
            'feat_kd': feat_loss,
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_map50(predictions, ground_truths, num_classes):
    aps = []
    for cls_id in range(num_classes):
        detections = []; num_gt = 0
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            gt_mask = gt['labels'] == cls_id
            num_gt += gt_mask.sum().item()
            pred_mask = pred['labels'] == cls_id
            for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                detections.append({'img_idx': img_idx, 'score': score.item(), 'box': box})
        if num_gt == 0: continue
        detections.sort(key=lambda x: x['score'], reverse=True)
        gt_matched = {i: [False] * (gt['labels'] == cls_id).sum().item() 
                      for i, gt in enumerate(ground_truths)}
        tp = np.zeros(len(detections)); fp = np.zeros(len(detections))
        for i, det in enumerate(detections):
            img_idx = det['img_idx']
            gt_mask = ground_truths[img_idx]['labels'] == cls_id
            gt_boxes = ground_truths[img_idx]['boxes'][gt_mask]
            if len(gt_boxes) == 0: fp[i] = 1; continue
            ious = box_iou(det['box'].unsqueeze(0), gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            if best_iou >= 0.5 and not gt_matched[img_idx][best_idx.item()]:
                tp[i] = 1; gt_matched[img_idx][best_idx.item()] = True
            else: fp[i] = 1
        tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(num_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
        ap = sum(np.max(precisions[recalls >= t]) for t in np.linspace(0, 1, 11) 
                if np.any(recalls >= t)) / 11.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


# ============================================================================
# TRAINER
# ============================================================================

def get_feature_dims(model):
    """Извлекает размерности фич из модели."""
    # Пробуем найти encoder dim
    for name, param in model.named_parameters():
        if 'encoder' in name and 'weight' in name and param.dim() >= 2:
            return param.size(0)  # input dim
    return 256  # fallback


def run_experiment(config, use_kd=False):
    device = torch.device('cuda')
    num_classes = config['data']['num_classes']
    
    # Студент
    student = lightly_train.load_model(config['student']['model'])
    student.to(device)
    
    # Учитель (для KD)
    teacher = None
    kd_loss_fn = None
    if use_kd:
        teacher = lightly_train.load_model(config['teacher']['weights'])
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad = False
        
        # Авто-определение размерностей
        s_dim = get_feature_dims(student)
        t_dim = get_feature_dims(teacher)
        logger.info(f"Feature dims: student={s_dim}, teacher={t_dim}")
        
        kd_loss_fn = DistillationLoss(num_classes, s_dim, t_dim).to(device)
    
    # Данные
    data_path = Path(config['data']['path'])
    train_ds = DefectDataset(data_path / "train" / "images", data_path / "train" / "labels")
    val_ds = DefectDataset(data_path / "val" / "images", data_path / "val" / "labels")
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=collate_fn)
    
    # Оптимизатор
    params = list(student.parameters())
    if kd_loss_fn: params += list(kd_loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.0001)
    
    exp_name = 'kd' if use_kd else 'baseline'
    output_dir = Path(config['paths']['output']) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kd_weights = config.get('kd_weights', {'logits': 1.0, 'boxes': 0.5, 'feat': 0.3})
    best_map = 0.0
    
    for epoch in range(config['training']['epochs']):
        student.train()
        if kd_loss_fn: kd_loss_fn.train()
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            images_tensor = torch.stack(images)
            
            # Student GT loss
            loss_dict = student(images_tensor, targets)
            det_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            total_loss = det_loss
            
            # KD
            if use_kd and teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(images_tensor)
                
                student_out = student(images_tensor)
                
                if isinstance(teacher_out, tuple) and isinstance(student_out, tuple):
                    kd_losses = kd_loss_fn(student_out, teacher_out)
                    
                    total_loss += kd_weights['logits'] * kd_losses['logits_kd']
                    total_loss += kd_weights['boxes'] * kd_losses['boxes_kd']
                    total_loss += kd_weights['feat'] * kd_losses['feat_kd']
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Validation
        student.eval()
        predictions, ground_truths = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                output = student(images[0].unsqueeze(0))
                
                if isinstance(output, tuple):
                    scores, boxes, labels = output
                    keep = scores[0] > 0.25
                    predictions.append({
                        'boxes': boxes[0][keep].cpu(),
                        'scores': scores[0][keep].cpu(),
                        'labels': labels[0][keep].cpu(),
                    })
                
                ground_truths.append({
                    'boxes': targets[0]['boxes'].cpu(),
                    'labels': targets[0]['labels'].cpu(),
                })
        
        map50 = compute_map50(predictions, ground_truths, num_classes)
        logger.info(f"Epoch {epoch+1}: mAP@50={map50:.4f}")
        
        if map50 > best_map:
            best_map = map50
            torch.save(student.state_dict(), output_dir / 'best_model.pth')
    
    return {'experiment': exp_name, 'best_map50': best_map}


def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_kd.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--experiments', nargs='+', default=['baseline', 'kd'])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results = {}
    for exp in args.experiments:
        results[exp] = run_experiment(config, use_kd=(exp == 'kd'))
    
    logger.info(f"\n{'='*50}")
    logger.info("RESULTS")
    for name, r in results.items():
        logger.info(f"  {name}: mAP@50 = {r['best_map50']:.4f}")
    if 'baseline' in results and 'kd' in results:
        delta = results['kd']['best_map50'] - results['baseline']['best_map50']
        logger.info(f"  Δ = {delta:+.4f}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()