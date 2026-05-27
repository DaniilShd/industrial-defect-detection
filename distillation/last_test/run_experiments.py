#!/usr/bin/env python3
"""
Эксперименты для магистерской:
1. Baseline: Faster R-CNN ResNet-18 (ImageNet) на mixed_full
2. Pseudo: Faster R-CNN ResNet-18 на mixed_full + 4000 псевдоразмеченных патчей
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import box_iou
from tqdm import tqdm

import lightly_train

# ============================================================================
# КОНФИГ
# ============================================================================

CONFIG = {
    'seed': 42,
    'img_size': (640, 640),
    'num_classes': 4,
    'batch_size': 8,
    'epochs': 50,
    'lr': 0.0001,
    'weight_decay': 0.0001,
    'patience': 15,
    'teacher_weights': '/app/data/experiment_v3/models/teacher/teacher_mixed_full_ssl/exported_models/exported_best.pt',
    'mixed_full_path': '/app/data/experiment_v3/datasets/mixed_full',
    'patches_dir': '/app/data/processed/defect_patches/images/train',
    'output_dir': '/app/distillation/last_test/experiments',
    'pseudo_count': 4000,
    'pseudo_conf': 0.7,
}

# ============================================================================
# ЛОГГЕР
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ПСЕВДОРАЗМЕТКИ
# ============================================================================

def generate_pseudo_labels(config: dict) -> Path:
    """
    Выбирает 4000 случайных патчей, прогоняет через LTDETR,
    сохраняет YOLO-разметку.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Generating pseudo-labels")
    logger.info("=" * 60)

    patches_dir = Path(config['patches_dir'])
    output_dir = Path(config['output_dir']) / 'pseudo_dataset'
    img_out = output_dir / 'images'
    lbl_out = output_dir / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Собираем все патчи
    all_patches = list(patches_dir.glob('*.jpg')) + list(patches_dir.glob('*.png'))
    logger.info(f"Found {len(all_patches)} patches in {patches_dir}")

    # Выбираем 4000 случайных
    random.seed(config['seed'])
    selected = random.sample(all_patches, min(config['pseudo_count'], len(all_patches)))
    logger.info(f"Selected {len(selected)} patches for pseudo-labeling")

    # Загружаем учителя
    logger.info(f"Loading teacher: {config['teacher_weights']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = lightly_train.load_model(config['teacher_weights'])
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    pseudo_count = 0
    total_boxes = 0

    for patch_path in tqdm(selected, desc="Pseudo-labeling"):
        try:
            img = Image.open(patch_path).convert('RGB')
            w, h = img.size

            # Прогоняем через учителя
            results = teacher.predict(str(patch_path), threshold=config['pseudo_conf'])

            boxes = results.get('bboxes', [])
            labels = results.get('labels', [])
            scores = results.get('scores', [])

            if len(boxes) == 0:
                continue

            yolo_lines = []
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()
                cls_id = int(label.item())

                # В YOLO формат (нормализованные)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Сохраняем
            img.save(img_out / patch_path.name)
            label_path = lbl_out / f"{patch_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            pseudo_count += 1
            total_boxes += len(yolo_lines)

        except Exception as e:
            logger.warning(f"Failed {patch_path.name}: {e}")

    logger.info(f"✅ Generated: {pseudo_count} images, {total_boxes} boxes")
    return output_dir


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class YOLODataset(Dataset):
    """Датасет в YOLO формате (изображения + .txt разметка)"""

    def __init__(self, images_dir: Path, labels_dir: Path,
                 num_classes: int = 4, img_size: Tuple[int, int] = (640, 640)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size

        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = sorted([
            f for f in self.images_dir.glob('*')
            if f.suffix.lower() in extensions
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        image = image.resize(self.img_size, Image.BILINEAR)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        boxes, labels = self._parse_yolo(
            self.labels_dir / f"{img_path.stem}.txt", orig_w, orig_h
        )

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32) if boxes else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes) if boxes else 0, dtype=torch.int64),
        }
        return img_tensor, target

    def _parse_yolo(self, label_path, orig_w, orig_h):
        boxes, labels = [], []
        if not label_path.exists():
            return boxes, labels

        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1 = max(0, (xc - w/2) * orig_w * scale_x)
                y1 = max(0, (yc - h/2) * orig_h * scale_y)
                x2 = min(self.img_size[1], (xc + w/2) * orig_w * scale_x)
                y2 = min(self.img_size[0], (yc + h/2) * orig_h * scale_y)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)  # 0 = background
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# МОДЕЛЬ
# ============================================================================

def create_model(num_classes: int) -> nn.Module:
    base = resnet18(weights='DEFAULT')
    backbone = _resnet_fpn_extractor(base, trainable_layers=5)
    return FasterRCNN(backbone, num_classes=num_classes + 1)


# ============================================================================
# МЕТРИКИ
# ============================================================================

def compute_map(predictions, ground_truths, num_classes):
    """COCO-style mAP: mAP@50, mAP@75, mAP@50:95 (как в lightly_train)"""
    results = {}
    
    def compute_ap_for_threshold(iou_thr):
        aps = []
        for cls_id in range(num_classes):
            detections, num_gt = [], 0
            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                gt_mask = gt['labels'] == cls_id
                num_gt += gt_mask.sum().item()
                pred_mask = pred['labels'] == cls_id
                for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                    detections.append({'image_id': img_idx, 'score': score.item(), 'bbox': box})
            if num_gt == 0:
                continue
            detections.sort(key=lambda x: x['score'], reverse=True)
            gt_matched = {i: [False] * (ground_truths[i]['labels'] == cls_id).sum().item() 
                         for i in range(len(ground_truths))}
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            for i, det in enumerate(detections):
                gt_boxes = ground_truths[det['image_id']]['boxes'][
                    ground_truths[det['image_id']]['labels'] == cls_id]
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                ious = box_iou(det['bbox'].unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(dim=0)
                if best_iou >= iou_thr and not gt_matched[det['image_id']][best_idx.item()]:
                    tp[i] = 1
                    gt_matched[det['image_id']][best_idx.item()] = True
                else:
                    fp[i] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / num_gt
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
            ap = 0.0
            for t in np.linspace(0, 1, 101):
                if np.any(recalls >= t):
                    ap += np.max(precisions[recalls >= t]) / 101.0
            aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0
    
    # mAP@50 и mAP@75
    results['mAP@50'] = compute_ap_for_threshold(0.5)
    results['mAP@75'] = compute_ap_for_threshold(0.75)
    
    # mAP@50:95 (главная метрика — как val_metric/map у LTDETR)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    map_values = [compute_ap_for_threshold(thr) for thr in iou_thresholds]
    results['mAP@50:95'] = float(np.mean(map_values))
    
    return results


# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================

def train_model(model, train_loader, val_loader, config, exp_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    out_dir = Path(config['output_dir']) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_map50 = 0.0
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)

        # Validation
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

        metrics = compute_map(predictions, ground_truths, config['num_classes'])
        marker = " ⭐" if metrics['mAP@50'] > best_map50 else ""
        logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
           f"mAP@50: {metrics['mAP@50']:.4f} | "
           f"mAP@75: {metrics['mAP@75']:.4f} | "
           f"mAP@50:95: {metrics['mAP@50:95']:.4f}{marker}")
        history.append({'epoch': epoch, 'train_loss': avg_loss, **metrics})

        if metrics['mAP@50'] > best_map50:
            best_map50 = metrics['mAP@50']
            best_epoch = epoch
            patience = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, out_dir / 'best_model.pth')
        else:
            patience += 1

        if patience >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()

    result = {
        'experiment': exp_name,
        'best_map50': best_map50,
        'best_epoch': best_epoch,
        'history': history,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"✅ {exp_name}: mAP@50 = {best_map50:.4f}")
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_pseudo', action='store_true', help='Skip pseudo-label generation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Шаг 1: Псевдоразметка
    pseudo_dir = None
    if not args.skip_pseudo:
        pseudo_dir = generate_pseudo_labels(CONFIG)
    else:
        pseudo_dir = Path(CONFIG['output_dir']) / 'pseudo_dataset'
        if not pseudo_dir.exists():
            logger.error("Pseudo dataset not found! Run without --skip_pseudo first.")
            sys.exit(1)

    # Шаг 2: Датасеты
    mixed_path = Path(CONFIG['mixed_full_path'])
    train_original = YOLODataset(
        mixed_path / 'train' / 'images',
        mixed_path / 'train' / 'labels',
        CONFIG['num_classes'], CONFIG['img_size']
    )
    val_dataset = YOLODataset(
        mixed_path / 'val' / 'images',
        mixed_path / 'val' / 'labels',
        CONFIG['num_classes'], CONFIG['img_size']
    )
    train_pseudo = YOLODataset(
        pseudo_dir / 'images',
        pseudo_dir / 'labels',
        CONFIG['num_classes'], CONFIG['img_size']
    )

    logger.info(f"Original train: {len(train_original)} images")
    logger.info(f"Pseudo train: {len(train_pseudo)} images")
    logger.info(f"Validation: {len(val_dataset)} images")

    train_combined = ConcatDataset([train_original, train_pseudo])
    logger.info(f"Combined train: {len(train_combined)} images")

    train_loader = DataLoader(train_original, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
    train_combined_loader = DataLoader(train_combined, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Шаг 3: Эксперименты
    results = {}

    # Эксперимент 1: Baseline (только mixed_full)
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: Baseline (mixed_full only)")
    logger.info("=" * 60)
    model1 = create_model(CONFIG['num_classes'])
    results['baseline'] = train_model(model1, train_loader, val_loader, CONFIG, 'baseline')

    # Эксперимент 2: С псевдоразметкой
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: mixed_full + pseudo-labels")
    logger.info("=" * 60)
    model2 = create_model(CONFIG['num_classes'])
    results['pseudo'] = train_model(model2, train_combined_loader, val_loader, CONFIG, 'pseudo')

    # Итоги
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for name, r in results.items():
        logger.info(f"{name}: mAP@50 = {r['best_map50']:.4f} (epoch {r['best_epoch']})")

    with open(Path(CONFIG['output_dir']) / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()