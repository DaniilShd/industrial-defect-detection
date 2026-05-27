#!/usr/bin/env python3
"""
Тестирование обученных детекторов на тестовом датасете mixed_full/test
с использованием torchmetrics.MeanAveragePrecision (COCO-совместимые метрики)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from tqdm import tqdm

# ============================================================================
# КОНФИГ
# ============================================================================

CONFIG = {
    'seed': 42,
    'img_size': (640, 640),
    'num_classes': 4,
    'batch_size': 1,
    'test_dataset_path': '/app/data/experiment_v3/datasets/mixed_full',
    'models': {
        'baseline': '/app/distillation/last_test/experiments/baseline/best_model.pth',
        'pseudo': '/app/distillation/last_test/experiments/pseudo/best_model.pth',
        # Добавьте учителя, если нужно
        # 'teacher_ltdetr': '/app/data/experiment_v3/models/teacher/teacher_mixed_full_ssl/exported_models/exported_best.pt',
    },
    'output_dir': '/app/distillation/last_test/test_results_coco',
    'score_threshold': 0.25,
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
# ДАТАСЕТ
# ============================================================================

class YOLODataset(Dataset):
    """Датасет в YOLO формате для тестирования (0-based классы)"""

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

        # Ground truth в COCO-формате (0-based классы)
        boxes, labels = self._parse_yolo(
            self.labels_dir / f"{img_path.stem}.txt", orig_w, orig_h
        )

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
        return img_tensor, target, img_path.name

    def _parse_yolo(self, label_path, orig_w, orig_h):
        """Парсинг YOLO разметки с 0-based классами (без +1 для background)"""
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
                if cls >= self.num_classes:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1 = max(0, (xc - w/2) * orig_w * scale_x)
                y1 = max(0, (yc - h/2) * orig_h * scale_y)
                x2 = min(self.img_size[1], (xc + w/2) * orig_w * scale_x)
                y2 = min(self.img_size[0], (yc + h/2) * orig_h * scale_y)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)  # 0-based, без +1
        return boxes, labels


def collate_fn(batch):
    images, targets, names = zip(*batch)
    return list(images), list(targets), list(names)


# ============================================================================
# МОДЕЛЬ
# ============================================================================

def create_model(num_classes: int) -> nn.Module:
    """Создание Faster R-CNN с ResNet-18 backbone"""
    base = resnet18(weights='DEFAULT')
    backbone = _resnet_fpn_extractor(base, trainable_layers=5)
    return FasterRCNN(backbone, num_classes=num_classes + 1)  # +1 для background


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Загрузка сохраненной модели"""
    logger.info(f"Loading model from: {model_path}")
    model = create_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Обрабатываем разные форматы чекпоинтов
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            logger.info(f"Training metrics: mAP@50={checkpoint['metrics'].get('mAP@50', 'N/A')}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# МЕТРИКИ (COCO через torchmetrics)
# ============================================================================

def compute_coco_metrics(predictions: List[Dict], 
                         ground_truths: List[Dict],
                         score_threshold: float = 0.25) -> Dict:
    """
    Вычисление COCO-метрик через torchmetrics.MeanAveragePrecision
    
    Args:
        predictions: список предсказаний в формате {"boxes":, "scores":, "labels":}
        ground_truths: список ground truth в формате {"boxes":, "labels":}
        score_threshold: порог уверенности для фильтрации предсказаний
    
    Returns:
        Dict с метриками mAP@50, mAP@75, mAP@50:95, per-class AP
    """
    metric = MeanAveragePrecision(
        iou_type="bbox",
        box_format="xyxy",
        class_metrics=True
    )
    
    # Фильтруем и конвертируем в нужный формат
    for pred, gt in zip(predictions, ground_truths):
        # Фильтрация по confidence
        keep = pred['scores'] > score_threshold
        
        pred_filtered = {
            'boxes': pred['boxes'][keep].cpu(),
            'scores': pred['scores'][keep].cpu(),
            'labels': pred['labels'][keep].cpu().long(),
        }
        
        gt_formatted = {
            'boxes': gt['boxes'].cpu(),
            'labels': gt['labels'].cpu().long(),
        }
        
        # Обновляем метрику для каждого изображения
        metric.update([pred_filtered], [gt_formatted])
    
    # Вычисляем финальные метрики
    result = metric.compute()
    
    metrics = {
        'mAP@50:95': float(result['map'].item()),
        'mAP@50': float(result['map_50'].item()),
        'mAP@75': float(result['map_75'].item()),
    }
    
    # Per-class AP на разных IoU
    if 'map_per_class' in result and result['map_per_class'].numel() > 0:
        per_class_ap = result['map_per_class'].tolist()
        for cls_id, ap in enumerate(per_class_ap):
            metrics[f'AP_class_{cls_id}'] = round(float(ap), 4)
    
    # MAR (Mean Average Recall)
    if 'mar_1' in result:
        metrics['mAR@1'] = float(result['mar_1'].item())
    if 'mar_10' in result:
        metrics['mAR@10'] = float(result['mar_10'].item())
    if 'mar_100' in result:
        metrics['mAR@100'] = float(result['mar_100'].item())
    
    return metrics


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

@torch.no_grad()
def test_model(model: nn.Module, 
               test_loader: DataLoader, 
               config: Dict,
               model_name: str) -> Dict:
    """Тестирует модель на датасете с использованием torchmetrics"""
    
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    inference_times = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {model_name}")
    logger.info(f"{'='*60}")
    
    for images, targets, image_names in tqdm(test_loader, desc=f"Testing {model_name}"):
        images = [img.to(device) for img in images]
        
        # Измеряем время инференса
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = time.time()
        
        outputs = model(images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)
        
        # Конвертируем предсказания в 0-based (без background)
        for output, target in zip(outputs, targets):
            all_predictions.append({
                'boxes': output['boxes'].cpu(),
                'scores': output['scores'].cpu(),
                'labels': (output['labels'] - 1).cpu(),  # -1 потому что модель обучалась с +1 для background
            })
            all_ground_truths.append({
                'boxes': target['boxes'].cpu(),
                'labels': target['labels'].cpu(),
            })
    
    # Вычисляем COCO-метрики
    metrics = compute_coco_metrics(
        all_predictions, 
        all_ground_truths, 
        config['score_threshold']
    )
    
    # Добавляем информацию о скорости
    if inference_times:
        avg_time = np.mean(inference_times) * 1000  # ms
        metrics['avg_inference_time_ms'] = round(avg_time, 2)
        metrics['fps'] = round(1000.0 / avg_time, 2) if avg_time > 0 else 0
    
    # Статистика по детекциям
    total_preds = sum(len(p['boxes'][p['scores'] > config['score_threshold']]) for p in all_predictions)
    total_gt = sum(len(g['boxes']) for g in all_ground_truths)
    metrics['total_predictions'] = total_preds
    metrics['total_ground_truth'] = total_gt
    
    # Выводим результаты
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {model_name} (COCO metrics via torchmetrics)")
    logger.info(f"{'='*60}")
    logger.info(f"mAP@50:95:  {metrics['mAP@50:95']:.4f}  ← главная COCO-метрика")
    logger.info(f"mAP@50:     {metrics['mAP@50']:.4f}")
    logger.info(f"mAP@75:     {metrics['mAP@75']:.4f}")
    
    if 'mAR@1' in metrics:
        logger.info(f"\nRecall metrics:")
        logger.info(f"mAR@1:      {metrics['mAR@1']:.4f}")
        logger.info(f"mAR@10:     {metrics['mAR@10']:.4f}")
        logger.info(f"mAR@100:    {metrics['mAR@100']:.4f}")
    
    logger.info(f"\nPer-class AP@50:95:")
    for key, value in metrics.items():
        if key.startswith('AP_class_'):
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"\nStatistics:")
    logger.info(f"Total predictions (after filtering): {total_preds}")
    logger.info(f"Total ground truth boxes: {total_gt}")
    logger.info(f"Score threshold: {config['score_threshold']}")
    
    if 'avg_inference_time_ms' in metrics:
        logger.info(f"\nSpeed:")
        logger.info(f"Avg inference time: {metrics['avg_inference_time_ms']} ms")
        logger.info(f"FPS: {metrics['fps']}")
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Создаем тестовый датасет
    test_path = Path(CONFIG['test_dataset_path'])
    
    if not (test_path / 'test').exists():
        logger.error(f"Test dataset not found at {test_path / 'test'}")
        logger.info("Expected structure: data/experiment_v3/datasets/mixed_full/test/")
        logger.info("  test/images/")
        logger.info("  test/labels/")
        sys.exit(1)
    
    test_dataset = YOLODataset(
        test_path / 'test' / 'images',
        test_path / 'test' / 'labels',
        CONFIG['num_classes'],
        CONFIG['img_size']
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)} images")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collate_fn
    )
    
    # Создаем директорию для результатов
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Тестируем все модели
    all_results = {}
    
    for model_name, model_path in CONFIG['models'].items():
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"{'='*60}")
        model = load_model(model_path, CONFIG['num_classes'], device)
        
        # Тестируем
        metrics = test_model(model, test_loader, CONFIG, model_name)
        all_results[model_name] = metrics
        
        # Сохраняем индивидуальные результаты
        model_results_path = output_dir / f"{model_name}_coco_metrics.json"
        with open(model_results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to: {model_results_path}")
    
    # Сохраняем сравнение
    if len(all_results) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("MODELS COMPARISON (COCO metrics)")
        logger.info(f"{'='*70}")
        
        # Заголовок таблицы
        header = f"{'Model':<20} {'mAP@50:95':>10} {'mAP@50':>8} {'mAP@75':>8} {'FPS':>8} {'Params':>10}"
        logger.info(header)
        logger.info("-" * 70)
        
        # Данные по моделям
        for model_name, metrics in all_results.items():
            fps = metrics.get('fps', 'N/A')
            logger.info(f"{model_name:<20} {metrics['mAP@50:95']:>10.4f} "
                       f"{metrics['mAP@50']:>8.4f} {metrics['mAP@75']:>8.4f} "
                       f"{fps:>8.1f}")
        
        # Сравнение улучшений (если есть baseline)
        if 'baseline' in all_results:
            logger.info(f"\n{'='*70}")
            logger.info("IMPROVEMENTS OVER BASELINE")
            logger.info(f"{'='*70}")
            
            baseline = all_results['baseline']
            for model_name, metrics in all_results.items():
                if model_name == 'baseline':
                    continue
                
                logger.info(f"\n{model_name}:")
                for metric in ['mAP@50:95', 'mAP@50', 'mAP@75']:
                    base_val = baseline[metric]
                    curr_val = metrics[metric]
                    improvement = curr_val - base_val
                    logger.info(f"  {metric}: {base_val:.4f} → {curr_val:.4f} "
                               f"({improvement:+.4f}, {improvement/base_val*100:+.1f}%)")
    
    # Сохраняем общий отчет
    summary = {
        'test_dataset': str(test_path / 'test'),
        'num_test_images': len(test_dataset),
        'score_threshold': CONFIG['score_threshold'],
        'metric_type': 'COCO via torchmetrics.MeanAveragePrecision',
        'results': all_results,
    }
    
    summary_path = output_dir / "coco_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Testing completed!")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()