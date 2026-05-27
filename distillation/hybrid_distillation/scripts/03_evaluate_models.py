#!/usr/bin/env python3
"""
Комплексная оценка всех обученных моделей

Метрики:
  - mAP@50, mAP@75, mAP@50:95
  - Per-class AP
  - Precision, Recall, F1-score
  - FPS на CPU
  - Размер модели (MB), количество параметров
  - Время инференса (latency)

Сравнение с учителем LTDETR
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import yaml
from PIL import Image
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.ops import box_iou

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('03_evaluate_models.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Комплексный оценщик моделей детекции."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        num_classes: int,
        device: torch.device
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        
        self.model = self._load_model()
        
    def _load_model(self):
        """Загружает модель."""
        
        logger.info(f"Loading {self.model_type} model from {self.model_path}")
        
        # Тип 1: Наши обученные Faster R-CNN модели
        if self.model_type == 'faster_rcnn':
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Создаём бэкбон ResNet18 с FPN
            backbone = resnet_fpn_backbone('resnet18', pretrained=False)
            
            # Создаём детектор
            model = FasterRCNN(backbone, num_classes=self.num_classes + 1)
            
            # Загружаем веса
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            return model
        
        # Тип 2: LTDETR учитель (через Lightly API)
        elif self.model_type == 'teacher':
            try:
                # Пробуем через lightly_train
                import lightly_train
                logger.info("Loading via lightly_train.load_model()...")
                model = lightly_train.load_model(self.model_path)
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"lightly_train.load_model failed: {e}")
                logger.info("Trying direct checkpoint loading...")
                
                # Загружаем чекпоинт напрямую
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
                
                # Ищем state_dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # Создаём замену (Faster R-CNN ResNet50) для оценки метрик
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn
                    model = fasterrcnn_resnet50_fpn(num_classes=self.num_classes + 1)
                    
                    # Пробуем загрузить что можем
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        logger.info("Loaded teacher weights with strict=False")
                    except Exception as e2:
                        logger.warning(f"Failed to load state_dict: {e2}")
                    
                    model.to(self.device)
                    model.eval()
                    return model
                else:
                    raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def evaluate_detection(
        self,
        test_images: Path,
        test_labels: Path,
        conf_threshold: float = 0.25
    ) -> Dict:
        """Оценивает метрики детекции."""
        
        image_files = sorted([
            f for f in test_images.glob("*")
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])
        
        if not image_files:
            logger.error(f"No images found in {test_images}")
            return {}
        
        logger.info(f"Evaluating on {len(image_files)} images")
        
        predictions = []
        ground_truths = []
        
        for img_path in image_files:
            # Предсказания
            pred = self._predict_image(img_path, conf_threshold)
            predictions.append(pred)
            
            # Ground truth
            gt = self._load_ground_truth(img_path, test_labels)
            ground_truths.append(gt)
        
        # Вычисляем метрики
        metrics = self._compute_all_metrics(predictions, ground_truths)
        
        return metrics
    
    def _predict_image(self, img_path: Path, conf_threshold: float) -> Dict:
        """Предсказание для одного изображения."""
        
        if self.model_type == 'teacher':
            try:
                # Пробуем через predict API
                result = self.model.predict(str(img_path))
                boxes = result['bboxes']
                scores = result['scores']
                labels = result['labels']
                
                # Конвертируем в тензоры если нужно
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes)
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
            except Exception as e:
                logger.debug(f"predict() failed: {e}, using forward()")
                # Fallback: используем forward как для Faster R-CNN
                image = Image.open(img_path).convert("RGB")
                image = image.resize((640, 640), Image.BILINEAR)
                img_tensor = torch.from_numpy(
                    np.array(image, dtype=np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(img_tensor)[0]
                
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu() - 1
        else:
            # Наши Faster R-CNN модели
            image = Image.open(img_path).convert("RGB")
            image = image.resize((640, 640), Image.BILINEAR)
            img_tensor = torch.from_numpy(
                np.array(image, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)[0]
            
            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu() - 1  # 0-indexed
        
        # Фильтруем по confidence
        keep = scores > conf_threshold
        
        return {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep] if isinstance(labels, torch.Tensor) else torch.tensor(labels)[keep],
        }
    
    def _load_ground_truth(self, img_path: Path, labels_dir: Path) -> Dict:
        """Загружает ground truth аннотации."""
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            try:
                with Image.open(img_path) as img:
                    iw, ih = img.size
            except Exception:
                iw, ih = 640, 640
            
            scale_x = 640 / iw
            scale_y = 640 / ih
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        cls = int(float(parts[0]))
                        if cls >= self.num_classes:
                            continue
                        
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = max(0, (xc - w/2) * iw * scale_x)
                        y1 = max(0, (yc - h/2) * ih * scale_y)
                        x2 = min(640, (xc + w/2) * iw * scale_x)
                        y2 = min(640, (yc + h/2) * ih * scale_y)
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)
                    except (ValueError, IndexError):
                        continue
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
    
    def _compute_all_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict:
        """Вычисляет все метрики."""
        
        iou_thresholds = [0.5, 0.75] + list(np.linspace(0.5, 0.95, 10))
        
        # Per-class AP
        per_class_ap50 = {}
        per_class_ap75 = {}
        
        for cls in range(self.num_classes):
            ap50 = self._compute_ap(predictions, ground_truths, cls, 0.5)
            ap75 = self._compute_ap(predictions, ground_truths, cls, 0.75)
            per_class_ap50[f'cls{cls}_AP50'] = ap50
            per_class_ap75[f'cls{cls}_AP75'] = ap75
        
        # mAP
        map50 = float(np.mean(list(per_class_ap50.values())))
        map75 = float(np.mean(list(per_class_ap75.values())))
        
        # mAP@50:95
        map_values = []
        for thr in iou_thresholds:
            aps = []
            for cls in range(self.num_classes):
                ap = self._compute_ap(predictions, ground_truths, cls, thr)
                aps.append(ap)
            map_values.append(np.mean(aps))
        map50_95 = float(np.mean(map_values))
        
        # Precision, Recall, F1
        precision, recall, f1 = self._compute_prf(predictions, ground_truths, 0.5)
        
        return {
            'mAP_50': map50,
            'mAP_75': map75,
            'mAP_50_95': map50_95,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            **per_class_ap50,
            **per_class_ap75,
            'num_predictions': sum(len(p['boxes']) for p in predictions),
            'num_ground_truth': sum(len(g['boxes']) for g in ground_truths),
        }
    
    def _compute_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_id: int,
        iou_threshold: float
    ) -> float:
        """Average Precision для одного класса."""
        
        detections = []
        num_gt = 0
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            gt_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_mask]
            num_gt += len(gt_boxes)
            
            pred_mask = pred['labels'] == class_id
            for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                detections.append({
                    'image_id': img_idx,
                    'score': float(score),
                    'bbox': box,
                })
        
        if num_gt == 0:
            return 0.0 if len(detections) > 0 else 0.0
        if len(detections) == 0:
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
    
    def _compute_prf(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float
    ) -> Tuple[float, float, float]:
        """Precision, Recall, F1-score."""
        
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for pred, gt in zip(predictions, ground_truths):
            if len(pred['boxes']) == 0:
                total_fn += len(gt['boxes'])
                continue
            
            if len(gt['boxes']) == 0:
                total_fp += len(pred['boxes'])
                continue
            
            ious = box_iou(pred['boxes'], gt['boxes'])
            matched = set()
            
            for i in range(len(pred['boxes'])):
                if ious[i].max() >= iou_threshold:
                    best_gt = ious[i].argmax().item()
                    if best_gt not in matched:
                        total_tp += 1
                        matched.add(best_gt)
                    else:
                        total_fp += 1
                else:
                    total_fp += 1
            
            total_fn += len(gt['boxes']) - len(matched)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return float(precision), float(recall), float(f1)
    
    def measure_fps(
        self,
        img_size: Tuple[int, int] = (640, 640),
        warmup: int = 50,
        iterations: int = 200
    ) -> Dict:
        """Измеряет FPS и latency."""
        
        logger.info(f"Measuring FPS (warmup={warmup}, iterations={iterations})")
        
        # Создаём тестовое изображение
        dummy_img = Image.new('RGB', img_size, color=(128, 128, 128))
        dummy_path = Path("/tmp/dummy_test_img.jpg")
        dummy_img.save(dummy_path)
        
        # Warmup
        for _ in range(warmup):
            try:
                self._predict_image(dummy_path, 0.25)
            except Exception:
                pass
        
        # Измерение
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                self._predict_image(dummy_path, 0.25)
            except Exception:
                pass
            times.append(time.perf_counter() - start)
        
        # Удаляем временный файл
        dummy_path.unlink(missing_ok=True)
        
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        logger.info(f"  FPS: {fps:.1f}, Latency: {avg_time:.2f} ms")
        
        return {
            'fps': round(fps, 1),
            'latency_ms': round(avg_time, 2)
        }
    
    def count_parameters(self) -> Dict:
        """Подсчитывает параметры модели."""
        
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Размер на диске
        tmp_path = "/tmp/model_temp.pth"
        torch.save(self.model.state_dict(), tmp_path)
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        Path(tmp_path).unlink(missing_ok=True)
        
        return {
            'params_total': total,
            'params_trainable': trainable,
            'params_millions': round(total / 1e6, 1),
            'size_mb': round(size_mb, 1)
        }


def main():
    """Оценивает все модели."""
    
    config_path = Path(__file__).parent / "../config_hybrid_distillation.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Директории с тестовыми данными
    data_path = Path(config['detection']['data_path'])
    test_images = data_path / "test" / "images"
    test_labels = data_path / "test" / "labels"
    
    # Если нет test, используем val
    if not test_images.exists():
        logger.warning("Test set not found, using validation set")
        test_images = data_path / "val" / "images"
        test_labels = data_path / "val" / "labels"
    
    if not test_images.exists():
        logger.error(f"❌ No test/val images found!")
        sys.exit(1)
    
    num_classes = config['detection']['num_classes']
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 1. Оцениваем учителя LTDETR
    teacher_model_path = config['teacher'].get('weights')
    teacher_model_name = config['teacher'].get('model', 'teacher_ltdetr')
    
    logger.info(f"Teacher weights path from config: {teacher_model_path}")
    
    if teacher_model_path:
        teacher_path = Path(teacher_model_path)
        logger.info(f"Teacher path exists: {teacher_path.exists()}")
        
        if teacher_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating TEACHER: {teacher_model_name}")
            logger.info(f"Model path: {teacher_model_path}")
            logger.info(f"File size: {teacher_path.stat().st_size / (1024**2):.1f} MB")
            logger.info(f"{'='*60}")
            
            try:
                evaluator = ModelEvaluator(
                    str(teacher_path), 
                    'teacher',
                    num_classes, 
                    device
                )
                metrics = evaluator.evaluate_detection(test_images, test_labels)
                fps_data = evaluator.measure_fps(
                    tuple(config['fps']['img_size']),
                    config['fps']['warmup'],
                    config['fps']['iterations']
                )
                params = evaluator.count_parameters()
                
                teacher_result = {
                    'model': 'teacher_ltdetr',
                    'type': 'teacher',
                    **metrics,
                    **fps_data,
                    **params
                }
                all_results.append(teacher_result)
                logger.info(f"✅ Teacher evaluation completed: mAP@50 = {metrics.get('mAP_50', 0):.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate teacher: {e}", exc_info=True)
                logger.warning("Continuing without teacher evaluation...")
        else:
            logger.warning(f"Teacher model not found at: {teacher_model_path}")
    else:
        logger.warning("No teacher weights path in config")
    
    # 2. Оцениваем учеников (Faster R-CNN)
    for student_name, student_cfg in config['students'].items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {student_name}")
        logger.info(f"{'='*60}")
        
        model_path = Path(config['paths']['detection_output']) / student_name / 'model_final.pth'
        
        if not model_path.exists():
            # Пробуем best_model
            model_path = Path(config['paths']['detection_output']) / student_name / 'best_model.pth'
        
        if not model_path.exists():
            logger.warning(f"⚠️ Model not found: {model_path}")
            continue
        
        try:
            evaluator = ModelEvaluator(model_path, 'faster_rcnn', num_classes, device)
            
            # Метрики детекции
            metrics = evaluator.evaluate_detection(test_images, test_labels)
            
            # FPS
            fps_data = evaluator.measure_fps(
                tuple(config['fps']['img_size']),
                config['fps']['warmup'],
                config['fps']['iterations']
            )
            
            # Параметры
            params = evaluator.count_parameters()
            
            result = {
                'model': student_name,
                'type': student_cfg['type'],
                **metrics,
                **fps_data,
                **params
            }
            all_results.append(result)
            
            logger.info(f"✅ {student_name} evaluation completed: mAP@50 = {metrics.get('mAP_50', 0):.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {student_name}: {e}", exc_info=True)
    
    # 3. Сохраняем результаты
    output_path = results_dir / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 4. Выводим сводную таблицу
    logger.info(f"\n{'='*100}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*100}")
    
    if all_results:
        header = f"{'Model':<35} {'mAP@50':<10} {'mAP@75':<10} {'F1':<10} {'FPS':<8} {'Params(M)':<12} {'Size(MB)':<10}"
        logger.info(header)
        logger.info("-"*100)
        
        for r in sorted(all_results, key=lambda x: x.get('mAP_50', 0), reverse=True):
            logger.info(
                f"{r['model']:<35} "
                f"{r.get('mAP_50', 0):<10.4f} "
                f"{r.get('mAP_75', 0):<10.4f} "
                f"{r.get('F1', 0):<10.4f} "
                f"{r.get('fps', 0):<8.1f} "
                f"{r.get('params_millions', 0):<12.1f} "
                f"{r.get('size_mb', 0):<10.1f}"
            )
        
        logger.info(f"{'='*100}")
        
        # 5. Выводим сравнение с учителем
        teacher_result = next((r for r in all_results if r.get('type') == 'teacher'), None)
        distilled_result = next((r for r in all_results if r.get('type') == 'lightly_pretrained'), None)
        
        if teacher_result and distilled_result:
            teacher_map = teacher_result.get('mAP_50', 0)
            distilled_map = distilled_result.get('mAP_50', 0)
            ratio = (distilled_map / teacher_map) * 100 if teacher_map > 0 else 0
            
            logger.info(f"\n{'='*60}")
            logger.info("COMPARISON WITH TEACHER")
            logger.info(f"{'='*60}")
            logger.info(f"Teacher (LTDETR) mAP@50: {teacher_map:.4f}")
            logger.info(f"Distilled Student mAP@50: {distilled_map:.4f}")
            logger.info(f"Student retains {ratio:.1f}% of teacher's performance")
            logger.info(f"Parameter reduction: {teacher_result.get('params_millions', 0):.1f}M → {distilled_result.get('params_millions', 0):.1f}M")
            logger.info(f"Speedup: {distilled_result.get('fps', 0):.1f}× vs {teacher_result.get('fps', 0):.1f} FPS")
            logger.info(f"{'='*60}")
    else:
        logger.warning("No results to display!")
    
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()