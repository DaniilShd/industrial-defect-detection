#!/usr/bin/env python3
"""
Оценка всех моделей с едиными COCO-метриками + анализ сложности датасета.

Метрики: mAP50:95, mAP50, mAP75, per-class AP, FPS, latency, параметры, размер.
Дополнительно: анализ распределения объектов по размерам (small/medium/large).

Учитель LTDETR → предсказания через модель LightlyTrain или torchvision
Ученики Faster R-CNN → torchvision forward
Всё через torchmetrics.MeanAveragePrecision (COCO-совместимо)
"""

import json
import logging
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("03_evaluate.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════

def load_yolo_gt(img_path: Path, labels_dir: Path, num_classes: int,
                  img_size: Tuple[int, int] = (640, 640)) -> Dict:
    """Загрузка ground truth из YOLO формата."""
    lbl_path = labels_dir / f"{img_path.stem}.txt"
    boxes, labels = [], []
    
    if not lbl_path.exists():
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    try:
        w0, h0 = Image.open(img_path).size
    except Exception:
        w0, h0 = img_size[1], img_size[0]
    
    sx, sy = img_size[1] / w0, img_size[0] / h0

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                if cls >= num_classes:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1 = max(0.0, (xc - w/2) * w0 * sx)
                y1 = max(0.0, (yc - h/2) * h0 * sy)
                x2 = min(float(img_size[1]), (xc + w/2) * w0 * sx)
                y2 = min(float(img_size[0]), (yc + h/2) * h0 * sy)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
            except (ValueError, IndexError):
                continue

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
    }


def predict_teacher(model, img_path: Path, img_size: Tuple[int, int],
                    conf: float = 0.25) -> Dict:
    if hasattr(model, 'predict'):
        try:
            results = model.predict(str(img_path))  # Без image_size
            boxes = results["bboxes"]
            scores = results["scores"]
            labels = results["labels"]
            
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu()
                scores = scores.cpu()
                labels = labels.cpu()
            
            keep = scores > conf
            return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        except Exception as e:
            logger.error(f"predict() failed: {e}")
            return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}
    else:
        # Fallback, если нет predict
        return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}

def predict_student(model: FasterRCNN, img_path: Path,
                    img_size: Tuple[int, int], device: torch.device,
                    conf: float = 0.25) -> Dict:
    """Предсказание студента (Faster R-CNN)."""
    try:
        img = Image.open(img_path).convert("RGB").resize(img_size, Image.BILINEAR)
        tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Ошибка загрузки {img_path}: {e}")
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64)}

    with torch.no_grad():
        out = model(tensor)[0]

    keep = out["scores"] > conf
    return {
        "boxes": out["boxes"][keep].cpu(),
        "scores": out["scores"][keep].cpu(),
        "labels": (out["labels"][keep] - 1).cpu().clamp(min=0),  # 0 = фон -> 0 = первый класс
    }


def evaluate_model(predict_fn, image_files: List[Path], labels_dir: Path,
                   num_classes: int, img_size: Tuple[int, int],
                   class_names: Optional[List[str]] = None) -> Dict:
    """Расчёт COCO-метрик для модели."""
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy",
                                   class_metrics=True)
    
    n_processed = 0
    for img_path in image_files:
        pred = predict_fn(img_path)
        gt = load_yolo_gt(img_path, labels_dir, num_classes, img_size)
        
        if len(pred["boxes"]) > 0 or len(gt["boxes"]) > 0:
            metric.update([pred], [gt])
            n_processed += 1

    logger.info(f"  Обработано изображений с объектами: {n_processed}/{len(image_files)}")
    
    result = metric.compute()

    out = {
        "map50_95": float(result["map"].item()),
        "map50": float(result["map_50"].item()),
        "map75": float(result["map_75"].item()),
    }

    # Per-class AP
    if "map_per_class" in result and result["map_per_class"].numel() > 0:
        per_cls = result["map_per_class"].tolist()
        names = class_names or [f"cls{i}" for i in range(len(per_cls))]
        for i, (name, ap) in enumerate(zip(names, per_cls)):
            out[f"AP50_{name}"] = round(float(ap), 4)

    return out


def measure_fps(predict_fn, img_path: Path, warmup: int = 30,
                iterations: int = 100) -> Dict:
    """Измерение FPS и задержки."""
    # Прогрев
    for _ in range(warmup):
        try:
            predict_fn(img_path)
        except Exception:
            pass

    # Измерения
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            predict_fn(img_path)
        except Exception:
            pass
        times.append(time.perf_counter() - t0)

    if not times:
        return {"fps": 0.0, "latency_ms": 0.0}

    avg_lat = np.mean(times) * 1000  # мс
    return {"fps": round(1000/avg_lat, 1), "latency_ms": round(avg_lat, 2)}


def model_stats(model: torch.nn.Module) -> Dict:
    """Подсчёт параметров и размера модели."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Размер на диске
    tmp = Path("/tmp/_eval_model.pth")
    torch.save(model.state_dict(), tmp)
    size_mb = tmp.stat().st_size / (1024**2)
    tmp.unlink(missing_ok=True)
    
    return {
        "params_M": round(total/1e6, 1),
        "trainable_M": round(trainable/1e6, 1),
        "size_mb": round(size_mb, 1)
    }


def analyze_dataset_difficulty(image_files: List[Path], labels_dir: Path,
                               num_classes: int, img_size: Tuple[int, int],
                               class_names: Optional[List[str]] = None) -> Dict:
    """
    Анализ сложности датасета: распределение объектов по размерам.
    
    COCO-определения размеров:
    - small: area < 32² pixels
    - medium: 32² <= area < 96² pixels  
    - large: area >= 96² pixels
    """
    stats = {
        'num_images': len(image_files),
        'total_objects': 0,
        'small': 0,
        'medium': 0,
        'large': 0,
        'objects_per_image': [],
        'objects_per_class': defaultdict(int),
        'areas': [],
    }
    
    logger.info(f"\n{'='*70}")
    logger.info("АНАЛИЗ СЛОЖНОСТИ ДАТАСЕТА")
    logger.info(f"{'='*70}")
    
    for img_path in tqdm(image_files, desc="Analyzing dataset"):
        gt = load_yolo_gt(img_path, labels_dir, num_classes, img_size)
        boxes = gt['boxes']
        labels = gt['labels']
        
        if len(boxes) == 0:
            stats['objects_per_image'].append(0)
            continue
        
        stats['total_objects'] += len(boxes)
        stats['objects_per_image'].append(len(boxes))
        
        for box, label in zip(boxes, labels):
            area = ((box[2] - box[0]) * (box[3] - box[1])).item()
            stats['areas'].append(area)
            stats['objects_per_class'][label.item()] += 1
            
            if area < 32**2:
                stats['small'] += 1
            elif area < 96**2:
                stats['medium'] += 1
            else:
                stats['large'] += 1
    
    # Вычисляем производные метрики
    total = stats['total_objects']
    stats['pct_small'] = stats['small'] / max(total, 1) * 100
    stats['pct_medium'] = stats['medium'] / max(total, 1) * 100
    stats['pct_large'] = stats['large'] / max(total, 1) * 100
    stats['avg_objects_per_image'] = np.mean(stats['objects_per_image'])
    stats['median_objects_per_image'] = np.median(stats['objects_per_image'])
    stats['max_objects_per_image'] = max(stats['objects_per_image']) if stats['objects_per_image'] else 0
    
    if stats['areas']:
        stats['avg_box_area'] = np.mean(stats['areas'])
        stats['median_box_area'] = np.median(stats['areas'])
        stats['min_box_area'] = np.min(stats['areas'])
        stats['max_box_area'] = np.max(stats['areas'])
    
    # Вывод результатов
    logger.info(f"Изображений: {stats['num_images']}")
    logger.info(f"Всего объектов: {stats['total_objects']}")
    logger.info(f"Среднее объектов на изображение: {stats['avg_objects_per_image']:.1f}")
    logger.info(f"Медиана объектов на изображение: {stats['median_objects_per_image']:.1f}")
    logger.info(f"Максимум объектов на изображение: {stats['max_objects_per_image']}")
    logger.info(f"\nРаспределение по размерам (COCO-определение):")
    logger.info(f"  Small (<32² px):     {stats['small']:5d} ({stats['pct_small']:5.1f}%)")
    logger.info(f"  Medium (32²-96² px): {stats['medium']:5d} ({stats['pct_medium']:5.1f}%)")
    logger.info(f"  Large (>96² px):     {stats['large']:5d} ({stats['pct_large']:5.1f}%)")
    
    if stats['areas']:
        logger.info(f"\nСтатистика площадей (px²):")
        logger.info(f"  Средняя: {stats['avg_box_area']:.1f}")
        logger.info(f"  Медиана: {stats['median_box_area']:.1f}")
        logger.info(f"  Мин:     {stats['min_box_area']:.1f}")
        logger.info(f"  Макс:    {stats['max_box_area']:.1f}")
    
    logger.info(f"\nРаспределение по классам:")
    for cls_id in sorted(stats['objects_per_class'].keys()):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
        count = stats['objects_per_class'][cls_id]
        logger.info(f"  {name}: {count:5d} ({count/max(total,1)*100:5.1f}%)")
    
    # Оценка сложности датасета
    logger.info(f"\n{'='*70}")
    logger.info("ОЦЕНКА СЛОЖНОСТИ:")
    
    if stats['pct_small'] > 50:
        logger.info("⚠️  В датасете ПРЕОБЛАДАЮТ МЕЛКИЕ ОБЪЕКТЫ (>50%)")
        logger.info("   Ожидайте низкие метрики mAP@50:95")
        logger.info("   Рекомендации: multi-scale training, увеличение разрешения")
    elif stats['pct_small'] > 30:
        logger.info("⚠️  Значительная доля мелких объектов (30-50%)")
        logger.info("   Рекомендации: использовать FPN, multi-scale аугментации")
    else:
        logger.info("✓  Разумный баланс размеров объектов")
    
    if stats['avg_objects_per_image'] < 2:
        logger.info("⚠️  Мало объектов на изображение (<2 в среднем)")
        logger.info("   Возможна проблема с редкими классами")
    
    logger.info(f"{'='*70}\n")
    
    return {k: v for k, v in stats.items() if k not in ['objects_per_image', 'areas']}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    
    if not cfg_path.exists():
        logger.error(f"Конфиг не найден: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")

    num_classes = cfg["detection"]["num_classes"]
    img_size = tuple(cfg["detection"]["img_size"])
    class_names = list(cfg["detection"].get("class_names", {}).values())
    data_path = Path(cfg["detection"]["data_path"])

    # Тестовые данные (fallback на val если test отсутствует)
    test_imgs = data_path / "test" / "images"
    test_lbls = data_path / "test" / "labels"
    if not test_imgs.exists():
        logger.warning("test/ не найден, используем val/")
        test_imgs = data_path / "val" / "images"
        test_lbls = data_path / "val" / "labels"

    if not test_imgs.exists():
        logger.error("❌ Нет изображений для оценки!")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(f for f in test_imgs.glob("*") if f.suffix.lower() in exts)
    
    if not image_files:
        logger.error("❌ Нет изображений в тестовой выборке!")
        sys.exit(1)
    
    logger.info(f"Оценка на {len(image_files)} изображениях")

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ═══════════════════════════════════════════════════
    # АНАЛИЗ СЛОЖНОСТИ ДАТАСЕТА
    # ═══════════════════════════════════════════════════
    dataset_stats = analyze_dataset_difficulty(
        image_files, test_lbls, num_classes, img_size, class_names
    )
    
    # Сохраняем статистику датасета
    stats_path = results_dir / "dataset_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    logger.info(f"Статистика датасета сохранена в: {stats_path}")

    # ═══════════════════════════════════════════════════
    # ОЦЕНКА УЧИТЕЛЯ (LTDETR)
    # ═══════════════════════════════════════════════════
    detector_path = cfg["teacher"].get("detector_path")
    if detector_path and Path(detector_path).exists():
        logger.info(f"\n{'='*70}\nОЦЕНКА УЧИТЕЛЯ (LTDETR)\n{'='*70}")

        try:
            import lightly_train
            logger.info(f"Загрузка учителя: {detector_path}")
            teacher_model = lightly_train.load_model(detector_path)
            teacher_model.eval()

            predict_fn = lambda p: predict_teacher(teacher_model, p, img_size)

            metrics = evaluate_model(predict_fn, image_files, test_lbls,
                                     num_classes, img_size, class_names)
            fps = measure_fps(predict_fn, image_files[0],
                              cfg["fps"]["warmup"], cfg["fps"]["iterations"])
            stats = model_stats(teacher_model)

            result = {
                "model": "teacher_ltdetr",
                "type": "teacher",
                **metrics, **fps, **stats,
            }
            all_results.append(result)

            logger.info(f"✅ Учитель | mAP50:95={metrics['map50_95']:.4f} | "
                        f"mAP50={metrics['map50']:.4f} | FPS={fps['fps']:.1f}")
        except Exception as e:
            logger.error(f"❌ Ошибка оценки учителя: {e}", exc_info=True)
    else:
        logger.warning("⚠️  Учитель не найден, пропускаем оценку")

    # ═══════════════════════════════════════════════════
    # ОЦЕНКА УЧЕНИКОВ (Faster R-CNN с разной инициализацией)
    # ═══════════════════════════════════════════════════
    for name, group_cfg in cfg["students"].items():
        det_out = Path(cfg["paths"]["detection_output"]) / name
        
        # Ищем чекпоинт
        ckpt = det_out / "model_final.pth"
        if not ckpt.exists():
            ckpt = det_out / "best_model.pth"
        if not ckpt.exists():
            logger.warning(f"⚠️  Чекпоинт не найден: {name}")
            continue

        logger.info(f"\n{'='*70}\nОЦЕНКА: {name} ({group_cfg['type']})\n{'='*70}")

        try:
            # Создаём и загружаем модель
            backbone = resnet_fpn_backbone("resnet18", pretrained=False)
            model = FasterRCNN(backbone, num_classes=num_classes + 1)
            
            ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ckpt_data.get("model_state_dict", ckpt_data)
            model.load_state_dict(sd)
            model.to(device).eval()

            predict_fn = lambda p, m=model: predict_student(m, p, img_size, device)

            metrics = evaluate_model(predict_fn, image_files, test_lbls,
                                     num_classes, img_size, class_names)
            fps = measure_fps(predict_fn, image_files[0],
                              cfg["fps"]["warmup"], cfg["fps"]["iterations"])
            stats = model_stats(model)

            result = {
                "model": name,
                "type": group_cfg["type"],
                **metrics, **fps, **stats,
            }
            all_results.append(result)

            logger.info(f"✅ {name} | mAP50:95={metrics['map50_95']:.4f} | "
                        f"mAP50={metrics['map50']:.4f} | FPS={fps['fps']:.1f}")
        except Exception as e:
            logger.error(f"❌ Ошибка оценки {name}: {e}", exc_info=True)

    # ═══════════════════════════════════════════════════
    # СОХРАНЕНИЕ И ВЫВОД РЕЗУЛЬТАТОВ
    # ═══════════════════════════════════════════════════
    out_path = results_dir / "evaluation.json"
    
    # Добавляем статистику датасета в результаты
    final_results = {
        "dataset_analysis": dataset_stats,
        "models": all_results,
        "evaluation_params": {
            "num_test_images": len(image_files),
            "img_size": img_size,
            "num_classes": num_classes,
            "class_names": class_names,
        }
    }
    
    out_path.write_text(json.dumps(final_results, indent=2))
    logger.info(f"\nРезультаты сохранены в: {out_path}")

    if not all_results:
        logger.warning("⚠️  Нет результатов для отображения")
        return

    # Итоговая таблица
    logger.info(f"\n{'='*90}")
    logger.info("ИТОГОВОЕ СРАВНЕНИЕ (COCO-метрики)")
    logger.info(f"{'='*90}")
    logger.info(f"{'Модель':<35} {'mAP50:95':>10} {'mAP50':>8} {'mAP75':>8} {'FPS':>7} {'Params':>8} {'Size':>7}")
    logger.info("-" * 90)

    for r in sorted(all_results, key=lambda x: x.get("map50_95", 0), reverse=True):
        logger.info(
            f"{r['model']:<35} {r.get('map50_95',0):>10.4f} "
            f"{r.get('map50',0):>8.4f} {r.get('map75',0):>8.4f} "
            f"{r.get('fps',0):>7.1f} {r.get('params_M',0):>7.1f}M "
            f"{r.get('size_mb',0):>6.1f}MB"
        )

    logger.info(f"{'='*90}")

    # Интерпретация результатов с учётом сложности датасета
    logger.info(f"\n{'='*70}")
    logger.info("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    logger.info(f"{'='*70}")
    
    if dataset_stats['pct_small'] > 30:
        logger.info(f"⚠️  Высокая доля мелких объектов ({dataset_stats['pct_small']:.1f}%)")
        logger.info("   объясняет относительно низкие значения mAP@50:95.")
        logger.info("   Для улучшения рекомендуется:")
        logger.info("   - Multi-scale training/testing")
        logger.info("   - Увеличение входного разрешения")
        logger.info("   - Аугментации, улучшающие детекцию мелких объектов")
    
    if dataset_stats['avg_objects_per_image'] < 3:
        logger.info(f"⚠️  Низкая плотность объектов ({dataset_stats['avg_objects_per_image']:.1f}/изобр.)")
        logger.info("   может приводить к нестабильным метрикам на малых выборках.")

    # Сравнение студента с учителем
    teacher_r = next((r for r in all_results if r.get("type") == "teacher"), None)
    distilled_r = next((r for r in all_results if r.get("type") == "lightly_pretrained"), None)
    scratch_r = next((r for r in all_results if r.get("type") == "scratch"), None)
    imagenet_r = next((r for r in all_results if r.get("type") == "imagenet_pretrained"), None)

    if teacher_r:
        logger.info(f"\n📊 Учитель (LTDETR):")
        logger.info(f"   mAP50:95 = {teacher_r.get('map50_95', 0):.4f}")
        logger.info(f"   FPS      = {teacher_r.get('fps', 0):.1f}")
        logger.info(f"   Параметры = {teacher_r.get('params_M', 0):.1f}M")

    if distilled_r:
        logger.info(f"\n📊 Студент (distilled):")
        logger.info(f"   mAP50:95 = {distilled_r.get('map50_95', 0):.4f}")
        logger.info(f"   FPS      = {distilled_r.get('fps', 0):.1f}")
        logger.info(f"   Параметры = {distilled_r.get('params_M', 0):.1f}M")
        
        if teacher_r:
            ratio = distilled_r["map50_95"] / max(teacher_r["map50_95"], 1e-6) * 100
            speedup = distilled_r["fps"] / max(teacher_r["fps"], 1e-6)
            logger.info(f"   Сохраняет {ratio:.1f}% точности учителя")
            logger.info(f"   Ускорение: ×{speedup:.1f}")
            logger.info(f"   Компрессия: {teacher_r['params_M']}M → {distilled_r['params_M']}M")

    if scratch_r and imagenet_r and distilled_r:
        logger.info(f"\n📊 Сравнение инициализаций ResNet18:")
        logger.info(f"   Scratch:   mAP50:95 = {scratch_r.get('map50_95', 0):.4f}")
        logger.info(f"   ImageNet:  mAP50:95 = {imagenet_r.get('map50_95', 0):.4f}")
        logger.info(f"   Distilled: mAP50:95 = {distilled_r.get('map50_95', 0):.4f}")
        
        distill_vs_scratch = distilled_r["map50_95"] / max(scratch_r["map50_95"], 1e-6)
        distill_vs_imagenet = distilled_r["map50_95"] / max(imagenet_r["map50_95"], 1e-6)
        logger.info(f"   Distilled / Scratch:  ×{distill_vs_scratch:.2f}")
        logger.info(f"   Distilled / ImageNet: ×{distill_vs_imagenet:.2f}")


if __name__ == "__main__":
    main()