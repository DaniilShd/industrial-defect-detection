#!/usr/bin/env python3
"""Визуализация предсказаний LT-DETR используя правильный API"""

import cv2
import sys
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_model(model_path):
    """Загрузка модели через lightly_train"""
    try:
        from lightly_train import load_model
        model = load_model(str(model_path))
        logger.info("Модель загружена")
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        return None

def predict_boxes(model, image_path, conf_threshold=0.001, use_sahi=False):
    """
    Предсказание bbox используя API модели
    
    Args:
        model: загруженная модель
        image_path: путь к изображению
        conf_threshold: порог уверенности
        use_sahi: использовать ли SAHI для лучшего обнаружения мелких объектов
    """
    try:
        # Используем официальный API модели
        if use_sahi:
            results = model.predict_sahi(image=str(image_path))
        else:
            results = model.predict(str(image_path))
        
        # Извлекаем результаты
        boxes = []
        
        # По документации результаты содержат ключи: "bboxes", "labels", "scores"
        if isinstance(results, dict):
            bboxes = results.get("bboxes", results.get("boxes", []))
            labels = results.get("labels", [])
            scores = results.get("scores", [])
            
            # Конвертируем тензоры в numpy если нужно
            if torch.is_tensor(bboxes):
                bboxes = bboxes.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            
            # Фильтруем по порогу уверенности
            for i in range(len(bboxes)):
                score = scores[i] if i < len(scores) else 1.0
                if score >= conf_threshold:
                    box = bboxes[i]
                    label = labels[i] if i < len(labels) else 0
                    
                    # Координаты уже в абсолютных пикселях (xmin, ymin, xmax, ymax)
                    if len(box) == 4:
                        x1, y1, x2, y2 = map(int, box)
                        boxes.append((int(label), x1, y1, x2, y2, float(score)))
        
        return boxes
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании для {image_path}: {e}", exc_info=True)
        return []

def read_yolo_labels(label_path, img_shape):
    """Чтение YOLO формата"""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    img_h, img_w = img_shape[:2]
                    x1 = int((x_center - width/2) * img_w)
                    y1 = int((y_center - height/2) * img_h)
                    x2 = int((x_center + width/2) * img_w)
                    y2 = int((y_center + height/2) * img_h)
                    boxes.append((int(class_id), x1, y1, x2, y2))
    
    return boxes

def draw_boxes(image, gt_boxes, pred_boxes, class_names=None):
    """Рисование bbox на изображении"""
    img_copy = image.copy()
    
    # Цвета
    gt_color = (0, 255, 0)      # Зеленый - GT
    pred_color = (255, 0, 0)    # Синий - предсказания
    
    # Рисуем GT box'ы
    for box in gt_boxes:
        class_id, x1, y1, x2, y2 = box[:5]
        class_name = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
        
        # Рисуем прямоугольник
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), gt_color, 2)
        
        # Добавляем подпись
        label = f"GT: {class_name}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x1, y1 - label_h - 5), (x1 + label_w, y1), gt_color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Рисуем предсказанные box'ы
    for box in pred_boxes:
        class_id, x1, y1, x2, y2, conf = box[:6]
        class_name = class_names.get(class_id, str(class_id)) if class_names else str(class_name)
        
        # Рисуем прямоугольник
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), pred_color, 2)
        
        # Добавляем подпись с уверенностью
        label = f"Pred: {class_name} ({conf:.2f})"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Позиционируем подпись сверху или снизу в зависимости от места
        if y1 - label_h - 5 > 0:
            cv2.rectangle(img_copy, (x1, y1 - label_h - 5), (x1 + label_w, y1), pred_color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(img_copy, (x1, y2 + 5), (x1 + label_w, y2 + label_h + 5), pred_color, -1)
            cv2.putText(img_copy, label, (x1, y2 + label_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_copy

def visualize_predictions(images_dir, labels_dir, output_dir, model, class_names=None, 
                         conf_threshold=0.001, use_sahi=False):
    """Визуализация с предсказаниями модели"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем все изображения
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(images_dir.glob(f"*{ext}"))
        images.extend(images_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"Найдено {len(images)} изображений")
    logger.info(f"Использовать SAHI: {use_sahi}")
    
    if len(images) == 0:
        logger.error(f"Нет изображений в {images_dir}")
        return
    
    stats = {
        'total_gt': 0,
        'total_pred': 0,
        'images_processed': 0,
        'max_pred': 0
    }
    
    for img_path in tqdm(images, desc="Визуализация"):
        # Загрузка изображения
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Не удалось загрузить {img_path}")
            continue
        
        # Чтение GT
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = read_yolo_labels(label_path, img.shape)
        stats['total_gt'] += len(gt_boxes)
        
        # Предсказание
        pred_boxes = predict_boxes(model, img_path, conf_threshold, use_sahi)
        stats['total_pred'] += len(pred_boxes)
        stats['max_pred'] = max(stats['max_pred'], len(pred_boxes))
        
        # Рисование
        result_img = draw_boxes(img, gt_boxes, pred_boxes, class_names)
        
        # Добавляем информацию
        info_text = f"GT: {len(gt_boxes)} | Pred: {len(pred_boxes)} | Conf Thresh: {conf_threshold}"
        cv2.putText(result_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(result_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Сохраняем
        output_path = output_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(output_path), result_img)
        stats['images_processed'] += 1
        
        # Логируем первые 10 изображений
        if stats['images_processed'] <= 10:
            logger.info(f"  {img_path.name}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}")
    
    # Сохраняем статистику
    stats_file = output_dir / "stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Images processed: {stats['images_processed']}\n")
        f.write(f"Total GT boxes: {stats['total_gt']}\n")
        f.write(f"Total Pred boxes: {stats['total_pred']}\n")
        f.write(f"Max Pred boxes per image: {stats['max_pred']}\n")
        if stats['images_processed'] > 0:
            f.write(f"Avg GT per image: {stats['total_gt']/stats['images_processed']:.2f}\n")
            f.write(f"Avg Pred per image: {stats['total_pred']/stats['images_processed']:.2f}\n")
    
    logger.info(f"\nРезультаты сохранены в {output_dir}")
    logger.info(f"Статистика: {stats}")
    
    return stats

def test_single_image(model, image_path, class_names):
    """Тест на одном изображении для отладки"""
    logger.info(f"\n=== Тест на одном изображении ===")
    logger.info(f"Изображение: {image_path}")
    
    # Пробуем оба метода
    for use_sahi in [False, True]:
        logger.info(f"\nМетод predict{' with SAHI' if use_sahi else ''}:")
        try:
            if use_sahi:
                results = model.predict_sahi(image=str(image_path))
            else:
                results = model.predict(str(image_path))
            
            logger.info(f"  Тип результата: {type(results)}")
            if isinstance(results, dict):
                logger.info(f"  Ключи: {list(results.keys())}")
                for key in results.keys():
                    value = results[key]
                    if torch.is_tensor(value):
                        logger.info(f"    {key}: shape={value.shape}")
                        if value.numel() > 0:
                            logger.info(f"      пример: {value[0] if key == 'bboxes' else value[:3]}")
                    else:
                        logger.info(f"    {key}: {type(value)}")
                
                # Показываем количество найденных объектов
                bboxes = results.get("bboxes", [])
                scores = results.get("scores", [])
                logger.info(f"  Найдено объектов: {len(bboxes)}")
                if len(bboxes) > 0:
                    logger.info(f"  Макс уверенность: {scores.max().item():.3f}" if torch.is_tensor(scores) else f"  Оценки: {scores[:3]}")
            else:
                logger.info(f"  Результат: {results}")
        except Exception as e:
            logger.error(f"  Ошибка: {e}")
    
    logger.info(f"===========================\n")

def main():
    # Пути
    images_dir = Path("/app/data/experiment/datasets/real_baseline/test/images")
    labels_dir = Path("/app/data/experiment/datasets/real_baseline/test/labels")
    output_dir = Path("/app/data/experiment/datasets/real_baseline/visual")
    model_path = Path("/app/experiments/models/exp1_frozen/real_baseline_frozen_seed42/exported_models/exported_best.pt")
    
    # Проверяем
    if not images_dir.exists():
        logger.error(f"Папка с изображениями не найдена: {images_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель
    model = load_model(model_path)
    if model is None:
        return
    
    # Загружаем названия классов
    class_names = {0: "object"}  # По умолчанию
    data_yaml = Path("/app/data/experiment/datasets/real_baseline/data.yaml")
    if data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                names = data_config['names']
                if isinstance(names, list):
                    class_names = {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = names
                logger.info(f"Загружены классы: {class_names}")
    
    # Тест на первом изображении
    test_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if test_images:
        test_single_image(model, test_images[0], class_names)
    
    # Визуализация
    logger.info("\n=== Начинаем визуализацию всех изображений ===")
    
    # Пробуем с разными порогами
    for threshold in [0.001, 0.01, 0.05, 0.1]:
        logger.info(f"\n--- Используем порог уверенности: {threshold} ---")
        stats = visualize_predictions(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir / f"thresh_{threshold}",
            model=model,
            class_names=class_names,
            conf_threshold=threshold,
            use_sahi=False  # Сначала без SAHI
        )
        
        if stats and stats['total_pred'] > 0:
            logger.info(f"✅ Найдены предсказания с порогом {threshold}!")
            break
        elif stats and stats['total_pred'] == 0 and threshold == 0.1:
            # Если все еще нет предсказаний, пробуем с SAHI
            logger.info("\n--- Пробуем с SAHI ---")
            stats = visualize_predictions(
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=output_dir / "with_sahi",
                model=model,
                class_names=class_names,
                conf_threshold=0.001,
                use_sahi=True
            )

if __name__ == "__main__":
    main()