#!/usr/bin/env python3
"""
Визуализация предсказаний LTDETR с правильной фильтрацией и цветами.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Цвета для классов предсказаний (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # class 0 - зеленый
    1: (255, 0, 0),    # class 1 - синий
    2: (0, 0, 255),    # class 2 - красный
    3: (255, 255, 0),  # class 3 - голубой
    4: (255, 0, 255),  # class 4 - фиолетовый
    5: (0, 255, 255),  # class 5 - желтый
}

# Фиксированные цвета для GT и Predictions
GT_COLOR = (0, 255, 0)      # зеленый
PRED_COLOR = (0, 0, 255)    # красный


def load_model(checkpoint_path: str):
    """Загружает модель."""
    from lightly_train import load_model as lightly_load_model
    
    logger.info(f"Загружаем модель: {checkpoint_path}")
    model = lightly_load_model(str(checkpoint_path))
    logger.info("✅ Модель загружена")
    return model


def find_test_data(data_yaml_path: Path):
    """Находит тестовые изображения и лейблы."""
    with open(data_yaml_path) as f:
        dc = yaml.safe_load(f)
    
    dataset_path = Path(dc['path'])
    
    # Ищем test split
    test_key = dc.get('test', dc.get('val', 'test/images'))
    
    # Возможные структуры папок
    possible_structures = [
        # Структура 1: dataset/test/images и dataset/test/labels
        (dataset_path / "test" / "images", dataset_path / "test" / "labels"),
        # Структура 2: dataset/images/test и dataset/labels/test
        (dataset_path / "images" / "test", dataset_path / "labels" / "test"),
        # Структура 3: dataset/test (смешанная)
        (dataset_path / test_key, dataset_path / test_key.replace("images", "labels")),
        # Структура 4: как указано в data.yaml
        (dataset_path / test_key, dataset_path / test_key.replace("images", "labels")),
    ]
    
    test_images_path = None
    test_labels_path = None
    
    for img_path, lbl_path in possible_structures:
        if img_path.exists():
            test_images_path = img_path
            if lbl_path.exists():
                test_labels_path = lbl_path
            elif (img_path.parent / "labels").exists():
                test_labels_path = img_path.parent / "labels"
            break
    
    # Если не нашли labels, ищем рядом
    if test_images_path and not test_labels_path:
        for variant in [
            test_images_path.parent / "labels",
            dataset_path / "test" / "labels",
            dataset_path / "labels" / "test",
        ]:
            if variant.exists():
                test_labels_path = variant
                break
    
    if not test_images_path:
        raise FileNotFoundError(f"Тестовые изображения не найдены для {data_yaml_path}")
    
    image_files = sorted([
        f for f in test_images_path.glob("*")
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])
    
    logger.info(f"Найдено {len(image_files)} изображений")
    logger.info(f"  Images: {test_images_path}")
    logger.info(f"  Labels: {test_labels_path if test_labels_path else 'НЕ НАЙДЕНЫ!'}")
    
    return image_files, test_labels_path, dc.get('names', {})


def load_annotations(label_path: Path, img_width: int, img_height: int):
    """Загружает YOLO-аннотации."""
    boxes = []
    classes = []
    
    if not label_path or not label_path.exists():
        return boxes, classes
    
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cls_id = int(float(parts[0]))
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = max(0, x_center - width / 2)
                    y1 = max(0, y_center - height / 2)
                    x2 = min(img_width, x_center + width / 2)
                    y2 = min(img_height, y_center + height / 2)
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls_id)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Ошибка парсинга строки в {label_path}: {line}")
                    continue
    
    return boxes, classes


def predict_image(model, image_path: Path, conf_threshold: float = 0.25):
    """
    Предсказание с правильной фильтрацией.
    Возвращает ТОЛЬКО боксы выше порога уверенности.
    """
    import inspect
    
    # Смотрим сигнатуру метода predict
    try:
        sig = inspect.signature(model.predict)
        params = list(sig.parameters.keys())
    except:
        params = []
    
    # Пробуем вызвать predict
    try:
        # Вызываем без параметра threshold - получим все предсказания
        results = model.predict(str(image_path))
        
        if isinstance(results, dict):
            labels = results.get("labels", [])
            bboxes = results.get("bboxes", [])
            scores = results.get("scores", [])
            
            # Конвертируем в numpy для удобства
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # ★★★ ВАЖНО: Фильтруем по уверенности ★★★
            if len(scores) > 0:
                # Оставляем только предсказания выше порога
                mask = scores >= conf_threshold
                bboxes = bboxes[mask]
                labels = labels[mask]
                scores = scores[mask]
                
                # Дополнительно: если слишком много боксов, оставляем top-N
                max_boxes = 50
                if len(scores) > max_boxes:
                    # Сортируем по уверенности и берем top-N
                    top_indices = np.argsort(scores)[-max_boxes:]
                    bboxes = bboxes[top_indices]
                    labels = labels[top_indices]
                    scores = scores[top_indices]
            
            return bboxes.tolist(), labels.tolist(), scores.tolist()
        
        return [], [], []
        
    except Exception as e:
        logger.error(f"Predict error for {image_path.name}: {e}")
        return [], [], []


def draw_gt_boxes(image, boxes, classes, class_names=None):
    """Рисует ground truth боксы ЗЕЛЕНЫМ цветом."""
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        
        # Всегда зеленый для GT
        color = (0, 255, 0)
        
        # Толстая линия для GT
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Текст
        if class_names and cls_id in class_names:
            text = f"GT: {class_names[cls_id]}"
        else:
            text = f"GT: class_{cls_id}"
        
        # Белый фон для текста
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_pred_boxes(image, boxes, classes, scores, class_names=None):
    """Рисует predicted боксы КРАСНЫМ цветом."""
    for box, cls_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Всегда красный для predictions
        color = (0, 0, 255)
        
        # Тонкая линия для predictions
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Текст с классом и уверенностью
        if class_names and cls_id in class_names:
            text = f"{class_names[cls_id]}: {score:.2f}"
        else:
            text = f"cls_{cls_id}: {score:.2f}"
        
        # Красный фон для текста
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def visualize_predictions(config_path: str = "config_teacher.yaml"):
    """Основная функция."""
    
    config_file = Path(config_path)
    if not config_file.exists():
        config_file = Path(__file__).parent / config_path
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    output_base = Path(config['paths'].get('results_dir', '/app/data/experiment_v3/reports'))
    viz_dir = output_base / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Сохраняем в: {viz_dir}")
    
    teachers = [k for k in config.keys() if k.startswith('teacher_')]
    
    for teacher_name in teachers:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 {teacher_name}")
        logger.info(f"{'='*60}")
        
        teacher_cfg = config[teacher_name]
        experiment_data = Path(config['paths']['experiment_data'])
        data_yaml_path = experiment_data / teacher_cfg['data_yaml']
        
        teacher_viz_dir = viz_dir / teacher_name
        teacher_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем данные
        try:
            image_files, labels_path, class_names = find_test_data(data_yaml_path)
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            continue
        
        # Ищем чекпоинт
        models_dir = Path(config['paths']['models_dir']) / teacher_name
        checkpoint = None
        for pattern in ["exported_models/exported_best.pt", "exported_models/exported_last.pt"]:
            c = models_dir / pattern
            if c.exists():
                checkpoint = c
                break
        
        if not checkpoint:
            logger.error(f"Чекпоинт не найден")
            continue
        
        # Загружаем модель
        try:
            model = load_model(str(checkpoint))
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            continue
        
        conf_threshold = config['training'].get('conf_threshold', 0.25)
        logger.info(f"Порог уверенности: {conf_threshold}")
        
        # ★★★ ДЕБАГ ПЕРВОГО ИЗОБРАЖЕНИЯ ★★★
        test_img = image_files[0]
        logger.info(f"\n🔍 Дебаг первого изображения: {test_img.name}")
        
        # Загружаем изображение
        img = cv2.imread(str(test_img))
        h, w = img.shape[:2]
        logger.info(f"Размер: {w}x{h}")
        
        # Проверяем GT
        if labels_path:
            label_file = labels_path / f"{test_img.stem}.txt"
            logger.info(f"Label файл: {label_file}")
            logger.info(f"Label существует: {label_file.exists()}")
            
            gt_boxes, gt_classes = load_annotations(label_file, w, h)
            logger.info(f"GT боксов: {len(gt_boxes)}")
            if gt_boxes:
                logger.info(f"  Пример GT: box={gt_boxes[0]}, class={gt_classes[0]}")
        else:
            logger.warning("⚠️  Labels path не найден!")
            gt_boxes, gt_classes = [], []
        
        # Проверяем Predictions
        pred_boxes, pred_classes, pred_scores = predict_image(model, test_img, conf_threshold)
        logger.info(f"Prediction боксов (conf>{conf_threshold}): {len(pred_boxes)}")
        if pred_boxes:
            logger.info(f"  Пример: box={pred_boxes[0]}, class={pred_classes[0]}, score={pred_scores[0]:.3f}")
        
        # Показываем классы из модели
        if hasattr(model, 'classes'):
            logger.info(f"Классы модели: {model.classes}")
        
        logger.info(f"\nПродолжаем обработку всех {len(image_files)} изображений...")
        
        # Обрабатываем все изображения
        for img_path in tqdm(image_files, desc=teacher_name):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            viz = img.copy()
            
            # GT (зеленые)
            if labels_path:
                label_file = labels_path / f"{img_path.stem}.txt"
                gt_boxes, gt_classes = load_annotations(label_file, w, h)
                if gt_boxes:
                    draw_gt_boxes(viz, gt_boxes, gt_classes, class_names)
            
            # Predictions (красные)
            pred_boxes, pred_classes, pred_scores = predict_image(model, img_path, conf_threshold)
            if pred_boxes:
                draw_pred_boxes(viz, pred_boxes, pred_classes, pred_scores, class_names)
            
            # Легенда
            cv2.putText(viz, "GREEN = Ground Truth", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(viz, "RED = Predictions", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(viz, f"Conf threshold: {conf_threshold}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Счетчики
            cv2.putText(viz, f"GT: {len(gt_boxes)} | Pred: {len(pred_boxes)}", 
                       (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out_path = teacher_viz_dir / f"{img_path.stem}_viz.jpg"
            cv2.imwrite(str(out_path), viz)
        
        logger.info(f"✅ Сохранено в {teacher_viz_dir}")
    
    logger.info(f"\n🎉 Готово! {viz_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_teacher.yaml")
    args = parser.parse_args()
    visualize_predictions(args.config)