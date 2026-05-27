#!/usr/bin/env python3
"""
Создает сетку изображений с сравнением GT и предсказаний.
Удобно для быстрого анализа результатов.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_comparison_grid(config_path: str, teacher_name: str = None, 
                          max_images: int = 16, grid_size: tuple = (4, 4)):
    """
    Создает сетку для сравнения GT vs Predictions.
    
    Args:
        config_path: путь к конфигу
        teacher_name: имя учителя (если None - первый из конфига)
        max_images: максимальное число изображений
        grid_size: размер сетки (rows, cols)
    """
    from visualize_predictions import (
        load_model, load_test_dataset, load_annotations, 
        predict, draw_boxes, CLASS_COLORS, GT_COLOR, PRED_COLOR
    )
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Выбираем учителя
    if teacher_name is None:
        teachers = [k for k in config.keys() if k.startswith('teacher_')]
        teacher_name = teachers[0] if teachers else None
    
    if not teacher_name or teacher_name not in config:
        logger.error(f"Учитель {teacher_name} не найден в конфиге")
        return
    
    teacher_cfg = config[teacher_name]
    experiment_data = Path(config['paths']['experiment_data'])
    data_yaml_path = experiment_data / teacher_cfg['data_yaml']
    
    # Загружаем данные и модель
    image_files, labels_path, class_names = load_test_dataset(data_yaml_path)
    image_files = image_files[:max_images]
    
    # Ищем и загружаем модель
    models_dir = Path(config['paths']['models_dir']) / teacher_name
    checkpoints = list(models_dir.glob('**/*.pt')) + list(models_dir.glob('**/*.ckpt'))
    
    if not checkpoints:
        logger.error(f"Чекпоинты не найдены в {models_dir}")
        return
    
    checkpoint_path = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    model, device = load_model(str(checkpoint_path), config)
    
    # Создаем сетку
    rows, cols = grid_size
    cell_h, cell_w = 480, 640
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, img_path in enumerate(tqdm(image_files[:rows * cols])):
        row = idx // cols
        col = idx % cols
        
        # Загружаем изображение
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        img_h, img_w = image.shape[:2]
        
        # Ресайзим для сетки
        scale = min(cell_w / img_w, cell_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Рисуем GT и предсказания на ресайзнутой копии
        viz = image_resized.copy()
        
        # GT boxes
        label_file = labels_path / f"{img_path.stem}.txt"
        gt_boxes, gt_classes = load_annotations(label_file, img_w, img_h)
        
        if gt_boxes:
            # Масштабируем GT боксы
            scaled_gt = [[x * scale, y * scale, x2 * scale, y2 * scale] 
                        for x, y, x2, y2 in gt_boxes]
            draw_boxes(viz, scaled_gt, gt_classes, GT_COLOR, 
                      prefix="GT:", class_names=class_names)
        
        # Predictions
        pred_boxes, pred_classes, pred_scores = predict(
            model, img_path, device,
            conf_threshold=config['training'].get('conf_threshold', 0.25)
        )
        
        if pred_boxes:
            scaled_pred = [[x * scale, y * scale, x2 * scale, y2 * scale] 
                          for x, y, x2, y2 in pred_boxes]
            draw_boxes(viz, scaled_pred, pred_classes, PRED_COLOR,
                      scores=pred_scores, prefix="PR:", class_names=class_names)
        
        # Размещаем в сетке
        y_start = row * cell_h
        x_start = col * cell_w
        
        # Центрируем изображение в ячейке
        y_offset = y_start + (cell_h - new_h) // 2
        x_offset = x_start + (cell_w - new_w) // 2
        
        grid[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = viz
        
        # Добавляем имя файла
        cv2.putText(grid, img_path.stem[:20], (x_offset, y_offset - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Сохраняем сетку
    output_path = Path(config['paths']['results_dir']) / "visualizations" / f"{teacher_name}_grid.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    
    logger.info(f"Сетка сохранена: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_teacher.yaml")
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=16)
    parser.add_argument("--grid", type=int, nargs=2, default=[4, 4])
    
    args = parser.parse_args()
    create_comparison_grid(args.config, args.teacher, args.max_images, tuple(args.grid))