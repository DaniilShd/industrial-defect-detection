#!/usr/bin/env python3
"""Утилиты для работы с датасетами"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def read_yolo_labels(label_path: Path) -> Tuple[List, List]:
    """
    Чтение YOLO-лейблов.
    
    Returns:
        bboxes: список [x_center, y_center, width, height] (нормализованные)
        class_labels: список классов
    """
    bboxes, class_labels = [], []
    
    if not label_path.exists() or label_path.stat().st_size == 0:
        return bboxes, class_labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                xc = np.clip(float(parts[1]), 0.0, 1.0)
                yc = np.clip(float(parts[2]), 0.0, 1.0)
                w = np.clip(float(parts[3]), 0.001, 1.0)
                h = np.clip(float(parts[4]), 0.001, 1.0)
                
                if w > 0.001 and h > 0.001:
                    bboxes.append([xc, yc, w, h])
                    class_labels.append(cls)
    
    return bboxes, class_labels


def write_yolo_labels(label_path: Path, bboxes: List, class_labels: List):
    """
    Запись YOLO-лейблов.
    Координаты сохраняются с 6 знаками после запятой.
    """
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            xc = np.clip(float(bbox[0]), 0.0, 1.0)
            yc = np.clip(float(bbox[1]), 0.0, 1.0)
            w = np.clip(float(bbox[2]), 0.001, 1.0)
            h = np.clip(float(bbox[3]), 0.001, 1.0)
            
            if w > 0.001 and h > 0.001:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def resize_image_and_labels(
    image: np.ndarray,
    bboxes: List,
    class_labels: List,
    target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, List, List]:
    """
    Ресайз изображения до target_size.
    
    Важно: YOLO-разметка нормализованная, поэтому координаты bbox 
    НЕ МЕНЯЮТСЯ при ресайзе. Но мы проверяем и клиппим на всякий случай.
    
    Args:
        image: исходное изображение (H, W, C)
        bboxes: список YOLO bbox [xc, yc, w, h]
        class_labels: список классов
        target_size: целевой размер (width, height)
    
    Returns:
        resized_image, bboxes, class_labels
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Ресайзим изображение
    if h != target_h or w != target_w:
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # YOLO bbox нормализованные — не требуют изменений
    # Но клиппим для безопасности
    safe_bboxes = []
    for bbox in bboxes:
        xc = np.clip(float(bbox[0]), 0.0, 1.0)
        yc = np.clip(float(bbox[1]), 0.0, 1.0)
        bw = np.clip(float(bbox[2]), 0.001, 1.0)
        bh = np.clip(float(bbox[3]), 0.001, 1.0)
        if bw > 0.001 and bh > 0.001:
            safe_bboxes.append([xc, yc, bw, bh])
    
    return image, safe_bboxes, class_labels


def create_data_yaml(
    dataset_path: Path,
    train_dir: str = "train/images",
    val_dir: str = "val/images",
    test_dir: Optional[str] = "test/images",
    num_classes: int = 4,
    class_names: Optional[dict] = None
) -> Path:
    """
    Создаёт data.yaml для YOLO-совместимого датасета.
    """
    if class_names is None:
        class_names = {i: f"defect{i+1}" for i in range(num_classes)}
    
    data_config = {
        'path': str(dataset_path.absolute()),
        'train': train_dir,
        'val': val_dir,
        'nc': num_classes,
        'names': class_names
    }
    
    if test_dir:
        data_config['test'] = test_dir
    
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    return yaml_path


def count_dataset_images(dataset_dir: Path, split: str) -> int:
    """Подсчёт количества изображений в сплите."""
    images_dir = dataset_dir / split / "images"
    if images_dir.exists():
        return len(list(images_dir.glob('*')))
    return 0


def validate_yolo_dataset(dataset_dir: Path) -> Tuple[bool, List[str]]:
    """
    Проверка корректности YOLO датасета.
    
    Проверяет:
    - Наличие images/ и labels/ для каждого сплита
    - Соответствие имён изображений и лейблов
    - Корректность формата лейблов
    - Размер изображений (должен быть 640x640)
    """
    errors = []
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            errors.append(f"{split}: нет директории images/")
            continue
        
        if not labels_dir.exists():
            errors.append(f"{split}: нет директории labels/")
            continue
        
        image_files = {f.stem for f in images_dir.glob('*') 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        label_files = {f.stem for f in labels_dir.glob('*.txt')}
        
        # Проверка соответствия имён
        missing_labels = image_files - label_files
        extra_labels = label_files - image_files
        
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} изображений без лейблов")
        if extra_labels:
            errors.append(f"{split}: {len(extra_labels)} лейблов без изображений")
        
        # Проверка размеров изображений
        for img_path in list(images_dir.glob('*'))[:5]:  # Проверяем первые 5
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                if h != 640 or w != 640:
                    errors.append(f"{split}/{img_path.name}: размер {w}x{h}, ожидается 640x640")
        
        # Проверка формата лейблов
        for lbl_path in list(labels_dir.glob('*.txt'))[:5]:
            try:
                bboxes, classes = read_yolo_labels(lbl_path)
                if len(bboxes) != len(classes):
                    errors.append(f"{split}/{lbl_path.name}: несоответствие bbox и классов")
                for bbox in bboxes:
                    if any(not (0.0 <= v <= 1.0) for v in bbox):
                        errors.append(f"{split}/{lbl_path.name}: координаты вне [0,1]")
            except Exception as e:
                errors.append(f"{split}/{lbl_path.name}: ошибка чтения - {e}")
    
    return len(errors) == 0, errors


def get_image_paths(directory: Path) -> List[Path]:
    """Получить список всех изображений в директории."""
    extensions = {'.jpg', '.jpeg', '.png'}
    return sorted([f for f in directory.glob('*') if f.suffix.lower() in extensions])