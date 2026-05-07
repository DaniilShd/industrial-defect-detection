#!/usr/bin/env python3
"""
06_validate_resize.py — Финальная проверка и ресайз всех собранных датасетов.
Проверяет:
  - Все изображения 640x640
  - Все лейблы в YOLO-формате (нормализованные координаты)
  - Соответствие изображений и лейблов
При несоответствии — исправляет (ресайз, пересчёт лейблов).
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset_utils import (
    read_yolo_labels,
    write_yolo_labels,
    validate_yolo_dataset,
    count_dataset_images,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_and_fix_image(
    img_path: Path,
    lbl_path: Path,
    target_size: Tuple[int, int] = (640, 640),
    jpeg_quality: int = 95
) -> dict:
    """
    Проверяет и исправляет одно изображение + лейблы.
    
    Returns:
        dict с информацией о проверке
    """
    target_w, target_h = target_size
    result = {
        'path': str(img_path),
        'was_resized': False,
        'was_fixed': False,
        'errors': []
    }
    
    # Проверяем изображение
    image = cv2.imread(str(img_path))
    if image is None:
        result['errors'].append('Не удалось загрузить')
        return result
    
    h, w = image.shape[:2]
    
    if h != target_h or w != target_w:
        result['was_resized'] = True
        result['was_fixed'] = True
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
    # Проверяем лейблы
    if lbl_path.exists():
        try:
            bboxes, class_labels = read_yolo_labels(lbl_path)
            
            # Проверяем корректность нормализованных координат
            fixed = False
            fixed_bboxes = []
            fixed_labels = []
            
            for bbox, cls in zip(bboxes, class_labels):
                xc, yc, bw, bh = bbox
                
                # Клиппим и проверяем
                xc = np.clip(xc, 0.0, 1.0)
                yc = np.clip(yc, 0.0, 1.0)
                bw = np.clip(bw, 0.001, 1.0)
                bh = np.clip(bh, 0.001, 1.0)
                
                if bw > 0.001 and bh > 0.001:
                    fixed_bboxes.append([xc, yc, bw, bh])
                    fixed_labels.append(cls)
                else:
                    fixed = True
            
            if fixed or len(fixed_bboxes) != len(bboxes):
                result['was_fixed'] = True
                write_yolo_labels(lbl_path, fixed_bboxes, fixed_labels)
            
            if len(fixed_bboxes) == 0 and len(bboxes) > 0:
                result['errors'].append('Все bbox отфильтрованы')
                
        except Exception as e:
            result['errors'].append(f'Ошибка чтения лейбла: {e}')
            result['was_fixed'] = True
    
    return result


def validate_and_fix_all_datasets(config: dict) -> dict:
    """
    Проверяет и исправляет все собранные датасеты.
    """
    paths = config['paths']
    target_size = tuple(config['image']['target_size'])
    jpeg_quality = config['image']['jpeg_quality']
    
    experiment_data = Path(paths['output_dir'])
    datasets_dir = experiment_data / "datasets"
    
    # Загружаем информацию о датасетах
    info_path = experiment_data / "datasets_info.yaml"
    if not info_path.exists():
        logger.error(f"Файл с информацией о датасетах не найден: {info_path}")
        logger.error("Сначала запустите 05_merge_datasets.py")
        return {}
    
    with open(info_path, 'r') as f:
        datasets_info = yaml.safe_load(f)
    
    logger.info(f"🔍 Проверка {len(datasets_info)} датасетов...")
    logger.info(f"   Целевой размер: {target_size[0]}x{target_size[1]}")
    
    stats = {}
    total_fixed = 0
    total_errors = 0
    
    for name, yaml_path in datasets_info.items():
        dataset_dir = Path(yaml_path).parent
        logger.info(f"\n📦 {name}: {dataset_dir}")
        
        dataset_stats = {
            'total_images': 0,
            'resized': 0,
            'fixed_labels': 0,
            'errors': 0
        }
        
        all_image_paths = []
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            images = list((split_dir / 'images').glob('*'))
            all_image_paths.extend(images)
        
        for img_path in tqdm(all_image_paths, desc=f"  Проверка {name}"):
            # Находим соответствующий лейбл
            rel_path = img_path.parent.parent.name  # train/val/test
            lbl_dir = dataset_dir / rel_path / 'labels'
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            result = check_and_fix_image(img_path, lbl_path, target_size, jpeg_quality)
            
            dataset_stats['total_images'] += 1
            if result['was_resized']:
                dataset_stats['resized'] += 1
            if result['was_fixed']:
                dataset_stats['fixed_labels'] += 1
            if result['errors']:
                dataset_stats['errors'] += 1
        
        total_fixed += dataset_stats['resized'] + dataset_stats['fixed_labels']
        total_errors += dataset_stats['errors']
        
        # Финальная валидация YOLO
        is_valid, errors = validate_yolo_dataset(dataset_dir)
        dataset_stats['yolo_valid'] = is_valid
        dataset_stats['yolo_errors'] = errors
        
        if is_valid:
            logger.info(f"   ✅ {dataset_stats['total_images']} изобр. | "
                       f"ресайз: {dataset_stats['resized']} | "
                       f"лейблы: {dataset_stats['fixed_labels']} | "
                       f"YOLO: OK")
        else:
            logger.warning(f"   ⚠️ {dataset_stats['total_images']} изобр. | "
                          f"ресайз: {dataset_stats['resized']} | "
                          f"лейблы: {dataset_stats['fixed_labels']} | "
                          f"ошибок: {dataset_stats['errors']} | "
                          f"YOLO: {len(errors)} проблем")
            for err in errors[:5]:
                logger.warning(f"      - {err}")
        
        stats[name] = dataset_stats
    
    # Сохраняем статистику
    stats_path = experiment_data / "validation_stats.yaml"
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 ИТОГИ ПРОВЕРКИ:")
    logger.info(f"   Всего изображений проверено: {sum(s['total_images'] for s in stats.values())}")
    logger.info(f"   Ресайзнуто: {sum(s['resized'] for s in stats.values())}")
    logger.info(f"   Исправлено лейблов: {sum(s['fixed_labels'] for s in stats.values())}")
    logger.info(f"   Ошибок: {total_errors}")
    logger.info(f"   Статистика сохранена: {stats_path}")
    logger.info(f"{'='*60}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    stats = validate_and_fix_all_datasets(config)