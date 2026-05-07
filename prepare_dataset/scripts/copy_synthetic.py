#!/usr/bin/env python3
"""
02_copy_synthetic.py — Копирование синтетики в experiment/
с ресайзом до 640x640 и сохранением YOLO-разметки.
"""

import logging
import shutil
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset_utils import (
    read_yolo_labels,
    write_yolo_labels,
    resize_image_and_labels,
    count_dataset_images,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def copy_synthetic_dataset(config: dict) -> Path:
    """
    Копирует синтетический датасет с ресайзом 640x640.
    Синтетика содержит только train.
    """
    paths = config['paths']
    target_size = tuple(config['image']['target_size'])
    jpeg_quality = config['image']['jpeg_quality']
    
    synth_src = Path(paths['best_synthetic'])
    output_dir = Path(paths['output_dir']) / "synthetic" / "train"
    
    logger.info(f"📦 Копирование синтетики: {synth_src} → {output_dir}")
    logger.info(f"   Ресайз: {target_size[0]}x{target_size[1]}")
    
    if not synth_src.exists():
        raise FileNotFoundError(f"Синтетический датасет не найден: {synth_src}")
    
    # Определяем структуру источника
    if (synth_src / 'train' / 'images').exists():
        src_images_dir = synth_src / 'train' / 'images'
        src_labels_dir = synth_src / 'train' / 'labels'
    elif (synth_src / 'images').exists():
        src_images_dir = synth_src / 'images'
        src_labels_dir = synth_src / 'labels'
    else:
        raise FileNotFoundError(f"Не найдены images/ в {synth_src}")
    
    # Очищаем выходную директорию
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Копируем с ресайзом
    src_images = list(src_images_dir.glob('*'))
    logger.info(f"   Найдено изображений: {len(src_images)}")
    
    for img_path in tqdm(src_images, desc="  Копирование"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Не удалось загрузить: {img_path.name}")
            continue
        
        # Читаем лейблы
        lbl_path = src_labels_dir / f"{img_path.stem}.txt"
        bboxes, class_labels = read_yolo_labels(lbl_path)
        
        # Ресайз
        image_resized, bboxes, class_labels = resize_image_and_labels(
            image, bboxes, class_labels, target_size
        )
        
        # Сохраняем
        cv2.imwrite(
            str(output_dir / 'images' / f"{img_path.stem}.jpg"),
            image_resized,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        
        write_yolo_labels(
            output_dir / 'labels' / f"{img_path.stem}.txt",
            bboxes, class_labels
        )
    
    n_images = len(list((output_dir / 'images').glob('*')))
    logger.info(f"   Скопировано: {n_images} изображений")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    result = copy_synthetic_dataset(config)
    print(f"\n✅ Синтетика скопирована: {result}")