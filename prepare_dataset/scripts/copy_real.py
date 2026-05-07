#!/usr/bin/env python3
"""
01_copy_real.py — Копирование реальных данных в experiment/
+ ресайз до 640x640 с сохранением YOLO-разметки.
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
    count_dataset_images
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def copy_real_dataset(config: dict) -> dict:
    """
    Копирует реальные данные с ресайзом 640x640.
    """
    paths = config['paths']
    target_size = tuple(config['image']['target_size'])
    jpeg_quality = config['image']['jpeg_quality']
    
    real_src = Path(paths['real_dataset'])
    output_dir = Path(paths['output_dir']) / "real"
    
    logger.info(f"📦 Копирование реальных данных: {real_src} → {output_dir}")
    logger.info(f"   Ресайз: {target_size[0]}x{target_size[1]}")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    paths_info = {}
    
    for split in ['train', 'val', 'test']:
        src_split = real_src / split
        dst_split = output_dir / split
        
        if not src_split.exists():
            logger.warning(f"⚠️  Сплит {split} не найден")
            continue
        
        (dst_split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Копируем и ресайзим изображения
        src_images = list((src_split / 'images').glob('*'))
        
        for img_path in tqdm(src_images, desc=f"  {split}"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Загружаем изображение
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Не удалось загрузить: {img_path.name}")
                continue
            
            # Читаем лейблы
            lbl_path = src_split / 'labels' / f"{img_path.stem}.txt"
            bboxes, class_labels = read_yolo_labels(lbl_path)
            
            # Ресайз (YOLO-разметка нормализованная — не требует изменений)
            image_resized, bboxes, class_labels = resize_image_and_labels(
                image, bboxes, class_labels, target_size
            )
            
            # Сохраняем
            cv2.imwrite(
                str(dst_split / 'images' / f"{img_path.stem}.jpg"),
                image_resized,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            
            write_yolo_labels(
                dst_split / 'labels' / f"{img_path.stem}.txt",
                bboxes, class_labels
            )
        
        n_images = count_dataset_images(output_dir, split)
        logger.info(f"   {split}: {n_images} изображений")
        paths_info[f'real_{split}'] = str(dst_split)
    
    return paths_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    result = copy_real_dataset(config)
    print(f"\n✅ Реальные данные скопированы: {result}")