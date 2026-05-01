#!/usr/bin/env python3
"""
04_augment_real.py — Создание аугментированных копий реальных изображений.
Генерирует столько же изображений, сколько в synthetic/train.
Оригиналы не включаются.
"""

import logging
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.augmentation import get_metal_augmentation
from utils.dataset_utils import (
    read_yolo_labels,
    write_yolo_labels,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def augment_real_dataset(config: dict) -> Path:
    """
    Создаёт аугментированные копии реальных изображений.
    Количество = количество синтетических изображений.
    """
    paths = config['paths']
    aug_cfg = config['augmentation']
    jpeg_quality = config['image']['jpeg_quality']
    
    output_dir = Path(paths['output_dir']) / "real_augmented" / "train"
    real_train = Path(paths['output_dir']) / "real" / "train"
    synth_train = Path(paths['output_dir']) / "synthetic" / "train"
    
    # Определяем целевое количество = количество синтетики
    synth_images_dir = synth_train / 'images'
    if not synth_images_dir.exists():
        raise FileNotFoundError(f"Синтетика не найдена: {synth_images_dir}")
    
    num_generate = len(list(synth_images_dir.glob('*')))
    
    # Очищаем выходную директорию
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Исходные изображения
    source_images = list(real_train.glob('images/*.jpg')) + \
                    list(real_train.glob('images/*.jpeg')) + \
                    list(real_train.glob('images/*.png'))
    
    logger.info(f"🎨 Аугментация реальных данных:")
    logger.info(f"   Исходных: {len(source_images)}")
    logger.info(f"   Целевое количество: {num_generate}")
    
    transform = get_metal_augmentation(aug_cfg)
    
    generated = 0
    attempts = 0
    max_attempts = num_generate * 10
    
    pbar = tqdm(total=num_generate, desc="  Генерация")
    
    while generated < num_generate and attempts < max_attempts:
        attempts += 1
        
        src_img = random.choice(source_images)
        lbl_path = real_train / 'labels' / f"{src_img.stem}.txt"
        
        image = cv2.imread(str(src_img))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes, class_labels = read_yolo_labels(lbl_path)
        
        try:
            if bboxes:
                augmented = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
            else:
                augmented = transform(image=image_rgb, bboxes=[], class_labels=[])
                aug_image = augmented['image']
                aug_bboxes, aug_labels = [], []
        except Exception:
            continue
        
        # Пропускаем если были bbox'ы но все потерялись
        if not aug_bboxes and bboxes:
            continue
        
        # Сохраняем
        new_name = f"real_aug_{generated:06d}_{src_img.stem}"
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(output_dir / 'images' / f"{new_name}.jpg"),
            aug_image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        write_yolo_labels(
            output_dir / 'labels' / f"{new_name}.txt",
            aug_bboxes, aug_labels
        )
        
        generated += 1
        pbar.update(1)
    
    pbar.close()
    
    efficiency = generated / attempts * 100 if attempts > 0 else 0
    logger.info(f"✅ Сгенерировано: {generated} (эффективность: {efficiency:.1f}%)")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    result = augment_real_dataset(config)
    print(f"\n✅ Результат: {result}")