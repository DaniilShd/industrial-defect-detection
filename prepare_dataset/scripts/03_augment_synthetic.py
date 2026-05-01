#!/usr/bin/env python3
"""
03_augment_synthetic.py — Аугментация синтетического датасета.
Каждое изображение аугментируется 1 раз.
Результат сохраняется в synthetic_augmented/train.
"""

import logging
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


def augment_synthetic_dataset(config: dict) -> Path:
    """
    Применяет аугментацию к синтетическому датасету.
    Каждое изображение → 1 аугментированная копия.
    """
    paths = config['paths']
    aug_cfg = config['augmentation']
    jpeg_quality = config['image']['jpeg_quality']
    
    synth_train = Path(paths['output_dir']) / "synthetic" / "train"
    output_dir = Path(paths['output_dir']) / "synthetic_augmented" / "train"
    
    if not synth_train.exists():
        raise FileNotFoundError(f"Синтетика не найдена: {synth_train}")
    
    # Очищаем выходную директорию
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    src_images = list(synth_train.glob('images/*.jpg')) + \
                 list(synth_train.glob('images/*.jpeg')) + \
                 list(synth_train.glob('images/*.png'))
    
    logger.info(f"🎨 Аугментация синтетики: {len(src_images)} изображений")
    
    transform = get_metal_augmentation(aug_cfg)
    
    success = 0
    failed = 0
    
    for img_path in tqdm(src_images, desc="  Аугментация"):
        lbl_path = synth_train / 'labels' / f"{img_path.stem}.txt"
        
        image = cv2.imread(str(img_path))
        if image is None:
            failed += 1
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
        except Exception as e:
            logger.warning(f"Ошибка аугментации {img_path.name}: {e}")
            failed += 1
            continue
        
        # Сохраняем
        new_name = f"synth_aug_{img_path.stem}"
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
        
        success += 1
    
    logger.info(f"✅ Аугментация синтетики: {success} успешно, {failed} ошибок")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    result = augment_synthetic_dataset(config)
    print(f"\n✅ Результат: {result}")