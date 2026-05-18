#!/usr/bin/env python3
"""
04b_augment_real_controlled.py — Контролируемая аугментация реальных данных.
Генерирует ровно N копий КАЖДОГО реального изображения.
С добивкой недостающих из-за потерянных bbox.
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
from utils.dataset_utils import read_yolo_labels, write_yolo_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def augment_real_controlled(config: dict, copies_per_image: int, output_subdir: str) -> Path:
    """
    Генерирует ровно copies_per_image аугментированных копий
    КАЖДОГО реального изображения.
    """
    paths = config['paths']
    aug_cfg = config['augmentation']
    jpeg_quality = config['image']['jpeg_quality']
    
    output_dir = Path(paths['output_dir']) / output_subdir / "train"
    real_train = Path(paths['output_dir']) / "real" / "train"
    
    source_images = list(real_train.glob('images/*.jpg')) + \
                    list(real_train.glob('images/*.jpeg')) + \
                    list(real_train.glob('images/*.png'))
    
    if not source_images:
        raise FileNotFoundError(f"Реальные изображения не найдены в {real_train / 'images'}")
    
    total_generate = len(source_images) * copies_per_image
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎨 Контролируемая аугментация ({output_subdir}):")
    logger.info(f"   Исходных изображений: {len(source_images)}")
    logger.info(f"   Копий на каждое: {copies_per_image}")
    logger.info(f"   Всего будет сгенерировано: {total_generate}")
    
    transform = get_metal_augmentation(aug_cfg)
    
    # ============ ФАЗА 1: гарантированные копии ============
    generated = 0
    failed = 0
    version_counter = {}
    
    pbar = tqdm(total=total_generate, desc=f"  {output_subdir}")
    
    for src_img in source_images:
        lbl_path = real_train / 'labels' / f"{src_img.stem}.txt"
        image = cv2.imread(str(src_img))
        if image is None:
            failed += copies_per_image
            pbar.update(copies_per_image)
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_labels(lbl_path)
        if bboxes:
            bboxes = np.clip(bboxes, 1e-7, 1.0 - 1e-7).tolist()
        
        for copy_idx in range(copies_per_image):
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
                failed += 1
                pbar.update(1)
                continue
            
            if not aug_bboxes and bboxes:
                failed += 1
                pbar.update(1)
                continue
            
            img_key = src_img.stem
            version_counter[img_key] = version_counter.get(img_key, 0) + 1
            
            new_name = f"{output_subdir}_c{version_counter[img_key]:02d}_{src_img.stem}"
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(output_dir / 'images' / f"{new_name}.jpg"),
                aug_image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            write_yolo_labels(output_dir / 'labels' / f"{new_name}.txt", aug_bboxes, aug_labels)
            
            generated += 1
            pbar.update(1)
    
    # ============ ★★★ ФАЗА 2: ДОБИВКА недостающих ★★★ ============
    actual = len(list((output_dir / 'images').glob('*')))
    if actual < total_generate:
        shortage = total_generate - actual
        logger.warning(f"⚠️ Недобор {shortage} из {total_generate}, добиваем...")
        
        attempt = 0
        max_attempts = shortage * 5
        
        while actual < total_generate and attempt < max_attempts:
            attempt += 1
            src_img = random.choice(source_images)
            lbl_path = real_train / 'labels' / f"{src_img.stem}.txt"
            
            image = cv2.imread(str(src_img))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, class_labels = read_yolo_labels(lbl_path)
            if bboxes:
                bboxes = np.clip(bboxes, 1e-7, 1.0 - 1e-7).tolist()
            
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
            
            if not aug_bboxes and bboxes:
                continue
            
            img_key = src_img.stem
            version_counter[img_key] = version_counter.get(img_key, 0) + 1
            
            new_name = f"{output_subdir}_c{version_counter[img_key]:02d}_{src_img.stem}"
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(output_dir / 'images' / f"{new_name}.jpg"),
                aug_image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            write_yolo_labels(output_dir / 'labels' / f"{new_name}.txt", aug_bboxes, aug_labels)
            
            generated += 1
            actual += 1
            pbar.update(1)
    
    pbar.close()
    
    final_count = len(list((output_dir / 'images').glob('*')))
    logger.info(f"✅ {output_subdir}: {final_count}/{total_generate} (цель достигнута: {final_count >= total_generate})")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--copies', type=int, required=True, help='Количество копий на изображение')
    parser.add_argument('--output', type=str, required=True, help='Имя выходной поддиректории')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    augment_real_controlled(config, args.copies, args.output)