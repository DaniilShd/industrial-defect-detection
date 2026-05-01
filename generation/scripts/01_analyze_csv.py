#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from utils import load_config, print_section, rle_to_mask
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()


def main():
    print_section("АНАЛИЗ TRAIN.CSV")
    
    df = pd.read_csv(cfg['paths']['train_csv'])
    logger.info(f"Строк: {len(df):,}, изображений: {df['ImageId'].nunique():,}")
    
    # Статистика классов
    images_per_class = defaultdict(set)
    masks_per_class = defaultdict(int)
    pixels_per_class = defaultdict(int)
    image_classes = defaultdict(set)
    
    for _, row in df.iterrows():
        img_id = row['ImageId']
        cls = row['ClassId']
        rle = row['EncodedPixels']
        
        images_per_class[cls].add(img_id)
        if pd.notna(rle) and str(rle).strip():
            masks_per_class[cls] += 1
            image_classes[img_id].add(cls)
            pixels = str(rle).split()
            pixels_per_class[cls] += sum(int(pixels[i]) for i in range(1, len(pixels), 2))
    
    total_masks = sum(masks_per_class.values())
    
    print(f"\n{'Класс':<10} {'Название':<18} {'Масок':<12} {'%':<10} {'Пикселей':<15}")
    print("-" * 70)
    for cls in sorted(cfg['classes']['names']):
        name = cfg['classes']['names'][cls]
        cnt = masks_per_class[cls]
        pct = cnt / total_masks * 100 if total_masks else 0
        print(f"{cls:<10} {name:<18} {cnt:<12,} {pct:<9.1f}% {pixels_per_class[cls]:<15,}")
    
    # Выводы
    most = max(masks_per_class, key=masks_per_class.get)
    least = min(masks_per_class, key=masks_per_class.get)
    
    print(f"\nДисбаланс: {cfg['classes']['names'][most]} в "
          f"{masks_per_class[most]/masks_per_class[least]:.1f}x чаще "
          f"{cfg['classes']['names'][least]}")
    print(f"Рекомендация: увеличить сэмплы класса {least}")
    
    logger.info("✅ Анализ завершён")


if __name__ == "__main__":
    main()