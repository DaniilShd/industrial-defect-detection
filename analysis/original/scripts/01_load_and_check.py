#!/usr/bin/env python3
"""Загрузка данных и проверка целостности."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from utils import ensure_dir, print_section
from utils import load_config
cfg = load_config(Path(__file__).parent.parent / "config.yaml")
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print_section("ЗАГРУЗКА И ПРОВЕРКА ДАННЫХ")
    
    p = cfg['paths']
    
    # Загрузка CSV
    df = pd.read_csv(p['train_csv'])
    logger.info(f"CSV: {len(df):,} строк, {df['ImageId'].nunique():,} уникальных изображений")
    
    # Проверка дубликатов
    dup_images = df['ImageId'].duplicated().sum()
    logger.info(f"Дубликатов ImageId: {dup_images}")
    
    # Статистика по RLE
    rle_empty = df['EncodedPixels'].isna().sum()
    rle_filled = len(df) - rle_empty
    logger.info(f"RLE заполнено: {rle_filled:,}, пустых: {rle_empty:,}")
    
    # Проверка изображений
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    logger.info(f"Файлов изображений: {len(image_files):,}")
    
    # Проверка соответствия
    csv_images = set(df['ImageId'].unique())
    disk_images = {f.name for f in image_files}
    
    missing_on_disk = csv_images - disk_images
    missing_in_csv = disk_images - csv_images
    
    if missing_on_disk:
        logger.warning(f"В CSV но не на диске: {len(missing_on_disk)}")
    if missing_in_csv:
        logger.warning(f"На диске но не в CSV: {len(missing_in_csv)}")
    
    if not missing_on_disk and not missing_in_csv:
        logger.info("✅ Все изображения соответствуют CSV")
    
    # Сохранение статистики
    stats = {
        'total_rows': len(df),
        'unique_images': df['ImageId'].nunique(),
        'rle_filled': rle_filled,
        'rle_empty': rle_empty,
        'disk_images': len(image_files),
        'missing_disk': len(missing_on_disk),
        'missing_csv': len(missing_in_csv),
    }
    
    report_dir = ensure_dir(p['reports_dir'])
    pd.DataFrame([stats]).to_csv(report_dir / "01_data_integrity.csv", index=False)
    logger.info(f"Отчёт сохранён: {report_dir / '01_data_integrity.csv'}")


if __name__ == "__main__":
    main()