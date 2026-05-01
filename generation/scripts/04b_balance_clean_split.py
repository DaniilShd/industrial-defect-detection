#!/usr/bin/env python3
"""Балансировка и разбиение чистых патчей с кластеризацией KMeans (70/15/15)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import shutil
import random
from sklearn.cluster import KMeans
from collections import defaultdict
from utils import load_config, ensure_dir, print_section
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()

np.random.seed(cfg['split']['random_seed'])
random.seed(cfg['split']['random_seed'])


def extract_features(img_path: Path):
    """Извлекает признаки изображения: среднее, std, гистограмма."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    return np.concatenate([[mean_val, std_val], hist[:10]])


def cluster_images(image_paths: list, n_clusters: int = 20):
    """Кластеризует изображения по признакам."""
    logger.info("Извлечение признаков...")
    
    features = []
    valid_images = []
    
    for img_path in image_paths:
        feat = extract_features(img_path)
        if feat is not None:
            features.append(feat)
            valid_images.append(img_path)
    
    features = np.array(features)
    logger.info(f"Проанализировано: {len(features)} изображений")
    
    logger.info(f"Кластеризация (KMeans, n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg['split']['random_seed'], n_init=10)
    clusters = kmeans.fit_predict(features)
    
    cluster_images = defaultdict(list)
    for img_path, cluster in zip(valid_images, clusters):
        cluster_images[cluster].append(img_path)
    
    return cluster_images


def select_balanced(images: list, target_count: int, n_clusters: int = 20):
    """Равномерный отбор изображений из кластеров."""
    total = len(images)
    
    if total <= target_count:
        logger.info(f"Изображений ({total}) меньше целевого ({target_count}), отбираем все")
        return images
    
    # Кластеризация
    cluster_images = cluster_images(images, n_clusters)
    logger.info(f"Кластеров: {len(cluster_images)}")
    
    # Равномерный отбор
    per_cluster = target_count // n_clusters
    remainder = target_count % n_clusters
    
    selected = []
    for cluster_id in range(n_clusters):
        available = cluster_images[cluster_id]
        to_select = min(per_cluster + (1 if cluster_id < remainder else 0), len(available))
        selected.extend(random.sample(available, to_select))
    
    logger.info(f"Отобрано: {len(selected)} из {total}")
    return selected


def split_and_copy(images: list, out_dir: Path):
    """Разбиение 70/15/15 и копирование."""
    sp = cfg['split']
    
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * sp['train_ratio'])
    val_end = train_end + int(total * sp['val_ratio'])
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    # Очистка и копирование
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    for split_name, split_images in splits.items():
        split_dir = ensure_dir(out_dir / split_name)
        for img_path in split_images:
            dst = split_dir / img_path.name
            if dst.exists():
                dst = split_dir / f"{img_path.stem}_dup{img_path.suffix}"
            shutil.copy2(img_path, dst)
        
        logger.info(f"  {split_name}: {len(split_images)} изображений")
    
    return splits


def main():
    print_section("БАЛАНСИРОВКА ЧИСТЫХ ПАТЧЕЙ (70/15/15) С КЛАСТЕРИЗАЦИЕЙ")
    
    p = cfg['paths']
    src_dir = Path(p['clean_patches_dir'])
    out_dir = Path(p['balanced_clean_patches_dir'])
    
    # Поиск всех изображений
    all_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        all_images.extend(src_dir.rglob(ext))
    
    logger.info(f"Найдено изображений: {len(all_images):,}")
    
    if not all_images:
        logger.error("Изображения не найдены!")
        return
    
    # Отбор с кластеризацией
    TARGET_COUNT = 2000
    n_clusters = 20
    
    selected = select_balanced(all_images, TARGET_COUNT, n_clusters)
    
    # Разбиение и копирование
    splits = split_and_copy(selected, out_dir)
    
    # Итог
    total = sum(len(v) for v in splits.values())
    
    print(f"\n{'='*50}")
    print("ГОТОВО!")
    print(f"{'='*50}")
    print(f"""
✅ ОТОБРАНО: {total} чистых патчей

📁 ВЫХОДНАЯ ПАПКА: {out_dir}
   ├── train/ ({len(splits['train'])} изображений)
   ├── val/   ({len(splits['val'])} изображений)
   └── test/  ({len(splits['test'])} изображений)

📊 РАЗБИЕНИЕ: {cfg['split']['train_ratio']*100:.0f}/"""
          f"""{cfg['split']['val_ratio']*100:.0f}/"""
          f"""{cfg['split']['test_ratio']*100:.0f}
""")
    
    logger.info(f"✅ Готово: {out_dir}")


if __name__ == "__main__":
    main()