#!/usr/bin/env python3
"""Балансировка чистых патчей с кластеризацией KMeans (70/15/15)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import shutil
import random
from sklearn.cluster import KMeans
from collections import defaultdict
import logging
from utils import load_config, ensure_dir, print_section

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()

np.random.seed(cfg['split']['random_seed'])
random.seed(cfg['split']['random_seed'])


def extract_features(img_path: Path) -> np.ndarray:
    """Извлечение признаков: среднее, std, гистограмма."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    return np.concatenate([[mean_val, std_val], hist[:10]])


def cluster_images(image_paths: list, n_clusters: int) -> dict:
    """Кластеризация изображений по признакам."""
    logger.info("Извлечение признаков...")
    
    features = []
    valid_images = []
    
    for img_path in image_paths:
        feat = extract_features(img_path)
        if feat is not None:
            features.append(feat)
            valid_images.append(img_path)
    
    features = np.array(features)
    logger.info(f"Проанализировано: {len(features)}")
    
    logger.info(f"KMeans (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg['split']['random_seed'], n_init=10)
    clusters = kmeans.fit_predict(features)
    
    cluster_dict = defaultdict(list)
    for img_path, cluster in zip(valid_images, clusters):
        cluster_dict[cluster].append(img_path)
    
    return cluster_dict


def select_balanced(images: list, target_count: int, n_clusters: int) -> list:
    """Равномерный отбор из кластеров."""
    total = len(images)
    
    if total <= target_count:
        logger.info(f"Изображений {total} ≤ {target_count}, отбираем все")
        return images
    
    cluster_dict = cluster_images(images, n_clusters)
    logger.info(f"Кластеров: {len(cluster_dict)}")
    
    per_cluster = target_count // n_clusters
    remainder = target_count % n_clusters
    
    selected = []
    for cluster_id in range(n_clusters):
        available = cluster_dict[cluster_id]
        to_select = min(per_cluster + (1 if cluster_id < remainder else 0), len(available))
        selected.extend(random.sample(available, to_select))
    
    logger.info(f"Отобрано: {len(selected)} из {total}")
    return selected


def split_and_copy(images: list, out_dir: Path) -> dict:
    """Разбиение и копирование."""
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
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    for split_name, split_images in splits.items():
        split_dir = ensure_dir(out_dir / split_name)
        for img_path in split_images:
            dst = split_dir / img_path.name
            if dst.exists():
                dst = split_dir / f"{img_path.stem}_dup{img_path.suffix}"
            shutil.copy2(img_path, dst)
        logger.info(f"  {split_name}: {len(split_images)}")
    
    return splits


def main():
    print_section("БАЛАНСИРОВКА ЧИСТЫХ ПАТЧЕЙ С КЛАСТЕРИЗАЦИЕЙ")
    
    p = cfg['paths']
    sp = cfg['split']
    
    src_dir = Path(p['clean_patches_dir'])
    out_dir = Path(p['balanced_clean_patches_dir'])
    
    all_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        all_images.extend(src_dir.rglob(ext))
    
    logger.info(f"Найдено: {len(all_images):,}")
    
    if not all_images:
        logger.error("Изображения не найдены!")
        return
    
    # Отбор с кластеризацией
    target = sp.get('clean_target_count', 2000)
    n_clusters = sp.get('clean_n_clusters', 20)
    
    selected = select_balanced(all_images, target, n_clusters)
    splits = split_and_copy(selected, out_dir)
    
    total = sum(len(v) for v in splits.values())
    
    print(f"""
{'='*50}
ГОТОВО!
{'='*50}
✅ ОТОБРАНО: {total} чистых патчей

📁 ВЫХОД: {out_dir}
   ├── train/ ({len(splits['train'])})
   ├── val/   ({len(splits['val'])})
   └── test/  ({len(splits['test'])})

📊 РАЗБИЕНИЕ: {sp['train_ratio']*100:.0f}/{sp['val_ratio']*100:.0f}/{sp['test_ratio']*100:.0f}
""")
    
    logger.info(f"✅ Готово: {out_dir}")


if __name__ == "__main__":
    main()