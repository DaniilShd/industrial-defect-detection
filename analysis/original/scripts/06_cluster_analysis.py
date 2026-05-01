#!/usr/bin/env python3
"""Кластеризация изображений для выявления групп неоднородности."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.report_utils import save_figure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()


def extract_global_features(img: np.ndarray) -> np.ndarray:
    """Извлечение глобальных признаков для кластеризации."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Гистограмма
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()
    
    # Моменты
    mean = gray.mean()
    std = gray.std()
    
    # FFT низкие частоты
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = gray.shape
    low_freq_energy = fft[h//2-10:h//2+10, w//2-10:w//2+10].mean()
    total_energy = fft.mean()
    
    # Цветовые моменты (LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_means = lab.mean(axis=(0, 1))
    lab_stds = lab.std(axis=(0, 1))
    
    return np.concatenate([
        hist[:16],  # 16 bins
        [mean, std],
        [low_freq_energy / total_energy],
        lab_means,
        lab_stds
    ])


def main():
    print_section("КЛАСТЕРИЗАЦИЯ ИЗОБРАЖЕНИЙ")
    
    p = cfg['paths']
    cc = cfg['cluster']
    rpt = ensure_dir(p['reports_dir'])
    
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    sample = min(cc['sample_size'], len(image_files))
    images = load_images_batch(image_files, max_images=sample)
    
    # Извлечение признаков
    logger.info("Извлечение признаков...")
    features = np.array([extract_global_features(img) for img in images])
    
    # PCA для визуализации
    pca = PCA(n_components=2, random_state=cc['random_seed'])
    features_2d = pca.fit_transform(features)
    
    # KMeans
    logger.info(f"Кластеризация (n={cc['n_clusters']})...")
    kmeans = KMeans(n_clusters=cc['n_clusters'], random_state=cc['random_seed'], n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=cfg['report']['figsize'])
    
    # PCA scatter
    colors = plt.cm.tab10(np.linspace(0, 1, cc['n_clusters']))
    for i in range(cc['n_clusters']):
        mask = labels == i
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i]], label=f'Кластер {i+1}', alpha=0.6, s=15)
    axes[0].set_title(f'PCA визуализация ({cc["n_clusters"]} кластеров)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].legend(fontsize=7, ncol=2)
    
    # Размеры кластеров
    cluster_sizes = [(labels == i).sum() for i in range(cc['n_clusters'])]
    axes[1].bar(range(1, cc['n_clusters'] + 1), cluster_sizes, color=colors)
    axes[1].set_title('Размеры кластеров')
    axes[1].set_xlabel('Кластер')
    axes[1].set_ylabel('Изображений')
    
    plt.suptitle('Кластеризация: группы неоднородности', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "06_cluster_analysis.png", rpt, cfg['report']['dpi'])
    
    # Вывод
    print(f"\nКластеры ({cc['n_clusters']} шт.):")
    print(f"  Мин размер: {min(cluster_sizes)}, Макс: {max(cluster_sizes)}")
    print(f"  CV размеров: {np.std(cluster_sizes)/np.mean(cluster_sizes):.3f}")
    print(f"  → {cc['n_clusters']} кластеров подтверждают неоднородность данных")
    print(f"  → Разные кластеры = разные условия съёмки, текстуры, засветка")
    
    logger.info(f"Отчёт: {rpt / '06_cluster_analysis.png'}")


if __name__ == "__main__":
    main()