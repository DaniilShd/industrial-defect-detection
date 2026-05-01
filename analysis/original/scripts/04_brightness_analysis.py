#!/usr/bin/env python3
"""Анализ яркости, засветов, недосветов, контраста."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.report_utils import save_figure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()


def analyze_brightness(img: np.ndarray, bc: dict) -> dict:
    """Анализ яркости одного изображения."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Глобальные метрики
    overexposed = (gray > bc['overexposure_threshold']).sum() / gray.size
    underexposed = (gray < bc['underexposure_threshold']).sum() / gray.size
    
    # Локальная яркость (разбиваем на блоки)
    h, w = gray.shape
    block_size = bc['local_block_size']
    blocks_h = h // block_size
    blocks_w = w // block_size
    
    block_means = []
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_means.append(block.mean())
    
    return {
        'mean_brightness': gray.mean(),
        'std_brightness': gray.std(),
        'overexposed_ratio': overexposed,
        'underexposed_ratio': underexposed,
        'contrast': gray.std() / gray.mean() if gray.mean() > 0 else 0,
        'local_brightness_std': np.std(block_means),
        'local_brightness_range': np.max(block_means) - np.min(block_means)
    }


def main():
    print_section("АНАЛИЗ ЯРКОСТИ И ЗАСВЕТОВ")
    
    p = cfg['paths']
    bc = cfg['brightness']
    rpt = ensure_dir(p['reports_dir'])
    
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    sample = min(500, len(image_files))
    images = load_images_batch(image_files, max_images=sample)
    
    # Сбор метрик
    brightness_data = [analyze_brightness(img, bc) for img in images]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Средняя яркость
    means = [d['mean_brightness'] for d in brightness_data]
    axes[0, 0].hist(means, bins=40, alpha=0.7, color='#FFD700')
    axes[0, 0].set_title('Средняя яркость')
    axes[0, 0].set_xlabel('Brightness (0-255)')
    
    # Std яркости (контраст)
    stds = [d['std_brightness'] for d in brightness_data]
    axes[0, 1].hist(stds, bins=40, alpha=0.7, color='#FF6B6B')
    axes[0, 1].set_title('Std яркости (глобальный контраст)')
    
    # Пересветы
    over = [d['overexposed_ratio'] * 100 for d in brightness_data]
    axes[0, 2].hist(over, bins=40, alpha=0.7, color='#FF4500')
    axes[0, 2].set_title('Доля пересвеченных пикселей (>{}'.format(bc['overexposure_threshold']) + ')')
    axes[0, 2].set_xlabel('%')
    
    # Недосветы
    under = [d['underexposed_ratio'] * 100 for d in brightness_data]
    axes[1, 0].hist(under, bins=40, alpha=0.7, color='#4169E1')
    axes[1, 0].set_title('Доля тёмных пикселей (<{}'.format(bc['underexposure_threshold']) + ')')
    axes[1, 0].set_xlabel('%')
    
    # Локальная вариация яркости
    local_std = [d['local_brightness_std'] for d in brightness_data]
    axes[1, 1].hist(local_std, bins=40, alpha=0.7, color='#4ECDC4')
    axes[1, 1].set_title('Локальная вариация яркости (неравномерность)')
    
    # Разброс локальной яркости
    local_range = [d['local_brightness_range'] for d in brightness_data]
    axes[1, 2].hist(local_range, bins=40, alpha=0.7, color='#96CEB4')
    axes[1, 2].set_title('Размах локальной яркости')
    
    plt.suptitle('Анализ яркости: засветы, недосветы, неравномерность', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "04_brightness_analysis.png", rpt, cfg['report']['dpi'])
    
    # Вывод
    print(f"\nЯркостные характеристики:")
    print(f"  Средняя яркость:    {np.mean(means):.1f} ± {np.std(means):.1f}")
    print(f"  Пересветы:          {np.mean(over):.2f}% (макс {np.max(over):.2f}%)")
    print(f"  Недосветы:          {np.mean(under):.2f}% (макс {np.max(under):.2f}%)")
    print(f"  Локальная вариация: {np.mean(local_std):.1f} ± {np.std(local_std):.1f}")
    print(f"  → Разброс локальной яркости указывает на неравномерность засветки")
    
    logger.info(f"Отчёт: {rpt / '04_brightness_analysis.png'}")


if __name__ == "__main__":
    main()