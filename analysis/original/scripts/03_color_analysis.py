#!/usr/bin/env python3
"""Цветовой анализ: оттенки, насыщенность, разнообразие."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import   ensure_dir, print_section, load_images_batch
from utils.color_utils import compute_color_stats
from utils.report_utils import save_figure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from utils import load_config
cfg = load_config(Path(__file__).parent.parent / "config.yaml")


def main():
    print_section("ЦВЕТОВОЙ АНАЛИЗ")
    
    p = cfg['paths']
    rpt = ensure_dir(p['reports_dir'])
    
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    sample = min(500, len(image_files))
    images = load_images_batch(image_files, max_images=sample)
    
    # Сбор цветовых признаков
    color_features = [compute_color_stats(img) for img in images]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # LAB L (яркость)
    l_values = [f['lab_ch0_mean'] for f in color_features]
    axes[0, 0].hist(l_values, bins=40, alpha=0.7, color='#FF6B6B')
    axes[0, 0].set_title('LAB L* (светлота)')
    axes[0, 0].axvline(np.mean(l_values), color='red', linestyle='--')
    
    # LAB A (зелёный-красный)
    a_values = [f['lab_ch1_mean'] for f in color_features]
    axes[0, 1].hist(a_values, bins=40, alpha=0.7, color='#4ECDC4')
    axes[0, 1].set_title('LAB a* (зелёный ↔ красный)')
    axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5)
    
    # LAB B (синий-жёлтый)
    b_values = [f['lab_ch2_mean'] for f in color_features]
    axes[0, 2].hist(b_values, bins=40, alpha=0.7, color='#45B7D1')
    axes[0, 2].set_title('LAB b* (синий ↔ жёлтый)')
    axes[0, 2].axvline(0, color='black', linestyle='-', alpha=0.5)
    
    # HSV H (тон)
    h_values = [f['hsv_ch0_mean'] for f in color_features]
    axes[1, 0].hist(h_values, bins=50, alpha=0.7, color='#96CEB4')
    axes[1, 0].set_title('HSV Hue (цветовой тон)')
    
    # HSV S (насыщенность)
    s_values = [f['hsv_ch1_mean'] for f in color_features]
    axes[1, 1].hist(s_values, bins=40, alpha=0.7, color='#FFA07A')
    axes[1, 1].set_title('HSV Saturation (насыщенность)')
    
    # LAB A vs B (цветовое пространство)
    axes[1, 2].scatter(a_values, b_values, c=l_values, cmap='viridis', alpha=0.5, s=10)
    axes[1, 2].set_xlabel('a* (зелёный → красный)')
    axes[1, 2].set_ylabel('b* (синий → жёлтый)')
    axes[1, 2].set_title('Цветовое распределение (a*b*)')
    axes[1, 2].axhline(0, color='gray', alpha=0.3)
    axes[1, 2].axvline(0, color='gray', alpha=0.3)
    
    plt.suptitle('Цветовой анализ: разнообразие оттенков металла', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "03_color_diversity.png", rpt, cfg['report']['dpi'])
    
    # Вывод
    print(f"\nЦветовая вариативность (CV):")
    print(f"  LAB L*:   CV = {np.std(l_values)/np.mean(l_values):.3f}")
    print(f"  LAB a*:   CV = {np.std(a_values)/max(np.abs(np.mean(a_values)), 1e-6):.3f}")
    print(f"  LAB b*:   CV = {np.std(b_values)/max(np.abs(np.mean(b_values)), 1e-6):.3f}")
    print(f"  HSV Hue:  std = {np.std(h_values):.1f} (разброс тонов)")
    print(f"  HSV Sat:  CV = {np.std(s_values)/max(np.mean(s_values), 1e-6):.3f}")
    print(f"  ↓ Высокий разброс = сильная неоднородность")
    
    logger.info(f"Отчёт: {rpt / '03_color_diversity.png'}")


if __name__ == "__main__":
    main()