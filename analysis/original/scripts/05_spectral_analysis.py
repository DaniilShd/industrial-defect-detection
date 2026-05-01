#!/usr/bin/env python3
"""Спектральный анализ: FFT, частотные характеристики."""
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


def compute_fft_features(gray: np.ndarray, n_bands: int = 8) -> dict:
    """Вычисление спектральных признаков через FFT."""
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    features = {}
    
    # Радиальные полосы
    max_radius = min(center_y, center_x)
    band_width = max_radius / n_bands
    
    for i in range(n_bands):
        r_inner = i * band_width
        r_outer = (i + 1) * band_width
        
        mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                if r_inner <= r < r_outer:
                    mask[y, x] = True
        
        band_energy = magnitude[mask].sum()
        features[f'fft_band_{i}_energy'] = float(band_energy)
    
    # Низкие vs высокие частоты
    low_bands = sum(features[f'fft_band_{i}_energy'] for i in range(n_bands // 2))
    high_bands = sum(features[f'fft_band_{i}_energy'] for i in range(n_bands // 2, n_bands))
    features['low_high_ratio'] = low_bands / high_bands if high_bands > 0 else 0
    
    return features


def main():
    print_section("СПЕКТРАЛЬНЫЙ АНАЛИЗ (FFT)")
    
    p = cfg['paths']
    sc = cfg['spectral']
    rpt = ensure_dir(p['reports_dir'])
    
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    sample = min(300, len(image_files))
    images = load_images_batch(image_files, max_images=sample)
    
    # сбор FFT признаков
    fft_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fft = compute_fft_features(gray, sc['fft_bands'])
        fft_features.append(fft)
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=cfg['report']['figsize'])
    
    # Энергия по частотным полосам (среднее ± std)
    bands = range(sc['fft_bands'])
    band_means = [np.mean([f[f'fft_band_{i}_energy'] for f in fft_features]) for i in bands]
    band_stds = [np.std([f[f'fft_band_{i}_energy'] for f in fft_features]) for i in bands]
    
    axes[0, 0].bar(bands, band_means, yerr=band_stds, color='#FF6B6B', capsize=3)
    axes[0, 0].set_title('Энергия по частотным полосам')
    axes[0, 0].set_xlabel('Полоса (0=низкие, 7=высокие)')
    axes[0, 0].set_ylabel('Энергия')
    
    # Low/High ratio
    lh_ratios = [f['low_high_ratio'] for f in fft_features]
    axes[0, 1].hist(lh_ratios, bins=30, alpha=0.7, color='#4ECDC4')
    axes[0, 1].set_title('Отношение низкие/высокие частоты')
    axes[0, 1].axvline(np.mean(lh_ratios), color='red', linestyle='--',
                       label=f'μ={np.mean(lh_ratios):.2f}')
    axes[0, 1].legend()
    
    # Пример FFT спектра для первых 4 изображений
    for i in range(min(4, len(images))):
        gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        fft_log = np.log1p(fft_mag)
        
        row, col = i // 2, i % 2
        axes[1, row].imshow(fft_log, cmap='inferno', aspect='auto')
        axes[1, row].set_title(f'FFT пример {i+1}')
        axes[1, row].axis('off')
    
    plt.suptitle('Спектральный анализ: частотные характеристики', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "05_spectral_analysis.png", rpt, cfg['report']['dpi'])
    
    # Вывод
    print(f"\nСпектральные характеристики:")
    print(f"  Low/High ratio: {np.mean(lh_ratios):.2f} ± {np.std(lh_ratios):.2f}")
    print(f"  CV Low/High:     {np.std(lh_ratios)/np.mean(lh_ratios):.3f}")
    print(f"  → Высокий разброс = разные частотные профили фона")
    
    logger.info(f"Отчёт: {rpt / '05_spectral_analysis.png'}")


if __name__ == "__main__":
    main()