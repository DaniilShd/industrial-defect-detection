#!/usr/bin/env python3
"""Анализ текстур: GLCM, LBP, Gabor, градиенты."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.texture_utils import compute_glcm_features, compute_lbp_features, compute_gabor_features
from utils.image_utils import compute_gradient_magnitude
from utils.report_utils import save_figure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config(Path(__file__).parent.parent / "config.yaml")


def main():
    print_section("АНАЛИЗ ТЕКСТУР")
    
    p = cfg['paths']
    tc = cfg['texture']
    rpt = ensure_dir(p['reports_dir'])
    
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    sample = min(500, len(image_files))
    images = load_images_batch(image_files, max_images=sample)
    
    glcm_features = []
    lbp_features = []
    gabor_features = []
    gradient_means = []
    
    for img in images:
        gray = np.mean(img, axis=2).astype(np.uint8)
        
        glcm = compute_glcm_features(gray, tc['glcm_distances'], tc['glcm_angles'])
        lbp = compute_lbp_features(gray, tc['lbp_radius'], tc['lbp_points'])
        gabor = compute_gabor_features(gray, tc['gabor_frequencies'], tc['gabor_angles'])
        grad = compute_gradient_magnitude(img)
        
        glcm_features.append(glcm)
        lbp_features.append(lbp)
        gabor_features.append(gabor)
        gradient_means.append(grad.mean())
    
    fig, axes = plt.subplots(2, 2, figsize=cfg['report']['figsize'])
    
    contrast_values = [f['glcm_contrast_mean'] for f in glcm_features]
    axes[0, 0].hist(contrast_values, bins=30, alpha=0.7, color='#FF6B6B')
    axes[0, 0].set_title('GLCM Contrast')
    axes[0, 0].axvline(np.mean(contrast_values), color='red', linestyle='--',
                       label=f'μ={np.mean(contrast_values):.1f}')
    axes[0, 0].legend()
    
    lbp_entropy = [f['lbp_entropy'] for f in lbp_features]
    axes[0, 1].hist(lbp_entropy, bins=30, alpha=0.7, color='#4ECDC4')
    axes[0, 1].set_title('LBP Entropy')
    axes[0, 1].axvline(np.mean(lbp_entropy), color='red', linestyle='--',
                       label=f'μ={np.mean(lbp_entropy):.3f}')
    axes[0, 1].legend()
    
    axes[1, 0].hist(gradient_means, bins=30, alpha=0.7, color='#45B7D1')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axvline(np.mean(gradient_means), color='red', linestyle='--',
                       label=f'μ={np.mean(gradient_means):.2f}')
    axes[1, 0].legend()
    
    gabor_key = 'gabor_t90_f0.3_mean'
    gabor_values = [f.get(gabor_key, 0) for f in gabor_features]
    axes[1, 1].hist(gabor_values, bins=30, alpha=0.7, color='#96CEB4')
    axes[1, 1].set_title('Gabor Response')
    axes[1, 1].axvline(np.mean(gabor_values), color='red', linestyle='--',
                       label=f'μ={np.mean(gabor_values):.4f}')
    axes[1, 1].legend()
    
    plt.suptitle('Анализ разнообразия текстур', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "02_texture_diversity.png", rpt, cfg['report']['dpi'])
    
    print(f"\nРазброс текстурных признаков (CV = std/mean):")
    print(f"  GLCM contrast: CV = {np.std(contrast_values)/np.mean(contrast_values):.2f}")
    print(f"  LBP entropy:   CV = {np.std(lbp_entropy)/np.mean(lbp_entropy):.2f}")
    print(f"  Gradient:      CV = {np.std(gradient_means)/np.mean(gradient_means):.2f}")
    
    logger.info(f"Отчёт: {rpt / '02_texture_diversity.png'}")


if __name__ == "__main__":
    main()