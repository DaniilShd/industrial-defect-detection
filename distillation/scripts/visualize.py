#!/usr/bin/env python3
"""Scatter plot: качество vs скорость"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plot(results: list, output_dir: Path):
    """Строит scatter plot mAP@50 vs FPS."""
    fig, ax = plt.subplots(figsize=(12, 8))

    models = {
        'teacher_ltdetr': ('LTDETR+DINOv3', '#E74C3C', 's', 200),
        'yolo_nano': ('YOLOv8 Nano', '#3498DB', 'o', 150),
        'yolo_nano_fgd': ('YOLOv8 Nano (FGD)', '#2980B9', 'o', 150),
        'picodet_s': ('PicoDet-S', '#2ECC71', '^', 150),
        'picodet_s_fgd': ('PicoDet-S (FGD)', '#27AE60', '^', 150),
        'faster_rcnn_r18': ('Faster R-CNN R18 (FGD)', '#F39C12', 'D', 150),
    }

    for key, (label, color, marker, size) in models.items():
        r = next((r for r in results if r.get('model') == key), None)
        if r:
            ax.scatter(r['fps'], r['map50'], c=color, marker=marker, s=size,
                      label=label, edgecolors='black', linewidth=1, zorder=5)
            ax.annotate(label, (r['fps'], r['map50']),
                       textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.set_xlabel('FPS (CPU)', fontsize=12)
    ax.set_ylabel('mAP@50', fontsize=12)
    ax.set_title('Качество vs Скорость: Дистилляция в лёгкие модели', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_vs_speed.png', dpi=300)
    plt.close()