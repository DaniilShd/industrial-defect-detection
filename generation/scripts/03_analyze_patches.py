#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from collections import defaultdict
from utils import load_config, ensure_dir, print_section
from utils.report_utils import save_figure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    p = cfg['paths']
    with open(Path(p['defect_patches_dir']) / 'annotations.json') as f:
        ann = json.load(f)
    with open(Path(p['defect_patches_dir']) / 'dataset.yaml') as f:
        ds = yaml.safe_load(f)
    return ann, ds


def class_stats(annotations):
    class_counts = defaultdict(int)
    patches_per_class = defaultdict(set)
    boxes_per_patch = []
    total = 0
    
    base = Path(cfg['paths']['defect_patches_dir']) / cfg['paths']['yolo_labels_subdir']
    
    for a in annotations:
        lbl = base / f"{a['saved_as']}.txt"
        if lbl.exists():
            classes = set()
            for line in open(lbl):
                if line.strip():
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
                    classes.add(cls)
                    total += 1
            for c in classes:
                patches_per_class[c].add(a['saved_as'])
            boxes_per_patch.append(len(classes))
        else:
            boxes_per_patch.append(0)
    
    return class_counts, patches_per_class, total, boxes_per_patch


def main():
    print_section("АНАЛИЗ ДЕФЕКТНЫХ ПАТЧЕЙ")
    
    rpt = ensure_dir(cfg['paths']['reports_dir'])
    ann, ds = load_data()
    class_counts, ppc, total, bpp = class_stats(ann)
    names = ds['names']
    colors = [cfg['classes']['colors'].get(i + 1, '#333') for i in sorted(class_counts)]
    
    logger.info(f"Патчей: {len(ann)}, Боксов: {total}")
    
    # Распределение
    fig, ax = plt.subplots(figsize=cfg['report']['figsize'])
    cls_ids = sorted(class_counts)
    x = range(len(cls_ids))
    ax.bar(x, [class_counts[i] for i in cls_ids], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] if i < len(names) else f"Cls_{i}" for i in cls_ids], rotation=45)
    ax.set_title('Боксы по классам')
    for i, v in enumerate([class_counts[c] for c in cls_ids]):
        ax.text(i, v + max(class_counts.values()) * 0.01, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "class_distribution.png", rpt, cfg['report']['dpi'])
    plt.close()
    
    # Размеры дефектов
    box_sizes = defaultdict(list)
    base = Path(cfg['paths']['defect_patches_dir']) / cfg['paths']['yolo_labels_subdir']
    for a in ann:
        lbl = base / f"{a['saved_as']}.txt"
        if lbl.exists():
            for line in open(lbl):
                if line.strip():
                    parts = line.strip().split()
                    box_sizes[int(parts[0])].append(float(parts[3]) * float(parts[4]))
    
    sorted_cls = sorted(box_sizes)
    rows = (len(sorted_cls) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes = axes.flatten() if len(sorted_cls) > 1 else [axes]
    
    for i, cls in enumerate(sorted_cls):
        if i < len(axes):
            areas = box_sizes[cls]
            axes[i].hist(areas, bins=50, alpha=0.7, color=cfg['classes']['colors'].get(cls + 1, '#333'))
            axes[i].set_title(f"{names[cls] if cls < len(names) else f'Cls_{cls}'}")
            axes[i].axvline(np.mean(areas), color='red', linestyle='--', label=f'μ={np.mean(areas):.4f}')
            axes[i].legend(fontsize=8)
    
    for i in range(len(sorted_cls), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure(fig, "defect_sizes.png", rpt, cfg['report']['dpi'])
    plt.close()
    
    # Вывод
    print(f"\nКласс | Боксов | Патчей | Боксов/патч")
    print("-" * 45)
    for cls in sorted(class_counts):
        name = names[cls] if cls < len(names) else f"Cls_{cls}"
        cnt = class_counts[cls]
        n_patches = len(ppc[cls])
        print(f"{name:<15} {cnt:<7} {n_patches:<7} {cnt/n_patches:.2f}" if n_patches else f"{name:<15} {cnt:<7} {n_patches:<7} -")
    
    logger.info(f"Отчёты: {rpt}")


if __name__ == "__main__":
    main()