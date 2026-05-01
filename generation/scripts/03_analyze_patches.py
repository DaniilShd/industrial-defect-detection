#!/usr/bin/env python3
"""Анализ дефектных патчей: распределения, размеры, совместное присутствие."""
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
    """Статистика: боксы, патчи, совместное присутствие."""
    class_counts = defaultdict(int)       # боксов на класс
    patches_per_class = defaultdict(set)  # патчей с каждым классом
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


def plot_pie_patches_per_class(patches_per_class: dict, names: list, rpt: Path):
    """Круговая диаграмма: количество патчей с каждым классом."""
    cls_ids = sorted(patches_per_class.keys())
    sizes = [len(patches_per_class[c]) for c in cls_ids]
    labels = [names[c] if c < len(names) else f"Cls_{c}" for c in cls_ids]
    colors_pie = [cfg['classes']['colors'].get(c + 1, '#333') for c in cls_ids]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors_pie, textprops={'fontsize': 11}
    )
    axes[0].set_title('Доля патчей, содержащих класс', fontweight='bold')
    
    # Bar chart
    bars = axes[1].bar(labels, sizes, color=colors_pie)
    axes[1].set_title('Патчей с классом', fontweight='bold')
    axes[1].set_ylabel('Количество патчей')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, v in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                     str(v), ha='center', fontweight='bold')
    
    plt.suptitle('Распределение патчей по классам', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "patches_per_class_pie.png", rpt, cfg['report']['dpi'])
    plt.close()
    logger.info(f"Круговая диаграмма сохранена: {rpt / 'patches_per_class_pie.png'}")


def plot_cooccurrence(patches_per_class: dict, names: list, rpt: Path):
    """Тепловая карта совместного присутствия классов."""
    cls_ids = sorted(patches_per_class.keys())
    n = len(cls_ids)
    labels = [names[c] if c < len(names) else f"Cls_{c}" for c in cls_ids]
    
    # Матрица пересечений
    matrix = np.zeros((n, n))
    for i, ci in enumerate(cls_ids):
        for j, cj in enumerate(cls_ids):
            if i == j:
                matrix[i][j] = len(patches_per_class[ci])
            else:
                matrix[i][j] = len(patches_per_class[ci] & patches_per_class[cj])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    
    # Цифры в ячейках
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{matrix[i][j]:.0f}', ha='center', va='center',
                    color='white' if matrix[i][j] > matrix.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax)
    ax.set_title('Совместное присутствие классов в патчах', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "class_cooccurrence.png", rpt, cfg['report']['dpi'])
    plt.close()
    logger.info(f"Тепловая карта сохранена: {rpt / 'class_cooccurrence.png'}")


def main():
    print_section("АНАЛИЗ ДЕФЕКТНЫХ ПАТЧЕЙ")
    
    rpt = ensure_dir(cfg['paths']['reports_dir'])
    ann, ds = load_data()
    class_counts, ppc, total, bpp = class_stats(ann)
    names = ds['names']
    colors_hex = [cfg['classes']['colors'].get(i + 1, '#333') for i in sorted(class_counts)]
    
    logger.info(f"Патчей: {len(ann)}, Боксов: {total}")
    
    # 1. Распределение боксов по классам (bar)
    fig, ax = plt.subplots(figsize=cfg['report']['figsize'])
    cls_ids = sorted(class_counts)
    x = range(len(cls_ids))
    ax.bar(x, [class_counts[i] for i in cls_ids], color=colors_hex)
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] if i < len(names) else f"Cls_{i}" for i in cls_ids], rotation=45)
    ax.set_title('Боксы по классам')
    for i, v in enumerate([class_counts[c] for c in cls_ids]):
        ax.text(i, v + max(class_counts.values()) * 0.01, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "boxes_per_class.png", rpt, cfg['report']['dpi'])
    plt.close()
    
    # 2. Круговая диаграмма патчей по классам (НОВОЕ)
    plot_pie_patches_per_class(ppc, names, rpt)
    
    # 3. Тепловая карта совместного присутствия (НОВОЕ)
    plot_cooccurrence(ppc, names, rpt)
    
    # 4. Размеры дефектов
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
    
    # 5. Текстовый вывод
    print(f"\n{'Класс':<15} {'Боксов':<8} {'Патчей':<8} {'Боксов/патч':<12}")
    print("-" * 45)
    for cls in sorted(class_counts):
        name = names[cls] if cls < len(names) else f"Cls_{cls}"
        cnt = class_counts[cls]
        n_patches = len(ppc[cls])
        avg = cnt / n_patches if n_patches else 0
        print(f"{name:<15} {cnt:<8} {n_patches:<8} {avg:.2f}")
    
    logger.info(f"✅ Все отчёты: {rpt}")
    logger.info(f"  - boxes_per_class.png")
    logger.info(f"  - patches_per_class_pie.png")
    logger.info(f"  - class_cooccurrence.png")
    logger.info(f"  - defect_sizes.png")


if __name__ == "__main__":
    main()