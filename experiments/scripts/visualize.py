#!/usr/bin/env python3
"""Визуализация результатов эксперимента"""

from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-paper')


def create_all_visualizations(all_results: list, stats: dict, config: dict, output_dir: Path):
    completed = [r for r in all_results if r.get('status') == 'completed']
    if not completed:
        return

    # Группируем по датасету + стратегии
    groups = defaultdict(list)
    for r in completed:
        key = f"{r['dataset_name']}_{r['strategy_name']}"
        groups[key].append(r['test_map50'])

    names = sorted(groups.keys())
    data = [groups[n] for n in names]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, name in zip(bp['boxes'], names):
        if 'frozen' in name:
            patch.set_facecolor('#3498DB')
        elif 'finetune' in name:
            patch.set_facecolor('#E74C3C')
        elif 'ssl' in name:
            patch.set_facecolor('#2ECC71')
    
    ax.set_ylabel('Test mAP@50')
    ax.set_title('Сравнение стратегий обучения по датасетам')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_map50.png', dpi=300)
    plt.close()


def get_color(name: str) -> str:
    if 'baseline' in name: return '#E74C3C'
    elif 'synthetic' in name: return '#3498DB'
    elif 'augmented' in name: return '#2ECC71'
    elif 'mixed' in name: return '#F39C12'
    return '#95A5A6'