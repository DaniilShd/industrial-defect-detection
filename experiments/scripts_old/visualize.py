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
    """Создаёт все графики"""
    
    completed = [r for r in all_results if r.get('status') == 'completed']
    if not completed:
        return
    
    # Группируем
    groups = defaultdict(list)
    for r in completed:
        groups[r['dataset_name']].append(r['test_map50'])
    
    names = sorted(groups.keys())
    
    # 1. Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [groups[n] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(get_color(name))
    ax.set_ylabel('Test mAP@50')
    ax.set_title('Влияние состава данных на качество детекции')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_map50.png', dpi=300)
    plt.close()
    
    # 2. Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [np.mean(groups[n]) for n in names]
    sems = [np.std(groups[n], ddof=1) / np.sqrt(len(groups[n])) for n in names]
    colors = [get_color(n) for n in names]
    ax.bar(names, means, yerr=sems, capsize=5, color=colors, edgecolor='black')
    ax.set_ylabel('Mean mAP@50 ± SEM')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'bar_chart_map50.png', dpi=300)
    plt.close()


def get_color(name: str) -> str:
    if 'baseline' in name: return '#E74C3C'
    elif 'synthetic' in name: return '#3498DB'
    elif 'augmented' in name: return '#2ECC71'
    elif 'mixed' in name: return '#F39C12'
    return '#95A5A6'