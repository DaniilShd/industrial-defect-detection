#!/usr/bin/env python3
"""Визуализации для эксперимента дистилляции"""

from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def create_all_visualizations(results: List[Dict], output_dir: Path):
    """Создаёт все визуализации."""
    create_quality_speed_scatter(results, output_dir)
    create_map_comparison(results, output_dir)
    create_distillation_gain(results, output_dir)
    create_params_comparison(results, output_dir)
    create_results_table(results, output_dir)


def create_quality_speed_scatter(results: List[Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'teacher': '#E74C3C', 'baseline': '#3498DB', 'kd': '#2ECC71', 'fitnet': '#F39C12', 'hybrid': '#9B59B6'}
    
    for r in results:
        model_name = r.get('model', 'unknown')
        fps = r.get('fps', 0)
        map50 = r.get('map50', 0)
        
        color = colors.get(r.get('distill_method', 'baseline'), '#95A5A6')
        ax.scatter(fps, map50, c=color, s=200, edgecolors='black', linewidth=1, zorder=5, label=model_name)
        ax.annotate(model_name, (fps, map50), textcoords="offset points", xytext=(10, 10), fontsize=8)
    
    ax.set_xlabel('FPS (CPU)', fontsize=12)
    ax.set_ylabel('mAP@50', fontsize=12)
    ax.set_title('Quality vs Speed: Knowledge Distillation', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_vs_speed.png', dpi=300)
    plt.close()


def create_map_comparison(results: List[Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    models = [r['model'].replace('_', ' ').title() for r in results]
    maps = [r['map50'] for r in results]
    
    colors = ['#E74C3C' if 'teacher' in m else '#2ECC71' if 'hybrid' in m else '#3498DB' for m in models]
    ax.bar(range(len(models)), maps, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('mAP@50')
    ax.set_title('Detection Performance Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'map_comparison.png', dpi=300)
    plt.close()


def create_distillation_gain(results: List[Dict], output_dir: Path):
    baseline = next((r for r in results if 'baseline' in r['model']), None)
    if not baseline:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_map = baseline['map50']
    
    distill_results = [r for r in results if r != baseline and 'teacher' not in r['model']]
    names = [r['model'].replace('_', ' ').title() for r in distill_results]
    gains = [r['map50'] - baseline_map for r in distill_results]
    
    colors = ['#2ECC71' if g > 0 else '#E74C3C' for g in gains]
    ax.bar(range(len(names)), gains, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Δ mAP@50 from Baseline')
    ax.set_title('Improvement from Knowledge Distillation')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'distillation_gain.png', dpi=300)
    plt.close()


def create_params_comparison(results: List[Dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    student_results = [r for r in results if 'teacher' not in r['model']]
    names = [r['model'].replace('_', ' ').title() for r in student_results]
    params = [r.get('params_M', 0) for r in student_results]
    
    ax.bar(range(len(names)), params, color='#3498DB')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Size Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'params_comparison.png', dpi=300)
    plt.close()


def create_results_table(results: List[Dict], output_dir: Path):
    import logging
    logger = logging.getLogger(__name__)
    
    lines = ["="*100]
    lines.append(f"{'Model':<35} {'mAP@50':<10} {'mAP@75':<10} {'FPS':<10} {'Params(M)':<12} {'Size(MB)':<10} {'Method':<15}")
    lines.append("-"*100)
    
    for r in sorted(results, key=lambda x: x.get('map50', 0), reverse=True):
        lines.append(
            f"{r['model']:<35} {r.get('map50', 0):<10.4f} {r.get('map75', 0):<10.4f} "
            f"{r.get('fps', 0):<10.1f} {r.get('params_M', 0):<12.1f} "
            f"{r.get('size_MB', 0):<10.1f} {r.get('distill_method', 'teacher'):<15}"
        )
    
    lines.append("="*100)
    
    with open(output_dir / 'results_table.txt', 'w') as f:
        f.write('\n'.join(lines))
    
    for line in lines:
        logger.info(line)