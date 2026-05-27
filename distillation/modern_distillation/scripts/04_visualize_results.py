#!/usr/bin/env python3
"""
Визуализация результатов сравнения методов инициализации бэкбона

Графики:
  1. Сравнение mAP@50 (bar chart)
  2. Кривые обучения (loss + mAP по эпохам)
  3. Quality vs Speed (mAP vs FPS scatter)
  4. Per-class AP сравнение
  5. Прирост относительно scratch baseline
  6. Precision/Recall/F1 сравнение
  7. Радарная диаграмма
  8. Скорость сходимости
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('04_visualize_results.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Визуализатор результатов экспериментов."""
    
    def __init__(self, config: dict, evaluation_results: List[Dict], training_histories: Dict[str, List[Dict]]):
        self.config = config
        self.evaluation_results = evaluation_results
        self.training_histories = training_histories
        self.output_dir = Path(config['paths']['results_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Цветовая схема
        self.colors = {
            'teacher': '#E74C3C',      # Красный
            'scratch': '#95A5A6',      # Серый
            'imagenet_pretrained': '#3498DB',  # Синий
            'lightly_pretrained': '#2ECC71',
            'modern_distilled': '#E67E22',
            'modern_distilled': '#F39C12',   # Зелёный
        }
        
        # Названия для легенд
        self.labels = {
            'teacher_ltdetr': 'LTDETR (учитель)',
            'faster_rcnn_r18_scratch': 'Случайная инициализация',
            'faster_rcnn_r18_imagenet': 'ImageNet предобучение',
            'faster_rcnn_r18_distilled': 'Дистилляция (предложенный)',
            'faster_rcnn_r18_modern_distilled': 'Современная (2025)',
        }
        
        # Настройка стиля
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })
    
    def create_all_visualizations(self):
        """Создаёт все графики."""
        
        logger.info("Creating visualizations...")
        logger.info(f"  Evaluation results: {len(self.evaluation_results)} models")
        for r in self.evaluation_results:
            logger.info(f"    - {r.get('model')}: type={r.get('type')}, mAP@50={r.get('mAP_50', 0):.4f}")
        logger.info(f"  Training histories: {len(self.training_histories)} models")
        for name, hist in self.training_histories.items():
            logger.info(f"    - {name}: {len(hist)} epochs")
        
        self.plot_map_comparison()
        self.plot_training_curves()
        self.plot_quality_vs_speed()
        self.plot_per_class_ap()
        self.plot_improvement_over_baseline()
        self.plot_prf_comparison()  # Замена confidence_distribution
        self.plot_radar_comparison()
        self.plot_convergence_speed()
        
        logger.info(f"All plots saved to {self.output_dir}")
    
    def plot_map_comparison(self):
        """Столбчатая диаграмма сравнения mAP@50."""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for mAP comparison")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sorted_results = sorted(
            self.evaluation_results,
            key=lambda x: x.get('mAP_50', 0)
        )
        
        models = [self.labels.get(r['model'], r['model']) for r in sorted_results]
        maps = [r.get('mAP_50', 0) for r in sorted_results]
        colors = [self.colors.get(r.get('type', ''), '#95A5A6') for r in sorted_results]
        
        bars = ax.barh(range(len(models)), maps, color=colors, edgecolor='black', linewidth=0.5)
        
        for i, (bar, val) in enumerate(zip(bars, maps)):
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}',
                va='center', fontsize=11, fontweight='bold'
            )
        
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel('mAP@50')
        ax.set_title('Сравнение качества детекции дефектов\nпри разных методах инициализации бэкбона')
        ax.set_xlim(0, max(maps) * 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['teacher'], label='Учитель (LTDETR)'),
            Patch(facecolor=self.colors['lightly_pretrained'], label='Дистилляция (предложенный)'),
            Patch(facecolor=self.colors['imagenet_pretrained'], label='ImageNet'),
            Patch(facecolor=self.colors['scratch'], label='Случайная иниц.'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_map_comparison.png')
        plt.close()
        logger.info("  ✓ mAP comparison saved")
    
    def plot_training_curves(self):
        """Кривые обучения: loss и mAP по эпохам."""
        
        if not self.training_histories:
            logger.warning("No training histories available")
            # Создаём информативный пустой график
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Нет данных об истории обучения\n(запустите 02_train_detectors.py)',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Training Curves (no data)')
            plt.tight_layout()
            plt.savefig(self.output_dir / '02_training_curves.png')
            plt.close()
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for model_name, history in self.training_histories.items():
            if not history:
                continue
            
            label = self.labels.get(model_name, model_name)
            color = self._get_model_color(model_name)
            
            epochs = [h['epoch'] for h in history]
            
            # Loss
            ax1 = axes[0]
            losses = [h.get('train_loss', 0) for h in history]
            ax1.plot(epochs, losses, color=color, label=label, linewidth=2, alpha=0.8)
            
            # mAP
            ax2 = axes[1]
            maps = [h.get('val_map50', 0) for h in history]
            ax2.plot(epochs, maps, color=color, label=label, linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Кривая обучения (Loss)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Validation mAP@50')
        axes[1].set_title('Качество детекции по эпохам')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_training_curves.png')
        plt.close()
        logger.info("  ✓ Training curves saved")
    
    def plot_quality_vs_speed(self):
        """Scatter plot: качество vs скорость."""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for quality vs speed")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for result in self.evaluation_results:
            model_name = result['model']
            fps = result.get('fps', 0)
            map50 = result.get('mAP_50', 0)
            params = result.get('params_millions', 0)
            
            color = self._get_model_color(model_name)
            label = self.labels.get(model_name, model_name)
            
            size = max(100, params * 3)
            
            ax.scatter(
                fps, map50,
                s=size, c=color, edgecolors='black', linewidth=1,
                zorder=5, label=label, alpha=0.8
            )
            
            offset = 15 if params > 50 else 10
            ax.annotate(
                f'{label}\n({params}M)', 
                (fps, map50),
                textcoords="offset points",
                xytext=(offset, offset),
                fontsize=8, ha='left'
            )
        
        ax.set_xlabel('FPS (CPU)')
        ax.set_ylabel('mAP@50')
        ax.set_title('Качество vs Скорость\nРазмер маркера ∝ количество параметров')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_quality_vs_speed.png')
        plt.close()
        logger.info("  ✓ Quality vs Speed saved")
    
    def plot_per_class_ap(self):
        """Per-class AP сравнение."""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for per-class AP")
            return
        
        num_classes = self.config['detection']['num_classes']
        class_names = self.config['detection']['class_names']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(num_classes)
        width = 0.2
        
        for i, result in enumerate(self.evaluation_results):
            model_name = result['model']
            color = self._get_model_color(model_name)
            label = self.labels.get(model_name, model_name)
            
            ap_values = []
            for cls in range(num_classes):
                ap = result.get(f'cls{cls}_AP50', 0)
                ap_values.append(ap)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, ap_values, width, label=label, color=color, edgecolor='black', linewidth=0.5)
            
            for bar, val in zip(bars, ap_values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90
                    )
        
        ax.set_xlabel('Класс дефекта')
        ax.set_ylabel('AP@50')
        ax.set_title('Per-class Average Precision')
        ax.set_xticks(x)
        ax.set_xticklabels([class_names.get(i, f'Class {i}') for i in range(num_classes)])
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_per_class_ap.png')
        plt.close()
        logger.info("  ✓ Per-class AP saved")
    
    def plot_improvement_over_baseline(self):
        """Прирост относительно scratch baseline."""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for improvement")
            return
        
        scratch = next(
            (r for r in self.evaluation_results if r.get('type') == 'scratch'),
            None
        )
        
        if not scratch:
            logger.warning("No scratch baseline found")
            return
        
        baseline_map = scratch.get('mAP_50', 0)
        
        other_models = [r for r in self.evaluation_results if r.get('type') != 'scratch']
        
        if not other_models:
            logger.warning("No other models to compare with baseline")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = [self.labels.get(r['model'], r['model']) for r in other_models]
        improvements = [r.get('mAP_50', 0) - baseline_map for r in other_models]
        improvements_pct = [(imp / baseline_map) * 100 for imp in improvements]
        colors = [self._get_model_color(r['model']) for r in other_models]
        
        bars = ax.bar(range(len(models)), improvements, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, imp, imp_pct in zip(bars, improvements, improvements_pct):
            y_pos = bar.get_height() + 0.005 if bar.get_height() > 0 else bar.get_height() - 0.015
            ax.text(
                bar.get_x() + bar.get_width()/2, y_pos,
                f'+{imp:.4f}\n(+{imp_pct:.1f}%)',
                ha='center', fontsize=11, fontweight='bold'
            )
        
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('Δ mAP@50')
        ax.set_title(f'Прирост качества относительно случайной инициализации\n(базовый mAP@50 = {baseline_map:.4f})')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_improvement_over_baseline.png')
        plt.close()
        logger.info("  ✓ Improvement over baseline saved")
    
    def plot_prf_comparison(self):
        """Precision, Recall, F1 сравнение (замена confidence distribution)."""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for PRF comparison")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = [self.labels.get(r['model'], r['model']) for r in self.evaluation_results]
        precision = [r.get('Precision', 0) for r in self.evaluation_results]
        recall = [r.get('Recall', 0) for r in self.evaluation_results]
        f1 = [r.get('F1', 0) for r in self.evaluation_results]
        
        x = range(len(models))
        width = 0.25
        
        ax.bar([i - width for i in x], precision, width, label='Precision', color='#3498DB', edgecolor='black')
        ax.bar(x, recall, width, label='Recall', color='#2ECC71', edgecolor='black')
        ax.bar([i + width for i in x], f1, width, label='F1-score', color='#E74C3C', edgecolor='black')
        
        # Добавляем значения
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', fontsize=8)
            ax.text(i, r + 0.01, f'{r:.3f}', ha='center', fontsize=8)
            ax.text(i + width, f + 0.01, f'{f:.3f}', ha='center', fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall и F1-score')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_prf_comparison.png')
        plt.close()
        logger.info("  ✓ PRF comparison saved")
    
    def plot_radar_comparison(self):
        """Радарная диаграмма сравнения методов."""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results for radar")
            return
        
        metrics = ['mAP_50', 'mAP_75', 'F1', 'fps', 'params_millions']
        metric_labels = ['mAP@50', 'mAP@75', 'F1-score', 'FPS', 'Params (M)']
        
        max_values = {}
        min_values = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in self.evaluation_results if metric in r]
            if values:
                max_values[metric] = max(values)
                min_values[metric] = min(values)
        
        if not max_values:
            logger.warning("No metrics for radar plot")
            return
        
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for result in self.evaluation_results:
            model_name = result['model']
            color = self._get_model_color(model_name)
            label = self.labels.get(model_name, model_name)
            
            values = []
            for metric in metrics:
                val = result.get(metric, 0)
                if metric == 'params_millions':
                    if max_values[metric] > min_values[metric]:
                        val = 1 - (val - min_values[metric]) / (max_values[metric] - min_values[metric])
                    else:
                        val = 0.5
                else:
                    if max_values[metric] > min_values[metric]:
                        val = (val - min_values[metric]) / (max_values[metric] - min_values[metric])
                    else:
                        val = 0.5
                values.append(val)
            
            values += values[:1]
            ax.fill(angles, values, alpha=0.15, color=color)
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, markersize=6)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_title('Многокритериальное сравнение\n(1.0 = лучшее значение)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_radar_comparison.png')
        plt.close()
        logger.info("  ✓ Radar comparison saved")
    
    def plot_convergence_speed(self):
        """Скорость сходимости."""
        
        if not self.training_histories:
            logger.warning("No training histories for convergence speed")
            # Создаём информативный пустой график
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Нет данных об истории обучения\n(запустите 02_train_detectors.py)',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Convergence Speed (no data)')
            plt.tight_layout()
            plt.savefig(self.output_dir / '08_convergence_speed.png')
            plt.close()
            return
        
        target_maps = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        has_data = False
        
        for model_name, history in self.training_histories.items():
            if not history:
                continue
            
            label = self.labels.get(model_name, model_name)
            color = self._get_model_color(model_name)
            
            epochs_to_target = []
            for target in target_maps:
                reached = False
                for h in history:
                    if h.get('val_map50', 0) >= target:
                        epochs_to_target.append(h['epoch'])
                        reached = True
                        break
                if not reached:
                    epochs_to_target.append(float('nan'))
            
            valid_targets = [t for t, e in zip(target_maps, epochs_to_target) if not np.isnan(e)]
            valid_epochs = [e for e in epochs_to_target if not np.isnan(e)]
            
            if valid_epochs:
                ax.plot(valid_targets, valid_epochs, 'o-', color=color, label=label, linewidth=2, markersize=8)
                has_data = True
        
        if has_data:
            ax.set_xlabel('Целевой mAP@50')
            ax.set_ylabel('Эпох до достижения')
            ax.set_title('Скорость сходимости')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Модели не достигли целевых mAP\n(недостаточно эпох обучения)',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '08_convergence_speed.png')
        plt.close()
        logger.info("  ✓ Convergence speed saved")
    
    def _get_model_color(self, model_name: str) -> str:
        """Возвращает цвет для модели."""
        for result in self.evaluation_results:
            if result['model'] == model_name:
                return self.colors.get(result.get('type', ''), '#95A5A6')
        # Если не нашли в evaluation_results, ищем в конфиге
        for student_name, student_cfg in self.config.get('students', {}).items():
            if student_name == model_name:
                return self.colors.get(student_cfg.get('type', ''), '#95A5A6')
        return '#95A5A6'


def load_training_histories(config: dict) -> Dict[str, List[Dict]]:
    """Загружает истории обучения из JSON файлов."""
    
    histories = {}
    detection_dir = Path(config['paths']['detection_output'])
    
    if not detection_dir.exists():
        logger.warning(f"Detection output directory not found: {detection_dir}")
        return histories
    
    for student_name in config['students'].keys():
        results_file = detection_dir / student_name / 'training_results.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    histories[student_name] = data.get('history', [])
                    logger.info(f"Loaded history for {student_name}: {len(histories[student_name])} epochs")
            except Exception as e:
                logger.warning(f"Failed to load history for {student_name}: {e}")
        else:
            logger.warning(f"No training history found for {student_name} at {results_file}")
    
    return histories


def main():
    """Создаёт все визуализации."""
    
    config_path = Path(__file__).parent / "../config_modern_distillation.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['paths']['results_dir'])
    
    # Загружаем результаты оценки
    eval_file = results_dir / 'evaluation_results.json'
    
    if not eval_file.exists():
        logger.error(f"❌ Evaluation results not found: {eval_file}")
        logger.error("Run 03_evaluate_all_models.py first")
        sys.exit(1)
    
    with open(eval_file, 'r') as f:
        evaluation_results = json.load(f)
    
    logger.info(f"Loaded {len(evaluation_results)} evaluation results")
    
    # Загружаем истории обучения
    training_histories = load_training_histories(config)
    
    # Создаём визуализации
    visualizer = ResultsVisualizer(config, evaluation_results, training_histories)
    visualizer.create_all_visualizations()
    
    logger.info(f"\n✅ All visualizations saved to {results_dir}")


if __name__ == "__main__":
    main()