"""Анализ распределения классов в синтетических данных"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import seaborn as sns
from typing import Dict, List, Tuple, Set
import json

from config import AnalysisConfig

# Настройка русского шрифта
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ClassDistributionAnalyzer:
    """Анализатор распределения классов в синтетических данных"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.class_names = config.class_analysis.class_names
        self.num_classes = config.class_analysis.num_classes
        self.results = {
            "summary": {},
            "class_stats": {},
            "bbox_statistics": {},
            "visualization": {}
        }
        
    def parse_yolo_label(self, label_path: Path) -> Tuple[List[int], int]:
        """Разбор YOLO формата разметки"""
        if not label_path.exists():
            return [], 0
        
        classes = []
        num_bboxes = 0
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    classes.append(class_id)
                    num_bboxes += 1
        
        return classes, num_bboxes
    
    def analyze(self, synthetic_dir: Path, output_dir: Path) -> Dict:
        """Полный анализ распределения классов"""
        labels_path = synthetic_dir / "labels"
        images_path = synthetic_dir / "images"
        
        print("=" * 80)
        print("📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
        print("=" * 80)
        
        # Поиск файлов
        all_label_files = sorted(list(labels_path.glob("*.txt")))
        print(f"\n📁 Найдено файлов разметки: {len(all_label_files)}")
        
        image_files = set()
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.update(f.stem for f in images_path.glob(ext))
        
        print(f"📁 Найдено изображений: {len(image_files)}")
        
        # Сбор статистики
        class_counter = Counter()
        images_per_class = defaultdict(set)
        images_with_multiple_classes = set()
        bboxes_per_image = []
        classes_per_image = []
        empty_images = []
        bbox_sizes = []
        class_areas = defaultdict(list)
        
        for label_file in all_label_files:
            classes, num_bboxes = self.parse_yolo_label(label_file)
            image_name = label_file.stem
            
            if num_bboxes == 0:
                empty_images.append(image_name)
            else:
                bboxes_per_image.append(num_bboxes)
                classes_per_image.append(len(set(classes)))
                
                if len(set(classes)) > 1:
                    images_with_multiple_classes.add(image_name)
                
                for cls in classes:
                    class_counter[cls] += 1
                    images_per_class[cls].add(image_name)
                
                if self.config.class_analysis.compute_bbox_statistics:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                w, h = float(parts[3]), float(parts[4])
                                area = w * h
                                bbox_sizes.append(area)
                                class_areas[cls_id].append(area)
        
        # Сводная статистика
        total_bboxes = sum(class_counter.values())
        total_images_with_defects = len(all_label_files) - len(empty_images)
        
        self.results['summary'] = {
            "total_images": len(image_files),
            "total_labels": len(all_label_files),
            "images_with_defects": total_images_with_defects,
            "empty_images": len(empty_images),
            "total_bboxes": total_bboxes,
            "avg_bboxes_per_image": total_bboxes / max(total_images_with_defects, 1),
            "images_with_multiple_classes": len(images_with_multiple_classes)
        }
        
        # Статистика по классам
        print("\n" + "-" * 80)
        print("РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
        print("-" * 80)
        print(f"{'Класс':<25} {'Рамок':<10} {'%':<10} {'Изображений':<12} {'Рамок/Изобр.':<15}")
        print("-" * 80)
        
        for class_id in range(self.num_classes):
            count = class_counter[class_id]
            pct = (count / total_bboxes * 100) if total_bboxes > 0 else 0
            images = len(images_per_class[class_id])
            avg_per_image = count / max(images, 1)
            
            class_name = self.class_names.get(class_id, f"класс_{class_id}")
            print(f"{class_name:<25} {count:<10} {pct:<10.1f}% {images:<12} {avg_per_image:<15.2f}")
            
            self.results['class_stats'][class_name] = {
                "class_id": class_id,
                "bbox_count": count,
                "percentage": round(pct, 2),
                "image_count": images,
                "avg_bboxes_per_image": round(avg_per_image, 2)
            }
        
        # Статистика размеров рамок
        if bbox_sizes:
            bbox_sizes = np.array(bbox_sizes)
            self.results['bbox_statistics'] = {
                "mean": float(np.mean(bbox_sizes)),
                "std": float(np.std(bbox_sizes)),
                "min": float(np.min(bbox_sizes)),
                "max": float(np.max(bbox_sizes)),
                "median": float(np.median(bbox_sizes)),
                "by_class": {}
            }
            
            for class_id in range(self.num_classes):
                if class_id in class_areas:
                    areas = np.array(class_areas[class_id])
                    self.results['bbox_statistics']['by_class'][self.class_names[class_id]] = {
                        "mean": float(np.mean(areas)),
                        "std": float(np.std(areas)),
                        "median": float(np.median(areas))
                    }
        
        # Визуализации
        self._create_visualizations(output_dir, class_counter, images_per_class, 
                                    bboxes_per_image, classes_per_image, 
                                    images_with_multiple_classes, total_images_with_defects)
        
        # Сохранение JSON
        json_path = output_dir / "результаты_распределения_классов.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, set):
                    return list(obj)
                return obj
            
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=convert)
        
        print(f"\n✅ Результаты сохранены: {json_path}")
        
        return self.results
    
    def _create_visualizations(self, output_dir: Path, class_counter: Counter,
                              images_per_class: Dict, bboxes_per_image: List,
                              classes_per_image: List, images_with_multiple_classes: Set,
                              total_images_with_defects: int):
        """Создание визуализаций распределения классов"""
        viz_config = self.config.visualization
        viz_dir = output_dir / self.config.paths.subdirs.get('visualizations', 'visualizations')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        classes = list(range(self.num_classes))
        counts = [class_counter[i] for i in classes]
        img_counts = [len(images_per_class[i]) for i in classes]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F9ED69', '#F08A5D']
        colors = colors[:self.num_classes]
        
        # 1. Основное распределение
        fig, axes = plt.subplots(2, 2, figsize=viz_config.figsize)
        
        # Количество рамок по классам
        ax1 = axes[0, 0]
        bars = ax1.bar(classes, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Класс', fontweight='bold')
        ax1.set_ylabel('Количество рамок', fontweight='bold')
        ax1.set_title('Распределение рамок по классам', fontweight='bold')
        ax1.set_xticks(classes)
        ax1.set_xticklabels([self.class_names[i] for i in classes], rotation=45, ha='right')
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Изображений на класс
        ax2 = axes[0, 1]
        bars2 = ax2.bar(classes, img_counts, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Класс', fontweight='bold')
        ax2.set_ylabel('Количество изображений', fontweight='bold')
        ax2.set_title('Изображения, содержащие каждый класс', fontweight='bold')
        ax2.set_xticks(classes)
        ax2.set_xticklabels([self.class_names[i] for i in classes], rotation=45, ha='right')
        
        for bar, count in zip(bars2, img_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(img_counts)*0.02,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Гистограмма рамок на изображение
        ax3 = axes[1, 0]
        if bboxes_per_image:
            max_bboxes = max(bboxes_per_image)
            ax3.hist(bboxes_per_image, bins=range(1, max_bboxes+2), 
                    color='#3498db', alpha=0.8, edgecolor='black', align='left')
            ax3.set_xlabel('Рамок на изображение', fontweight='bold')
            ax3.set_ylabel('Частота', fontweight='bold')
            ax3.set_title('Распределение количества рамок на изображение', fontweight='bold')
            ax3.set_xticks(range(1, max_bboxes+1))
        
        # Гистограмма классов на изображение
        ax4 = axes[1, 1]
        if classes_per_image:
            max_classes = max(classes_per_image)
            hist_data = [classes_per_image.count(i) for i in range(1, max_classes+1)]
            bars4 = ax4.bar(range(1, max_classes+1), hist_data, 
                          color='#e74c3c', alpha=0.85, edgecolor='black')
            ax4.set_xlabel('Классов на изображение', fontweight='bold')
            ax4.set_ylabel('Частота', fontweight='bold')
            ax4.set_title('Распределение количества классов на изображение', fontweight='bold')
            ax4.set_xticks(range(1, max_classes+1))
            
            for bar, count in zip(bars4, hist_data):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hist_data)*0.02,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Анализ распределения классов синтетических данных', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        for fmt in viz_config.save_formats:
            plt.savefig(viz_dir / f'распределение_классов.{fmt}', 
                       dpi=viz_config.dpi, bbox_inches='tight')
        plt.close()
        
        # 2. Круговая диаграмма
        if sum(counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, texts, autotexts = ax.pie(
                counts, 
                labels=[self.class_names[i] for i in classes],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            ax.set_title('Распределение классов (по количеству рамок)', fontsize=14, fontweight='bold')
            
            for fmt in viz_config.save_formats:
                plt.savefig(viz_dir / f'круговая_диаграмма_классов.{fmt}',
                          dpi=viz_config.dpi, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Визуализации сохранены в: {viz_dir}")


def run_class_analysis(config: AnalysisConfig) -> Dict:
    """Запуск анализа распределения классов"""
    analyzer = ClassDistributionAnalyzer(config)
    
    results = analyzer.analyze(
        synthetic_dir=config.paths.synthetic_dir,
        output_dir=config.paths.output_dir
    )
    
    return results