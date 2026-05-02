#!/usr/bin/env python3
"""Визуализация синтетических изображений: side-by-side, bbox, grid"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import AnalysisConfig
from utils.io_utils import read_yolo_labels

logger = logging.getLogger(__name__)


class SyntheticVisualizer:
    """Визуализатор синтетических данных"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.colors = {
            0: (0, 255, 0),      # green
            1: (255, 0, 0),      # blue
            2: (0, 0, 255),      # red
            3: (255, 255, 0)     # cyan
        }
    
    def draw_bboxes(self, img: np.ndarray, label_path: Path) -> np.ndarray:
        """Отрисовка YOLO bbox на изображении"""
        if not label_path.exists():
            return img
        
        h, w = img.shape[:2]
        label_data = read_yolo_labels(label_path)
        
        for cls, bbox in zip(label_data['classes'], label_data['bboxes']):
            xc, yc, bw, bh = bbox
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            
            color = self.colors.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Class {cls}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
    
    def create_side_by_side(self, original_path: Path, synthetic_path: Path,
                           orig_label: Path, synth_label: Path) -> np.ndarray:
        """Создание side-by-side сравнения оригинал vs синтетика"""
        orig = cv2.imread(str(original_path))
        synth = cv2.imread(str(synthetic_path))
        
        if orig is None or synth is None:
            return None
        
        # Ресайз до одинакового размера
        target_size = 640
        orig = cv2.resize(orig, (target_size, target_size))
        synth = cv2.resize(synth, (target_size, target_size))
        
        # Отрисовка bbox
        orig = self.draw_bboxes(orig, orig_label)
        synth = self.draw_bboxes(synth, synth_label)
        
        # Разделительная линия
        h, w = orig.shape[:2]
        result = np.ones((h + 40, w * 2 + 10, 3), dtype=np.uint8) * 255
        
        result[20:20+h, 0:w] = orig
        result[20:20+h, w+10:w*2+10] = synth
        
        # Подписи
        cv2.putText(result, "ORIGINAL", (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(result, "SYNTHETIC", (w + 20, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result
    
    def create_grid(self, image_paths: List[Path], labels_dir: Path,
                   rows: int = 4, cols: int = 5) -> np.ndarray:
        """Создание сетки изображений с bbox"""
        target_size = 640
        grid_h = rows * target_size + (rows + 1) * 10
        grid_w = cols * target_size + (cols + 1) * 10
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240
        
        for idx, img_path in enumerate(image_paths):
            if idx >= rows * cols:
                break
            
            row = idx // cols
            col = idx % cols
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.resize(img, (target_size, target_size))
            
            label_path = labels_dir / f"{img_path.stem}.txt"
            img = self.draw_bboxes(img, label_path)
            
            y1 = 10 + row * (target_size + 10)
            x1 = 10 + col * (target_size + 10)
            grid[y1:y1+target_size, x1:x1+target_size] = img
        
        return grid
    
    def create_defect_comparison(self, original_dir: Path, synthetic_dir: Path,
                                output_dir: Path, num_samples: int = 20):
        """Создание сравнений дефектов"""
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Поиск изображений
        orig_images = sorted(list((original_dir / "images").glob("*.png")) +
                           list((original_dir / "images").glob("*.jpg")))
        synth_images = sorted(list((synthetic_dir / "images").glob("*.png")) +
                            list((synthetic_dir / "images").glob("*.jpg")))
        
        if not orig_images or not synth_images:
            logger.warning("No images found for visualization")
            return
        
        # Side-by-side сравнения
        logger.info("Creating side-by-side comparisons...")
        
        comparisons_dir = viz_dir / "comparisons"
        comparisons_dir.mkdir(exist_ok=True)
        
        # Подбор пар
        orig_by_stem = {img.stem: img for img in orig_images}
        
        pairs = []
        for synth_img in synth_images[:num_samples]:
            # Ищем оригинал с похожим именем
            synth_stem = synth_img.stem.replace('syn_', '').split('_v')[0]
            for orig_stem, orig_img in orig_by_stem.items():
                if synth_stem in orig_stem or orig_stem in synth_stem:
                    pairs.append((orig_img, synth_img))
                    break
        
        for i, (orig, synth) in enumerate(pairs[:num_samples]):
            orig_label = original_dir / "labels" / f"{orig.stem}.txt"
            synth_label = synthetic_dir / "labels" / f"{synth.stem}.txt"
            
            comparison = self.create_side_by_side(orig, synth, orig_label, synth_label)
            if comparison is not None:
                cv2.imwrite(str(comparisons_dir / f"comparison_{i:03d}.png"), comparison)
        
        logger.info(f"Created {len(pairs[:num_samples])} comparisons")
        
        # Grid визуализации
        logger.info("Creating grid visualizations...")
        
        # Original grid
        orig_grid = self.create_grid(orig_images[:20], original_dir / "labels")
        cv2.imwrite(str(viz_dir / "original_grid.png"), orig_grid)
        
        # Synthetic grid
        synth_grid = self.create_grid(synth_images[:20], synthetic_dir / "labels")
        cv2.imwrite(str(viz_dir / "synthetic_grid.png"), synth_grid)
        
        logger.info("Grid visualizations created")
    
    def create_difference_maps(self, original_dir: Path, synthetic_dir: Path,
                              output_dir: Path, num_samples: int = 10):
        """Создание карт различий между оригиналом и синтетикой"""
        diff_dir = output_dir / "visualizations" / "difference_maps"
        diff_dir.mkdir(parents=True, exist_ok=True)
        
        orig_images = {}
        for ext in ['*.png', '*.jpg']:
            for img in (original_dir / "images").glob(ext):
                orig_images[img.stem] = img
        
        synth_images = list((synthetic_dir / "images").glob("*.png")) + \
                      list((synthetic_dir / "images").glob("*.jpg"))
        
        pairs = []
        for synth_img in synth_images:
            synth_stem = synth_img.stem.replace('syn_', '').split('_v')[0]
            for orig_stem, orig_img in orig_images.items():
                if synth_stem in orig_stem or orig_stem in synth_stem:
                    pairs.append((orig_img, synth_img))
                    break
        
        logger.info("Creating difference maps...")
        
        for i, (orig_path, synth_path) in enumerate(pairs[:num_samples]):
            orig = cv2.imread(str(orig_path))
            synth = cv2.imread(str(synth_path))
            
            if orig is None or synth is None:
                continue
            
            # Ресайз
            target_size = 640
            orig = cv2.resize(orig, (target_size, target_size))
            synth = cv2.resize(synth, (target_size, target_size))
            
            # Разница
            diff = cv2.absdiff(orig, synth)
            diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            
            # Тепловая карта
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            
            # Композиция: оригинал | синтетика | разница | тепловая карта
            h, w = orig.shape[:2]
            composite = np.ones((h + 40, w * 4 + 30, 3), dtype=np.uint8) * 255
            
            composite[20:20+h, 0:w] = orig
            composite[20:20+h, w+10:w*2+10] = synth
            composite[20:20+h, w*2+20:w*3+20] = diff_enhanced
            composite[20:20+h, w*3+30:w*4+30] = diff_heatmap
            
            labels = ["ORIGINAL", "SYNTHETIC", "DIFFERENCE", "HEATMAP"]
            for j, label in enumerate(labels):
                cv2.putText(composite, label, (10 + j*(w+10), 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.imwrite(str(diff_dir / f"diff_{i:03d}.png"), composite)
        
        logger.info(f"Created {len(pairs[:num_samples])} difference maps")
    
    def create_class_samples_grid(self, synthetic_dir: Path, output_dir: Path):
        """Создание сеток примеров для каждого класса"""
        class_dir = output_dir / "visualizations" / "by_class"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        labels_dir = synthetic_dir / "labels"
        images_dir = synthetic_dir / "images"
        
        # Группируем изображения по классам
        class_images = {i: [] for i in range(4)}
        
        for label_file in labels_dir.glob("*.txt"):
            label_data = read_yolo_labels(label_file)
            classes_in_image = set(label_data['classes'])
            
            for cls in classes_in_image:
                img_path = images_dir / f"{label_file.stem}.png"
                if not img_path.exists():
                    img_path = images_dir / f"{label_file.stem}.jpg"
                
                if img_path.exists():
                    class_images[cls].append(img_path)
        
        # Создаем сетки
        for cls, img_paths in class_images.items():
            if not img_paths:
                continue
            
            samples = random.sample(img_paths, min(16, len(img_paths)))
            grid = self.create_grid(samples, labels_dir, rows=4, cols=4)
            cv2.imwrite(str(class_dir / f"class_{cls}_samples.png"), grid)
        
        logger.info("Class sample grids created")
    
    def analyze(self, original_dir: Path, synthetic_dir: Path, output_dir: Path):
        """Запуск всех визуализаций"""
        logger.info("=" * 80)
        logger.info("📊 SYNTHETIC DATA VISUALIZATION")
        logger.info("=" * 80)
        
        num_samples = self.config.visualization.random_samples
        
        # Side-by-side сравнения
        self.create_defect_comparison(original_dir, synthetic_dir, 
                                      output_dir, num_samples)
        
        # Карты различий
        self.create_difference_maps(original_dir, synthetic_dir, 
                                    output_dir, min(num_samples, 10))
        
        # Сетки по классам
        self.create_class_samples_grid(synthetic_dir, output_dir)
        
        logger.info("Visualization complete")


def run_visualization(config: AnalysisConfig) -> None:
    """Запуск визуализации"""
    visualizer = SyntheticVisualizer(config)
    
    visualizer.analyze(
        original_dir=config.paths.original_dir,
        synthetic_dir=config.paths.synthetic_dir,
        output_dir=config.paths.output_dir
    )