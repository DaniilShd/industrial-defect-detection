#!/usr/bin/env python3
"""CLI для визуализации YOLO bbox. Цвета из config.yaml."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from utils import load_config
from utils.visualization_utils import get_class_colors, visualize_batch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    p = cfg['paths']
    
    parser = argparse.ArgumentParser(description="Визуализация YOLO bbox")
    parser.add_argument("--images", type=str, default=None, help="Путь к изображениям")
    parser.add_argument("--labels", type=str, default=None, help="Путь к разметке")
    parser.add_argument("--output", type=str, default=None, help="Выходная директория")
    parser.add_argument("--samples", type=int, default=20, help="Количество примеров (0 = все)")
    parser.add_argument("--show", action="store_true", help="Показать изображения")
    args = parser.parse_args()
    
    # По умолчанию — дефектные патчи
    images_dir = Path(args.images) if args.images else Path(p['defect_patches_dir']) / p['yolo_images_subdir']
    labels_dir = Path(args.labels) if args.labels else Path(p['defect_patches_dir']) / p['yolo_labels_subdir']
    output_dir = Path(args.output) if args.output else Path(p['reports_dir']) / "visualized_bboxes"
    
    # Цвета и имена из конфига
    class_colors = get_class_colors(cfg['classes'])
    class_names = {int(k) - 1: v for k, v in cfg['classes']['names'].items()}
    
    visualize_batch(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        num_samples=args.samples,
        class_names=class_names,
        class_colors=class_colors,
        show=args.show
    )


if __name__ == "__main__":
    main()