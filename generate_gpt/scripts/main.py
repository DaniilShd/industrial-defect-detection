#!/usr/bin/env python3
"""Точка входа — генерация синтетического датасета"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, update_config_from_args
from scripts.generate_dataset import PoissonDefectGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SD фон + SD дефект + Масштабирование + Color Correction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к YAML конфигу")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--rle_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--variants", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sd_strength", type=float, default=None)
    args = parser.parse_args()
    
    # Загрузка конфига
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    # Переопределение путей из аргументов
    if args.input_dir:
        config.paths.input_dir = args.input_dir
    if args.rle_csv:
        config.paths.rle_csv = args.rle_csv
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    
    logger.info(f"Input: {config.paths.input_dir}")
    logger.info(f"RLE: {config.paths.rle_csv}")
    logger.info(f"Output: {config.paths.output_dir}")
    
    generator = PoissonDefectGenerator(config)
    total = generator.generate_dataset(
        Path(config.paths.input_dir),
        Path(config.paths.rle_csv),
        Path(config.paths.output_dir),
        config.generation.variants,
        config.generation.limit
    )
    
    logger.info(f"Готово! Сгенерировано {total} изображений")


if __name__ == "__main__":
    main()