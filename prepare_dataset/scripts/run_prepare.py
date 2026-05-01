#!/usr/bin/env python3
"""
run_prepare.py — Оркестратор подготовки данных.
Запускает все этапы последовательно:
  1. Копирование реальных данных (с ресайзом 640x640)
  2. Копирование синтетики (с ресайзом 640x640)
  3. Аугментация синтетики
  4. Аугментация реальных данных
  5. Сборка финальных датасетов
  6. Финальная валидация и ресайз
"""

import logging
import sys
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import (
    copy_real,
    copy_synthetic,
    augment_synthetic,
    augment_real,
    merge_datasets,
    validate_resize,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    logger.info("=" * 60)
    logger.info("🚀 ПОДГОТОВКА ДАННЫХ ДЛЯ ЭКСПЕРИМЕНТА")
    logger.info("=" * 60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # MLflow
    mlflow_cfg = config['mlflow']
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg['experiment_name'])
    
    with mlflow.start_run(run_name=config['experiment']['name']):
        mlflow.log_dict(config, "config.yaml")
        
        # ── Этап 1: Копирование реальных данных ──
        logger.info("\n" + "─" * 40)
        logger.info("📦 ЭТАП 1/6: Копирование реальных данных (640x640)")
        real_paths = copy_real.copy_real_dataset(config)
        mlflow.log_dict(real_paths, "real_paths.yaml")
        
        # ── Этап 2: Копирование синтетики ──
        logger.info("\n" + "─" * 40)
        logger.info("📦 ЭТАП 2/6: Копирование синтетических данных (640x640)")
        try:
            synth_path = copy_synthetic.copy_synthetic_dataset(config)
            mlflow.log_param("synthetic_path", str(synth_path))
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            logger.error("Укажите путь к лучшей синтетике в config.yaml → paths.best_synthetic")
            raise
        
        # ── Этап 3: Аугментация синтетики ──
        logger.info("\n" + "─" * 40)
        logger.info("🎨 ЭТАП 3/6: Аугментация синтетических данных")
        synth_aug_path = augment_synthetic.augment_synthetic_dataset(config)
        mlflow.log_param("synth_augmented_path", str(synth_aug_path) if synth_aug_path else "None")
        
        # ── Этап 4: Аугментация реальных данных ──
        logger.info("\n" + "─" * 40)
        logger.info("🎨 ЭТАП 4/6: Аугментация реальных данных")
        real_aug_path = augment_real.augment_real_dataset(config)
        mlflow.log_param("real_augmented_path", str(real_aug_path) if real_aug_path else "None")
        
        # ── Этап 5: Сборка финальных датасетов ──
        logger.info("\n" + "─" * 40)
        logger.info("📦 ЭТАП 5/6: Сборка финальных датасетов")
        datasets = merge_datasets.merge_all_datasets(config)
        mlflow.log_dict({k: str(v) for k, v in datasets.items()}, "datasets.yaml")
        
        # ── Этап 6: Финальная валидация и ресайз ──
        logger.info("\n" + "─" * 40)
        logger.info("🔍 ЭТАП 6/6: Финальная проверка и ресайз")
        stats = validate_resize.validate_and_fix_all_datasets(config)
        mlflow.log_dict(stats, "validation_stats.yaml")
        
        # Считаем итоговые метрики
        total_images = sum(s['total_images'] for s in stats.values())
        total_resized = sum(s['resized'] for s in stats.values())
        total_fixed = sum(s['fixed_labels'] for s in stats.values())
        total_errors = sum(s['errors'] for s in stats.values())
        all_valid = all(s.get('yolo_valid', False) for s in stats.values())
        
        # Итоги
        output_dir = Path(config['paths']['output_dir'])
        total_size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e9
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
        logger.info(f"   Датасетов:              {len(datasets)}")
        logger.info(f"   Всего изображений:      {total_images}")
        logger.info(f"   Ресайзнуто:              {total_resized}")
        logger.info(f"   Исправлено лейблов:      {total_fixed}")
        logger.info(f"   Ошибок:                  {total_errors}")
        logger.info(f"   Все валидны (YOLO):      {all_valid}")
        logger.info(f"   Общий размер:            {total_size_gb:.2f} ГБ")
        logger.info(f"   Данные:                  {output_dir}")
        logger.info("=" * 60)
        
        # Сводка по датасетам
        logger.info("\n📊 Собранные датасеты:")
        datasets_cfg = config['datasets']
        for name in datasets:
            cfg = datasets_cfg[name]
            logger.info(f"   {name} ({cfg['size']}): {cfg['description']}")
        
        mlflow.log_metrics({
            "num_datasets": len(datasets),
            "total_images": total_images,
            "total_resized": total_resized,
            "total_fixed_labels": total_fixed,
            "total_errors": total_errors,
            "total_size_gb": round(total_size_gb, 2),
            "all_valid": int(all_valid)
        })


if __name__ == "__main__":
    main()