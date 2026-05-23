#!/usr/bin/env python3
"""
Учитель 2: real_augmented с полным finetune (без SSL)
Стратегия: LTDETR с размороженным бэкбоном
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.scripts.train_ltdetr import train_ltdetr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEACHER_NAME = "teacher_real_augmented_finetune"


def _prepare_mlflow_metrics(result: dict) -> dict:
    """Собирает ВСЕ метрики для MLflow."""
    metrics = {
        'test_map50': result.get('test_map50', 0),
        'test_map75': result.get('test_map75', 0),
        'test_map50_95': result.get('test_map50_95', 0),
        'val_map50': result.get('val_map50', 0),
        'training_time_hours': result.get('training_time_hours', 0),
        'n_epochs': result.get('n_epochs', 0),
        'n_images': result.get('n_images', 0),
    }
    for k, v in result.items():
        if k.startswith('test_cls'):
            metrics[k] = v
    return metrics


def main():
    config_path = Path(__file__).parent / "config_teacher.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    models_dir = Path(config['paths']['models_dir']) / TEACHER_NAME
    report_dir = Path(config['paths']['report_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    teacher_cfg = config['teacher_real_augmented_finetune']
    seed = config['seeds'][0]

    with mlflow.start_run(run_name=TEACHER_NAME):
        mlflow.log_param("teacher", TEACHER_NAME)
        mlflow.log_param("dataset", teacher_cfg['dataset'])
        mlflow.log_param("strategy", "finetune")
        mlflow.log_param("ssl_enabled", False)
        mlflow.log_param("model", config['training']['model'])
        mlflow.log_param("epochs", config['training']['fixed_epochs'])
        mlflow.log_param("seed", seed)

        try:
            # LTDETR с размороженным бэкбоном (без SSL)
            logger.info(f"\n{'='*60}")
            logger.info(f"LT-DETR training для {teacher_cfg['dataset']} (finetune)")
            logger.info(f"{'='*60}")

            run_cfg = {
                'run_name': TEACHER_NAME,
                'dataset_name': teacher_cfg['dataset'],
                'data_yaml': teacher_cfg['data_yaml'],
                'strategy_name': 'finetune',
                'freeze_backbone': False,
                'seed': seed,
            }

            result = train_ltdetr(config, run_cfg, models_dir)

            mlflow.log_metrics(_prepare_mlflow_metrics(result))
            mlflow.set_tag("status", "completed")

            # Сохраняем отчет
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = report_dir / f"{TEACHER_NAME}_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(str(report_path))

            logger.info(f"\n{'='*60}")
            logger.info(f"✅ {TEACHER_NAME}: mAP50={result.get('test_map50', 0):.4f}")
            logger.info(f"   mAP50-95={result.get('test_map50_95', 0):.4f}")
            logger.info(f"   Время: {result.get('training_time_hours', 0):.2f} ч")
            logger.info(f"   Модель: {result.get('model_path', 'N/A')}")
            logger.info(f"   Отчет: {report_path}")
            logger.info(f"{'='*60}")

        except Exception as e:
            logger.error(f"❌ Ошибка в {TEACHER_NAME}: {e}", exc_info=True)
            mlflow.set_tag("status", "failed")
            raise


if __name__ == "__main__":
    main()