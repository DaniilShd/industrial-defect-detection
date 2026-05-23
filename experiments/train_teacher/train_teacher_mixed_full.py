#!/usr/bin/env python3
"""
Учитель 1: mixed_full с SSL дообучением бэкбона
Стратегия: distillation → LTDETR с размороженным SSL-бэкбоном
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.scripts.train_ltdetr import train_ltdetr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEACHER_NAME = "teacher_mixed_full_ssl"


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


def _get_backbone_from_ltdetr(full_model: str) -> str:
    """Извлекает чистый бэкбон из имени LTDETR-модели."""
    mapping = {
        'convnext-large': 'dinov3/convnext-large',
        'convnext-base': 'dinov3/convnext-base',
        'convnext-small': 'dinov3/convnext-small',
        'convnext-tiny': 'dinov3/convnext-tiny',
        'vitt16plus': 'dinov3/vitt16plus',
        'vitt16': 'dinov3/vitt16',
        'vits16plus': 'dinov3/vits16plus',
        'vits16': 'dinov3/vits16',
        'vitb16': 'dinov3/vitb16',
        'vitl16': 'dinov3/vitl16',
    }
    for key, backbone in mapping.items():
        if key in full_model:
            return backbone
    logger.warning(f"Неизвестная модель {full_model}, fallback на dinov3/vits16")
    return 'dinov3/vits16'


def ssl_pretrain_for_teacher(config: dict, models_dir: Path) -> Path:
    """
    SSL дообучение бэкбона методом distillation.
    """
    from lightly_train import pretrain

    teacher_cfg = config['teacher_mixed_full_ssl']
    ssl_cfg = teacher_cfg['ssl']

    experiment_data = Path(config['paths']['experiment_data'])
    data_yaml_path = experiment_data / teacher_cfg['data_yaml']

    with open(data_yaml_path) as f:
        dc = yaml.safe_load(f)

    train_key = dc.get('train', 'train/images')
    unlabeled_path = Path(dc['path']) / train_key

    if (unlabeled_path / 'images').exists():
        unlabeled_path = unlabeled_path / 'images'

    if not unlabeled_path.exists():
        raise FileNotFoundError(f"No images for SSL: {unlabeled_path}")

    n_images = len(list(unlabeled_path.glob("*")))

    full_model = config['training']['model']  # ← как в exp3
    backbone = _get_backbone_from_ltdetr(full_model)
    teacher_model = ssl_cfg['teacher']

    ssl_out = models_dir / "ssl_pretrain"

    logger.info(f"SSL distillation for {teacher_cfg['dataset']}: {n_images} unlabeled images")
    logger.info(f"  Student: {backbone}")
    logger.info(f"  Teacher: {teacher_model}")
    logger.info(f"  Epochs: {ssl_cfg['epochs']}, Batch: {ssl_cfg['batch_size']}")

    pretrain(
        out=str(ssl_out),
        data=str(unlabeled_path),
        model=backbone,
        method=ssl_cfg['method'],
        method_args={"teacher": teacher_model},
        epochs=ssl_cfg['epochs'],
        batch_size=ssl_cfg['batch_size'],
        precision=config['training'].get('precision', 'bf16-mixed'),
        seed=config['seeds'][0],
        overwrite=True,
    )

    backbone_path = ssl_out / "exported_models" / "exported_last.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")

    logger.info(f"Distilled backbone saved: {backbone_path}")
    return backbone_path


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

    teacher_cfg = config['teacher_mixed_full_ssl']
    seed = config['seeds'][0]

    with mlflow.start_run(run_name=TEACHER_NAME):
        mlflow.log_param("teacher", TEACHER_NAME)
        mlflow.log_param("dataset", teacher_cfg['dataset'])
        mlflow.log_param("strategy", "ssl")
        mlflow.log_param("ssl_teacher", teacher_cfg['ssl']['teacher'])
        mlflow.log_param("ssl_epochs", teacher_cfg['ssl']['epochs'])
        mlflow.log_param("model", config['training']['model'])
        mlflow.log_param("epochs", config['training']['fixed_epochs'])
        mlflow.log_param("seed", seed)

        try:
            # Шаг 1: SSL дообучение бэкбона (distillation)
            logger.info(f"\n{'='*60}")
            logger.info(f"Шаг 1: SSL pretrain для {teacher_cfg['dataset']}")
            logger.info(f"{'='*60}")

            backbone_path = ssl_pretrain_for_teacher(config, models_dir)
            mlflow.log_param("ssl_backbone_path", str(backbone_path))

            # Шаг 2: LTDETR с SSL-дообученным бэкбоном
            logger.info(f"\n{'='*60}")
            logger.info(f"Шаг 2: LT-DETR training для {teacher_cfg['dataset']}")
            logger.info(f"{'='*60}")

            run_cfg = {
                'run_name': TEACHER_NAME,
                'dataset_name': teacher_cfg['dataset'],
                'data_yaml': teacher_cfg['data_yaml'],
                'strategy_name': 'ssl',
                'freeze_backbone': False,
                'seed': seed,
            }

            result = train_ltdetr(
                config, run_cfg, models_dir,
                extra_model_args={"backbone_weights": str(backbone_path)},
            )

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