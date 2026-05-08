#!/usr/bin/env python3
"""Эксперимент 3: SSL-дообучение бэкбона + LT-DETR (DINOv3)
SSL дообучение проводится ИНДИВИДУАЛЬНО для каждого датасета
(используются только train/images этого датасета без разметки)."""

import itertools
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
from experiments.scripts.statistical_analysis import run_statistical_analysis
from experiments.scripts.visualize import create_all_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp3_ssl"
STRATEGY_NAME = "ssl"


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


def _get_backbone_and_teacher(full_model: str) -> tuple:
    """Определяет бэкбон и учителя по имени LTDETR модели."""
    if 'convnext-tiny' in full_model:
        return 'convnext_tiny', 'dinov2/vits14'
    elif 'convnext-small' in full_model:
        return 'convnext_small', 'dinov2/vitb14'
    elif 'convnext-base' in full_model:
        return 'convnext_base', 'dinov2/vitb14'
    elif 'convnext-large' in full_model:
        return 'convnext_large', 'dinov2/vitl14'
    else:
        return 'convnext_tiny', 'dinov2/vits14'


def ssl_pretrain_for_dataset(
    config: dict,
    models_dir: Path,
    ds_name: str,
) -> Path:
    """
    SSL дообучение бэкбона ТОЛЬКО на данных конкретного датасета.
    
    Использует train/images датасета как неразмеченные данные.
    Лейблы игнорируются — LightlyTrain читает только изображения.
    """
    from lightly_train import pretrain

    ssl_out = models_dir / f"ssl_pretrain_{ds_name}"
    experiment_data = Path(config['paths']['experiment_data'])
    
    # Берём train/images только этого датасета
    unlabeled_path = experiment_data / ds_name / "train" / "images"
    
    if not unlabeled_path.exists():
        raise FileNotFoundError(f"No images for SSL: {unlabeled_path}")
    
    n_images = len(list(unlabeled_path.glob("*")))
    logger.info(f"SSL for {ds_name}: {n_images} unlabeled images from {unlabeled_path}")
    
    full_model = config['training']['model']
    backbone, teacher = _get_backbone_and_teacher(full_model)
    
    logger.info(f"Distillation: {teacher} → {backbone}")
    
    pretrain(
        out=str(ssl_out),
        data=str(unlabeled_path),       # Только данные этого датасета
        model=backbone,                 # Студент (ConvNeXt)
        method="distillation",          # Дистилляция
        method_args={"teacher": teacher},  # Учитель (DINOv2 ViT)
        epochs=config.get('ssl', {}).get('epochs', 10),
        batch_size=config.get('ssl', {}).get('batch_size', 32),
        seed=config['seeds'][0],
        overwrite=True,
    )

    backbone_path = ssl_out / "exported_models" / "exported_last.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")
    
    logger.info(f"SSL backbone saved: {backbone_path}")
    return backbone_path


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    datasets = config['datasets']
    seeds = config['seeds']
    # Каждый датасет: 1 SSL + 1 LTDETR = 2 запуска на датасет
    total_runs = len(datasets) * len(seeds)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_{EXPERIMENT_NAME}")

    results_dir = Path(config['paths']['results_dir']) / EXPERIMENT_NAME
    models_dir = Path(config['paths']['models_dir']) / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    completed_count = 0

    with mlflow.start_run(run_name=EXPERIMENT_NAME):
        mlflow.log_param("experiment", EXPERIMENT_NAME)
        mlflow.log_param("strategy", STRATEGY_NAME)
        mlflow.log_param("total_runs", total_runs)
        mlflow.log_param("ssl_backbone", config['training']['model'])

        for (ds_name, ds_cfg), seed in itertools.product(datasets.items(), seeds):
            run_cfg = {
                'run_name': f"{ds_name}_{STRATEGY_NAME}_seed{seed}",
                'dataset_name': ds_name,
                'data_yaml': ds_cfg['data_yaml'],
                'strategy_name': STRATEGY_NAME,
                'freeze_backbone': False,
                'seed': seed,
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"Запуск: {run_cfg['run_name']}")
            logger.info(f"{'='*60}")

            with mlflow.start_run(run_name=run_cfg['run_name'], nested=True):
                mlflow.log_params(run_cfg)
                
                try:
                    # 🔥 Шаг 1: SSL дообучение ДЛЯ ЭТОГО ДАТАСЕТА
                    logger.info(f"SSL pretrain for {ds_name}...")
                    backbone_path = ssl_pretrain_for_dataset(
                        config, models_dir, ds_name
                    )
                    mlflow.log_param("ssl_backbone_path", str(backbone_path))
                    
                    # 🔥 Шаг 2: LTDETR с SSL бэкбоном
                    logger.info(f"LT-DETR training for {ds_name}...")
                    result = train_ltdetr(
                        config, run_cfg, models_dir / run_cfg['run_name'],
                        extra_model_args={"backbone_weights": str(backbone_path)},
                    )
                    
                    all_results.append(result)
                    completed_count += 1
                    mlflow.log_metrics(_prepare_mlflow_metrics(result))
                    mlflow.set_tag("status", "completed")
                    
                    logger.info(f"✅ {run_cfg['run_name']}: "
                               f"mAP50={result.get('test_map50', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка в {run_cfg['run_name']}: {e}", exc_info=True)
                    mlflow.set_tag("status", "failed")

        # Сохраняем результаты
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))

        # Статистический анализ
        if completed_count >= 2:
            stats = run_statistical_analysis(all_results, config)
            stats_path = results_dir / f"stats_{timestamp}.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            mlflow.log_artifact(str(stats_path))

            figs_dir = results_dir / "figures"
            figs_dir.mkdir(exist_ok=True)
            create_all_visualizations(all_results, stats, config, figs_dir)
            for p in figs_dir.glob('*.png'):
                mlflow.log_artifact(str(p))

        logger.info(f"\nЗавершено: {completed_count}/{total_runs}")


if __name__ == "__main__":
    main()