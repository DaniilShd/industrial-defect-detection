#!/usr/bin/env python3
"""Эксперимент 3: SSL-дообучение бэкбона + LT-DETR (DINOv3)"""

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


def ssl_pretrain(config: dict, models_dir: Path) -> Path:
    """
    SSL-дообучение бэкбона ConvNeXt на неразмеченных данных.
    
    Для DINOv3/ConvNeXt используется метод MoCo v3 или SimCLR,
    так как DINOv2 метод предназначен только для ViT архитектур.
    """
    from lightly_train import pretrain

    ssl_out = models_dir / "ssl_pretrain"
    unlabeled_data = config.get('ssl', {}).get('unlabeled_data', '/app/data/unlabeled')
    
    # ★ Определяем бэкбон из основной модели LT-DETR
    # "dinov3/convnext-tiny-ltdetr-coco" → бэкбон = convnext_tiny
    # "dinov3/convnext-large-ltdetr-coco" → бэкбон = convnext_large
    full_model = config['training']['model']
    
    # Извлекаем тип бэкбона из названия модели
    if 'convnext-tiny' in full_model:
        backbone_model = 'convnext_tiny'
    elif 'convnext-small' in full_model:
        backbone_model = 'convnext_small'
    elif 'convnext-base' in full_model:
        backbone_model = 'convnext_base'
    elif 'convnext-large' in full_model:
        backbone_model = 'convnext_large'
    elif 'vits' in full_model:
        backbone_model = full_model.split('/')[1].split('-')[0]  # vits16, vitb8, etc.
    else:
        backbone_model = 'convnext_tiny'  # fallback
    
    # ★ SSL method для CNN: MoCo v2/v3 или SimCLR
    # DINOv2 работает только с ViT
    ssl_method = config.get('ssl', {}).get('method', 'moco')
    
    logger.info(f"SSL pretrain: backbone={backbone_model}, method={ssl_method}")
    logger.info(f"Unlabeled data: {unlabeled_data}")
    
    pretrain(
        out=str(ssl_out),
        data=unlabeled_data,
        model=backbone_model,  # ★ Исправлено: convnext_tiny/large вместо dinov2/vits14
        method=ssl_method,      # ★ Исправлено: moco/simclr вместо dinov2
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
    total_runs = len(datasets) * len(seeds) + 1  # +1 за SSL

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

        # Шаг 1: SSL-дообучение (один раз)
        logger.info("Шаг 1: SSL-дообучение бэкбона...")
        try:
            backbone_path = ssl_pretrain(config, models_dir)
            mlflow.log_artifact(str(backbone_path))
            logger.info(f"SSL backbone: {backbone_path}")
        except Exception as e:
            logger.error(f"SSL failed: {e}")
            return

        # Шаг 2: LT-DETR с SSL-бэкбоном на всех датасетах
        for (ds_name, ds_cfg), seed in itertools.product(datasets.items(), seeds):
            run_cfg = {
                'run_name': f"{ds_name}_{STRATEGY_NAME}_seed{seed}",
                'dataset_name': ds_name,
                'data_yaml': ds_cfg['data_yaml'],
                'strategy_name': STRATEGY_NAME,
                'freeze_backbone': False,
                'seed': seed,
            }

            logger.info(f"Запуск: {run_cfg['run_name']}")
            with mlflow.start_run(run_name=run_cfg['run_name'], nested=True):
                mlflow.log_params(run_cfg)
                try:
                    result = train_ltdetr(
                        config, run_cfg, models_dir / run_cfg['run_name'],
                        extra_model_args={"backbone_weights": str(backbone_path)},
                    )
                    all_results.append(result)
                    completed_count += 1
                    mlflow.log_metrics({
                        'test_map50': result['test_map50'],
                        'test_map75': result['test_map75'],
                        'test_map50_95': result['test_map50_95'],
                        'val_map50': result['val_map50'],
                        'training_time_hours': result['training_time_hours'],
                    })
                    mlflow.set_tag("status", "completed")
                except Exception as e:
                    logger.error(f"Ошибка: {e}", exc_info=True)
                    mlflow.set_tag("status", "failed")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))

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

        logger.info(f"Завершено: {completed_count}/{total_runs - 1}")


if __name__ == "__main__":
    main()