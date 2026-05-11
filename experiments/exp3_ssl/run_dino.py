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
    """
    Определяет чистый бэкбон и учителя по имени LTDETR-модели.
    
    Пример:
      'dinov3/convnext-large-ltdetr-coco' → ('dinov3/convnext-large', 'dinov3/vitl16')
      'dinov3/vitt16-ltdetr-coco'         → ('dinov3/vitt16', 'dinov3/vits16')
    """
    # Сопоставление LTDETR-модель → (чистый бэкбон, учитель)
    if 'convnext-large' in full_model:
        return 'dinov3/convnext-large', 'dinov3/vitl16'
    elif 'convnext-base' in full_model:
        return 'dinov3/convnext-base', 'dinov3/vitl16'
    elif 'convnext-small' in full_model:
        return 'dinov3/convnext-small', 'dinov3/vitb16'
    elif 'convnext-tiny' in full_model:
        return 'dinov3/convnext-tiny', 'dinov3/vits16'
    elif 'vitt16plus' in full_model:
        return 'dinov3/vitt16plus', 'dinov3/vits16'
    elif 'vitt16' in full_model:
        return 'dinov3/vitt16', 'dinov3/vits16'
    elif 'vits16plus' in full_model:
        return 'dinov3/vits16plus', 'dinov3/vitb16'
    elif 'vits16' in full_model:
        return 'dinov3/vits16', 'dinov3/vitb16'
    elif 'vitb16' in full_model:
        return 'dinov3/vitb16', 'dinov3/vitl16'
    elif 'vitl16' in full_model:
        return 'dinov3/vitl16', 'dinov3/vitl16'
    else:
        logger.warning(f"Неизвестная модель {full_model}, fallback на convnext-tiny")
        return 'dinov3/convnext-tiny', 'dinov3/vits16'


def ssl_pretrain_for_dataset(
    config: dict,
    models_dir: Path,
    ds_name: str,
    seed: int,
) -> Path:
    """
    SSL дообучение бэкбона методом DINO (без внешнего учителя).
    
    DINO сам строит учителя через exponential moving average ученика.
    Не требует внешнего ImageNet-учителя → учится на ваших данных.
    """
    from lightly_train import pretrain

    ssl_out = models_dir / f"ssl_pretrain_{ds_name}_seed{seed}"
    experiment_data = Path(config['paths']['experiment_data'])
    
    unlabeled_path = experiment_data / ds_name / "train" / "images"
    
    if not unlabeled_path.exists():
        raise FileNotFoundError(f"No images for SSL: {unlabeled_path}")
    
    n_images = len(list(unlabeled_path.glob("*")))
    logger.info(f"SSL DINO for {ds_name}: {n_images} unlabeled images")
    
    full_model = config['training']['model']  # dinov3/vits16-ltdetr-coco
    backbone, _ = _get_backbone_and_teacher(full_model)  # dinov3/vits16
    
    logger.info(f"DINO SSL: model={backbone} (no external teacher needed)")
    logger.info(f"SSL config: epochs={config['ssl']['epochs']}, "
                f"batch_size={config['ssl']['batch_size']}")
    
    # ★ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: method="dino" без method_args
    pretrain(
        out=str(ssl_out),
        data=str(unlabeled_path),
        model=backbone,                    # dinov3/vits16
        method="dino",                     # ← DINO вместо distillation
        # method_args не нужны! DINO сам создает учителя из ученика
        epochs=config['ssl']['epochs'],    # 400
        batch_size=config['ssl']['batch_size'],  # 64
        precision=config['training'].get('precision', 'bf16-mixed'),
        seed=seed,
        overwrite=True,
    )

    backbone_path = ssl_out / "exported_models" / "exported_last.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")
    
    logger.info(f"DINO backbone saved: {backbone_path}")
    return backbone_path


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    datasets = config['datasets']
    seeds = config['seeds']
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
        mlflow.log_param("base_model", config['training']['model'])

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
                    # Шаг 1: SSL дообучение бэкбона на неразмеченных данных этого датасета
                    logger.info(f"SSL pretrain for {ds_name} (seed={seed})...")
                    backbone_path = ssl_pretrain_for_dataset(
                        config, models_dir, ds_name, seed
                    )
                    mlflow.log_param("ssl_backbone_path", str(backbone_path))
                    
                    # Шаг 2: LTDETR с SSL-дообученным бэкбоном
                    logger.info(f"LT-DETR training for {ds_name} (seed={seed})...")
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