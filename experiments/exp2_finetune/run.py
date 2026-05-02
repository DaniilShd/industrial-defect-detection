#!/usr/bin/env python3
"""Эксперимент 2: Дообучение бэкбона внутри LT-DETR"""

import itertools
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
from experiments.scripts.statistical_analysis import run_statistical_analysis
from experiments.scripts.visualize import create_all_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp2_finetune"
STRATEGY_NAME = "finetune"


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
                    result = train_ltdetr(config, run_cfg, models_dir / run_cfg['run_name'])
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

        logger.info(f"Завершено: {completed_count}/{total_runs}")


if __name__ == "__main__":
    main()