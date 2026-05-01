#!/usr/bin/env python3
"""
Оркестратор эксперимента: полный цикл с MLflow-логированием.
Запуск: python scripts/run_experiment.py
"""

import itertools
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.scripts.train_single import train_single_model
from experiments.scripts.statistical_analysis import run_statistical_analysis
from experiments.scripts.visualize import create_all_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_experiment_grid(config: dict) -> list:
    """Генерирует сетку: датасет × стратегия бэкбона × seed"""
    datasets = config['datasets']
    strategies = config['backbone_strategies']
    seeds = config['seeds']
    
    grid = []
    for (ds_name, ds_cfg), (strat_name, strat_cfg), seed in itertools.product(
        datasets.items(), strategies.items(), seeds
    ):
        grid.append({
            'run_name': f"{ds_name}_{strat_name}_seed{seed}",
            'dataset_name': ds_name,
            'dataset_type': ds_cfg['type'],
            'data_yaml': ds_cfg['data_yaml'],
            'strategy_name': strat_name,
            'freeze_backbone': strat_cfg['freeze_backbone'],
            'backbone_lr_factor': strat_cfg['backbone_lr_factor'],
            'seed': seed,
        })
    
    return grid


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    
    logger.info("=" * 60)
    logger.info(f"🔬 ЭКСПЕРИМЕНТ: {config['experiment']['name']}")
    logger.info(f"   Датасетов: {len(config['datasets'])}")
    logger.info(f"   Стратегий бэкбона: {len(config['backbone_strategies'])}")
    logger.info(f"   Повторов: {config['repeats']}")
    logger.info("=" * 60)
    
    # Генерируем сетку
    grid = generate_experiment_grid(config)
    total_runs = len(grid)
    logger.info(f"Всего запусков: {total_runs}")
    
    # MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Создаём директории
    results_dir = Path(config['paths']['results_dir'])
    models_dir = Path(config['paths']['models_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Запускаем все эксперименты
    with mlflow.start_run(run_name=config['experiment']['name']):
        mlflow.log_dict(config, "experiment_config.yaml")
        mlflow.log_param("total_runs", total_runs)
        
        for idx, run_cfg in enumerate(grid):
            logger.info(f"\n{'='*50}")
            logger.info(f"Запуск {idx+1}/{total_runs}: {run_cfg['run_name']}")
            logger.info(f"{'='*50}")
            
            with mlflow.start_run(run_name=run_cfg['run_name'], nested=True):
                mlflow.log_params(run_cfg)
                
                try:
                    result = train_single_model(
                        config=config,
                        run_cfg=run_cfg,
                        models_dir=models_dir / run_cfg['run_name'],
                    )
                    
                    all_results.append(result)
                    
                    # Логируем в MLflow
                    mlflow.log_metrics({
                        'test_map50': result.get('test_map50', 0),
                        'test_map75': result.get('test_map75', 0),
                        'val_map50': result.get('val_map50', 0),
                        'training_time_h': result.get('training_time_hours', 0),
                    })
                    
                    logger.info(f"✅ mAP@50={result.get('test_map50', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка: {e}")
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))
                    all_results.append({
                        'run_name': run_cfg['run_name'],
                        'status': 'failed',
                        'error': str(e),
                    })
        
        # Сохраняем все результаты
        results_path = results_dir / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))
        
        # Статистический анализ
        logger.info("\n" + "=" * 60)
        logger.info("📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ")
        stats_results = run_statistical_analysis(all_results, config)
        stats_path = results_dir / "statistical_analysis.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        mlflow.log_artifact(str(stats_path))
        
        # Визуализация
        logger.info("\n📈 ВИЗУАЛИЗАЦИЯ")
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        create_all_visualizations(all_results, stats_results, config, figures_dir)
        for fig_path in figures_dir.glob('*.png'):
            mlflow.log_artifact(str(fig_path))
        
        # Итоги
        completed = [r for r in all_results if r.get('status') == 'completed']
        logger.info(f"\n✅ Завершено: {len(completed)}/{total_runs}")
        
        if stats_results.get('significant_pairs'):
            logger.info("\n📊 Значимые различия:")
            for pair in stats_results['significant_pairs']:
                logger.info(f"   {pair}")


if __name__ == "__main__":
    main()