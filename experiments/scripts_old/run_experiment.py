#!/usr/bin/env python3
"""
Оркестратор эксперимента: полный цикл с MLflow-логированием.
Запуск: python experiments/scripts/run_experiment.py
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.scripts.train_single import train_single_model
from experiments.scripts.statistical_analysis import run_statistical_analysis
from experiments.scripts.visualize import create_all_visualizations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_experiment_grid(config: dict) -> list:
    """Генерирует сетку экспериментов: датасет × стратегия × seed."""
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
            'freeze_backbone': strat_cfg.get('freeze_backbone', False),
            'backbone_lr_factor': strat_cfg.get('backbone_lr_factor', 0.1),
            'seed': seed,
        })
    
    return grid


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    
    logger.info("=" * 60)
    logger.info(f"ЭКСПЕРИМЕНТ: {config['experiment']['name']}")
    logger.info(f"  Модель: {config['training']['model']}")
    logger.info(f"  Датасетов: {len(config['datasets'])}")
    logger.info(f"  Стратегий: {len(config['backbone_strategies'])}")
    logger.info(f"  Сидов: {len(config['seeds'])}")
    logger.info("=" * 60)
    
    # Генерируем сетку
    grid = generate_experiment_grid(config)
    total_runs = len(grid)
    logger.info(f"Всего запусков: {total_runs}")
    
    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Создаём директории
    results_dir = Path(config['paths']['results_dir'])
    models_dir = Path(config['paths']['models_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    completed_count = 0
    failed_count = 0
    
    # Parent run
    with mlflow.start_run(run_name=config['experiment']['name']):
        mlflow.log_dict(config, "experiment_config.yaml")
        mlflow.log_param("total_runs", total_runs)
        mlflow.log_param("model", config['training']['model'])
        
        for idx, run_cfg in enumerate(grid):
            logger.info(f"\n{'='*50}")
            logger.info(f"Запуск {idx+1}/{total_runs}: {run_cfg['run_name']}")
            logger.info(f"  Датасет: {run_cfg['dataset_name']} ({run_cfg['dataset_type']})")
            logger.info(f"  Стратегия: {run_cfg['strategy_name']}")
            logger.info(f"  Seed: {run_cfg['seed']}")
            logger.info(f"{'='*50}")
            
            with mlflow.start_run(run_name=run_cfg['run_name'], nested=True):
                # Логируем параметры
                mlflow.log_params({
                    'run_name': run_cfg['run_name'],
                    'dataset_name': run_cfg['dataset_name'],
                    'dataset_type': run_cfg['dataset_type'],
                    'strategy_name': run_cfg['strategy_name'],
                    'seed': run_cfg['seed'],
                })
                
                try:
                    start_time = time.time()
                    
                    result = train_single_model(
                        config=config,
                        run_cfg=run_cfg,
                        models_dir=models_dir / run_cfg['run_name'],
                    )
                    
                    elapsed = time.time() - start_time
                    all_results.append(result)
                    completed_count += 1
                    
                    # Логируем метрики
                    mlflow.log_metrics({
                        'test_map50': result.get('test_map50', 0),
                        'test_map75': result.get('test_map75', 0),
                        'test_map50_95': result.get('test_map50_95', 0),
                        'val_map50': result.get('val_map50', 0),
                        'final_val_map50': result.get('final_val_map50', 0),
                        'training_time_hours': result.get('training_time_hours', 0),
                        'best_val_step': result.get('best_val_step', 0),
                    })
                    
                    # Логируем per-class метрики
                    for k, v in result.items():
                        if k.startswith('test_cls'):
                            mlflow.log_metric(k, v)
                    
                    # Логируем train.log
                    train_log = result.get('train_log_path')
                    if train_log and Path(train_log).exists():
                        mlflow.log_artifact(train_log)
                    
                    # Логируем result.json
                    result_path = models_dir / run_cfg['run_name'] / "result.json"
                    if result_path.exists():
                        mlflow.log_artifact(str(result_path))
                    
                    mlflow.set_tag("status", "completed")
                    
                    logger.info(f"mAP@50: test={result.get('test_map50', 0):.4f}, "
                               f"val={result.get('val_map50', 0):.4f}")
                    logger.info(f"Время: {elapsed/60:.1f} мин")
                    
                except Exception as e:
                    logger.error(f"Ошибка в {run_cfg['run_name']}: {e}", exc_info=True)
                    failed_count += 1
                    
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))
                    mlflow.set_tag("status", "failed")
                    
                    all_results.append({
                        'run_name': run_cfg['run_name'],
                        'dataset_name': run_cfg['dataset_name'],
                        'strategy_name': run_cfg['strategy_name'],
                        'seed': run_cfg['seed'],
                        'status': 'failed',
                        'error': str(e),
                    })
        
        # Сохраняем все результаты
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"all_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))
        
        # Итоговые метрики
        mlflow.log_metrics({
            'completed_runs': completed_count,
            'failed_runs': failed_count,
            'completion_rate': completed_count / total_runs if total_runs > 0 else 0,
        })
        
        # Статистический анализ (только если есть успешные запуски)
        if completed_count >= 2:
            logger.info("\n" + "=" * 60)
            logger.info("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
            try:
                stats_results = run_statistical_analysis(all_results, config)
                stats_path = results_dir / f"statistical_analysis_{timestamp}.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats_results, f, indent=2, default=str)
                mlflow.log_artifact(str(stats_path))
                
                if stats_results.get('significant_pairs'):
                    logger.info("Значимые различия:")
                    for pair in stats_results['significant_pairs']:
                        logger.info(f"  {pair}")
            except Exception as e:
                logger.error(f"Ошибка статистического анализа: {e}")
            
            # Визуализация
            logger.info("\nВИЗУАЛИЗАЦИЯ")
            try:
                figures_dir = results_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                create_all_visualizations(all_results, stats_results, config, figures_dir)
                for fig_path in figures_dir.glob('*.png'):
                    mlflow.log_artifact(str(fig_path))
            except Exception as e:
                logger.error(f"Ошибка визуализации: {e}")
        
        # Финальный отчёт
        logger.info(f"\n{'='*60}")
        logger.info(f"Завершено: {completed_count}/{total_runs}")
        logger.info(f"Ошибок: {failed_count}/{total_runs}")
        logger.info(f"Результаты: {results_path}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()