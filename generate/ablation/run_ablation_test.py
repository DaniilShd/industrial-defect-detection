#!/usr/bin/env python3
"""
run_ablation.py — Ablation study полного цикла
Запуск: python generate/ablation/run_ablation.py
"""

import itertools
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path

import mlflow
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate.ablation.generate_synthetic import generate_synthetic_dataset
from generate.ablation.train_ltdetr import train_ltdetr
from generate.ablation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ablation_config(config_path: str = "config_test.yaml") -> dict:
    """Загрузка конфигурации ablation study."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_combinations(grid: dict) -> list:
    """Генерация всех комбинаций гиперпараметров."""
    keys = [
        'sd_defect_strength',
        'sd_background_strength',
        'high_freq_alpha',
        'variants',
        'balance_strategy'
    ]
    values = [grid[k] for k in keys]
    repeats = grid.get('repeats', 1)
    
    combos = []
    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        for r in range(repeats):
            p = params.copy()
            p['run_id'] = f"abl_{i:03d}_r{r}"
            p['repeat'] = r
            combos.append(p)
    return combos


def setup_dataset_dir(run_dir: Path, real_train: Path, real_val: Path, 
                      real_test: Path, synth_dir: Path) -> Path:
    """Объединяет реальные и синтетические данные в один датасет."""
    ds_dir = run_dir / "dataset"
    
    for split, sources in [
        ('train', [real_train, synth_dir]),
        ('val', [real_val]),
        ('test', [real_test])
    ]:
        for sub in ['images', 'labels']:
            (ds_dir / split / sub).mkdir(parents=True, exist_ok=True)
        
        img_dst = ds_dir / split / 'images'
        lbl_dst = ds_dir / split / 'labels'
        
        for src in sources:
            s_img = src / 'images'
            s_lbl = src / 'labels'
            
            if not s_img.exists():
                continue
            
            for img_path in s_img.glob("*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                shutil.copy2(img_path, img_dst / img_path.name)
                
                lbl_path = s_lbl / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
    
    # Создаём data.yaml для LightlyTrain
    data_yaml = ds_dir / "data.yaml"
    with open(data_yaml, 'w') as f:
        yaml.dump({
            'format': 'yolo',
            'path': str(ds_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                0: 'defect1',
                1: 'defect2',
                2: 'defect3',
                3: 'defect4'
            }
        }, f)
    
    return data_yaml


def main():
    config_path = Path(__file__).parent / "config_test.yaml"
    cfg = load_ablation_config(str(config_path))

    grid = cfg['grid']
    ltdetr_cfg = cfg['ltdetr']
    fixed = cfg['fixed_generation']
    paths = cfg['paths']
    
    real_train = Path(paths['real_train'])
    real_val = Path(paths['real_val'])
    real_test = Path(paths['real_test'])
    rle_csv = real_train / "train_rle.csv"
    results_base = Path(paths['results_dir'])
    results_base.mkdir(parents=True, exist_ok=True)
    
    combos = get_combinations(grid)
    logger.info(f"Total runs: {len(combos)}")
    
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    
    all_results = []
    
    for idx, run_params in enumerate(combos):
        run_id = run_params['run_id']
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {idx+1}/{len(combos)}: {run_id}")
        logger.info(f"{'='*60}")
        
        with mlflow.start_run(run_name=run_id):
            # Логируем все параметры одним вызовом
            all_params = {
                **run_params,
                **{f"ltdetr_{k}": v for k, v in ltdetr_cfg.items()}
            }
            mlflow.log_params(all_params)
            mlflow.set_tag("run_type", "ablation")
            
            try:
                run_dir = results_base / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                
                # =============================================
                # Шаг 1/4: Генерация синтетического датасета
                # =============================================
                logger.info("Step 1/4: Generating synthetic dataset...")
                synth_dir = run_dir / "synthetic"
                total_synth = generate_synthetic_dataset(
                    run_params, fixed, synth_dir, real_train, rle_csv
                )
                mlflow.log_metric("synthetic_images", total_synth)
                logger.info(f"Generated {total_synth} synthetic images")
                
                # =============================================
                # Шаг 2/4: Подготовка датасета
                # =============================================
                logger.info("Step 2/4: Preparing dataset...")
                data_yaml = setup_dataset_dir(
                    run_dir, real_train, real_val, real_test, synth_dir
                )
                
                # Статистика датасета
                num_train_images = len(list(
                    (run_dir / "dataset/train/images").glob("*")
                ))
                num_val_images = len(list(
                    (run_dir / "dataset/val/images").glob("*")
                ))
                effective_batch = ltdetr_cfg['batch_size']
                steps_per_epoch = max(1, num_train_images / effective_batch)
                epochs = ltdetr_cfg['max_steps'] / steps_per_epoch
                
                logger.info(f"Dataset: {num_train_images} train + "
                           f"{num_val_images} val images, "
                           f"~{epochs:.1f} epochs over {ltdetr_cfg['max_steps']} steps")
                
                mlflow.log_metrics({
                    "num_train_images": num_train_images,
                    "num_val_images": num_val_images,
                    "effective_epochs": round(epochs, 1),
                })
                
                # =============================================
                # Шаг 3/4: Обучение LT-DETR
                # =============================================
                logger.info("Step 3/4: Training LT-DETR...")
                ltdetr_dir = run_dir / "ltdetr"
                train_result = train_ltdetr(
                    data_yaml=data_yaml,
                    out_dir=ltdetr_dir,
                    max_steps=ltdetr_cfg['max_steps'],
                    val_every_steps=ltdetr_cfg.get('val_every_steps', 500),
                    lr=ltdetr_cfg['lr'],
                    batch_size=ltdetr_cfg['batch_size'],
                    freeze_backbone=ltdetr_cfg.get('freeze_backbone', False),
                    seed=ltdetr_cfg.get('seed', 42),
                )
                
                # Логируем артефакты обучения
                train_log = ltdetr_dir / "train.log"
                if train_log.exists():
                    mlflow.log_artifact(str(train_log), "training_logs")
                
                for event_file in ltdetr_dir.glob("events.out.tfevents.*"):
                    mlflow.log_artifact(str(event_file), "training_logs")
                
                # Логируем result.json если есть
                result_json = ltdetr_dir / "result.json"
                if result_json.exists():
                    mlflow.log_artifact(str(result_json), "training_logs")
                
                # Логируем data.yaml
                if data_yaml.exists():
                    mlflow.log_artifact(str(data_yaml), "dataset_config")
                
                # =============================================
                # Шаг 4/4: Оценка на тестовом наборе
                # =============================================
                logger.info("Step 4/4: Evaluating on test set...")
                default_metrics = {
                    'mAP_50': 0.0,
                    'mAP_75': 0.0,
                    'mAP_50_95': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1': 0.0,
                    'num_predictions': 0,
                    'num_ground_truth': 0
                }
                metrics = default_metrics.copy()
                
                if train_result.get('model_path'):
                    try:
                        metrics = evaluate_model(
                            model_path=train_result['model_path'],
                            test_images=real_test / "images",
                            test_labels=real_test / "labels",
                        )
                        logger.info(
                            f"Test results: "
                            f"mAP_50={metrics.get('mAP_50', 0):.4f}, "
                            f"mAP_75={metrics.get('mAP_75', 0):.4f}, "
                            f"mAP_50_95={metrics.get('mAP_50_95', 0):.4f}"
                        )
                    except Exception as e:
                        logger.error(f"Evaluation failed: {e}")
                        traceback.print_exc()
                else:
                    logger.error("No model found for evaluation")
                
                # Добавляем мета-информацию
                metrics['training_time_h'] = train_result.get('training_time_hours', 0)
                
                # Логируем все метрики в MLflow
                mlflow.log_metrics(metrics)
                
                # Сохраняем результат
                all_results.append({**run_params, **metrics})
                mlflow.set_tag("status", "completed")
                
                logger.info(f"Run {run_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                traceback.print_exc()
                
                mlflow.set_tag("status", "failed")
                mlflow.log_metric("failed", 1)
                
                all_results.append({
                    **run_params,
                    'mAP_50': 0.0,
                    'mAP_75': 0.0,
                    'mAP_50_95': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1': 0.0,
                    'training_time_h': 0,
                    'status': 'failed'
                })
            
            finally:
                # Очистка
                torch.cuda.empty_cache()
                for temp_dir in ['dataset', 'synthetic', 'ltdetr']:
                    temp_path = run_dir / temp_dir
                    if temp_path.exists():
                        shutil.rmtree(temp_path, ignore_errors=True)
                        logger.debug(f"Cleaned up {temp_path}")
    
    # =============================================
    # Финальная сводка
    # =============================================
    if all_results:
        summary_path = results_base / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        with mlflow.start_run(run_name="ablation_summary"):
            mlflow.log_artifact(str(summary_path), "summary")
            
            successful_runs = [r for r in all_results if r.get('status') != 'failed']
            
            if successful_runs:
                best = max(successful_runs, key=lambda x: x.get('mAP_50', 0))
                
                logger.info(f"\n{'='*60}")
                logger.info(f"BEST RUN: {best['run_id']}")
                logger.info(f"  mAP_50:     {best.get('mAP_50', 0):.4f}")
                logger.info(f"  mAP_75:     {best.get('mAP_75', 0):.4f}")
                logger.info(f"  mAP_50_95:  {best.get('mAP_50_95', 0):.4f}")
                logger.info(f"  Precision:  {best.get('Precision', 0):.4f}")
                logger.info(f"  Recall:     {best.get('Recall', 0):.4f}")
                logger.info(f"  F1:         {best.get('F1', 0):.4f}")
                logger.info(f"  Parameters:")
                logger.info(f"    defect_strength:    {best.get('sd_defect_strength', '?')}")
                logger.info(f"    bg_strength:        {best.get('sd_background_strength', '?')}")
                logger.info(f"    high_freq_alpha:    {best.get('high_freq_alpha', '?')}")
                logger.info(f"    variants:           {best.get('variants', '?')}")
                logger.info(f"    balance_strategy:   {best.get('balance_strategy', '?')}")
                logger.info(f"  Training time: {best.get('training_time_h', 0):.2f}h")
                logger.info(f"{'='*60}")
                
                mlflow.log_metrics({
                    'best_mAP_50': best.get('mAP_50', 0),
                    'best_mAP_75': best.get('mAP_75', 0),
                    'best_mAP_50_95': best.get('mAP_50_95', 0),
                    'total_runs': len(all_results),
                    'successful_runs': len(successful_runs),
                    'failed_runs': len(all_results) - len(successful_runs),
                })
                mlflow.log_params({f"best_{k}": v for k, v in best.items() 
                                  if k.startswith('sd_') or k.startswith('high_') 
                                  or k in ['variants', 'balance_strategy']})
            
            logger.info(f"\nTotal: {len(all_results)} runs "
                       f"({len(successful_runs)} successful, "
                       f"{len(all_results) - len(successful_runs)} failed)")
            logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()