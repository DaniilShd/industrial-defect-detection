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

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate.ablation.generate_synthetic import generate_synthetic_dataset
from generate.ablation.train_ltdetr import train_ltdetr
from generate.ablation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ablation_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_combinations(grid: dict) -> list:
    keys = [
        'sd_defect_strength',
        'sd_background_strength',
        'high_freq_alpha',
        'synthetic_total',
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


def setup_dataset_dir(run_dir: Path, real_train: Path, real_val: Path, real_test: Path, 
                      synth_dir: Path, target_size: int = 640) -> Path:
    """Объединяет реальные и синтетические данные + ресайз до target_size"""
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
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Ресайз если нужно
                if h != target_size or w != target_size:
                    img = cv2.resize(img, (target_size, target_size), 
                                    interpolation=cv2.INTER_LANCZOS4)
                
                cv2.imwrite(str(img_dst / img_path.name), img)
                
                # Копируем лейбл (YOLO нормализован — ресайз не влияет)
                lbl_path = s_lbl / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
    
    # data.yaml
    data_yaml = ds_dir / "data.yaml"
    with open(data_yaml, 'w') as f:
        yaml.dump({
            'format': 'yolo', 'path': str(ds_dir),
            'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
            'nc': 4, 'names': {0: 'defect1', 1: 'defect2', 2: 'defect3', 3: 'defect4'}
        }, f)
    
    return data_yaml


def main():
    # Загрузка конфига
    config_path = Path(__file__).parent / "config.yaml"
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
    
    combos = get_combinations(grid)
    logger.info(f"Total runs: {len(combos)}")
    
    # MLflow
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    
    all_results = []
    
    for run_params in combos:
        run_id = run_params['run_id']
        logger.info(f"\n{'='*60}\n{run_id}\n{'='*60}")
        
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params(run_params)
            mlflow.log_params(ltdetr_cfg)
            
            try:
                run_dir = results_base / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Генерация синтетики
                synth_dir = run_dir / "synthetic"
                total_synth = generate_synthetic_dataset(
                    run_params, fixed, synth_dir, real_train, rle_csv
                )
                mlflow.log_metric("synthetic_images", total_synth)
                
                # 2. Подготовка датасета
                data_yaml = setup_dataset_dir(run_dir, real_train, real_val, real_test, synth_dir)
                
                # 3. Обучение
                ltdetr_dir = run_dir / "ltdetr"
                train_result = train_ltdetr(
                    data_yaml, ltdetr_dir,
                    steps=ltdetr_cfg['steps'],
                    lr=ltdetr_cfg['lr'],
                    batch_size=ltdetr_cfg['batch_size']
                )
                
                # 4. Оценка
                if train_result['model_path']:
                    metrics = evaluate_model(
                        train_result['model_path'],
                        real_test / "images",
                        real_test / "labels"
                    )
                else:
                    metrics = {'mAP@50': 0.0, 'mAP@75': 0.0}
                
                metrics['training_time_h'] = train_result['training_time_hours']
                
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(data_yaml))
                
                all_results.append({**run_params, **metrics})
                
                logger.info(f"mAP@50={metrics['mAP@50']:.4f}, mAP@75={metrics['mAP@75']:.4f}")
                
            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                traceback.print_exc()
                mlflow.log_param("status", "failed")
            
            finally:
                torch.cuda.empty_cache()
    
    # Итоги
    if all_results:
        best = max(all_results, key=lambda x: x['mAP@50'])
        logger.info(f"\nBest run: {best['run_id']} mAP@50={best['mAP@50']:.4f}")
        logger.info(f"  defect_strength={best.get('sd_defect_strength', '?')}, "
                   f"bg_strength={best.get('sd_background_strength', '?')}, "
                   f"hf_alpha={best.get('high_freq_alpha', '?')}, "
                   f"n={best.get('synthetic_total', '?')}, "
                   f"bal={best.get('balance_strategy', '?')}")
        
        summary_path = results_base / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(summary_path))


if __name__ == "__main__":
    main()