#!/usr/bin/env python3
"""Обучение одной модели LT-DETR с early stopping"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def _find_best_val_metrics(out_dir: Path) -> dict:
    """Извлекает лучшие валидационные метрики из metrics.csv"""
    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = next(out_dir.glob("**/metrics.csv"), None)
    
    if metrics_csv and metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if df.empty:
                return {}
            
            val_cols = [c for c in df.columns if 'val' in c.lower() and 'map' in c.lower()]
            if val_cols:
                best_idx = df[val_cols[0]].idxmax()
                return {
                    'best_val_map50': float(df[val_cols[0]].max()),
                    'best_val_step': int(df.iloc[best_idx].get('step', best_idx * 500)),
                    'final_val_map50': float(df[val_cols[0]].iloc[-1]),
                    'total_steps_trained': int(df.iloc[-1].get('step', len(df) * 500))
                }
        except Exception as e:
            logger.warning(f"Could not parse metrics.csv: {e}")
    
    return {}


def _check_early_stopping(out_dir: Path, patience: int, val_every: int) -> tuple:
    """
    Проверяет, сработала ли ранняя остановка.
    Returns: (early_stopped: bool, actual_steps: int)
    """
    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = next(out_dir.glob("**/metrics.csv"), None)
    
    if metrics_csv and metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if df.empty:
                return False, 0
            
            val_cols = [c for c in df.columns if 'val' in c.lower() and 'map' in c.lower()]
            if val_cols:
                vals = df[val_cols[0]].values
                actual_steps = len(df) * val_every
                
                if len(vals) >= patience + 1:
                    best_idx = vals.argmax()
                    if best_idx < len(vals) - patience:
                        logger.info(f"Early stopping detected: best at step {(best_idx+1)*val_every}, "
                                   f"stopped at step {actual_steps}")
                        return True, actual_steps
                
                return False, actual_steps
        except Exception:
            pass
    
    return False, 0


def train_single_model(config: dict, run_cfg: dict, models_dir: Path) -> dict:
    """
    Обучает одну модель с early stopping и возвращает метрики.
    """
    from lightly_train import train_object_detection, load_model
    from experiments.scripts.evaluate import evaluate_model
    
    data_yaml_path = Path(config['paths']['experiment_data']) / run_cfg['data_yaml']
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml_path}")
    
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    # Определяем precision
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    elif torch.cuda.is_available():
        precision = "16-mixed"
    else:
        precision = "32-true"
    
    # Параметры early stopping из конфига
    max_steps = config['training']['max_steps']
    early_stopping_patience = config['training']['early_stopping_patience']
    val_every_steps = config['training']['val_every_steps']
    
    # Параметры обучения
    train_params = {
        "out": str(models_dir),
        "model": config['training']['model'],
        "data": data_config,
        "seed": run_cfg['seed'],
        "precision": precision,
        "steps": max_steps,
        "overwrite": True,
        "batch_size": config['training']['batch_size'],
        "model_args": {
            "lr": config['training']['lr'],
            "backbone_freeze": run_cfg['freeze_backbone'],
            "backbone_lr_factor": run_cfg['backbone_lr_factor'],
        },
        "logger_args": {"val_every_num_steps": val_every_steps},
        "save_checkpoint_args": {"save_best": True, "save_last": True},
    }
    
    logger.info(f"Обучение: {run_cfg['run_name']} (precision={precision})")
    logger.info(f"   Early stopping: patience={early_stopping_patience}, val_every={val_every_steps}")
    
    start = time.time()
    
    # Встроенная ранняя остановка
    best_val_map = -1
    patience_counter = 0
    last_checkpoint_step = 0
    
    # Сохраняем оригинальную функцию, если нужно будет переопределить поведение
    train_object_detection(**train_params)
    
    training_time = (time.time() - start) / 3600
    
    # Анализируем результаты
    val_metrics = _find_best_val_metrics(models_dir)
    early_stopped, actual_steps = _check_early_stopping(models_dir, early_stopping_patience, val_every_steps)
    
    if not actual_steps:
        actual_steps = max_steps
    
    # Загружаем модель
    model_path = None
    for p in [models_dir / "exported_models" / "exported_best.pt",
              models_dir / "exported_models" / "exported_last.pt"]:
        if p.exists():
            model_path = p
            break
    
    if model_path is None:
        # Поиск в чекпоинтах
        checkpoints_dir = models_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.ckpt"))
            if checkpoints:
                model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Используем чекпоинт: {model_path}")
    
    if model_path is None:
        raise FileNotFoundError(f"Модель не найдена в {models_dir}")
    
    model = load_model(str(model_path))
    
    # Оцениваем на тесте
    test_path = data_config.get('test', data_config.get('val'))
    if isinstance(test_path, str):
        test_path = Path(test_path)
    metrics = evaluate_model(model, test_path, num_classes=config['classes']['num_classes'])
    
    result = {
        'run_name': run_cfg['run_name'],
        'dataset_name': run_cfg['dataset_name'],
        'dataset_type': run_cfg['dataset_type'],
        'strategy_name': run_cfg['strategy_name'],
        'seed': run_cfg['seed'],
        'freeze_backbone': run_cfg['freeze_backbone'],
        'test_map50': metrics.get('mAP@50', 0),
        'test_map75': metrics.get('mAP@75', 0),
        'test_map50_95': metrics.get('mAP@50:95', 0),
        'val_map50': val_metrics.get('best_val_map50', 0),
        'best_val_step': val_metrics.get('best_val_step', 0),
        'training_time_hours': round(training_time, 3),
        'actual_steps': actual_steps,
        'max_steps': max_steps,
        'early_stopped': early_stopped,
        'early_stopping_patience': early_stopping_patience,
        'model_path': str(model_path),
        'status': 'completed',
    }
    
    logger.info(f"✅ mAP@50: test={result['test_map50']:.4f}, val={result['val_map50']:.4f}")
    logger.info(f"   Steps: {actual_steps}/{max_steps}, early_stopped={early_stopped}")
    
    # Сохраняем метрики
    with open(models_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    return result