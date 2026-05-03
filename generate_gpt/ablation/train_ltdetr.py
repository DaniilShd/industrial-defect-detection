#!/usr/bin/env python3
"""Обучение LT-DETR с early stopping и логированием в MLflow"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
import torch
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def _find_best_val_metrics(out_dir: Path) -> Dict:
    """Извлекает лучшие валидационные метрики из результатов LightlyTrain."""
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
            
            loss_cols = [c for c in df.columns if 'loss' in c.lower() and 'val' in c.lower()]
            if loss_cols:
                best_idx = df[loss_cols[0]].idxmin()
                return {
                    'best_val_loss': float(df[loss_cols[0]].min()),
                    'best_val_step': int(df.iloc[best_idx].get('step', best_idx * 500)),
                    'final_val_loss': float(df[loss_cols[0]].iloc[-1]),
                    'total_steps_trained': int(df.iloc[-1].get('step', len(df) * 500))
                }
        except Exception as e:
            logger.warning(f"Could not parse metrics.csv: {e}")
    
    return {}


def train_ltdetr(
    data_yaml: Path,
    out_dir: Path,
    max_steps: int = 5000,           # Максимальное количество шагов
    early_stopping_patience: int = 3, # Остановка после N валидаций без улучшений
    val_every_steps: int = 500,      # Валидация каждые N шагов
    # min_delta: float = 0.001,        # Минимальное улучшение для сброса patience
    lr: float = 1e-4,
    batch_size: int = 8,
    seed: int = 42,
) -> Dict:
    """
    Обучение LT-DETR с early stopping.
    
    Early stopping: если val-метрика не улучшается patience валидаций подряд,
    обучение останавливается. Это даёт честное сравнение моделей.
    
    Args:
        max_steps: максимальное количество шагов (защита от бесконечного обучения)
        early_stopping_patience: количество валидаций без улучшений до остановки
        val_every_steps: интервал валидации
        min_delta: минимальное улучшение метрики для сброса patience
    """
    from lightly_train import train_object_detection
    
    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"
    
    # Эффективное количество шагов с учётом early stopping
    patience_steps = early_stopping_patience * val_every_steps
    
    params = {
        "out": str(out_dir),
        "model": "dinov3/convnext-tiny-ltdetr-coco",
        "data": str(data_yaml),
        "seed": seed,
        "precision": precision,
        "steps": max_steps,  # Максимум, early stopping остановит раньше
        "overwrite": True,
        "batch_size": batch_size,
        "model_args": {
            "lr": lr,
            "backbone_freeze": True,
            "backbone_lr_factor": 0.1,
        },
        "logger_args": {"val_every_num_steps": val_every_steps},
        "save_checkpoint_args": {"save_best": True, "save_last": True},
    }
    
    logger.info(f"Training LT-DETR: max_steps={max_steps}, "
                f"early_stopping_patience={early_stopping_patience} "
                f"(эффективно: {patience_steps} шагов без улучшений), "
                f"val_every={val_every_steps} steps")
    
    start = time.time()
    train_object_detection(**params)
    elapsed = time.time() - start
    
    # Анализ результатов обучения
    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = next(out_dir.glob("**/metrics.csv"), None)
    
    actual_steps = max_steps
    early_stopped = False
    
    if metrics_csv and metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if not df.empty:
                actual_steps = len(df) * val_every_steps
                
                # Определяем, был ли early stopping
                val_cols = [c for c in df.columns if 'val' in c.lower() and 'map' in c.lower()]
                if val_cols:
                    vals = df[val_cols[0]].values
                    # Проверяем: если последние N значений не улучшались — early stopping сработал
                    if len(vals) >= early_stopping_patience + 1:
                        best_idx = vals.argmax()
                        if best_idx < len(vals) - early_stopping_patience:
                            early_stopped = True
                            logger.info(f"Early stopping detected: best at step {(best_idx+1)*val_every_steps}, "
                                       f"stopped at step {actual_steps}")
        except Exception as e:
            logger.warning(f"Could not analyze metrics: {e}")
    
    # Логирование кривых обучения
    if metrics_csv and metrics_csv.exists():
        mlflow.log_artifact(str(metrics_csv), "training_curves")
        
        # График
        try:
            df = pd.read_csv(metrics_csv)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss
            loss_cols = [c for c in df.columns if 'loss' in c.lower()]
            for col in loss_cols:
                axes[0].plot(df[col], label=col)
            axes[0].set_xlabel('Validation #')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training & Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # mAP
            map_cols = [c for c in df.columns if 'map' in c.lower()]
            for col in map_cols:
                axes[1].plot(df[col], label=col, marker='o')
            axes[1].set_xlabel('Validation #')
            axes[1].set_ylabel('mAP')
            axes[1].set_title(f'Validation mAP (early_stopped={early_stopped})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plot_path = out_dir / "learning_curves.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(str(plot_path), "training_curves")
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
    
    # Val-метрики
    val_metrics = _find_best_val_metrics(out_dir)
    if val_metrics:
        mlflow.log_metrics(val_metrics)
    
    # Поиск модели
    model_path = None
    for p in [out_dir / "exported_models" / "exported_best.pt",
              out_dir / "exported_models" / "exported_last.pt"]:
        if p.exists():
            model_path = str(p)
            break
    
    result = {
        "model_path": model_path,
        "training_time_hours": elapsed / 3600,
        "val_metrics": val_metrics,
        "early_stopped": early_stopped,
        "actual_steps": actual_steps,
        "max_steps": max_steps
    }
    
    mlflow.log_metrics({
        "early_stopped": int(early_stopped),
        "actual_steps": actual_steps,
        "training_time_hours": elapsed / 3600
    })
    
    return result