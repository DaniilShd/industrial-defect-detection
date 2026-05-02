#!/usr/bin/env python3
"""Обучение YOLOv8 Nano с защитой от переобучения"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _check_yolo_overfitting(results_dir: Path) -> dict:
    """Анализирует кривые обучения YOLO на переобучение."""
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return {'overfitting_detected': False}

    try:
        df = pd.read_csv(csv_path)
        if 'metrics/mAP50(B)' not in df.columns or 'metrics/mAP50-95(B)' not in df.columns:
            return {'overfitting_detected': False}

        val_col = 'metrics/mAP50(B)'
        train_loss_col = 'train/box_loss' if 'train/box_loss' in df.columns else None

        best_idx = df[val_col].idxmax()
        best_val = df[val_col].max()

        # Признаки переобучения
        warning_signs = []

        # 1. Val растёт, потом падает более 2 пунктов
        if best_idx < len(df) - 10:
            final_val = df[val_col].iloc[-10:].mean()
            if best_val - final_val > 0.02:
                warning_signs.append(f"val_map50 упала с {best_val:.4f} до {final_val:.4f}")

        # 2. Train loss продолжает падать при падающей валидации
        if train_loss_col and best_idx < len(df) - 10:
            train_loss_best = df[train_loss_col].iloc[best_idx]
            train_loss_final = df[train_loss_col].iloc[-10:].mean()
            if train_loss_final < train_loss_best * 0.8:
                warning_signs.append("train_loss падает при падающей валидации")

        if warning_signs:
            logger.warning(f"⚠️  Признаки переобучения: {'; '.join(warning_signs)}")

        return {
            'overfitting_detected': len(warning_signs) > 0,
            'warning_signs': warning_signs,
            'best_val_map50': float(best_val),
            'best_val_epoch': int(best_idx),
        }
    except Exception as e:
        logger.warning(f"Не удалось проанализировать YOLO-метрики: {e}")
        return {'overfitting_detected': False}


def train_yolo(config: dict, models_dir: Path) -> dict:
    """Обучает YOLOv8 Nano с early stopping и мониторингом."""
    from ultralytics import YOLO

    data_yaml_path = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "data.yaml"
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)

    yolo_data = {
        'path': str(Path(config['paths']['experiment_data']) / config['teacher']['dataset']),
        'train': data_config.get('train', 'train/images'),
        'val': data_config.get('val', 'val/images'),
        'test': data_config.get('test', 'test/images'),
        'nc': config['classes']['num_classes'],
        'names': list(config['classes']['names'].values()),
    }

    yolo_yaml = models_dir / "yolo_data.yaml"
    with open(yolo_yaml, 'w') as f:
        yaml.dump(yolo_data, f)

    cfg = config['students']['yolo_nano']
    out_dir = models_dir / "yolo_nano"

    model = YOLO(f"{cfg['model']}.pt")
    model.train(
        data=str(yolo_yaml),
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        lr0=cfg['lr'],
        project=str(models_dir),
        name="yolo_nano",
        exist_ok=True,
        # Защита от переобучения
        patience=15,              # Early stopping после 15 эпох без улучшений
        cos_lr=True,              # Cosine annealing LR
        warmup_epochs=3,          # Разогрев learning rate
        close_mosaic=10,          # Отключение mosaic augmentation в конце
        dropout=0.0,              # Dropout (можно увеличить до 0.1 при сильном переобучении)
        weight_decay=0.0005,      # L2-регуляризация
        # Сохранение
        save=True,                # Сохранять чекпоинты
        save_period=10,           # Каждые 10 эпох
    )

    overfitting = _check_yolo_overfitting(out_dir)

    model_path = out_dir / "weights" / "best.pt"
    return {
        "model_path": str(model_path),
        "status": "completed",
        "overfitting": overfitting,
    }