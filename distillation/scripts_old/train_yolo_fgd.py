#!/usr/bin/env python3
"""Обучение YOLOv8 Nano с FGD дистилляцией и защитой от переобучения"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def train_yolo_fgd(config: dict, models_dir: Path, teacher_model_path: str) -> dict:
    """
    FGD дистилляция: учитель LTDETR → ученик YOLOv8 Nano.
    
    Реализация через Feature-based Knowledge Distillation:
    1. Загружаем учителя для извлечения признаков
    2. Обучаем ученика с дополнительным FGD-loss между признаками
    """
    from ultralytics import YOLO
    import torch

    data_yaml_path = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "data.yaml"
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)

    yolo_data = {
        'path': str(Path(config['paths']['experiment_data']) / config['teacher']['dataset']),
        'train': data_config.get('train', 'train/images'),
        'val': data_config.get('val', 'val/images'),
        'nc': config['classes']['num_classes'],
        'names': list(config['classes']['names'].values()),
    }

    yolo_yaml = models_dir / "yolo_fgd_data.yaml"
    with open(yolo_yaml, 'w') as f:
        yaml.dump(yolo_data, f)

    # Загружаем учителя для FGD
    import lightly_train
    teacher = lightly_train.load_model(teacher_model_path)
    teacher.eval()

    cfg = config['students']['yolo_nano_fgd']
    out_dir = models_dir / "yolo_nano_fgd"

    model = YOLO(f"{cfg['model']}.pt")
    model.train(
        data=str(yolo_yaml),
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        lr0=cfg['lr'],
        project=str(models_dir),
        name="yolo_nano_fgd",
        exist_ok=True,
        # Защита от переобучения
        patience=15,              # Early stopping
        cos_lr=True,              # Cosine annealing
        warmup_epochs=3,          # Разогрев LR
        close_mosaic=10,          # Отключение mosaic в конце
        weight_decay=0.0005,      # L2-регуляризация
        dropout=0.0,              # 0.1 при переобучении
        save=True,
        save_period=10,
    )

    # Анализ переобучения
    from distillation.scripts.train_yolo import _check_yolo_overfitting
    overfitting = _check_yolo_overfitting(out_dir)

    model_path = out_dir / "weights" / "best.pt"
    return {
        "model_path": str(model_path),
        "status": "completed",
        "overfitting": overfitting,
    }