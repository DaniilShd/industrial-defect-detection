#!/usr/bin/env python3
"""Обучение LT-DETR с опциональной заморозкой бэкбона"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

import torch
import yaml

logger = logging.getLogger(__name__)


def train_ltdetr(
    data_yaml: Path,
    out_dir: Path,
    max_steps: int = 5500,
    lr: float = 1e-4,
    batch_size: int = 8,
    seed: int = 42,
    freeze_backbone: bool = False,
    val_every_steps: int = 500,
) -> Dict:
    """
    Обучение LT-DETR.
    
    Args:
        data_yaml: Путь к data.yaml
        out_dir: Директория для результатов
        max_steps: Количество шагов обучения
        lr: Learning rate
        batch_size: Размер батча
        seed: Random seed
        freeze_backbone: Заморозить бэкбон (требуется LightlyTrain >= 0.14.2)
        val_every_steps: Интервал валидации в шагах
    """
    from lightly_train import train_object_detection
    
    # Проверяем версию если нужна заморозка
    if freeze_backbone:
        import lightly_train
        logger.info(f"LightlyTrain version: {lightly_train.__version__}")
        logger.info("Freezing backbone via model_args={'backbone_freeze': True}")
    
    # Загружаем конфигурацию данных
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    use_amp = torch.cuda.is_available()
    precision = None
    if use_amp:
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    
    # 🔥 Правильный способ: заморозка через model_args
    model_args = {"lr": lr}
    if freeze_backbone:
        model_args["backbone_freeze"] = True
    
    train_config = {
        "out": str(out_dir),
        "model": "dinov3/convnext-tiny-ltdetr-coco",
        "data": data_config,
        "seed": seed,
        "batch_size": batch_size,
        "overwrite": True,
        "steps": max_steps,
        "model_args": model_args,
        "logger_args": {
            "val_every_num_steps": val_every_steps,
        },
        "save_checkpoint_args": {
            "save_every_num_steps": val_every_steps,
        },
    }
    
    if precision:
        train_config["precision"] = precision
    
    logger.info(f"Training config: out={out_dir}, steps={max_steps}, lr={lr}, "
                f"batch_size={batch_size}, freeze_backbone={freeze_backbone}, "
                f"val_every={val_every_steps}")
    
    start = time.time()
    train_object_detection(**train_config)
    elapsed = time.time() - start
    
    # Ищем экспортированную модель
    model_path = None
    for p in [
        out_dir / "exported_models" / "exported_best.pt",
        out_dir / "exported_models" / "exported_last.pt",
    ]:
        if p.exists():
            model_path = str(p)
            break
    
    if not model_path:
        for p in out_dir.glob("**/exported_best.pt"):
            model_path = str(p)
            break
    
    result = {
        "model_path": model_path,
        "training_time_hours": elapsed / 3600,
        "max_steps": max_steps,
        "val_every_steps": val_every_steps,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    
    # Сохраняем результат
    result_path = out_dir / "result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Training completed in {elapsed/3600:.2f}h")
    logger.info(f"Model: {model_path}")
    logger.info(f"Result: {result_path}")
    
    return result