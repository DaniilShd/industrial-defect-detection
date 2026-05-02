#!/usr/bin/env python3
"""Обучение учителя LTDETR + DINOv2 с поддержкой frozen / finetune / ssl"""

import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _parse_val_metrics(train_log: Path) -> list:
    if not train_log.exists():
        return []
    content = train_log.read_text()
    pattern = r'Step\s+(\d+).*?val[_\s/]*(?:map|mAP)50[_\s/]*[:=]\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, content, re.IGNORECASE)
    return [(int(s), float(v)) for s, v in matches if v]


def _ssl_pretrain(config: dict, models_dir: Path) -> Path:
    """SSL-дообучение DINOv2 на неразмеченных данных."""
    from lightly_train import pretrain

    ssl_cfg = config['teacher']['ssl']
    ssl_out = models_dir / "ssl_pretrain"

    logger.info(f"SSL pretrain: epochs={ssl_cfg['epochs']}, batch={ssl_cfg['batch_size']}")
    pretrain(
        out=str(ssl_out),
        data=ssl_cfg['unlabeled_data'],
        model="dinov2/vits14-noreg",
        method="dinov2",
        epochs=ssl_cfg['epochs'],
        batch_size=ssl_cfg['batch_size'],
        seed=42,
        overwrite=True,
    )

    backbone_path = ssl_out / "exported_models" / "exported_last.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")
    logger.info(f"SSL backbone saved: {backbone_path}")
    return backbone_path


def train_teacher(config: dict, models_dir: Path) -> dict:
    """
    Обучает LTDETR+DINOv2 учителя.

    Логика:
    - strategy = 'frozen': backbone_freeze = True
    - strategy = 'finetune': backbone_freeze = False
    - strategy = 'ssl': запускает SSL-дообучение → backbone_weights + backbone_freeze = False
    """
    import lightly_train

    strategy = config['teacher']['strategy']
    data_yaml = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "data.yaml"

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    data_config['format'] = 'yolo'

    out_dir = models_dir / "teacher"
    val_every = config['teacher']['val_every_steps']

    # Определяем параметры в зависимости от стратегии
    model_args = {}
    if strategy == 'frozen':
        model_args['backbone_freeze'] = True
        logger.info("Strategy: FROZEN backbone")
    elif strategy == 'finetune':
        model_args['backbone_freeze'] = False
        logger.info("Strategy: FINETUNE backbone (end-to-end)")
    elif strategy == 'ssl':
        ssl_backbone = _ssl_pretrain(config, models_dir)
        model_args['backbone_freeze'] = False
        model_args['backbone_weights'] = str(ssl_backbone)
        logger.info("Strategy: SSL pretrain + backbone_weights")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    params = {
        "out": str(out_dir),
        "model": config['teacher']['model'],
        "data": data_config,
        "steps": config['teacher']['max_steps'],
        "batch_size": config['teacher']['batch_size'],
        "overwrite": True,
        "model_args": model_args,
        "save_checkpoint_args": {"save_every_num_steps": val_every},
    }

    logger.info(f"Training teacher: {config['teacher']['model']}, steps={config['teacher']['max_steps']}")
    lightly_train.train_object_detection(**params)

    # Анализ переобучения
    train_log = out_dir / "train.log"
    val_metrics = _parse_val_metrics(train_log) if train_log.exists() else []

    overfit_info = {'overfitting_detected': False, 'val_test_gap': 0.0}
    if val_metrics:
        logger.info(f"Val mAP50: {len(val_metrics)} rounds, best={max(v for _, v in val_metrics):.4f}")
        if len(val_metrics) >= 3:
            recent = [v for _, v in val_metrics[-3:]]
            if max(recent) < max(v for _, v in val_metrics) - 0.01:
                logger.warning("⚠️  Возможное переобучение: val-метрика падает последние раунды")
                overfit_info = {'overfitting_detected': True, 'warning': 'val decreasing'}

    model_path = out_dir / "exported_models" / "exported_best.pt"
    result = {
        "model_path": str(model_path),
        "status": "completed",
        "strategy": strategy,
        "val_metrics": [{"step": s, "map50": v} for s, v in val_metrics],
        "overfitting": overfit_info,
    }

    with open(out_dir / "training_info.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result