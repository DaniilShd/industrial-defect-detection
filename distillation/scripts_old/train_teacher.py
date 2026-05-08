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
    import lightly_train

    strategy = config['teacher']['strategy']
    data_yaml = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "data.yaml"

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    data_config['format'] = 'yolo'

    out_dir = models_dir / "teacher"

    # Определяем параметры
    model_args = {}
    if strategy == 'frozen':
        model_args['backbone_freeze'] = True
    elif strategy == 'finetune':
        model_args['backbone_freeze'] = False
    elif strategy == 'ssl':
        # 🔥 Для ConvNeXt-Large: distillation, не DINOv2 pretrain!
        from lightly_train import pretrain
        ssl_out = models_dir / "ssl_pretrain"
        unlabeled_path = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "train" / "images"
        
        pretrain(
            out=str(ssl_out),
            data=str(unlabeled_path),
            model="convnext_large",
            method="distillation",
            method_args={"teacher": "dinov2/vitl14"},
            epochs=config['teacher'].get('ssl_epochs', 10),
            batch_size=config['teacher'].get('ssl_batch', 32),
            seed=42,
            overwrite=True,
        )
        model_args['backbone_weights'] = str(ssl_out / "exported_models" / "exported_last.pt")
        model_args['backbone_freeze'] = False

    params = {
        "out": str(out_dir),
        "model": config['teacher']['model'],
        "data": data_config,
        "steps": config['teacher']['max_steps'],
        "batch_size": config['teacher']['batch_size'],
        "overwrite": True,
        "model_args": model_args,
        "save_checkpoint_args": {"save_every_num_steps": config['teacher']['val_every_steps']},
    }

    lightly_train.train_object_detection(**params)

    model_path = out_dir / "exported_models" / "exported_best.pt"
    return {
        "model_path": str(model_path),
        "status": "completed",
        "strategy": strategy,
    }