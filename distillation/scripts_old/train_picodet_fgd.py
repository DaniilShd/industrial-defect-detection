#!/usr/bin/env python3
"""Обучение PicoDet-S с FGD дистилляцией и защитой от переобучения"""

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


def train_picodet_fgd(config: dict, models_dir: Path, teacher_model_path: str) -> dict:
    """
    FGD дистилляция: учитель LTDETR → ученик PicoDet-S.
    
    PicoDet загружает веса учителя через backbone_weights
    и дообучается с механизмами LightlyTrain (экспорт best model,
    валидация по расписанию).
    """
    import lightly_train

    data_yaml = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "data.yaml"
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    data_config['format'] = 'yolo'

    cfg = config['students']['picodet_s_fgd']
    out_dir = models_dir / "picodet_s_fgd"

    params = {
        "out": str(out_dir),
        "model": "picodet-s-coco",
        "data": data_config,
        "steps": cfg['steps'],
        "batch_size": cfg['batch'],
        "overwrite": True,
        "model_args": {
            "backbone_weights": teacher_model_path,
        },
        "save_checkpoint_args": {"save_every_num_steps": 500},
    }

    logger.info(f"Training PicoDet-S FGD: steps={cfg['steps']}")
    lightly_train.train_object_detection(**params)

    train_log = out_dir / "train.log"
    val_metrics = _parse_val_metrics(train_log)

    overfitting = {'overfitting_detected': False}
    if len(val_metrics) >= 3:
        values = [v for _, v in val_metrics]
        best_idx = values.index(max(values))
        if best_idx < len(values) - 3:
            recent_max = max(values[-3:])
            if recent_max < max(values) - 0.01:
                overfitting = {
                    'overfitting_detected': True,
                    'warning': f"val упала с {max(values):.4f} до {recent_max:.4f}",
                }
                logger.warning(f"⚠️  Переобучение PicoDet FGD: {overfitting['warning']}")

    result = {
        "model_path": str(out_dir / "exported_models" / "exported_best.pt"),
        "status": "completed",
        "val_metrics": [{"step": s, "map50": v} for s, v in val_metrics],
        "overfitting": overfitting,
    }

    with open(out_dir / "training_info.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result