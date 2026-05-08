#!/usr/bin/env python3
"""Обучение PicoDet-S с защитой от переобучения"""

import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _parse_val_metrics(train_log: Path) -> list:
    """Извлекает историю val_map50 из train.log."""
    if not train_log.exists():
        return []
    content = train_log.read_text()
    pattern = r'Step\s+(\d+).*?val[_\s/]*(?:map|mAP)50[_\s/]*[:=]\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, content, re.IGNORECASE)
    return [(int(s), float(v)) for s, v in matches if v]


def _check_overfitting(val_metrics: list) -> dict:
    """Проверяет признаки переобучения."""
    if len(val_metrics) < 3:
        return {'overfitting_detected': False, 'warning': 'insufficient_data'}

    values = [v for _, v in val_metrics]
    best_idx = values.index(max(values))
    best_val = values[best_idx]

    warning_signs = []

    # Падение val после достижения пика
    if best_idx < len(values) - 3:
        recent = values[-3:]
        if max(recent) < best_val - 0.01:
            warning_signs.append(f"val упала с {best_val:.4f} до {max(recent):.4f}")

    # Монотонный рост val (подозрительно хорошо)
    if len(values) >= 5:
        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        if increasing:
            warning_signs.append("Монотонный рост val — возможно переобучение")

    if warning_signs:
        logger.warning(f"⚠️  {'; '.join(warning_signs)}")

    return {
        'overfitting_detected': len(warning_signs) > 0,
        'warning_signs': warning_signs,
        'best_val_map50': best_val,
        'best_val_step': val_metrics[best_idx][0],
    }


def train_picodet(config: dict, models_dir: Path) -> dict:
    """Обучает PicoDet-S с мониторингом переобучения."""
    import lightly_train

    data_yaml = Path(config['paths']['experiment_data']) / config['teacher']['dataset'] / "data.yaml"
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    data_config['format'] = 'yolo'

    cfg = config['students']['picodet_s']
    out_dir = models_dir / "picodet_s"

    params = {
        "out": str(out_dir),
        "model": "picodet-s-coco",
        "data": data_config,
        "steps": cfg['steps'],
        "batch_size": cfg['batch'],
        "overwrite": True,
        "model_args": {},
        "save_checkpoint_args": {"save_every_num_steps": 500},
    }

    logger.info(f"Training PicoDet-S: steps={cfg['steps']}")
    lightly_train.train_object_detection(**params)

    train_log = out_dir / "train.log"
    val_metrics = _parse_val_metrics(train_log)
    overfitting = _check_overfitting(val_metrics)

    result = {
        "model_path": str(out_dir / "exported_models" / "exported_best.pt"),
        "status": "completed",
        "val_metrics": [{"step": s, "map50": v} for s, v in val_metrics],
        "overfitting": overfitting,
    }

    with open(out_dir / "training_info.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result