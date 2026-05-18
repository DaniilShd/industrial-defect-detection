#!/usr/bin/env python3
"""Обучение LT-DETR — общая функция"""

import json
import logging
import re
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def _parse_train_log(out_dir: Path) -> dict:
    """Извлекает валидационные метрики из train.log или metrics.csv"""
    import pandas as pd
    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = next(out_dir.glob("**/metrics.csv"), None)
    
    if metrics_csv and metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if not df.empty:
                val_cols = [c for c in df.columns if 'val' in c.lower() and 'map' in c.lower()]
                if val_cols:
                    vals = df[val_cols[0]].values
                    best_idx = vals.argmax()
                    return {
                        'best_val_map50': float(vals.max()),
                        'best_val_step': int((best_idx + 1) * 500),
                        'final_val_map50': float(vals[-1]),
                        'num_val_rounds': len(vals),
                    }
        except Exception as e:
            logger.warning(f"Не удалось прочитать metrics.csv: {e}")
    
    log_path = out_dir / "train.log"
    if not log_path.exists():
        candidates = list(out_dir.glob("**/train.log"))
        log_path = candidates[0] if candidates else None
    if not log_path:
        logger.warning(f"train.log не найден в {out_dir}")
        return {}

    try:
        content = log_path.read_text()
        pattern = r'val[_\s/]*(?:metric/)?(?:map|mAP)50[_\s/]*[:=]\s*([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            values = [float(m) for m in matches if m]
            if values:
                return {'best_val_map50': max(values), 'final_val_map50': values[-1], 'num_val_rounds': len(values)}
        return {}
    except Exception as e:
        logger.warning(f"Ошибка парсинга train.log: {e}")
        return {}


def _find_model_path(out_dir: Path) -> Path:
    candidates = [
        out_dir / "exported_models" / "exported_best.pt",
        out_dir / "exported_models" / "exported_last.pt",
    ]
    candidates.extend(out_dir.glob("**/exported_models/exported_best.pt"))
    candidates.extend(out_dir.glob("**/exported_models/exported_last.pt"))
    for c in candidates:
        if c.exists():
            return c
    ckpts = list(out_dir.glob("**/checkpoints/*.ckpt"))
    if ckpts:
        return max(ckpts, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Модель не найдена в {out_dir}")


def train_ltdetr(
    config: dict,
    run_cfg: dict,
    models_dir: Path,
    extra_model_args: dict = None,
) -> dict:
    """
    Обучение LT-DETR.
    """
    from lightly_train import train_object_detection, load_model
    from experiments.scripts.evaluate import evaluate_model

    data_yaml_path = Path(config['paths']['experiment_data']) / run_cfg['data_yaml']
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml_path}")

    model_args = {"lr": config['training'].get('lr', 1e-4)}
    # Всегда добавляем ключ backbone_freeze (API требует именно так)
    model_args["backbone_freeze"] = run_cfg.get('freeze_backbone', False)

    if extra_model_args:
        model_args.update(extra_model_args)

    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"

    # ★ ИСПРАВЛЕНИЕ 1: Адаптивные шаги из словаря
    max_steps = config['training']['max_steps']
    if isinstance(max_steps, dict):
        max_steps = max_steps.get(run_cfg['dataset_name'], 7000)
    
    # ★ ИСПРАВЛЕНИЕ 2: gradient_accumulation_steps из конфига
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)

    train_params = {
        "out": str(models_dir),
        "model": config['training']['model'],
        "data": str(data_yaml_path),
        "seed": run_cfg['seed'],
        "precision": precision,
        "steps": max_steps,  # ← теперь адаптивные
        "overwrite": True,
        "batch_size": config['training']['batch_size'],
        "gradient_accumulation_steps": grad_accum,  # ← из конфига
        "model_args": model_args,
        "logger_args": {
            "val_every_num_steps": config['training'].get('val_every_steps', 500),
        },
        "save_checkpoint_args": {
            "save_best": True,
            "save_last": True,
        },
    }

    # Логируем информацию о датасете и эпохах
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    train_path = Path(data_config['path']) / 'train'
    n_images = len(list((train_path / 'images').glob('*')))
    effective_batch = config['training']['batch_size'] * grad_accum
    steps_per_epoch = n_images / effective_batch
    epochs = max_steps / steps_per_epoch
    
    logger.info(f"Датасет: {n_images} img | Батч: {config['training']['batch_size']}×{grad_accum}={effective_batch}")
    logger.info(f"Шагов: {max_steps} | ~{epochs:.1f} эпох | Валидация каждые {train_params['logger_args']['val_every_num_steps']} шагов")
    logger.info(f"Обучение: {run_cfg['run_name']}, backbone_freeze={model_args['backbone_freeze']}")
    
    start = time.time()
    train_object_detection(**train_params)
    training_time = (time.time() - start) / 3600

    val_metrics = _parse_train_log(models_dir)
    model_path = _find_model_path(models_dir)
    model = load_model(str(model_path))

    # Определяем тестовый путь
    test_path = data_config.get('test', data_config.get('val'))
    if isinstance(test_path, str):
        test_path = Path(test_path)
        if not test_path.is_absolute():
            test_path = Path(data_config['path']) / test_path

    if (test_path / "images").exists():
        test_images = test_path / "images"
        test_labels = test_path / "labels"
    else:
        test_images = test_path
        test_labels = test_path.parent / "labels" if (test_path.parent / "labels").exists() else test_path.parent

    logger.info(f"Оценка на тесте: images={test_images}, labels={test_labels}")

    # ★ ИСПРАВЛЕНИЕ 3: Правильный порог для оценки
    eval_conf = config['training'].get('eval_conf_threshold', 0.001)
    
    metrics = evaluate_model(
        model, test_images=test_images, test_labels=test_labels,
        num_classes=config['classes']['num_classes'],
        conf_threshold=eval_conf,  # ← 0.001 для честного сравнения
    )

    result = {
        'run_name': run_cfg['run_name'],
        'dataset_name': run_cfg['dataset_name'],
        'strategy_name': run_cfg['strategy_name'],
        'seed': run_cfg['seed'],
        'test_map50': metrics.get('mAP_50', 0),
        'test_map75': metrics.get('mAP_75', 0),
        'test_map50_95': metrics.get('mAP_50_95', 0),
        'val_map50': val_metrics.get('best_val_map50', 0),
        'best_val_step': val_metrics.get('best_val_step', 0),
        'training_time_hours': round(training_time, 3),
        'model_path': str(model_path),
        'status': 'completed',
        'n_epochs': round(epochs, 1),  # ← для отчётности
        'n_images': n_images,
    }

    for k, v in metrics.items():
        if k.startswith('cls'):
            result[f'test_{k}'] = v

    logger.info(f"mAP@50: test={result['test_map50']:.4f}, val={result['val_map50']:.4f}")
    with open(models_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result