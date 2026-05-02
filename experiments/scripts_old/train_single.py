#!/usr/bin/env python3
"""Обучение одной модели LT-DETR"""

import json
import logging
import re
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def _parse_train_log(out_dir: Path) -> dict:
    """
    Парсит train.log для извлечения val-метрик.
    
    Ищет вложенные структуры (например, abl_000_r0/ltdetr/train.log).
    """
    log_path = out_dir / "train.log"
    if not log_path.exists():
        candidates = list(out_dir.glob("**/train.log"))
        if candidates:
            log_path = candidates[0]
        else:
            logger.warning(f"train.log не найден в {out_dir}")
            return {}
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Ищем mAP50 в форматах:
        # "val_map50=X.XX", "val/mAP50: X.XX", "val_metric/map: X.XX"
        pattern = r'val[_\s/]*(?:metric/)?(?:map|mAP)50[_\s/]*[:=]\s*([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        if matches:
            values = [float(m) for m in matches if m]
            if values:
                return {
                    'best_val_map50': max(values),
                    'final_val_map50': values[-1],
                    'num_val_rounds': len(values),
                }
        
        # Альтернативный формат: "Step N: ... val map50 X.XX"
        pattern2 = r'Step\s+(\d+).*?val[_\s]*(?:map|mAP)50\D*([0-9]*\.?[0-9]+)'
        matches2 = re.findall(pattern2, content, re.IGNORECASE)
        if matches2:
            valid = [(int(m[0]), float(m[1])) for m in matches2 if m[0] and m[1]]
            if valid:
                best_step, best_val = max(valid, key=lambda x: x[1])
                return {
                    'best_val_map50': best_val,
                    'best_val_step': best_step,
                    'final_val_map50': valid[-1][1],
                    'total_steps_logged': valid[-1][0],
                    'num_val_rounds': len(valid),
                }
        
        logger.warning(f"Не удалось извлечь метрики из {log_path}")
        return {}
        
    except Exception as e:
        logger.warning(f"Ошибка парсинга train.log: {e}")
        return {}


def _find_model_path(out_dir: Path) -> Path:
    """Ищет модель для инференса с учётом вложенных структур."""
    candidates = [
        out_dir / "exported_models" / "exported_best.pt",
        out_dir / "exported_models" / "exported_last.pt",
    ]
    candidates.extend(out_dir.glob("**/exported_models/exported_best.pt"))
    candidates.extend(out_dir.glob("**/exported_models/exported_last.pt"))
    
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Найдена модель: {candidate}")
            return candidate
    
    ckpt_candidates = list(out_dir.glob("**/checkpoints/*.ckpt"))
    if ckpt_candidates:
        latest = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
        logger.info(f"Используем чекпоинт: {latest}")
        return latest
    
    raise FileNotFoundError(f"Модель не найдена в {out_dir}")


def train_single_model(config: dict, run_cfg: dict, models_dir: Path) -> dict:
    """Обучает одну модель LT-DETR и возвращает метрики."""
    from lightly_train import train_object_detection, load_model
    
    # Загружаем YAML с конфигурацией датасета
    data_yaml_path = Path(config['paths']['experiment_data']) / run_cfg['data_yaml']
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml_path}")
    
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    # Добавляем format, если отсутствует (КРИТИЧЕСКИ!)
    if 'format' not in data_config:
        data_config['format'] = 'yolo'
        logger.info("Добавлен format=yolo в data_config")
    
    # Параметры обучения из конфига
    max_steps = config['training']['max_steps']
    lr = config['training']['lr']
    batch_size = config['training']['batch_size']
    model_name = config['training']['model']
    seed = run_cfg['seed']
    
    # Параметры, соответствующие документации API
    train_params = {
        "out": str(models_dir),
        "model": model_name,
        "data": data_config,
        "seed": seed,
        "steps": max_steps,
        "overwrite": True,
        "batch_size": batch_size,
        "model_args": {
            "backbone_freeze": run_cfg.get('freeze_backbone', False),
        },
        "save_checkpoint_args": {
            "save_every_num_steps": config['training'].get('val_every_steps', 500),
        },
    }
    
    logger.info(f"Обучение: {run_cfg['run_name']}")
    logger.info(f"  Модель: {model_name}, steps={max_steps}")
    logger.info(f"  lr={lr}, batch_size={batch_size}, seed={seed}")
    
    start = time.time()
    train_object_detection(**train_params)
    training_time = (time.time() - start) / 3600
    
    time.sleep(1)  # Даём файлам дописаться
    
    # Парсим метрики
    val_metrics = _parse_train_log(models_dir)
    
    # Загружаем модель
    model_path = _find_model_path(models_dir)
    model = load_model(str(model_path))
    
    # Оценка на тесте
    from experiments.scripts.evaluate import evaluate_model
    
    test_path = data_config.get('test', data_config.get('val'))
    if isinstance(test_path, str):
        test_path = Path(test_path)
        if not test_path.is_absolute():
            test_path = Path(data_config.get('path', '')) / test_path
    
    if (test_path / "images").exists():
        test_images = test_path / "images"
        test_labels = test_path / "labels" if (test_path / "labels").exists() else test_path
    else:
        test_images = test_path
        test_labels = test_path
    
    metrics = evaluate_model(
        model,
        test_images=test_images,
        test_labels=test_labels,
        num_classes=config['classes']['num_classes'],
        conf_threshold=config['training'].get('conf_threshold', 0.25),
    )
    
    result = {
        'run_name': run_cfg['run_name'],
        'dataset_name': run_cfg['dataset_name'],
        'dataset_type': run_cfg['dataset_type'],
        'strategy_name': run_cfg['strategy_name'],
        'seed': seed,
        'model_name': model_name,
        'test_map50': metrics.get('mAP_50', 0),
        'test_map75': metrics.get('mAP_75', 0),
        'test_map50_95': metrics.get('mAP_50_95', 0),
        'val_map50': val_metrics.get('best_val_map50', 0),
        'best_val_step': val_metrics.get('best_val_step', 0),
        'final_val_map50': val_metrics.get('final_val_map50', 0),
        'num_val_rounds': val_metrics.get('num_val_rounds', 0),
        'training_time_hours': round(training_time, 3),
        'max_steps': max_steps,
        'model_path': str(model_path),
        'status': 'completed',
    }
    
    # Per-class метрики
    for k, v in metrics.items():
        if k.startswith('cls'):
            result[f'test_{k}'] = v
    
    logger.info(f"✅ mAP@50: test={result['test_map50']:.4f}, val={result['val_map50']:.4f}")
    
    with open(models_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    return result