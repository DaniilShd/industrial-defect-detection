#!/usr/bin/env python3
"""
Пайплайн обучения LTDETR-модели с выбором стратегии через конфиг.

Стратегии:
  - frozen:   замороженный бэкбон (baseline)
  - finetune: размороженный бэкбон, полное обучение
  - ssl:      SSL дообучение бэкбона → LTDETR с этим бэкбоном
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_teacher")


# ============================================================
# Вспомогательные функции
# ============================================================

def _get_backbone_from_ltdetr(full_model: str) -> str:
    """Извлекает чистый бэкбон из имени LTDETR-модели."""
    mapping = {
        'convnext-large': 'dinov3/convnext-large',
        'convnext-base': 'dinov3/convnext-base',
        'convnext-small': 'dinov3/convnext-small',
        'convnext-tiny': 'dinov3/convnext-tiny',
        'vitt16plus': 'dinov3/vitt16plus',
        'vitt16': 'dinov3/vitt16',
        'vits16plus': 'dinov3/vits16plus',
        'vits16': 'dinov3/vits16',
        'vitb16': 'dinov3/vitb16',
        'vitl16': 'dinov3/vitl16',
    }
    for key, backbone in mapping.items():
        if key in full_model:
            return backbone
    return 'dinov3/convnext-tiny'


def _find_model_path(out_dir: Path) -> Path:
    """Находит лучшую экспортированную модель."""
    candidates = [
        out_dir / "exported_models" / "exported_best.pt",
        out_dir / "exported_models" / "exported_last.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: чекпоинты
    ckpts = list(out_dir.glob("**/checkpoints/*.ckpt"))
    if ckpts:
        return max(ckpts, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Модель не найдена в {out_dir}")


def _read_data_yaml(yaml_path: Path) -> dict:
    """Читает data.yaml и возвращает словарь для LightlyTrain."""
    with open(yaml_path) as f:
        dc = yaml.safe_load(f)
    return {
        "format": "yolo",
        "path": dc['path'],
        "train": dc.get('train', 'images/train'),
        "val": dc.get('val', 'images/val'),
        "names": dc.get('names', {}),
    }


def _run_ssl_pretrain(config: dict, models_dir: Path, seed: int) -> Path:
    """SSL дообучение бэкбона. Возвращает путь к exported_last.pt."""
    from lightly_train import pretrain

    ssl_cfg = config['ssl']
    experiment_data = Path(config['paths']['experiment_data'])
    data_yaml_path = experiment_data / config['dataset']['data_yaml']

    with open(data_yaml_path) as f:
        dc = yaml.safe_load(f)

    unlabeled_path = Path(dc['path']) / dc.get('train', 'images/train') / 'images' \
        if 'images' not in dc.get('train', 'images/train') \
        else Path(dc['path']) / dc['train']

    # Если train = "images/train", идём в images/train/images
    if not unlabeled_path.exists():
        unlabeled_path = Path(dc['path']) / dc.get('train', 'images/train')

    # Берём только папку с картинками
    if (unlabeled_path / 'images').exists():
        unlabeled_path = unlabeled_path / 'images'

    full_model = config['model']['architecture']
    backbone = _get_backbone_from_ltdetr(full_model)

    ssl_out = models_dir / "ssl_pretrain"

    logger.info(f"SSL: teacher={ssl_cfg['teacher']} → student={backbone}")
    logger.info(f"SSL: {len(list(unlabeled_path.glob('*')))} img из {unlabeled_path}")

    pretrain(
        out=str(ssl_out),
        data=str(unlabeled_path),
        model=backbone,
        method=ssl_cfg['method'],
        method_args={"teacher": ssl_cfg['teacher']},
        epochs=ssl_cfg['epochs'],
        batch_size=ssl_cfg['batch_size'],
        precision=config['training'].get('precision', 'bf16-mixed'),
        seed=seed,
        overwrite=True,
    )

    backbone_path = ssl_out / "exported_models" / "exported_last.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")

    return backbone_path


# ============================================================
# Основной пайплайн
# ============================================================

def train_teacher(config: dict):
    """Обучение LTDETR согласно стратегии из конфига."""
    from lightly_train import train_object_detection, load_model
    from experiments.scripts.evaluate import evaluate_model

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    strategy = config['strategy']['name']
    seed = config['training']['seed']

    experiment_data = Path(config['paths']['experiment_data'])
    models_dir = Path(config['paths']['models_dir']) / f"{strategy}_{ts}"
    report_dir = Path(config['paths']['report_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_path = experiment_data / config['dataset']['data_yaml']
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml_path}")

    data_dict = _read_data_yaml(data_yaml_path)

    # MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # =====================
    # Шаг 0: Выбор стратегии
    # =====================
    freeze_backbone = False
    extra_model_args = {}

    logger.info("=" * 60)
    logger.info(f"Стратегия: {strategy}")
    logger.info(f"Модель: {config['model']['architecture']}")
    logger.info("=" * 60)

    if strategy == "frozen":
        freeze_backbone = True
        logger.info("Бэкбон заморожен")

    elif strategy == "finetune":
        freeze_backbone = False
        logger.info("Бэкбон разморожен, полное обучение")

    elif strategy == "ssl":
        freeze_backbone = False
        logger.info("Шаг 0: SSL дообучение бэкбона...")
        backbone_path = _run_ssl_pretrain(config, models_dir, seed)
        extra_model_args = {"backbone_weights": str(backbone_path)}
        logger.info(f"SSL бэкбон: {backbone_path}")

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    # =====================
    # Шаг 1: Обучение LTDETR
    # =====================
    logger.info("=" * 60)
    logger.info("Шаг 1: Обучение LTDETR")
    logger.info("=" * 60)

    with mlflow.start_run(run_name=f"{strategy}_{ts}"):
        mlflow.log_params({
            'strategy': strategy,
            'model': config['model']['architecture'],
            'freeze_backbone': freeze_backbone,
            'seed': seed,
            'dataset': config['dataset']['data_yaml'],
        })

        model_args = {
            "lr": config['training'].get('lr', 1e-4),
        }
        if freeze_backbone:
            model_args["backbone_freeze"] = True
        if extra_model_args:
            model_args.update(extra_model_args)

        precision = config['training'].get('precision', 'bf16-mixed')

        train_params = {
            "out": str(models_dir),
            "model": config['model']['architecture'],
            "data": data_dict,
            "seed": seed,
            "precision": precision,
            "steps": config['training'].get('steps', 'auto'),
            "overwrite": True,
            "batch_size": config['training']['batch_size'],
            "model_args": model_args,
            "logger_args": {
                "val_every_num_steps": config['training'].get('val_every_steps', 500),
            },
            "save_checkpoint_args": {
                "save_best": True,
                "save_last": True,
            },
            "metric_args": {
                "watch_metric": config['metrics'].get('watch_metric', 'val_metric/map'),
            },
        }

        logger.info(f"Параметры: steps={train_params['steps']}, "
                    f"batch_size={train_params['batch_size']}, "
                    f"freeze_backbone={freeze_backbone}")

        t0 = time.time()
        train_object_detection(**train_params)
        training_time = (time.time() - t0) / 3600

        # =====================
        # Шаг 2: Загрузка модели и оценка
        # =====================
        logger.info("=" * 60)
        logger.info("Шаг 2: Оценка модели")
        logger.info("=" * 60)

        model_path = _find_model_path(models_dir)
        model = load_model(str(model_path))

        # Определяем тестовый сплит
        with open(data_yaml_path) as f:
            dc = yaml.safe_load(f)

        test_split = dc.get('test', dc.get('val', 'images/val'))
        test_path = Path(dc['path']) / test_split

        if (test_path / "images").exists():
            test_images = test_path / "images"
            test_labels = test_path / "labels"
        else:
            test_images = test_path
            test_labels = test_path.parent / "labels" \
                if (test_path.parent / "labels").exists() \
                else test_path.parent

        logger.info(f"Тест: images={test_images}, labels={test_labels}")

        eval_conf = config['training'].get('eval_conf_threshold', 0.001)
        num_classes = len(dc.get('names', {})) or 4

        metrics = evaluate_model(
            model,
            test_images=test_images,
            test_labels=test_labels,
            num_classes=num_classes,
            conf_threshold=eval_conf,
        )

        # =====================
        # Шаг 3: Отчёт
        # =====================
        logger.info("=" * 60)
        logger.info("Шаг 3: Формирование отчёта")
        logger.info("=" * 60)

        result = {
            'timestamp': ts,
            'strategy': strategy,
            'model': config['model']['architecture'],
            'freeze_backbone': freeze_backbone,
            'seed': seed,
            'test_map50': metrics.get('mAP_50', 0),
            'test_map75': metrics.get('mAP_75', 0),
            'test_map50_95': metrics.get('mAP_50_95', 0),
            'precision': metrics.get('Precision', 0),
            'recall': metrics.get('Recall', 0),
            'f1': metrics.get('F1', 0),
            'training_time_h': round(training_time, 2),
            'model_path': str(model_path),
        }

        # Per-class метрики
        for k, v in metrics.items():
            if k.startswith('cls'):
                result[f'test_{k}'] = v

        report_path = report_dir / f"report_{strategy}_{ts}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Логи в MLflow
        mlflow_metrics = {
            'test_map50': result['test_map50'],
            'test_map75': result['test_map75'],
            'test_map50_95': result['test_map50_95'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'training_time_h': training_time,
        }
        for k, v in result.items():
            if k.startswith('test_cls'):
                mlflow_metrics[k] = v

        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(report_path))
        mlflow.set_tag("status", "completed")

        # Финальный вывод
        logger.info("=" * 60)
        logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        logger.info(f"Стратегия: {strategy}")
        logger.info(f"mAP@50: {result['test_map50']:.4f}")
        logger.info(f"mAP@50-95: {result['test_map50_95']:.4f}")
        logger.info(f"Время: {training_time:.2f} ч")
        logger.info(f"Модель: {model_path}")
        logger.info(f"Отчёт: {report_path}")
        logger.info("=" * 60)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train_teacher(config)