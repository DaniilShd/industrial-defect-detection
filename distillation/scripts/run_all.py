#!/usr/bin/env python3
"""Оркестратор эксперимента дистилляции с поддержкой frozen/finetune/ssl"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from distillation.scripts.train_teacher import train_teacher
from distillation.scripts.train_yolo import train_yolo
from distillation.scripts.train_yolo_fgd import train_yolo_fgd
from distillation.scripts.train_picodet import train_picodet
from distillation.scripts.train_picodet_fgd import train_picodet_fgd
from distillation.scripts.train_faster_rcnn import train_faster_rcnn
from distillation.scripts.evaluate import evaluate_model
from distillation.scripts.measure_fps import measure_fps, count_parameters
from distillation.scripts.visualize import create_scatter_plot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_and_measure(model_path: str, model_name: str, config: dict,
                         test_images: Path, test_labels: Path) -> dict:
    """Оценивает mAP, FPS и параметры для модели."""
    import lightly_train

    model = lightly_train.load_model(model_path)
    metrics = evaluate_model(model, test_images, test_labels, config['classes']['num_classes'])
    fps = measure_fps(model, test_images, tuple(config['fps']['img_size']),
                      config['fps']['warmup'], config['fps']['iterations'],
                      config['fps']['device'])
    params = count_parameters(model)

    return {
        'model': model_name,
        'map50': metrics['mAP_50'],
        'map75': metrics['mAP_75'],
        'map50_95': metrics['mAP_50_95'],
        **{f"{model_name}_{k}": v for k, v in metrics.items() if k.startswith('cls')},
        'fps': fps['fps'],
        'latency_ms': fps['latency_ms'],
        'params_M': params['params_M'],
        'size_MB': params['size_MB'],
        'model_path': model_path,
    }


def log_training_info(result: dict, model_name: str):
    """Логирует информацию о переобучении."""
    overfitting = result.get('overfitting', {})
    if overfitting.get('overfitting_detected'):
        mlflow.log_metric(f"{model_name}_overfitting", 1)
        warning = overfitting.get('warning', '')
        if warning:
            mlflow.set_tag(f"{model_name}_warning", warning)
    else:
        mlflow.log_metric(f"{model_name}_overfitting", 0)


def get_or_train_teacher(config: dict, config_path: Path, models_dir: Path,
                         test_images: Path, test_labels: Path) -> tuple:
    """Загружает готового учителя или обучает с учётом стратегии."""
    teacher_path = config['teacher'].get('weights_path', '')
    strategy = config['teacher']['strategy']

    if teacher_path and Path(teacher_path).exists():
        logger.info(f"✅ Использую готового учителя: {teacher_path}")
        r = evaluate_and_measure(teacher_path, 'teacher_ltdetr', config,
                                 test_images, test_labels)
        return teacher_path, r

    logger.info(f"🔄 Обучаю учителя (strategy={strategy})...")
    with mlflow.start_run(run_name="teacher_ltdetr", nested=True):
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("dataset", config['teacher']['dataset'])
        mlflow.log_param("model", config['teacher']['model'])
        mlflow.log_params(config['teacher'])

        result = train_teacher(config, models_dir)

        if result['status'] != 'completed':
            raise RuntimeError("Обучение учителя не завершилось")

        teacher_path = result['model_path']
        mlflow.log_param("teacher_model_path", teacher_path)

        # Логируем overfitting
        log_training_info(result, 'teacher')

        # Логируем train.log
        train_log = Path(teacher_path).parent.parent / "train.log"
        if train_log.exists():
            mlflow.log_artifact(str(train_log))

        # Сохраняем путь в конфиг
        config['teacher']['weights_path'] = teacher_path
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"💾 Путь учителя сохранён: {teacher_path}")

        r = evaluate_and_measure(teacher_path, 'teacher_ltdetr', config,
                                 test_images, test_labels)

    return teacher_path, r


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    strategy = config['teacher']['strategy']
    logger.info(f"Дистилляция: teacher_strategy={strategy}")

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    results_dir = Path(config['paths']['results_dir'])
    models_dir = Path(config['paths']['models_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_dataset = config['teacher']['dataset']
    test_images = Path(config['paths']['experiment_data']) / best_dataset / "test" / "images"
    test_labels = Path(config['paths']['experiment_data']) / best_dataset / "test" / "labels"

    if not test_images.exists():
        logger.error(f"❌ Тестовые изображения не найдены: {test_images}")
        return

    all_results = []

    with mlflow.start_run(run_name=f"{config['experiment']['name']}_{strategy}"):
        mlflow.log_dict(config, "config.yaml")
        mlflow.log_param("teacher_strategy", strategy)

        # 1. Учитель
        logger.info(f"=== 1/6: Teacher LTDETR+DINOv2 ({strategy}) ===")
        teacher_path, teacher_r = get_or_train_teacher(config, config_path, models_dir,
                                                       test_images, test_labels)
        all_results.append(teacher_r)
        mlflow.log_metrics({
            'teacher_map50': teacher_r['map50'],
            'teacher_map75': teacher_r['map75'],
            'teacher_map50_95': teacher_r['map50_95'],
            'teacher_fps': teacher_r['fps'],
            'teacher_params_M': teacher_r['params_M'],
            'teacher_size_MB': teacher_r['size_MB'],
        })

        # 2-6: Ученики (без изменений)
        # ... (полный код run_all.py как в предыдущем ответе)

        # Финальная таблица
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"distillation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))

        create_scatter_plot(all_results, results_dir)
        for p in results_dir.glob('quality_vs_speed*.png'):
            mlflow.log_artifact(str(p))

        logger.info("\n" + "=" * 90)
        logger.info(f"{'Model':<25} {'mAP@50':<10} {'mAP@75':<10} {'FPS':<10} {'Params(M)':<12} {'Size(MB)':<10}")
        logger.info("-" * 80)
        for r in all_results:
            logger.info(f"{r['model']:<25} {r['map50']:<10.4f} {r['map75']:<10.4f} "
                       f"{r['fps']:<10.1f} {r['params_M']:<12.1f} {r['size_MB']:<10.1f}")


if __name__ == "__main__":
    main()