#!/usr/bin/env python3
"""
Главный скрипт для последовательного обучения двух учителей:
  1. teacher_mixed_full_ssl:      mixed_full + SSL дообучение бэкбона
  2. teacher_real_augmented_finetune: real_augmented + finetune (без SSL)

Запускает каждый скрипт как отдельный процесс (исполняемый файл).
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "teacher_training_v3"


def run_script(script_path: Path) -> bool:
    """Запускает Python-скрипт и возвращает True если успешно."""
    logger.info(f"Запуск: {script_path.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,  # Вывод в реальном времени
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"❌ Скрипт {script_path.name} завершился с ошибкой (код {result.returncode})")
        return False
    return True


def main():
    config_path = Path(__file__).parent / "config_teacher.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(EXPERIMENT_NAME)

    report_dir = Path(config['paths']['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    script_dir = Path(__file__).parent

    # Список скриптов для запуска
    scripts = [
        script_dir / "train_teacher_mixed_full.py",
        script_dir / "train_teacher_real_aug.py",
    ]

    results = []

    with mlflow.start_run(run_name="all_teachers_training"):
        mlflow.log_param("experiment", EXPERIMENT_NAME)
        mlflow.log_param("total_teachers", len(scripts))
        mlflow.log_param("model", config['training']['model'])

        for i, script_path in enumerate(scripts, 1):
            logger.info("\n" + "=" * 80)
            logger.info(f"🎓 ЗАПУСК УЧИТЕЛЯ {i}/{len(scripts)}: {script_path.stem}")
            logger.info("=" * 80)

            success = run_script(script_path)
            results.append({
                'script': script_path.stem,
                'success': success,
            })

        total_time = (time.time() - total_start) / 3600

        summary = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_time_hours': round(total_time, 2),
            'scripts': results,
        }

        summary_path = report_dir / f"summary_all_teachers_{summary['timestamp']}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(str(summary_path))

        # Финальный вывод
        logger.info("\n" + "=" * 80)
        logger.info("📊 СВОДКА ЗАПУСКА УЧИТЕЛЕЙ")
        logger.info("=" * 80)

        for r in results:
            status = "✅ УСПЕШНО" if r['success'] else "❌ ОШИБКА"
            logger.info(f"{r['script']:40s} | {status}")

        logger.info(f"\nОбщее время: {total_time:.1f} ч")
        logger.info(f"Сводный отчет: {summary_path}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()