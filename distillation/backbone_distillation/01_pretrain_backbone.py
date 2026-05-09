#!/usr/bin/env python3
"""
Предобучение бэкбона ResNet18 через дистилляцию знаний от LTDETR-дефектоскописта

Процесс:
  1. Загрузка конфигурации
  2. Проверка неразмеченных данных
  3. Дистилляция через LightlyTrain API (distillationv3)
  4. Сохранение предобученного бэкбона

Учитель: dinov3/convnext-large (230M, дообучен на дефектах)
Ученик: torchvision/resnet18
Метод: distillation (глобальные + локальные признаки)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import lightly_train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('01_pretrain_backbone.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def count_images(directory: Path) -> int:
    """Подсчитывает количество изображений в директории."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f"*{ext}")))
        count += len(list(directory.rglob(f"*{ext.upper()}")))
    return count


def verify_unlabeled_data(data_path: Path) -> bool:
    """Проверяет наличие и объём неразмеченных данных."""
    
    if not data_path.exists():
        logger.error(f"❌ Directory not found: {data_path}")
        return False
    
    num_images = count_images(data_path)
    
    logger.info(f"Found {num_images} images in {data_path}")
    
    if num_images == 0:
        logger.error("❌ No images found!")
        return False
    
    if num_images < 100:
        logger.warning(f"⚠️ Only {num_images} images. Distillation works better with >= 10,000 images")
        logger.warning("Performance may be suboptimal with limited data")
    elif num_images < 1000:
        logger.info(f"🟡 {num_images} images - acceptable for preliminary experiments")
    elif num_images < 10000:
        logger.info(f"🟢 {num_images} images - good for distillation")
    else:
        logger.info(f"✅ {num_images} images - excellent for distillation")
    
    return True


def find_pretrained_model(output_dir: Path) -> Optional[Path]:
    """Ищет уже предобученную модель в разных возможных локациях."""
    
    possible_paths = [
        output_dir / "exported_models" / "exported_last.pt",
        output_dir / "exported_models" / "exported_best.pt",
        output_dir / "checkpoints" / "last.ckpt",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def pretrain_backbone(config: dict) -> str:
    """
    Основная функция предобучения бэкбона.
    
    Returns:
        str: Путь к предобученной модели
    """
    
    pretrain_cfg = config['pretrain']
    teacher_cfg = config['teacher']
    paths_cfg = config['paths']
    
    output_dir = Path(paths_cfg['pretrain_output']) / "resnet18_distilled"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("PHASE 1: BACKBONE PRETRAINING VIA DISTILLATION")
    logger.info("=" * 80)
    logger.info(f"Teacher:      {teacher_cfg['model']}")
    logger.info(f"Teacher desc: {teacher_cfg['description']}")
    logger.info(f"Student:      torchvision/resnet18")
    logger.info(f"Method:       {pretrain_cfg['method']}")
    logger.info(f"Epochs:       {pretrain_cfg['epochs']}")
    logger.info(f"Batch size:   {pretrain_cfg['batch_size']}")
    logger.info(f"Image size:   {pretrain_cfg['image_size']}")
    logger.info(f"Output:       {output_dir}")
    logger.info("=" * 80)
    
    # Проверяем, не обучена ли уже модель
    existing_model = find_pretrained_model(output_dir)
    if existing_model:
        logger.info(f"✅ Pretrained model already exists: {existing_model}")
        logger.info(f"Size: {existing_model.stat().st_size / (1024**2):.1f} MB")
        logger.info("Delete the file to retrain, or continue to next step")
        return str(existing_model)
    
    # Проверяем данные
    unlabeled_path = Path(pretrain_cfg['unlabeled_data'])
    if not verify_unlabeled_data(unlabeled_path):
        # Пробуем fallback - используем train изображения без разметки
        detection_data = Path(config['detection']['data_path'])
        fallback_path = detection_data / "train" / "images"
        
        if fallback_path.exists() and count_images(fallback_path) > 0:
            logger.warning(f"Using training images as unlabeled data: {fallback_path}")
            unlabeled_path = fallback_path
        else:
            raise FileNotFoundError(
                f"No unlabeled data available.\n"
                f"Primary: {unlabeled_path}\n"
                f"Fallback: {fallback_path}\n"
                f"Please add images to continue."
            )
    
    # Настройка метода дистилляции
    method_args = {
        "teacher": teacher_cfg['model'],
    }
    
    # Если есть собственные веса учителя (дообученного на дефектах)
    if teacher_cfg.get('weights') and Path(teacher_cfg['weights']).exists():
        logger.info(f"Using custom teacher weights: {teacher_cfg['weights']}")
        method_args["teacher_weights"] = teacher_cfg['weights']
    else:
        logger.info("Using default pretrained teacher weights")
    
    # Запуск предобучения
    try:
        logger.info("\nStarting distillation pretraining...")
        start_time = datetime.now()
        
        lightly_train.pretrain(
            out=str(output_dir),
            data=str(unlabeled_path),
            model="torchvision/resnet18",
            method=pretrain_cfg['method'],
            method_args=method_args,
            epochs=pretrain_cfg['epochs'],
            batch_size=pretrain_cfg['batch_size'],
            transform_args={
                "image_size": tuple(pretrain_cfg['image_size'])
            },
            loggers={
                "mlflow": {
                    "experiment_name": config['mlflow']['experiment_name'],
                    "run_name": "resnet18_backbone_distillation",
                    "tracking_uri": config['mlflow']['tracking_uri'],
                },
                "tensorboard": {}
            },
        )
        
        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
        logger.info(f"\n✅ Pretraining completed in {elapsed_hours:.2f} hours")
        
    except Exception as e:
        logger.error(f"❌ Pretraining failed: {e}", exc_info=True)
        raise
    
    # Проверяем результат
    exported_model = find_pretrained_model(output_dir)
    if exported_model:
        file_size = exported_model.stat().st_size / (1024**2)
        logger.info(f"\n✅ Backbone pretrained successfully!")
        logger.info(f"   Path: {exported_model}")
        logger.info(f"   Size: {file_size:.1f} MB")
        
        # Логируем содержимое output директории
        logger.info(f"\nOutput directory contents:")
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size / (1024**2)
                logger.info(f"   {item.relative_to(output_dir)} ({size:.1f} MB)")
        
        return str(exported_model)
    else:
        logger.error(f"❌ Model not found after training!")
        logger.error(f"Contents of {output_dir}:")
        for item in output_dir.rglob("*"):
            logger.error(f"   {item}")
        raise FileNotFoundError(f"Expected model not found in {output_dir}")


def main():
    """Точка входа."""
    
    config_path = Path(__file__).parent / "config_pretrain_comparison.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Hypothesis: {config['experiment']['hypothesis']}")
    
    try:
        pretrained_path = pretrain_backbone(config)
        
        # Сохраняем путь для использования в следующих шагах
        cache_dir = Path(config['paths']['pretrain_output'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "pretrained_model_path.txt"
        cache_file.write_text(pretrained_path)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PRETRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Model path: {pretrained_path}")
        logger.info(f"Path cached to: {cache_file}")
        logger.info(f"\nNext step: python 02_train_detectors.py")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pretraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pretraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()