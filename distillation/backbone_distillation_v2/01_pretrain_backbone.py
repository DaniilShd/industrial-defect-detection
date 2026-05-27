#!/usr/bin/env python3
"""
Предобучение бэкбона ResNet18 через дистилляцию от LTDETR.

Учитель: дообученный LTDETR (DINOv3 backbone)
Ученик: torchvision/resnet18
Метод: distillation (LightlyTrain distillationv3)

ВНИМАНИЕ:
  - LightlyTrain сам загружает веса учителя из teacher_weights
  - Не нужно предварительно извлекать backbone учителя
  - teacher_weights указывает прямо на чекпоинт LTDETR (.ckpt)
  - Method "distillation" = distillationv3 (оптимален для DINOv3 → detection)
  - Добавлен фикс для PyTorch 2.6 (weights_only=True по умолчанию)
"""

import logging
import sys
from pathlib import Path

import torch
import torch.serialization
import yaml
import lightly_train

# ============================================================
# ФИКС ДЛЯ PyTorch 2.6 + LightlyTrain 0.15.0
# LightlyTrain вызывает torch.load(..., weights_only=True),
# но .ckpt файлы содержат DataLoader, который не разрешён.
# Разрешаем его глобально.
# ============================================================
torch.serialization.add_safe_globals([torch.utils.data.DataLoader])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("01_pretrain_backbone.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def count_images(directory: Path) -> int:
    """Подсчёт всех изображений в директории рекурсивно."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(
        len(list(directory.rglob(f"*{e}")))
        for e in exts | {e.upper() for e in exts}
    )


def validate_teacher_weights(teacher_weights: Path) -> bool:
    """
    Проверяет, что файл учителя существует и может быть загружен.
    """
    if not teacher_weights.exists():
        logger.error(f"❌ Файл учителя не найден: {teacher_weights}")
        return False

    try:
        # Для проверки загружаем с weights_only=False
        checkpoint = torch.load(teacher_weights, map_location="cpu", weights_only=False)
        
        if isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state = checkpoint

        logger.info(f"✅ Файл учителя загружен, ключей: {len(state) if isinstance(state, dict) else 'N/A'}")
        logger.info(f"   Размер файла: {teacher_weights.stat().st_size / (1024**2):.1f} MB")
        
        if isinstance(state, dict):
            first_keys = list(state.keys())[:5]
            logger.info(f"   Примеры ключей: {first_keys}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки учителя: {e}")
        return False


def main():
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Конфиг не найден: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pretrain_cfg = cfg["pretrain"]
    teacher_cfg = cfg["teacher"]
    paths_cfg = cfg["paths"]

    out_dir = Path(paths_cfg["pretrain_output"]) / "resnet18_distilled"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем, не завершена ли уже дистилляция
    exported_path = out_dir / "exported_models" / "exported_last.pt"
    if exported_path.exists():
        logger.info(f"✅ Предобученный backbone уже существует: {exported_path}")
        cache_file = Path(paths_cfg["pretrain_output"]) / "pretrained_path.txt"
        cache_file.write_text(str(exported_path))
        return

    # Данные для дистилляции (неразмеченные изображения дефектов)
    unlabeled_path = Path(pretrain_cfg["unlabeled_data"])
    
    if not unlabeled_path.exists() or count_images(unlabeled_path) == 0:
        # Fallback на обучающие изображения детектора
        fallback = Path(cfg["detection"]["data_path"]) / "train" / "images"
        if fallback.exists() and count_images(fallback) > 0:
            logger.warning(f"unlabeled_data не найдены, fallback: {fallback}")
            unlabeled_path = fallback
        else:
            logger.error("❌ Нет изображений для дистилляции!")
            sys.exit(1)

    n_images = count_images(unlabeled_path)
    logger.info(f"Найдено {n_images} изображений для дистилляции")
    if n_images < 1000:
        logger.warning("⚠️  Мало изображений (<1000), качество дистилляции может быть низким")

    # Путь к весам учителя (.ckpt чекпоинт LTDETR)
    teacher_weights = Path(teacher_cfg["teacher_weights"])
    
    # Валидация учителя
    if not validate_teacher_weights(teacher_weights):
        logger.error("❌ Веса учителя невалидны, прерывание")
        sys.exit(1)

    # Параметры метода
    method_args = {
        "teacher": teacher_cfg["base_model"],
        "teacher_weights": str(teacher_weights),
    }

    logger.info("=" * 70)
    logger.info("КОНФИГУРАЦИЯ ДИСТИЛЛЯЦИИ")
    logger.info("=" * 70)
    logger.info(f"Учитель (архитектура): {method_args['teacher']}")
    logger.info(f"Веса учителя (файл):    {method_args['teacher_weights']}")
    logger.info(f"Ученик (архитектура):   torchvision/resnet18")
    logger.info(f"Метод дистилляции:      {pretrain_cfg['method']} (distillationv3)")
    logger.info(f"Данные:                 {unlabeled_path} ({n_images} изображений)")
    logger.info(f"Эпохи:                  {pretrain_cfg['epochs']}")
    logger.info(f"Batch size:             {pretrain_cfg['batch_size']}")
    logger.info(f"Image size:             {pretrain_cfg['image_size']}")
    logger.info(f"Выходная директория:    {out_dir}")
    logger.info("=" * 70)

    try:
        lightly_train.pretrain(
            out=str(out_dir),
            data=str(unlabeled_path),
            model="torchvision/resnet18",
            method=pretrain_cfg["method"],
            method_args=method_args,
            epochs=pretrain_cfg["epochs"],
            batch_size=pretrain_cfg["batch_size"],
            transform_args={"image_size": tuple(pretrain_cfg["image_size"])},
            overwrite=True,
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Дистилляция прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ошибка дистилляции: {e}", exc_info=True)
        sys.exit(1)

    # Проверка результата
    if not exported_path.exists():
        logger.error(f"❌ Экспортированная модель не найдена после дистилляции: {exported_path}")
        logger.error("   Проверьте логи LightlyTrain выше")
        sys.exit(1)

    file_size_mb = exported_path.stat().st_size / (1024 ** 2)
    logger.info(f"✅ Дистилляция завершена успешно!")
    logger.info(f"   Модель студента: {exported_path}")
    logger.info(f"   Размер: {file_size_mb:.1f} MB")

    # Сохраняем путь для следующего этапа
    cache_file = Path(paths_cfg["pretrain_output"]) / "pretrained_path.txt"
    cache_file.write_text(str(exported_path))
    logger.info(f"   Путь сохранён в: {cache_file}")


if __name__ == "__main__":
    main()