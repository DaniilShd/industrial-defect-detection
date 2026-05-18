
#!/usr/bin/env python3
"""
run_prepare.py — Оркестратор подготовки данных для эксперимента.
Этапы:
  1. Копирование реальных данных (с ресайзом 640x640)
  2. Копирование синтетики (с ресайзом 640x640)
  3. Аугментация синтетики (1 копия каждого)
  4. Аугментация реальных данных:
     a. Базовая (S копий)
     b. Контролируемая ×3 (для real_augmented_3x)
     c. Массивная (для real_augmented_massive)
  5. Сборка финальных датасетов
  6. Финальная валидация и ресайз
"""

import logging
import sys
from pathlib import Path

import mlflow
import yaml

# Добавляем scripts в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импортируем из конкретных файлов (старый работающий стиль)
from scripts import copy_real
from scripts import copy_synthetic
from scripts import augment_synthetic
from scripts import augment_real
from scripts import augment_real_controlled
from scripts import merge_datasets
from scripts import validate_resize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_augmentation_copies(config: dict) -> tuple:
    """
    Вычисляет количество копий для разных типов аугментации
    на основе реальных данных R и синтетики S.
    
    Returns:
        (copies_for_3x, copies_for_massive, R, S, k)
    """
    paths = config['paths']
    output_dir = Path(paths['output_dir'])
    real_train = output_dir / "real" / "train"
    synth_train = output_dir / "synthetic" / "train"
    
    real_images = list(real_train.glob('images/*.jpg')) + \
                  list(real_train.glob('images/*.jpeg')) + \
                  list(real_train.glob('images/*.png'))
    synth_images = list(synth_train.glob('images/*.jpg')) + \
                   list(synth_train.glob('images/*.jpeg')) + \
                   list(synth_train.glob('images/*.png'))
    
    R = len(real_images)
    S = len(synth_images)
    k = S / R
    
    # real_augmented_3x: размер = 2S = 2kR, копий = 2k-1 на изображение
    copies_3x = max(1, int(2 * k - 1))
    
    # real_augmented_massive: размер = R + 3S = R(1 + 3k), копий = 3k
    copies_massive = max(1, int(3 * k))
    
    return copies_3x, copies_massive, R, S, k


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    logger.info("=" * 60)
    logger.info("🚀 ПОДГОТОВКА ДАННЫХ ДЛЯ ЭКСПЕРИМЕНТА")
    logger.info("=" * 60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # MLflow
    mlflow_cfg = config['mlflow']
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg['experiment_name'])
    
    with mlflow.start_run(run_name=config['experiment']['name']):
        mlflow.log_dict(config, "config.yaml")
        
        # ── Этап 1: Копирование реальных данных ──
        logger.info("\n" + "─" * 40)
        logger.info("📦 ЭТАП 1/7: Копирование реальных данных (640x640)")
        real_paths = copy_real.copy_real_dataset(config)
        mlflow.log_dict(real_paths, "real_paths.yaml")
        
        # ── Этап 2: Копирование синтетики ──
        logger.info("\n" + "─" * 40)
        logger.info("📦 ЭТАП 2/7: Копирование синтетических данных (640x640)")
        try:
            synth_path = copy_synthetic.copy_synthetic_dataset(config)
            mlflow.log_param("synthetic_path", str(synth_path))
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            logger.error("Укажите путь к лучшей синтетике в config.yaml → paths.best_synthetic")
            raise
        
        # Вычисляем параметры аугментации
        copies_3x, copies_massive, R, S, k = compute_augmentation_copies(config)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 СТАТИСТИКА ДАННЫХ:")
        logger.info(f"   Реальных изображений (R): {R}")
        logger.info(f"   Синтетических изображений (S): {S}")
        logger.info(f"   Коэффициент k = S/R: {k:.1f}")
        logger.info(f"   Копий для real_augmented_3x: {copies_3x} на изображение")
        logger.info(f"   Копий для real_augmented_massive: {copies_massive} на изображение")
        logger.info("=" * 60)
        
        mlflow.log_metrics({
            "R": R,
            "S": S,
            "k": k,
            "copies_3x": copies_3x,
            "copies_massive": copies_massive
        })
        
        # ── Этап 3: Аугментация синтетики ──
        logger.info("\n" + "─" * 40)
        logger.info("🎨 ЭТАП 3/7: Аугментация синтетических данных")
        synth_aug_path = augment_synthetic.augment_synthetic_dataset(config)
        mlflow.log_param("synth_augmented_path", str(synth_aug_path) if synth_aug_path else "None")
        
        # ── Этап 4a: Базовая аугментация реальных (S копий) ──
        logger.info("\n" + "─" * 40)
        logger.info("🎨 ЭТАП 4a/7: Базовая аугментация реальных данных (real_augmented)")
        real_aug_path = augment_real.augment_real_dataset(config)
        mlflow.log_param("real_augmented_path", str(real_aug_path) if real_aug_path else "None")
        
        # ── Этап 4b: Контролируемая аугментация ×(2k-1) ──
        logger.info("\n" + "─" * 40)
        logger.info(f"🎨 ЭТАП 4b/7: Аугментация реальных ×{copies_3x} (real_augmented_3x)")
        real_aug_3x_path = augment_real_controlled.augment_real_controlled(
            config,
            copies_per_image=copies_3x,
            output_subdir="real_augmented_3x"
        )
        mlflow.log_param("real_augmented_3x_path", str(real_aug_3x_path) if real_aug_3x_path else "None")
        
        # ── Этап 4c: Массивная аугментация реальных ×3k ──
        logger.info("\n" + "─" * 40)
        logger.info(f"🎨 ЭТАП 4c/7: Массивная аугментация реальных ×{copies_massive} (real_augmented_massive)")
        real_aug_massive_path = augment_real_controlled.augment_real_controlled(
            config,
            copies_per_image=copies_massive,
            output_subdir="real_augmented_massive"
        )
        mlflow.log_param("real_augmented_massive_path", str(real_aug_massive_path) if real_aug_massive_path else "None")
        
        # ── Этап 5: Сборка финальных датасетов ──
        logger.info("\n" + "─" * 40)
        logger.info("📦 ЭТАП 5/7: Сборка финальных датасетов")
        datasets = merge_datasets.merge_all_datasets(config)
        mlflow.log_dict({k: str(v) for k, v in datasets.items()}, "datasets.yaml")
        
        # ── Этап 6: Финальная валидация и ресайз ──
        logger.info("\n" + "─" * 40)
        logger.info("🔍 ЭТАП 6/7: Финальная проверка и ресайз")
        stats = validate_resize.validate_and_fix_all_datasets(config)
        mlflow.log_dict(stats, "validation_stats.yaml")
        
        # Считаем итоговые метрики
        total_images = sum(s['total_images'] for s in stats.values())
        total_resized = sum(s['resized'] for s in stats.values())
        total_fixed = sum(s['fixed_labels'] for s in stats.values())
        total_errors = sum(s['errors'] for s in stats.values())
        all_valid = all(s.get('yolo_valid', False) for s in stats.values())
        
        # Итоги
        output_dir = Path(config['paths']['output_dir'])
        total_size_gb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e9
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
        logger.info(f"   Датасетов:              {len(datasets)}")
        logger.info(f"   Всего изображений:      {total_images}")
        logger.info(f"   Ресайзнуто:              {total_resized}")
        logger.info(f"   Исправлено лейблов:      {total_fixed}")
        logger.info(f"   Ошибок:                  {total_errors}")
        logger.info(f"   Все валидны (YOLO):      {all_valid}")
        logger.info(f"   Общий размер:            {total_size_gb:.2f} ГБ")
        logger.info(f"   Данные:                  {output_dir}")
        logger.info("=" * 60)
        
        # Сводка по датасетам
        logger.info("\n📊 СОБРАННЫЕ ДАТАСЕТЫ:")
        datasets_cfg = config['datasets']
        for name in datasets:
            cfg = datasets_cfg[name]
            logger.info(f"   {name} ({cfg['size']}): {cfg['description']}")
        
        # Проверка ключевых гипотез
        logger.info("\n🎯 ПРОВЕРКА ГИПОТЕЗ:")
        logger.info(f"   1. real_augmented vs real_plus_synthetic_aug (размер R+S={R+S})")
        logger.info(f"      → Синтетика даёт лучшие аугментации, чем аугментация реальных?")
        logger.info(f"   2. real_plus_synthetic_aug vs real_plus_synthetic_original (размер R+S={R+S})")
        logger.info(f"      → Аугментация синтетики даёт прирост?")
        logger.info(f"   3. synthetic_full vs real_augmented_3x (размер 2S={2*S})")
        logger.info(f"      → Синтетика может заменить реальные данные?")
        logger.info(f"   4. mixed_full vs real_augmented_massive (размер R+3S={R+3*S})")
        logger.info(f"      → Синтетика даёт уникальное разнообразие?")
        
        mlflow.log_metrics({
            "num_datasets": len(datasets),
            "total_images": total_images,
            "total_resized": total_resized,
            "total_fixed_labels": total_fixed,
            "total_errors": total_errors,
            "total_size_gb": round(total_size_gb, 2),
            "all_valid": int(all_valid)
        })


if __name__ == "__main__":
    main()
