#!/usr/bin/env python3
"""
05_merge_datasets.py — Сборка финальных датасетов для каждой
конфигурации эксперимента.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset_utils import create_data_yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_one_dataset(
    name: str,
    dataset_cfg: Dict,
    experiment_data: Path,
    output_base: Path,
    yolo_config: Dict,
) -> Path:
    """
    Собирает один датасет из указанных источников.
    """
    output_dir = output_base / name
    description = dataset_cfg.get('description', '')
    size_info = dataset_cfg.get('size', '?')
    
    logger.info(f"\n📦 {name} ({size_info}): {description}")
    
    # Очищаем
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Собираем train
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    
    for source in dataset_cfg['train']:
        src_path = experiment_data / source['source']
        src_images = src_path / 'images'
        src_labels = src_path / 'labels'
        
        if not src_images.exists():
            logger.warning(f"   ⚠️  Источник не найден: {src_images}")
            continue
        
        image_files = [f for f in src_images.glob('*') 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        logger.info(f"   + {source['source']}: {len(image_files)} изображений")
        
        # Копируем с префиксом источника (избегаем конфликтов имён)
        prefix = source['source'].replace('/', '_')
        
        for img_path in tqdm(image_files, desc=f"     {source['source']}", leave=False):
            new_stem = f"{prefix}_{img_path.stem}"
            
            # Копируем изображение
            shutil.copy2(img_path, train_img_dir / f"{new_stem}.jpg")
            
            # Копируем лейбл
            lbl_src = src_labels / f"{img_path.stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, train_lbl_dir / f"{new_stem}.txt")
            
            total_train += 1
    
    # Копируем val
    val_source = experiment_data / dataset_cfg['val']
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    val_images = list((val_source / 'images').glob('*'))
    for img_path in val_images:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        lbl_path = val_source / 'labels' / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy2(lbl_path, val_lbl_dir / lbl_path.name)
    
    # Копируем test
    test_source = experiment_data / dataset_cfg['test']
    test_img_dir = output_dir / "test" / "images"
    test_lbl_dir = output_dir / "test" / "labels"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    test_images = list((test_source / 'images').glob('*'))
    for img_path in test_images:
        shutil.copy2(img_path, test_img_dir / img_path.name)
        lbl_path = test_source / 'labels' / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy2(lbl_path, test_lbl_dir / lbl_path.name)
    
    # Создаём data.yaml
    num_classes = yolo_config['num_classes']
    class_names = {int(k): v for k, v in yolo_config['class_names'].items()}
    
    create_data_yaml(
        dataset_path=output_dir,
        train_dir='train/images',
        val_dir='val/images',
        test_dir='test/images',
        num_classes=num_classes,
        class_names=class_names
    )
    
    logger.info(f"   ✅ train: {total_train}, val: {len(val_images)}, test: {len(test_images)}")
    
    return output_dir


def merge_all_datasets(config: Dict) -> Dict[str, Path]:
    """
    Собирает все конфигурации датасетов.
    
    Returns:
        {имя: путь_к_data.yaml}
    """
    paths = config['paths']
    datasets_cfg = config['datasets']
    yolo_config = config['yolo']
    
    experiment_data = Path(paths['output_dir'])
    datasets_output = experiment_data / "datasets"
    
    datasets_output.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for name, dataset_cfg in datasets_cfg.items():
        output_dir = merge_one_dataset(
            name=name,
            dataset_cfg=dataset_cfg,
            experiment_data=experiment_data,
            output_base=datasets_output,
            yolo_config=yolo_config,
        )
        results[name] = output_dir / "data.yaml"
    
    # Сохраняем информацию
    info = {name: str(yaml_path) for name, yaml_path in results.items()}
    info_path = experiment_data / "datasets_info.yaml"
    with open(info_path, 'w') as f:
        yaml.dump(info, f, default_flow_flow=False)
    
    # Сводка
    logger.info(f"\n{'='*60}")
    logger.info("📊 СОБРАННЫЕ ДАТАСЕТЫ:")
    for name, yaml_path in results.items():
        cfg = datasets_cfg[name]
        logger.info(f"  {name} ({cfg['size']}): {cfg['description']}")
    logger.info(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = merge_all_datasets(config)
    print(f"\n✅ Собрано датасетов: {len(datasets)}")
    for name, path in datasets.items():
        print(f"   {name}: {path}")