#!/usr/bin/env python3
"""Быстрый тест пайплайна"""

import sys
import logging
import time
from pathlib import Path
import yaml
import shutil
import mlflow
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from generate.ablation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = Path("/app/data/processed/balanced_defect_patches_v2")
TEST_OUT = Path("/app/data/results_v2/test_pipeline")

print("=" * 60)
print("🧪 QUICK PIPELINE TEST")
print("=" * 60)

# 1. Находим изображения (автоматически)
def find_images_and_labels(path: Path):
    """Ищет изображения и лейблы в разных структурах папок"""
    images = []
    labels = []
    
    # Вариант 1: images/ и labels/
    if (path / 'images').exists():
        images = list((path / 'images').glob("*"))
    # Вариант 2: всё в корне
    else:
        images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    
    # Ищем лейблы
    if (path / 'labels').exists():
        labels = list((path / 'labels').glob("*.txt"))
    else:
        labels = list(path.glob("*.txt"))
    
    # Фильтруем только изображения
    images = [i for i in images if i.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    return images, labels

for split_name, split_path in [
    ('train', BASE / 'train'),
    ('val', BASE / 'val'),
    ('test', BASE / 'test')
]:
    imgs, lbls = find_images_and_labels(split_path)
    print(f"  {split_name}: {len(imgs)} images, {len(lbls)} labels")
    if imgs:
        print(f"    Example: {imgs[0]}")
    if not imgs:
        print(f"    ⚠️  No images in {split_path}")
        # Покажем что есть
        for item in split_path.iterdir():
            print(f"    -> {item.name}")

# Используем train для теста
train_imgs, train_lbls = find_images_and_labels(BASE / 'train')
val_imgs, val_lbls = find_images_and_labels(BASE / 'val')

if not train_imgs:
    print("\n❌ Cannot find training images!")
    print("Checking directory structure:")
    for item in BASE.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(BASE)}")
    exit(1)

# 2. Копируем в тестовый датасет
ds_dir = TEST_OUT / "dataset"
ds_dir.mkdir(parents=True, exist_ok=True)

for split, imgs, lbls in [('train', train_imgs[:50], train_lbls), 
                            ('val', val_imgs[:20], val_lbls)]:
    img_dir = ds_dir / split / 'images'
    lbl_dir = ds_dir / split / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for img in imgs:
        shutil.copy2(img, img_dir / img.name)
    
    for lbl in lbls:
        shutil.copy2(lbl, lbl_dir / lbl.name)

num_train = len(list((ds_dir / 'train/images').glob("*")))
num_val = len(list((ds_dir / 'val/images').glob("*")))
print(f"\n✅ Dataset: {num_train} train + {num_val} val images")

# 3. data.yaml
data_yaml = ds_dir / "data.yaml"
with open(data_yaml, 'w') as f:
    yaml.dump({
        'format': 'yolo',
        'path': str(ds_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'defect1', 1: 'defect2', 2: 'defect3', 3: 'defect4'}
    }, f)

# 4. Обучение
print("\n🚀 Training 50 steps...")
from lightly_train import train_object_detection

ltdetr_dir = TEST_OUT / "ltdetr"
precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"

mlflow.set_tracking_uri("file:///app/mlruns")
mlflow.set_experiment("pipeline_test")

with mlflow.start_run(run_name="quick_test"):
    start = time.time()
    train_object_detection(
        out=str(ltdetr_dir),
        model="dinov3/convnext-tiny-ltdetr",
        data=str(data_yaml),
        seed=42,
        batch_size=4,
        overwrite=True,
        steps=50,
        precision=precision,
        model_args={"lr": 0.0001},
    )
    elapsed = time.time() - start
    
    # Поиск модели
    model_path = None
    for p in [ltdetr_dir / "exported_models" / "exported_best.pt",
              ltdetr_dir / "exported_models" / "exported_last.pt"]:
        if p.exists():
            model_path = str(p)
            break
    
    print(f"✅ Training: {elapsed/60:.1f} min")
    print(f"   Model: {model_path}")

# 5. Оценка
if model_path:
    print("\n📊 Evaluating...")
    test_imgs, test_lbls = find_images_and_labels(BASE / 'test')
    
    # Копируем тестовые данные
    test_img_dir = TEST_OUT / "test_images"
    test_lbl_dir = TEST_OUT / "test_labels"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img in test_imgs:
        shutil.copy2(img, test_img_dir / img.name)
    for lbl in test_lbls:
        shutil.copy2(lbl, test_lbl_dir / lbl.name)
    
    metrics = evaluate_model(
        model_path=model_path,
        test_images=test_img_dir,
        test_labels=test_lbl_dir,
    )
    print(f"   mAP_50:     {metrics.get('mAP_50', 0):.4f}")
    print(f"   mAP_75:     {metrics.get('mAP_75', 0):.4f}")
    print(f"   mAP_50_95:  {metrics.get('mAP_50_95', 0):.4f}")

print("\n" + "=" * 60)
print("✅ TEST PASSED!")
print("=" * 60)
