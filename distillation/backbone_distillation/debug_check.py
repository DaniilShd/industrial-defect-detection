#!/usr/bin/env python3
"""Проверка наличия всех файлов и путей."""
import yaml
from pathlib import Path

with open('config_pretrain_comparison.yaml') as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("ПРОВЕРКА ПУТЕЙ")
print("=" * 60)

# Проверка учителя
teacher_weights = config['teacher'].get('weights')
print(f"\nTeacher weights path: {teacher_weights}")
if teacher_weights:
    tp = Path(teacher_weights)
    print(f"  Exists: {tp.exists()}")
    if tp.exists():
        print(f"  Size: {tp.stat().st_size / (1024**2):.1f} MB")

# Проверка данных
data_path = Path(config['detection']['data_path'])
print(f"\nDataset path: {data_path}")
print(f"  Exists: {data_path.exists()}")
for split in ['train', 'val', 'test']:
    split_path = data_path / split / 'images'
    if split_path.exists():
        num_files = len(list(split_path.glob("*")))
        print(f"  {split}: {num_files} images ✓")
    else:
        print(f"  {split}: ✗ (not found)")

# Проверка обученных моделей
detection_output = Path(config['paths']['detection_output'])
print(f"\nDetection output: {detection_output}")
print(f"  Exists: {detection_output.exists()}")
for student_name in config['students'].keys():
    model_path = detection_output / student_name / 'model_final.pth'
    hist_path = detection_output / student_name / 'training_results.json'
    print(f"  {student_name}:")
    print(f"    model_final.pth: {'✓' if model_path.exists() else '✗'}")
    print(f"    training_results.json: {'✓' if hist_path.exists() else '✗'}")

# Проверка предобученного бэкбона
cache_file = Path(config['paths']['pretrain_output']) / "pretrained_model_path.txt"
print(f"\nPretrained backbone cache: {cache_file}")
print(f"  Exists: {cache_file.exists()}")
if cache_file.exists():
    cached_path = cache_file.read_text().strip()
    print(f"  Cached path: {cached_path}")
    print(f"  Cached file exists: {Path(cached_path).exists()}")
else:
    # Ищем напрямую
    pretrain_dir = Path(config['paths']['pretrain_output']) / "resnet18_distilled"
    exported = pretrain_dir / "exported_models" / "exported_last.pt"
    print(f"  Looking for: {exported}")
    print(f"  Found: {exported.exists()}")

# Проверка отчётов
results_dir = Path(config['paths']['results_dir'])
print(f"\nResults directory: {results_dir}")
print(f"  Exists: {results_dir.exists()}")
eval_file = results_dir / 'evaluation_results.json'
print(f"  evaluation_results.json: {'✓' if eval_file.exists() else '✗'}")

# Проверка наличия teacher_backbone_weights
teacher_bb = Path("/app/backbone_distillation/teacher_backbone_weights1.pt")
print(f"\nTeacher backbone weights: {teacher_bb}")
print(f"  Exists: {teacher_bb.exists()}")
if teacher_bb.exists():
    print(f"  Size: {teacher_bb.stat().st_size / (1024**2):.1f} MB")