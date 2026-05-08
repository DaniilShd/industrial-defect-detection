#!/usr/bin/env python3
"""Проверка датасета balanced_defect_patches"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import ast

def check_dataset():
    """Полная проверка датасета"""
    
    base_path = Path("/app/data/processed/balanced_defect_patches_v2")
    
    print("=" * 70)
    print("📊 ПРОВЕРКА ДАТАСЕТА ПАТЧЕЙ")
    print("=" * 70)
    print(f"Путь: {base_path}")
    print()
    
    # 1. Базовая статистика по сплитам
    print("📁 1. БАЗОВАЯ СТАТИСТИКА")
    print("-" * 70)
    
    total_images = 0
    total_labels = 0
    split_stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        n_images = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        n_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        split_stats[split] = {'images': n_images, 'labels': n_labels}
        total_images += n_images
        total_labels += n_labels
        
        print(f"\n  {split.upper()}:")
        print(f"    Изображения: {n_images}")
        print(f"    Метки: {n_labels}")
        
        if n_images != n_labels:
            print(f"    ⚠️ НЕСООТВЕТСТВИЕ: images ≠ labels!")
    
    print(f"\n  ВСЕГО:")
    print(f"    Изображения: {total_images}")
    print(f"    Метки: {total_labels}")
    
    # 2. Проверка дубликатов патчей
    print("\n" + "=" * 70)
    print("🔍 2. ПРОВЕРКА ДУБЛИКАТОВ ПАТЧЕЙ")
    print("-" * 70)
    
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        patches = [f.stem for f in labels_dir.glob('*.txt')]
        unique = set(patches)
        
        print(f"\n  {split.upper()}:")
        print(f"    Всего: {len(patches)}")
        print(f"    Уникальных: {len(unique)}")
        
        if len(patches) != len(unique):
            duplicates = len(patches) - len(unique)
            print(f"    ⚠️ ДУБЛИКАТОВ: {duplicates}")
            
            counter = Counter(patches)
            print(f"    Примеры повторяющихся патчей:")
            for patch, count in counter.items():
                if count > 1:
                    print(f"      - {patch}: {count} раза")
                    break
        else:
            print(f"    ✅ Дубликатов нет")
    
    # 3. Проверка data leakage
    print("\n" + "=" * 70)
    print("🔒 3. ПРОВЕРКА DATA LEAKAGE")
    print("-" * 70)
    
    img_to_splits = defaultdict(set)
    
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            patch_name = label_file.stem
            # Извлекаем исходное изображение (всё до "_x")
            if '_x' in patch_name:
                orig_img = patch_name.split('_x')[0]
            else:
                orig_img = patch_name
            img_to_splits[orig_img].add(split)
    
    leaked = {img: splits for img, splits in img_to_splits.items() if len(splits) > 1}
    
    if leaked:
        print(f"\n  ⚠️ НАЙДЕНА УТЕЧКА ДАННЫХ!")
        print(f"  Количество изображений в нескольких сплитах: {len(leaked)}")
        print(f"\n  Примеры:")
        for img, splits in list(leaked.items())[:5]:
            print(f"    {img} -> {splits}")
    else:
        print(f"\n  ✅ Утечки данных не обнаружено")
        print(f"  Все {len(img_to_splits)} изображений в одном сплите")
    
    # 4. Анализ классов
    print("\n" + "=" * 70)
    print("🏷️ 4. АНАЛИЗ КЛАССОВ")
    print("-" * 70)
    
    class_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    patches_with_multiple = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            classes_in_patch = set()
            with open(label_file) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    patches_with_multiple += 1
                
                for line in lines:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_stats[class_id][split] += 1
                        classes_in_patch.add(class_id)
    
    print(f"\n  Патчей с несколькими дефектами: {patches_with_multiple}")
    
    print(f"\n  РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
    print(f"  {'Класс':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Всего':<10}")
    print(f"  {'-'*50}")
    
    for class_id in sorted(class_stats.keys()):
        train_c = class_stats[class_id]['train']
        val_c = class_stats[class_id]['val']
        test_c = class_stats[class_id]['test']
        total = train_c + val_c + test_c
        print(f"  {class_id:<10} {train_c:<10} {val_c:<10} {test_c:<10} {total:<10}")
    
    # 5. Проверка RLE файлов
    print("\n" + "=" * 70)
    print("📄 5. ПРОВЕРКА RLE ФАЙЛОВ")
    print("-" * 70)
    
    for split in ['train', 'val', 'test']:
        rle_file = base_path / split / f"{split}_rle.csv"
        if rle_file.exists():
            df = pd.read_csv(rle_file)
            print(f"\n  {split.upper()} RLE:")
            print(f"    Записей: {len(df)}")
            print(f"    Уникальных изображений: {df['ImageId'].nunique()}")
            print(f"    Классы: {sorted(df['ClassId'].unique())}")
        else:
            print(f"\n  {split.upper()}: rle.csv не найден")
    
    # 6. Итоговый вердикт
    print("\n" + "=" * 70)
    print("📋 ИТОГОВЫЙ ВЕРДИКТ")
    print("=" * 70)
    
    issues = []
    
    # Проверяем соответствие images/labels
    for split, stats in split_stats.items():
        if stats['images'] != stats['labels']:
            issues.append(f"В {split}: images ({stats['images']}) ≠ labels ({stats['labels']})")
    
    # Проверяем дубликаты в патчах
    if total_labels != len(set()):
        pass  # уже проверили
    
    # Проверяем утечку
    if leaked:
        issues.append(f"Data leakage: {len(leaked)} изображений в нескольких сплитах")
    
    if issues:
        print("\n  ⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print("  Датасет корректно сформирован для обучения.")


def main():
    check_dataset()


if __name__ == "__main__":
    main()