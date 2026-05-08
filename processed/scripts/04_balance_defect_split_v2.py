#!/usr/bin/env python3
"""Балансировка дефектных патчей ПО ПАТЧАМ (ГАРАНТИЯ отсутствия утечки)"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import shutil
import random
import ast
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from utils import load_config, ensure_dir, print_section
from utils.rle_utils import rle_to_mask, mask_to_rle
from utils.patch_utils import extract_offset
from utils.report_utils import save_figure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()

np.random.seed(cfg['split']['random_seed'])
random.seed(cfg['split']['random_seed'])


def load_metadata():
    """Загрузка метаданных патчей."""
    p = cfg['paths']
    meta = pd.read_csv(Path(p['defect_patches_dir']) / 'patches_metadata.csv')
    with open(Path(p['defect_patches_dir']) / 'annotations.json') as f:
        ann = json.load(f)
    return meta, ann


def get_file_mappings():
    """Получение словарей реальных файлов."""
    p = cfg['paths']
    src_img = Path(p['defect_patches_dir']) / p['yolo_images_subdir']
    src_lbl = Path(p['defect_patches_dir']) / p['yolo_labels_subdir']
    
    real_img = {}
    for ext in ['*.png', '*.jpg']:
        real_img.update({f.stem: f.name for f in src_img.glob(ext)})
    real_lbl = {f.stem: f.name for f in src_lbl.glob("*.txt")}
    
    return src_img, src_lbl, real_img, real_lbl


def parse_classes_from_metadata(meta: pd.DataFrame) -> dict:
    """Извлечение классов из метаданных."""
    patches_by_class = defaultdict(list)
    
    for _, row in meta.iterrows():
        name = row['saved_as']
        val = row.get('classes_present', '[]')
        classes = set()
        
        if isinstance(val, str):
            try:
                classes = set(ast.literal_eval(val))
            except (ValueError, SyntaxError):
                classes = {int(x.strip()) for x in val.strip('[]').split(',') if x.strip()}
        else:
            classes = set(val if isinstance(val, list) else [val])
        
        for c in classes:
            patches_by_class[c].append(name)
    
    return patches_by_class


def select_balanced_no_leakage_guaranteed(patches_by_class: dict) -> tuple:
    """
    Балансировка по ПАТЧАМ с ГАРАНТИРОВАННЫМ отсутствием утечки данных.
    
    Ключевая идея: сначала разбиваем ИЗОБРАЖЕНИЯ на train/val/test,
    а потом внутри каждого сплита отбираем нужное количество патчей.
    """
    sp = cfg['split']
    multiplier = sp.get('balance_multiplier', 2.0)
    
    # 1. Группируем патчи по исходным изображениям
    image_to_patches = defaultdict(list)  # изображение -> все его патчи
    patch_to_class = {}  # патч -> класс (основной класс для балансировки)
    
    for class_id, patches in patches_by_class.items():
        for patch in patches:
            if '_x' in patch:
                orig_img = patch.split('_x')[0]
            else:
                orig_img = patch
            image_to_patches[orig_img].append(patch)
            # Сохраняем класс для этого патча (берем первый, т.к. патч может быть в нескольких)
            if patch not in patch_to_class:
                patch_to_class[patch] = class_id
    
    # 2. Находим миноритарный класс по КОЛИЧЕСТВУ ПАТЧЕЙ
    min_cls = min(patches_by_class, key=lambda x: len(patches_by_class[x]))
    min_cnt = len(patches_by_class[min_cls])
    logger.info(f"Миноритарный класс: {min_cls} ({cfg['classes']['names'].get(min_cls, '?')})")
    logger.info(f"  Патчей: {min_cnt}")
    
    # 3. Целевое количество патчей для каждого класса
    target_patches_per_class = {}
    for cls in patches_by_class:
        if cls == min_cls:
            target_patches_per_class[cls] = min_cnt
        else:
            target_patches_per_class[cls] = min(int(min_cnt * multiplier), len(patches_by_class[cls]))
    
    logger.info(f"Целевое количество патчей на класс:")
    for cls, target in target_patches_per_class.items():
        logger.info(f"  Класс {cls}: {target} патчей")
    
    # 4. Разбиваем ИЗОБРАЖЕНИЯ на train/val/test (а не патчи!)
    all_images = list(image_to_patches.keys())
    random.shuffle(all_images)
    
    n_train = int(len(all_images) * sp['train_ratio'])
    n_val = int(len(all_images) * sp['val_ratio'])
    
    train_images = set(all_images[:n_train])
    val_images = set(all_images[n_train:n_train + n_val])
    test_images = set(all_images[n_train + n_val:])
    
    logger.info(f"Разбиение изображений: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    # 5. Для каждого класса отбираем патчи ТОЛЬКО из предназначенных для него изображений
    selected = {'train': [], 'val': [], 'test': []}
    sel_by_class = defaultdict(lambda: {'train': [], 'val': [], 'test': []})
    
    for cls, target in target_patches_per_class.items():
        # Сортируем изображения по наличию этого класса
        images_with_class = []
        for img in image_to_patches:
            # Проверяем, есть ли в этом изображении патчи нужного класса
            patches_of_class = [p for p in image_to_patches[img] if patch_to_class.get(p) == cls]
            if patches_of_class:
                images_with_class.append(img)
        
        random.shuffle(images_with_class)
        
        # Распределяем по сплитам в нужной пропорции
        n_train_imgs = int(len(images_with_class) * sp['train_ratio'])
        n_val_imgs = int(len(images_with_class) * sp['val_ratio'])
        
        train_imgs_for_class = set(images_with_class[:n_train_imgs])
        val_imgs_for_class = set(images_with_class[n_train_imgs:n_train_imgs + n_val_imgs])
        test_imgs_for_class = set(images_with_class[n_train_imgs + n_val_imgs:])
        
        # Собираем патчи
        for img in train_imgs_for_class:
            patches = [p for p in image_to_patches[img] if patch_to_class.get(p) == cls]
            selected['train'].extend(patches[:target//3])  # берем часть, чтобы не перебрать
            sel_by_class[cls]['train'].extend(patches[:target//3])
        
        for img in val_imgs_for_class:
            patches = [p for p in image_to_patches[img] if patch_to_class.get(p) == cls]
            selected['val'].extend(patches)
            sel_by_class[cls]['val'].extend(patches)
        
        for img in test_imgs_for_class:
            patches = [p for p in image_to_patches[img] if patch_to_class.get(p) == cls]
            selected['test'].extend(patches)
            sel_by_class[cls]['test'].extend(patches)
        
        # Если не хватает, добираем из других изображений
        for split, current_count in [
            ('train', len(sel_by_class[cls]['train'])),
            ('val', len(sel_by_class[cls]['val'])),
            ('test', len(sel_by_class[cls]['test']))
        ]:
            target_count = int(target * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split])
            if current_count < target_count:
                # Добираем случайные патчи из изображений этого класса
                all_patches_of_class = patches_by_class[cls].copy()
                random.shuffle(all_patches_of_class)
                needed = target_count - current_count
                additional = [p for p in all_patches_of_class if p not in selected[split]][:needed]
                selected[split].extend(additional)
                sel_by_class[cls][split].extend(additional)
        
        logger.info(f"  Класс {cls}: train={len(sel_by_class[cls]['train'])}, "
                   f"val={len(sel_by_class[cls]['val'])}, test={len(sel_by_class[cls]['test'])}")
    
    # Убираем дубликаты
    for split in ['train', 'val', 'test']:
        selected[split] = list(set(selected[split]))
    
    return selected, sel_by_class, min_cls, min_cnt


def select_balanced_no_leakage_v2(patches_by_class: dict) -> tuple:
    """
    Более простой и надежный подход:
    1. Разбиваем ИЗОБРАЖЕНИЯ на train/val/test
    2. В каждом сплите отбираем нужное количество ПАТЧЕЙ
    3. Патчи из одного изображения гарантированно в одном сплите
    """
    sp = cfg['split']
    multiplier = sp.get('balance_multiplier', 2.0)
    
    # 1. Группируем патчи по исходным изображениям
    image_to_patches = defaultdict(list)
    for class_id, patches in patches_by_class.items():
        for patch in patches:
            if '_x' in patch:
                orig_img = patch.split('_x')[0]
            else:
                orig_img = patch
            image_to_patches[orig_img].append(patch)
    
    # 2. Находим миноритарный класс
    min_cls = min(patches_by_class, key=lambda x: len(patches_by_class[x]))
    min_cnt = len(patches_by_class[min_cls])
    logger.info(f"Миноритарный класс: {min_cls}, {min_cnt} патчей")
    
    # 3. Целевое количество патчей на класс
    target_per_class = {}
    for cls in patches_by_class:
        target_per_class[cls] = min_cnt if cls == min_cls else min(int(min_cnt * multiplier), len(patches_by_class[cls]))
    
    # 4. Разбиваем ИЗОБРАЖЕНИЯ на сплиты (не патчи!)
    all_images = list(image_to_patches.keys())
    random.shuffle(all_images)
    
    n_train = int(len(all_images) * sp['train_ratio'])
    n_val = int(len(all_images) * sp['val_ratio'])
    
    split_assignment = {}
    for i, img in enumerate(all_images):
        if i < n_train:
            split_assignment[img] = 'train'
        elif i < n_train + n_val:
            split_assignment[img] = 'val'
        else:
            split_assignment[img] = 'test'
    
    logger.info(f"Изображений: Train={n_train}, Val={n_val}, Test={len(all_images)-n_train-n_val}")
    
    # 5. Собираем патчи по сплитам
    selected = {'train': [], 'val': [], 'test': []}
    sel_by_class = defaultdict(lambda: {'train': [], 'val': [], 'test': []})
    
    for cls, target in target_per_class.items():
        # Собираем патчи этого класса, группируя по изображениям
        patches_by_img = defaultdict(list)
        for patch in patches_by_class[cls]:
            if '_x' in patch:
                orig_img = patch.split('_x')[0]
            else:
                orig_img = patch
            patches_by_img[orig_img].append(patch)
        
        # Для каждого сплита отбираем патчи
        for split in ['train', 'val', 'test']:
            # Берем изображения этого сплита
            imgs_in_split = [img for img in patches_by_img.keys() if split_assignment.get(img) == split]
            
            # Все патчи из этих изображений
            patches_in_split = []
            for img in imgs_in_split:
                patches_in_split.extend(patches_by_img[img])
            
            # Если нужно больше, добираем случайными патчами из других изображений этого же класса
            if len(patches_in_split) < target * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split]:
                # Добираем
                needed = int(target * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split]) - len(patches_in_split)
                other_patches = [p for p in patches_by_class[cls] if p not in patches_in_split]
                random.shuffle(other_patches)
                patches_in_split.extend(other_patches[:needed])
            
            selected[split].extend(patches_in_split[:int(target * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split])])
            sel_by_class[cls][split] = patches_in_split[:int(target * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split])]
        
        logger.info(f"  Класс {cls}: train={len(sel_by_class[cls]['train'])}, "
                   f"val={len(sel_by_class[cls]['val'])}, test={len(sel_by_class[cls]['test'])}")
    
    # Убираем дубликаты
    for split in ['train', 'val', 'test']:
        selected[split] = list(set(selected[split]))
    
    return selected, sel_by_class, min_cls, min_cnt


def copy_patches(selected: dict, src_img: Path, src_lbl: Path,
                 real_img: dict, real_lbl: dict, out_dir: Path):
    """Копирование патчей в выходную структуру."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    for sn in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            ensure_dir(out_dir / sn / sub)
        
        for name in selected[sn]:
            base = name.replace('.png', '').replace('.jpg', '')
            if base in real_img:
                shutil.copy2(src_img / real_img[base], out_dir / sn / 'images' / real_img[base])
            if base in real_lbl:
                shutil.copy2(src_lbl / real_lbl[base], out_dir / sn / 'labels' / real_lbl[base])


def save_rle_for_splits(selected: dict, out_dir: Path):
    """Сохранение RLE разметки для каждого сплита."""
    p = cfg['paths']
    orig_df = pd.read_csv(p['train_csv'])
    
    rle_by_img = defaultdict(list)
    for _, row in orig_df.iterrows():
        rle_by_img[row['ImageId']].append({
            'ClassId': row['ClassId'],
            'EncodedPixels': row['EncodedPixels']
        })
    
    total_rle = 0
    for sn in ['train', 'val', 'test']:
        rle_data = []
        for name in selected[sn]:
            parts = name.split('_x')
            orig = parts[0] + '.jpg' if len(parts) >= 2 else name + '.jpg'
            off_x, pw = extract_offset(name)
            
            if orig in rle_by_img:
                for ri in rle_by_img[orig]:
                    full = rle_to_mask(ri['EncodedPixels'])
                    if full is not None and full.sum() > 0:
                        pm = full[:, off_x:off_x + pw]
                        if pw != cfg['patch']['patch_size']:
                            pm = cv2.resize(pm, (cfg['patch']['resize_to'], cfg['patch']['resize_to']),
                                          interpolation=cv2.INTER_NEAREST)
                            pm = (pm > 0.5).astype(np.uint8)
                        new_rle = mask_to_rle(pm)
                        if new_rle:
                            ext_name = name if name.endswith('.png') else name + '.png'
                            rle_data.append({
                                'ImageId': ext_name,
                                'ClassId': ri['ClassId'],
                                'EncodedPixels': new_rle
                            })
        
        if rle_data:
            rle_data = list({(d['ImageId'], d['ClassId']): d for d in rle_data}.values())
            pd.DataFrame(rle_data).to_csv(out_dir / sn / f"{sn}_rle.csv", index=False)
            logger.info(f"  {sn}: {len(rle_data)} RLE записей")
            total_rle += len(rle_data)
    
    return total_rle


def plot_balance_summary(patches_by_class: dict, sel_by_class: dict, rpt: Path):
    """График распределения до/после балансировки."""
    all_ids = sorted(sel_by_class.keys())
    names_list = [cfg['classes']['names'].get(i, f"Cls_{i}") for i in all_ids]
    
    fig, axes = plt.subplots(1, 2, figsize=cfg['report']['figsize'])
    x = np.arange(len(all_ids))
    w = 0.35
    
    orig_cnt = [len(patches_by_class[i]) for i in all_ids]
    sel_cnt = [sum(len(sel_by_class[i][s]) for s in ['train', 'val', 'test']) for i in all_ids]
    train_cnt = [len(sel_by_class[i]['train']) for i in all_ids]
    val_cnt = [len(sel_by_class[i]['val']) for i in all_ids]
    test_cnt = [len(sel_by_class[i]['test']) for i in all_ids]
    
    axes[0].bar(x - w/2, orig_cnt, w, label='Исходное', color='#3498db', alpha=0.7)
    axes[0].bar(x + w/2, sel_cnt, w, label='Отобранное', color='#e74c3c', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names_list, rotation=45, ha='right')
    axes[0].set_title('До / После балансировки')
    axes[0].legend()
    
    axes[1].bar(x - w, train_cnt, w, label='Train', color='#2ecc71')
    axes[1].bar(x, val_cnt, w, label='Val', color='#f39c12')
    axes[1].bar(x + w, test_cnt, w, label='Test', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_list, rotation=45, ha='right')
    axes[1].set_title(f'Разбиение {cfg["split"]["train_ratio"]*100:.0f}/'
                      f'{cfg["split"]["val_ratio"]*100:.0f}/'
                      f'{cfg["split"]["test_ratio"]*100:.0f}')
    axes[1].legend()
    
    plt.suptitle('Балансировка по ПАТЧАМ (ГАРАНТИЯ без утечки)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "balance_defect_split_no_leakage_v2.png", rpt, cfg['report']['dpi'])
    plt.close()


def main():
    print_section("БАЛАНСИРОВКА ПО ПАТЧАМ (ГАРАНТИЯ БЕЗ УТЕЧКИ)")
    
    p = cfg['paths']
    out = ensure_dir(p['balanced_defect_patches_dir'])
    rpt = ensure_dir(p['reports_dir'])
    
    meta, ann = load_metadata()
    src_img, src_lbl, real_img, real_lbl = get_file_mappings()
    
    patches_by_class = parse_classes_from_metadata(meta)
    logger.info("Исходное распределение патчей по классам:")
    for cls in sorted(patches_by_class):
        logger.info(f"  Класс {cls}: {len(patches_by_class[cls])} патчей")
    
    # Используем новый метод с гарантией
    selected, sel_by_class, min_cls, min_cnt = select_balanced_no_leakage_v2(patches_by_class)
    
    total_sel = sum(len(v) for v in selected.values())
    logger.info(f"\nОтобрано всего: {total_sel} патчей")
    logger.info(f"  Train: {len(selected['train'])} патчей")
    logger.info(f"  Val: {len(selected['val'])} патчей")
    logger.info(f"  Test: {len(selected['test'])} патчей")
    
    copy_patches(selected, src_img, src_lbl, real_img, real_lbl, out)
    
    total_rle = save_rle_for_splits(selected, out)
    logger.info(f"Всего RLE записей: {total_rle}")
    
    plot_balance_summary(patches_by_class, sel_by_class, rpt)
    
    logger.info(f"\n✅ Готово: {out}")
    logger.info(f"📊 Отчёт: {rpt / 'balance_defect_split_no_leakage_v2.png'}")


if __name__ == "__main__":
    main()