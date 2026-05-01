#!/usr/bin/env python3
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
from utils import load_config, ensure_dir, print_section
from utils.rle_utils import rle_to_mask, mask_to_rle
from utils.patch_utils import extract_offset
from utils.report_utils import save_figure
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()

np.random.seed(cfg['split']['random_seed'])
random.seed(cfg['split']['random_seed'])


def main():
    print_section("БАЛАНСИРОВКА + RLE (70/15/15)")
    
    p = cfg['paths']
    out = ensure_dir(p['balanced_patches_dir'])
    rpt = ensure_dir(p['reports_dir'])
    
    # Загрузка
    meta = pd.read_csv(Path(p['defect_patches_dir']) / 'patches_metadata.csv')
    with open(Path(p['defect_patches_dir']) / 'annotations.json') as f:
        ann = json.load(f)
    
    src_img = Path(p['defect_patches_dir']) / p['yolo_images_subdir']
    src_lbl = Path(p['defect_patches_dir']) / p['yolo_labels_subdir']
    
    real_img = {}
    for ext in ['*.png', '*.jpg']:
        real_img.update({f.stem: f.name for f in src_img.glob(ext)})
    real_lbl = {f.stem: f.name for f in src_lbl.glob("*.txt")}
    
    # Разбор классов
    patches_by_class = defaultdict(list)
    for _, row in meta.iterrows():
        name = row['saved_as']
        val = row.get('classes_present', '[]')
        classes = set()
        if isinstance(val, str):
            try:
                classes = set(ast.literal_eval(val))
            except:
                classes = {int(x.strip()) for x in val.strip('[]').split(',') if x.strip()}
        else:
            classes = set(val if isinstance(val, list) else [val])
        
        for c in classes:
            patches_by_class[c].append(name)
    
    # Миноритарный класс
    min_cls = min(patches_by_class, key=lambda x: len(patches_by_class[x]))
    min_cnt = len(patches_by_class[min_cls])
    logger.info(f"Миноритарный: класс {min_cls} ({min_cnt})")
    
    # Отбор
    selected = {'train': [], 'val': [], 'test': []}
    sel_by_class = defaultdict(lambda: {'train': [], 'val': [], 'test': []})
    
    for cls in patches_by_class:
        patches = patches_by_class[cls]
        random.shuffle(patches)
        target = min_cnt if cls == min_cls else min(min_cnt * 2, len(patches))
        sel = patches[:target]
        random.shuffle(sel)
        
        t_end = int(target * cfg['split']['train_ratio'])
        v_end = t_end + int(target * cfg['split']['val_ratio'])
        
        for sn, sp in zip(['train', 'val', 'test'], [sel[:t_end], sel[t_end:v_end], sel[v_end:]]):
            selected[sn].extend(sp)
            sel_by_class[cls][sn] = sp
    
    # Копирование файлов
    if out.exists():
        shutil.rmtree(out)
    
    for sn in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            ensure_dir(out / sn / sub)
        for name in selected[sn]:
            base = name.replace('.png', '').replace('.jpg', '')
            if base in real_img:
                shutil.copy2(src_img / real_img[base], out / sn / 'images' / real_img[base])
            if base in real_lbl:
                shutil.copy2(src_lbl / real_lbl[base], out / sn / 'labels' / real_lbl[base])
    
    # RLE сохранение
    orig_df = pd.read_csv(p['train_csv'])
    rle_by_img = defaultdict(list)
    for _, row in orig_df.iterrows():
        rle_by_img[row['ImageId']].append({'ClassId': row['ClassId'], 'EncodedPixels': row['EncodedPixels']})
    
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
                        if pw != 256:
                            pm = cv2.resize(pm, (256, 256), interpolation=cv2.INTER_NEAREST)
                            pm = (pm > 0.5).astype(np.uint8)
                        new_rle = mask_to_rle(pm)
                        if new_rle:
                            ext_name = name if name.endswith('.png') else name + '.png'
                            rle_data.append({'ImageId': ext_name, 'ClassId': ri['ClassId'], 'EncodedPixels': new_rle})
        
        if rle_data:
            pd.DataFrame(rle_data).to_csv(out / sn / f"{sn}_rle.csv", index=False)
            logger.info(f"{sn}: {len(rle_data)} RLE")
    
    # График
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
    axes[0].legend()
    
    axes[1].bar(x - w, train_cnt, w, label='Train', color='#2ecc71')
    axes[1].bar(x, val_cnt, w, label='Val', color='#f39c12')
    axes[1].bar(x + w, test_cnt, w, label='Test', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_list, rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    save_figure(fig, "balance_split.png", rpt, cfg['report']['dpi'])
    plt.close()
    
    logger.info(f"Готово: {out}")


if __name__ == "__main__":
    main()