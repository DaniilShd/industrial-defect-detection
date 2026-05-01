#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import json
import yaml
from tqdm import tqdm
from collections import defaultdict
from utils import load_config, ensure_dir, print_section
from utils.rle_utils import create_masks_by_class
from utils.patch_utils import has_black_background, resize_with_padding
from utils.yolo_utils import masks_to_yolo_boxes
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()


def main():
    print_section("ИЗВЛЕЧЕНИЕ ДЕФЕКТНЫХ ПАТЧЕЙ 256×256")
    
    p = cfg['paths']
    pc = cfg['patch']
    
    out_dir = Path(p['defect_patches_dir'])
    img_dir = ensure_dir(out_dir / p['yolo_images_subdir'])
    lbl_dir = ensure_dir(out_dir / p['yolo_labels_subdir'])
    
    df = pd.read_csv(p['train_csv'])
    logger.info(f"CSV загружен: {len(df):,} строк")
    
    masks_by_image = {}
    for img_id, grp in tqdm(df.groupby('ImageId'), desc="Маски"):
        masks_by_image[img_id] = create_masks_by_class(grp)
    
    images = list(Path(p['train_images_dir']).glob("*.jpg"))
    logger.info(f"Изображений: {len(images):,}")
    
    all_patches = []
    class_counts = defaultdict(int)
    total_boxes = 0
    rejected = 0
    
    for img_path in tqdm(images, desc="Патчи"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = masks_by_image.get(img_path.name, {})
        if not masks:
            continue
        
        for x in range(0, 1600 - pc['patch_size'] + 1, pc['stride']):
            patch = img_rgb[:, x:x + pc['patch_size']]
            
            if pc['reject_black'] and has_black_background(patch, pc['black_threshold'], pc['max_black_ratio']):
                rejected += 1
                continue
            
            patch_masks = {}
            total_defect = 0
            for cls_id, full_mask in masks.items():
                pm = full_mask[:, x:x + pc['patch_size']]
                defect = np.sum(pm)
                if defect > 0:
                    patch_masks[cls_id] = pm
                    total_defect += defect
            
            if total_defect < pc['min_defect_area']:
                continue
            
            boxes = masks_to_yolo_boxes(patch_masks, pc['min_box_area'])
            if not boxes:
                continue
            
            if patch.shape[1] != pc['resize_to']:
                patch, boxes = resize_with_padding(patch, boxes, pc['resize_to'])
            
            name = f"{img_path.stem}_x{x}_w{pc['patch_size']}"
            img_save = img_dir / f"{name}.{pc['save_format']}"
            lbl_save = lbl_dir / f"{name}.txt"
            
            img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_save), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3] if pc['save_format'] == 'png' else [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            with open(lbl_save, 'w') as f:
                for box in boxes:
                    yolo_cls = box[0] - 1
                    f.write(f"{yolo_cls} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                    class_counts[box[0]] += 1
                    total_boxes += 1
            
            all_patches.append({
                'image': img_path.name, 'x': x, 'width': pc['patch_size'],
                'defect_area': int(total_defect), 'num_boxes': len(boxes),
                'classes_present': list(set(b[0] for b in boxes)),
                'saved_as': name, 'split': 'train'
            })
    
    # Сохраняем dataset.yaml, метаданные
    yaml_data = {
        'path': str(out_dir.absolute()), 'train': p['yolo_images_subdir'],
        'val': p['yolo_images_subdir'], 'nc': len(cfg['classes']['names']),
        'names': list(cfg['classes']['names'].values())
    }
    with open(out_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    if all_patches:
        pd.DataFrame(all_patches).to_csv(out_dir / 'patches_metadata.csv', index=False)
        with open(out_dir / 'annotations.json', 'w') as f:
            json.dump(all_patches, f, indent=2)
    
    print(f"\nПатчей: {len(all_patches):,} | Bbox: {total_boxes:,} | Отбраковано: {rejected:,}")
    for cls_id in [1, 2, 3, 4]:
        print(f"  {cfg['classes']['names'][cls_id]}: {class_counts[cls_id]:,}")
    
    logger.info(f"Готово: {out_dir}")


if __name__ == "__main__":
    main()