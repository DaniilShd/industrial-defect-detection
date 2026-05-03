# generate/testing_lora/generate_lora_synthetic.py
#!/usr/bin/env python3
"""Генератор синтетики с LoRA-дообученной SD"""

import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.blending import apply_multiscale_blend
from utils.color_correction import adaptive_color_correction
from utils.rle_utils import rle_to_defect_bboxes
from utils.scaling import scale_defect_and_mask

logger = logging.getLogger(__name__)


class LoRADefectGenerator:
    def __init__(self, config, lora_weights_path: str):
        self.config = config
        from sd_generator_lora import SDDefectGeneratorLoRA
        self.sd_generator = SDDefectGeneratorLoRA(config, lora_weights_path)
        
        random.seed(config['generation']['random_seed'])
        np.random.seed(config['generation']['random_seed'])
    
    def generate_dataset(self, input_dir: Path, rle_csv: Path, output_dir: Path, variants: int = 3):
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')
        
        all_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in input_dir.glob(ext):
                all_images[img.name] = img
                all_images[img.stem] = img
        
        stats = {'total': 0, 'errors': 0}
        gen_cfg = self.config['generation']
        
        for image_id, group in tqdm(groups, desc="LoRA SD generation"):
            img_path = all_images.get(image_id)
            if img_path is None:
                continue
            
            original = cv2.imread(str(img_path))
            if original is None:
                continue
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            img_h, img_w = original.shape[:2]
            
            all_bboxes = []
            for _, row in group.iterrows():
                rle = row.get('EncodedPixels')
                if pd.isna(rle) or str(rle).strip().lower() in ['', 'nan']:
                    continue
                all_bboxes.extend(rle_to_defect_bboxes(str(rle), int(row['ClassId']) - 1))
            
            if not all_bboxes:
                continue
            
            for v in range(variants):
                try:
                    # Генерация фона
                    original_pil = Image.fromarray(original)
                    background = self.sd_generator.generate(
                        original_pil,
                        prompt="industrial steel surface, brushed metal, uniform texture",
                        negative="defect, scratch, crack, rust",
                        strength=0.08,
                        steps=gen_cfg['sd_steps'],
                        guidance=gen_cfg['sd_guidance_scale'],
                        seed=gen_cfg['random_seed'] + stats['total'] * 1000 + v * 100
                    )
                    background = np.clip(background, 0, 255).astype(np.uint8)
                    result = background.copy().astype(np.float32)
                    
                    yolo_annotations = []
                    
                    for bbox in all_bboxes:
                        # ... (аналогично оригинальному process_image_all_defects)
                        pass
                    
                    # Сохранение
                    filename = f"lora_syn_{stats['total']:06d}_{img_path.stem}_v{v}.png"
                    cv2.imwrite(
                        str(output_dir / "images" / filename),
                        cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    )
                    
                    stats['total'] += 1
                    
                except Exception as e:
                    logger.error(f"Error: {e}")
                    stats['errors'] += 1
                
                if stats['total'] % 10 == 0:
                    torch.cuda.empty_cache()
        
        logger.info(f"✅ Generated {stats['total']} images, {stats['errors']} errors")
        return stats['total']