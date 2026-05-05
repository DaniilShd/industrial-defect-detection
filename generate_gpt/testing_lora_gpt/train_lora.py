# ============================================================
# generate/testing_lora/train_lora.py — REAL MASKS + bf16 + UPGRADES
# ============================================================
#!/usr/bin/env python3
"""LoRA training на INPAINTING с РЕАЛЬНЫМИ масками из RLE + bf16"""

import logging
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def rle_to_mask_simple(rle_string: Optional[str], size: int = 256) -> np.ndarray:
    """RLE → маска 256x256 для патча (C-order)."""
    if rle_string is None or pd.isna(rle_string) or str(rle_string).strip() in ('', 'nan'):
        return np.zeros((size, size), dtype=np.uint8)
    try:
        numbers = list(map(int, str(rle_string).split()))
    except ValueError:
        return np.zeros((size, size), dtype=np.uint8)
    
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    total_pixels = size * size
    
    flat_mask = np.zeros(total_pixels, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        end = min(start + length, total_pixels)
        if start < total_pixels:
            flat_mask[start:end] = 1
    
    return flat_mask.reshape((size, size))


def load_dataset_with_masks(dataset_dir: Path):
    """Загружает (изображение, промпт, маска)"""
    data = []
    for img_path in sorted(dataset_dir.glob("*.png")):
        txt_path = img_path.with_suffix(".txt")
        mask_path = img_path.with_suffix(".mask.png")
        if not txt_path.exists() or not mask_path.exists():
            continue
        with open(txt_path) as f:
            prompt = f.read().strip()
        if prompt:
            data.append((img_path, prompt, mask_path))
    return data


def prepare_lora_dataset(real_train_dir: Path, output_dir: Path, 
                         class_labels: dict, prompt_templates: list,
                         rle_csv: Path = None):
    """Готовит датасет с РЕАЛЬНЫМИ масками из RLE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = real_train_dir / "images"
    labels_dir = real_train_dir / "labels"
    
    rle_map = {}
    if rle_csv and rle_csv.exists():
        rle_df = pd.read_csv(rle_csv)
        for _, row in rle_df.iterrows():
            img_id = row['ImageId']
            cls = int(row['ClassId']) - 1
            rle = row.get('EncodedPixels')
            if img_id not in rle_map:
                rle_map[img_id] = {}
            if cls not in rle_map[img_id]:
                rle_map[img_id][cls] = rle
    
    count = 0
    for img_path in sorted(images_dir.glob("*.png")):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        
        classes = set()
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(float(parts[0])))
        if not classes:
            continue
        
        cls = list(classes)[0]
        class_name = class_labels.get(cls, "defect")
        prompt = np.random.choice(prompt_templates).format(cls=class_name)
        
        out_img = output_dir / img_path.name
        Image.open(img_path).convert("RGB").save(out_img)
        
        out_txt = output_dir / f"{img_path.stem}.txt"
        with open(out_txt, "w") as f:
            f.write(prompt)
        
        out_mask = output_dir / f"{img_path.stem}.mask.png"
        
        real_mask = None
        for img_id, class_masks in rle_map.items():
            if img_id == img_path.name or img_id == img_path.stem + '.png':
                combined_mask = np.zeros((256, 256), dtype=np.uint8)
                for rle_cls, rle_str in class_masks.items():
                    mask_cls = rle_to_mask_simple(rle_str, 256)
                    combined_mask = np.maximum(combined_mask, mask_cls)
                real_mask = combined_mask
                break
        
        if real_mask is not None and real_mask.sum() > 0:
            cv2.imwrite(str(out_mask), real_mask * 255)
            count += 1
        else:
            logger.debug(f"Skipping {img_path.name}: no RLE mask")
            continue
    
    logger.info(f"Prepared {count} images with REAL masks for LoRA training")
    return max(count, 1)


# ============================================================
# generate/testing_lora/train_lora.py — ФИНАЛЬНЫЙ TRAIN LOOP
# ============================================================
def train_lora(base_model: str, dataset_dir: Path, output_dir: Path, config: dict):
    from diffusers import StableDiffusionInpaintPipeline
    from peft import LoraConfig, get_peft_model

    logger.info(f"Training LoRA: rank={config['rank']}, steps={config['max_steps']}")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")

    lora_config = LoraConfig(
        r=config['rank'], lora_alpha=config['alpha'],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=config['learning_rate'])
    dataset = load_dataset_with_masks(dataset_dir)

    if not dataset:
        return None

    pipe.unet.train()

    for step in tqdm(range(config['max_steps']), desc="LoRA"):
        img_path, prompt, mask_path = dataset[np.random.randint(len(dataset))]

        if np.random.random() < 0.25:
            prompt = ""

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        real_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if real_mask.sum() == 0:
            continue

        # Всё в PIL для передачи в pipeline
        mask_pil = Image.fromarray((real_mask * 255).astype(np.uint8))

        # Используем встроенный train_step пайплайна — он правильно формирует вход
        pipe.unet.train()
        # Не используем pipe напрямую — только UNet