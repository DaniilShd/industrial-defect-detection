# ============================================================
# generate/testing_lora/generate_lora_synthetic.py
# ============================================================
#!/usr/bin/env python3
"""Генератор синтетики: SD Inpainting + LoRA + Color Correction + Blending"""

import logging
import random
import sys
from pathlib import Path
from typing import Optional

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
from utils.spectral import inject_high_freq

logger = logging.getLogger(__name__)

# В начало файла:
CLASS_LABELS = {0: "crack", 1: "rust", 2: "scratch", 3: "dent"}

PROMPT_TEMPLATES = [
    "photo of {cls} on steel sheet, industrial lighting, ultra realistic, 4k",
    "close-up photo of {cls}, metallic surface, sharp details, realistic texture",
    "{cls} defect on brushed steel, high frequency detail, macro photography",
    "industrial defect: {cls}, steel surface, realistic lighting, no blur",
]

NEGATIVE_PROMPT = "blurry, low quality, artifacts, distorted, smooth, plastic, painting"


def create_soft_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    mask_float = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
    sharp = np.clip(mask_float * 0.7 + blurred * 0.3, 0, 1)
    return (sharp * 255).astype(np.uint8)


class LoRADefectGenerator:
    def __init__(self, config, lora_weights_path: str):
        self.config = config
        from sd_generator_lora import SDDefectGeneratorLoRA
        self.sd_generator = SDDefectGeneratorLoRA(config, lora_weights_path)

        random.seed(config['generation']['random_seed'])
        np.random.seed(config['generation']['random_seed'])

    def generate_dataset(
        self,
        input_dir: Path,
        rle_csv: Path,
        output_dir: Path,
        variants: int = 3,
        limit: Optional[int] = None
    ) -> int:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')

        all_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in input_dir.glob(ext):
                all_images[img.name] = img
                all_images[img.stem] = img

        stats = {'total': 0, 'errors': 0, 'defects': 0}
        gen_cfg = self.config['generation']

        for image_id, group in tqdm(groups, desc="LoRA Inpainting"):
            if limit and stats['total'] >= limit:
                break

            img_path = all_images.get(image_id)
            if img_path is None:
                continue

            original = cv2.imread(str(img_path))
            if original is None:
                continue
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            img_h, img_w = original.shape[:2]

            # Собираем все дефекты на изображении
            all_bboxes = []
            for _, row in group.iterrows():
                rle = row.get('EncodedPixels')
                if pd.isna(rle) or str(rle).strip().lower() in ['', 'nan']:
                    continue
                all_bboxes.extend(
                    rle_to_defect_bboxes(str(rle), int(row['ClassId']) - 1)
                )

            if not all_bboxes:
                continue

            # Генерируем варианты
            for v in range(variants):
                if limit and stats['total'] >= limit:
                    break

                try:
                    result = original.copy().astype(np.float32)
                    yolo_lines = []

                    for bbox in all_bboxes:
                        try:
                            cls = bbox['class']
                            class_name = CLASS_LABELS.get(cls, "defect")
                            prompt = random.choice(PROMPT_TEMPLATES).format(cls=class_name)

                            pad = 32
                            bx, by, bw, bh = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                            x1 = max(0, bx - pad); y1 = max(0, by - pad)
                            x2 = min(img_w, bx + bw + pad); y2 = min(img_h, by + bh + pad)

                            crop_img = result[y1:y2, x1:x2].copy()
                            crop_mask = bbox['component_mask'][y1:y2, x1:x2].copy()

                            # Расширение маски
                            kernel = np.ones((7, 7), np.uint8)
                            expanded_mask = cv2.dilate(crop_mask.astype(np.uint8), kernel, iterations=1)
                            soft_mask = create_soft_mask(expanded_mask, kernel_size=15)

                            mask_pil = Image.fromarray(soft_mask)
                            image_pil = Image.fromarray(crop_img.astype(np.uint8))

                            # ✅ Адаптивный strength по размеру дефекта
                            area = bw * bh / (img_w * img_h)
                            if area < 0.01:
                                strength = random.uniform(0.35, 0.50)
                            elif area < 0.05:
                                strength = random.uniform(0.25, 0.40)
                            else:
                                strength = random.uniform(0.20, 0.35)

                            seed = gen_cfg['random_seed'] + stats['total'] * 1000 + v * 100 + cls

                            generated = self.sd_generator.inpaint(
                                image=image_pil, mask=mask_pil,
                                prompt=prompt, negative=NEGATIVE_PROMPT,
                                strength=strength, steps=gen_cfg['sd_steps'],
                                guidance=gen_cfg['sd_guidance_scale'], seed=seed
                            )
                            generated = np.clip(generated, 0, 255)

                            # Цветокоррекция
                            corrected = adaptive_color_correction(
                                generated.astype(np.float32),
                                crop_img.astype(np.float32),
                                crop_mask,
                                strength=gen_cfg['color_correction']
                            )

                            # ✅ Edge-aware blur — сохраняет текстуру
                            edges = cv2.Canny(corrected.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
                            edges_3ch = np.stack([edges] * 3, axis=-1)
                            blurred = cv2.GaussianBlur(corrected.astype(np.float32), (5, 5), 0)
                            corrected = corrected * (0.7 + 0.3 * edges_3ch) + blurred * (0.3 * (1 - edges_3ch))

                            # Смешивание
                            blended = apply_multiscale_blend(
                                corrected.astype(np.float32),
                                crop_img.astype(np.float32),
                                crop_mask.astype(np.float32)
                            )
                            result[y1:y2, x1:x2] = blended.astype(np.float32)

                            # YOLO
                            xc = (bx + bw / 2) / img_w; yc = (by + bh / 2) / img_h
                            ww = bw / img_w; hh = bh / img_h
                            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

                        except Exception as e:
                            logger.warning(f"Inpainting failed: {e}")
                            continue


                    # ✅ После цикла по bbox, high-freq injection ТОЛЬКО на фон:
                    if gen_cfg.get('use_high_freq_injection', True) and yolo_lines:
                        mask_total = np.zeros((img_h, img_w), dtype=np.float32)
                        for bbox in all_bboxes:
                            m = bbox['component_mask']
                            h, w = m.shape
                            bx, by = bbox['x'], bbox['y']
                            # Безопасное копирование с проверкой границ
                            h = min(h, img_h - by)
                            w = min(w, img_w - bx)
                            if h > 0 and w > 0:
                                mask_total[by:by+h, bx:bx+w] += m[:h, :w]
                        mask_total = np.clip(mask_total, 0, 1)
                        mask_3ch = np.stack([mask_total] * 3, axis=-1)

                        hf_result = inject_high_freq(result, original.astype(np.float32), alpha=gen_cfg['high_freq_alpha'])
                        result = hf_result * (1 - mask_3ch) + result * mask_3ch 

                    # High-frequency injection — возвращаем микротекстуру
                    # if gen_cfg.get('use_high_freq_injection', True):
                    #     result = inject_high_freq(
                    #         result,
                    #         original.astype(np.float32),
                    #         alpha=gen_cfg['high_freq_alpha']
                    #     )

                    # Сохранение
                    stem = img_path.stem
                    filename = f"lora_syn_{stats['total']:06d}_{stem}_v{v}.png"

                    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
                    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(
                        str(output_dir / "images" / filename),
                        result_bgr,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3]
                    )

                    # Сохранение лейблов
                    label_path = output_dir / "labels" / f"{Path(filename).stem}.txt"
                    with open(label_path, "w") as f:
                        f.write("\n".join(yolo_lines))

                    stats['total'] += 1
                    stats['defects'] += len(yolo_lines)

                except Exception as e:
                    logger.error(f"Error generating variant {v}: {e}")
                    stats['errors'] += 1

                if stats['total'] % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"✅ Generated {stats['total']} images, {stats['defects']} defects, {stats['errors']} errors")
        return stats['total']