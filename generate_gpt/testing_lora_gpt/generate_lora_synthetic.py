# ============================================================
# generate/testing_lora/generate_lora_synthetic.py
# ============================================================
#!/usr/bin/env python3
"""Генератор синтетики: Context Swapping + Affine Jitter + Poisson Blend"""

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


def create_soft_mask(mask: np.ndarray, kernel_size: int = 9) -> Image.Image:
    mask = mask.astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    mask = np.clip(mask, 0, 255)
    return Image.fromarray(mask, mode="L")


def affine_transform(mask: np.ndarray, bbox: dict) -> tuple:
    """Случайные аффинные искажения маски и bbox."""
    h, w = mask.shape
    cx, cy = w/2, h/2
    
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h
    
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    
    mask_warped = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    # Обновляем bbox
    bx, by, bw, bh = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    corners = np.array([[bx, by], [bx+bw, by], [bx, by+bh], [bx+bw, by+bh]], dtype=np.float32)
    corners = cv2.transform(corners.reshape(1, 4, 2), matrix).reshape(4, 2)
    new_x = corners[:, 0].min()
    new_y = corners[:, 1].min()
    new_w = corners[:, 0].max() - new_x
    new_h = corners[:, 1].max() - new_y
    
    updated_bbox = bbox.copy()
    updated_bbox['x'] = int(new_x)
    updated_bbox['y'] = int(new_y)
    updated_bbox['w'] = int(new_w)
    updated_bbox['h'] = int(new_h)
    
    return mask_warped, updated_bbox


class LoRADefectGenerator:
    def __init__(self, config, lora_weights_path: str):
        self.config = config
        from sd_generator_lora import SDDefectGeneratorLoRA
        self.sd_generator = SDDefectGeneratorLoRA(config, lora_weights_path)

        random.seed(config['generation']['random_seed'])
        np.random.seed(config['generation']['random_seed'])
        self.class_labels = {int(k): v for k, v in config['generation']['class_labels'].items()}
        self.prompt_templates = config['generation']['prompt_templates']
        self.negative_prompt = config['generation']['negative_prompt']
        
        # Загружаем чистые фоны
        self.clean_backgrounds = []
        if config['generation'].get('use_clean_backgrounds', False):
            clean_path = Path(config['paths'].get('clean_patches', ''))
            if clean_path.exists():
                self.clean_backgrounds = sorted(clean_path.glob("*.png")) + sorted(clean_path.glob("*.jpg"))
                logger.info(f"Loaded {len(self.clean_backgrounds)} clean backgrounds")

    def _get_background(self, original: np.ndarray) -> np.ndarray:
        gen_cfg = self.config['generation']
        if self.clean_backgrounds and random.random() < gen_cfg.get('clean_background_ratio', 0.5):
            bg_path = random.choice(self.clean_backgrounds)
            self._last_bg_name = bg_path.stem  # ← сохраняем имя
            bg = cv2.imread(str(bg_path))
            if bg is not None:
                return cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        self._last_bg_name = None
        return original.copy()

    def generate_dataset(self, input_dir: Path, rle_csv: Path, output_dir: Path,
                     variants: int = 3, limit: Optional[int] = None) -> int:
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

            all_bboxes = []
            for _, row in group.iterrows():
                rle = row.get('EncodedPixels')
                if pd.isna(rle) or str(rle).strip().lower() in ['', 'nan']:
                    continue
                all_bboxes.extend(rle_to_defect_bboxes(str(rle), int(row['ClassId']) - 1))

            if not all_bboxes:
                continue

            for v in range(variants):
                if limit and stats['total'] >= limit:
                    break

                try:
                    # ✅ НОВЫЙ фон для каждого варианта
                    background = self._get_background(original)
                    bg_h, bg_w = background.shape[:2]
                    result = background.copy().astype(np.float32)
                    yolo_lines = []

                    for bbox in all_bboxes:
                        try:
                            cls = bbox['class']
                            class_name = self.class_labels.get(cls, "defect")
                            prompt = random.choice(self.prompt_templates).format(cls=class_name)

                            # ✅ Копируем bbox — не модифицируем оригинал
                            bx, by, bw, bh = bbox['x'], bbox['y'], bbox['w'], bbox['h']

                            # ✅ Jitter только если чистое изображение ТОГО ЖЕ размера
                            if gen_cfg.get('jitter_bbox', False) and bg_h == img_h and bg_w == img_w:
                                bx = max(0, min(bg_w - bw, bx + random.randint(-20, 20)))
                                by = max(0, min(bg_h - bh, by + random.randint(-20, 20)))

                            pad = gen_cfg['pad']
                            x1 = max(0, bx - pad)
                            y1 = max(0, by - pad)
                            x2 = min(bg_w, bx + bw + pad)
                            y2 = min(bg_h, by + bh + pad)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            crop_w = x2 - x1
                            crop_h = y2 - y1
                            crop_img = result[y1:y2, x1:x2].copy()

                            # ✅ Маска правильного размера
                            crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                            mask_from_rle = bbox['component_mask']
                            mh, mw = mask_from_rle.shape

                            # Координаты маски внутри кропа
                            mx1 = max(0, bx - x1)
                            my1 = max(0, by - y1)

                            # Сколько реально копируем
                            copy_w = min(mw, crop_w - mx1)
                            copy_h = min(mh, crop_h - my1)

                            if copy_w > 0 and copy_h > 0:
                                crop_mask[my1:my1+copy_h, mx1:mx1+copy_w] = mask_from_rle[:copy_h, :copy_w]

                            if crop_mask.sum() == 0:
                                continue

                            kernel = np.ones((gen_cfg['mask_dilate_kernel'], gen_cfg['mask_dilate_kernel']), np.uint8)
                            expanded_mask = cv2.dilate(crop_mask, kernel, iterations=gen_cfg['mask_dilate_iterations'])
                            mask_pil = create_soft_mask(expanded_mask, gen_cfg['mask_blur_kernel'])
                            image_pil = Image.fromarray(crop_img.astype(np.uint8))

                            strength = random.uniform(gen_cfg['strength_min'], gen_cfg['strength_max'])
                            seed = gen_cfg['random_seed'] + stats['total'] * 1000 + v * 100 + cls
                            
                            
                            #  Сохранить маску и кроп для проверки
                            # debug_dir = Path("/app/data/results/lora_test_v3/debug")
                            # debug_dir.mkdir(parents=True, exist_ok=True)
                            # cv2.imwrite(str(debug_dir / f"crop_{stats['total']:06d}.png"), crop_img)
                            # cv2.imwrite(str(debug_dir / f"mask_{stats['total']:06d}.png"), expanded_mask * 255)

                            generated = self.sd_generator.inpaint(
                                image=image_pil, mask=mask_pil,
                                prompt=prompt, negative=self.negative_prompt,
                                strength=strength, steps=gen_cfg['sd_steps'],
                                guidance=gen_cfg['sd_guidance_scale'], seed=seed
                            )
                            generated = np.clip(generated, 0, 255)

                            if generated.shape[:2] != (crop_h, crop_w):
                                generated = cv2.resize(generated, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

                            # Простая цветокоррекция — без adaptive (она меняет всё изображение)
                            # corrected = generated.astype(np.float32) * 0.7 + crop_img.astype(np.float32) * 0.3

                            # Простое смешивание
                            # mask_3ch = np.stack([crop_mask.astype(np.float32)] * 3, axis=-1)
                            # blended = corrected * mask_3ch + crop_img.astype(np.float32) * (1 - mask_3ch)
                            # result[y1:y2, x1:x2] = blended.astype(np.float32)

                            # Заменить блок смешивания (строки ~220-235) на:

                            # Мягкая маска для плавного перехода
                            soft_mask = cv2.GaussianBlur(crop_mask.astype(np.float32), (5, 5), 0)
                            mask_3ch = np.stack([soft_mask] * 3, axis=-1)

                            # 80% генерация внутри маски, плавный переход к фону
                            blended = generated.astype(np.float32) * mask_3ch + crop_img.astype(np.float32) * (1 - mask_3ch)
                            result[y1:y2, x1:x2] = blended.astype(np.float32)

                            xc = (bx + bw / 2) / bg_w
                            yc = (by + bh / 2) / bg_h
                            ww = bw / bg_w
                            hh = bh / bg_h
                            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

                        except Exception as e:
                            logger.warning(f"Inpainting failed: {e}")
                            continue

                    if not yolo_lines:
                        continue

                    # High-freq injection
                    if gen_cfg.get('use_high_freq_injection', True):
                        result = inject_high_freq(result, background.astype(np.float32), alpha=gen_cfg['high_freq_alpha'])

                    # stem = img_path.stem if img_path else f"img_{stats['total']:06d}"
                    if background is not original and hasattr(self, '_last_bg_name'):
                        bg_stem = self._last_bg_name
                    else:
                        bg_stem = img_path.stem if img_path else f"img_{stats['total']:06d}"

                    # filename = f"lora_syn_{stats['total']:06d}_{stem}_on_{bg_stem}_v{v}.png"
                    # filename = f"lora_syn_{stats['total']:06d}_{stem}_v{v}.png"

                    orig_stem = img_path.stem if img_path else f"img_{stats['total']:06d}"
                    if self._last_bg_name:
                        filename = f"lora_syn_{stats['total']:06d}_{orig_stem}_on_{self._last_bg_name}_v{v}.png"
                    else:
                        filename = f"lora_syn_{stats['total']:06d}_{orig_stem}_v{v}.png"

                    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
                    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_dir / "images" / filename), result_bgr,
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])

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