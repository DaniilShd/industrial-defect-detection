# ============================================================
# generate/testing_lora/train_lora.py — PSEUDO-DEFECT MASKS
# ============================================================
#!/usr/bin/env python3
"""LoRA training на INPAINTING с реалистичной геометрией дефектов"""

import logging
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_dataset(dataset_dir: Path):
    data = []
    for img_path in sorted(dataset_dir.glob("*.png")):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        with open(txt_path) as f:
            prompt = f.read().strip()
        if prompt:
            data.append((img_path, prompt))
    return data


def prepare_lora_dataset(real_train_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = real_train_dir / "images"
    labels_dir = real_train_dir / "labels"

    CLASS_LABELS = {0: "crack", 1: "rust", 2: "scratch", 3: "dent"}
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

        class_name = CLASS_LABELS.get(list(classes)[0], "defect")
        PROMPT_TEMPLATES = [
            "photo of {cls} on steel sheet, industrial lighting, ultra realistic, 4k",
            "close-up photo of {cls}, metallic surface, sharp details, realistic texture",
            "{cls} defect on brushed steel, high frequency detail, macro photography",
            "industrial defect: {cls}, steel surface, realistic lighting, no blur",
        ]
        prompt = np.random.choice(PROMPT_TEMPLATES).format(cls=class_name)

        out_img = output_dir / img_path.name
        out_txt = output_dir / f"{img_path.stem}.txt"

        Image.open(img_path).convert("RGB").save(out_img)
        with open(out_txt, "w") as f:
            f.write(prompt)
        count += 1

    logger.info(f"Prepared {count} images for LoRA training")
    return count


def generate_pseudo_defect_mask(size: int = 512, mask_blur_kernel: int = 31) -> torch.Tensor:
    """
    Генерирует маску, похожую на реальные дефекты Severstal:
    - Тонкие линии (scratches)
    - Ломаные (cracks)
    - Нерегулярные пятна (rust/dents)
    """
    mask = np.zeros((size, size), dtype=np.float32)

    defect_type = np.random.choice(['scratch', 'crack', 'spot', 'combo'])

    if defect_type == 'scratch':
        num_lines = np.random.randint(1, 4)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, size, 2)
            x2 = x1 + np.random.randint(-size//4, size//4)
            y2 = y1 + np.random.randint(-size//4, size//4)
            thickness = np.random.randint(2, 8)
            cv2.line(mask, (x1, y1), (x2, y2), color=1.0, thickness=thickness)

    elif defect_type == 'crack':
        num_segments = np.random.randint(3, 8)
        points = [(np.random.randint(0, size), np.random.randint(0, size))]
        for _ in range(num_segments - 1):
            last = points[-1]
            points.append((
                last[0] + np.random.randint(-size//6, size//6),
                last[1] + np.random.randint(-size//6, size//6)
            ))
        for i in range(len(points) - 1):
            thickness = np.random.randint(2, 10)
            cv2.line(mask, points[i], points[i+1], color=1.0, thickness=thickness)

    elif defect_type == 'spot':
        num_spots = np.random.randint(1, 5)
        for _ in range(num_spots):
            cx, cy = np.random.randint(size//4, 3*size//4, 2)
            rx, ry = np.random.randint(10, size//4), np.random.randint(5, size//6)
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, color=1.0, thickness=-1)

    else:  # combo
        # Scratch + spot
        x1, y1 = np.random.randint(0, size, 2)
        x2, y2 = np.random.randint(0, size, 2)
        cv2.line(mask, (x1, y1), (x2, y2), color=1.0, thickness=np.random.randint(2, 6))
        cx, cy = np.random.randint(size//4, 3*size//4, 2)
        cv2.circle(mask, (cx, cy), np.random.randint(5, size//8), color=1.0, thickness=-1)

    # ✅ Блюр маски — убирает distribution mismatch с inference
    mask = cv2.GaussianBlur(mask, (mask_blur_kernel, mask_blur_kernel), 0)
    mask = np.clip(mask, 0, 1)

    return torch.tensor(mask).unsqueeze(0).unsqueeze(0)


def train_lora(base_model: str, dataset_dir: Path, output_dir: Path, config: dict):
    from diffusers import StableDiffusionInpaintPipeline
    from peft import LoraConfig, get_peft_model

    logger.info(f"Training LoRA on INPAINTING: rank={config['rank']}, steps={config['max_steps']}")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    lora_config = LoraConfig(
        r=config['rank'],
        lora_alpha=config['alpha'],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=config['learning_rate'])
    dataset = load_dataset(dataset_dir)

    if not dataset:
        logger.error("No training data!")
        return None

    logger.info(f"Training on {len(dataset)} image-prompt pairs")
    pipe.unet.train()

    for step in tqdm(range(config['max_steps']), desc="LoRA inpainting"):
        img_path, prompt = dataset[np.random.randint(len(dataset))]

        # ✅ Prompt dropout — 10% без текста (учит безусловную генерацию)
        if np.random.random() < config['prompt_dropout']:
            prompt = ""

        image = Image.open(img_path).convert("RGB").resize((config['image_size'], config['image_size']))

        # ✅ Pseudo-defect mask вместо прямоугольников
        mask = generate_pseudo_defect_mask(512, config['mask_blur_kernel']).to("cuda")

        
        inputs = pipe.tokenizer(
            prompt, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(
                inputs.input_ids.to("cuda")
            )[0]
            image = pipe.image_processor.preprocess(image).to("cuda").to(torch.float16)

            latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215

        mask = torch.nn.functional.interpolate(mask, size=(latents.shape[2], latents.shape[3]), mode='bilinear')

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device="cuda")
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        masked_latents = latents * (1 - mask)
        mask = mask.to(noisy_latents.dtype)

        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "mask": mask,
                "masked_image_latents": masked_latents
            }
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % config['save_every'] == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            pipe.unet.save_pretrained(output_dir / f"checkpoint-{step+1}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "lora_final"
    pipe.unet.save_pretrained(final_path)
    lora_config.save_pretrained(final_path)
    
    logger.info(f"Final weights: {final_path}")
    return final_path 