# =========================
# FIXED: train_lora.py
# =========================

import logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_dataset(dataset_dir: Path):
    data = []
    for img_path in dataset_dir.glob("*.png"):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        with open(txt_path) as f:
            prompt = f.read().strip()
        data.append((img_path, prompt))
    return data


def train_lora(base_model, dataset_dir, output_dir, config):
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    lora_config = LoraConfig(
        r=config['rank'],
        lora_alpha=config['alpha'],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)

    dataset = load_dataset(dataset_dir)
    pipe.unet.train()

    for step in tqdm(range(config['max_steps'])):
        img_path, prompt = dataset[np.random.randint(len(dataset))]

        image = Image.open(img_path).convert("RGB").resize((512, 512))

        inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(
                inputs.input_ids.to("cuda")
            )[0]

            latents = pipe.vae.encode(
                pipe.image_processor.preprocess(image).unsqueeze(0).to("cuda")
            ).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,), device="cuda")
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % config['save_every'] == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pipe.unet.state_dict(), output_dir / f"lora_{step+1}.pt")

    final_path = output_dir / "lora_final.pt"
    torch.save(pipe.unet.state_dict(), final_path)
    return final_path


# =========================
# FIXED: prepare_lora_dataset
# =========================

def prepare_lora_dataset(real_train_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = real_train_dir / "images"
    labels_dir = real_train_dir / "labels"

    class_names = {0: "defect_1", 1: "defect_2", 2: "defect_3", 3: "defect_4"}

    for img_path in images_dir.glob("*.png"):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        classes = set()
        with open(lbl_path) as f:
            for line in f:
                cls = int(float(line.split()[0]))
                classes.add(cls)

        if not classes:
            continue

        class_name = class_names[list(classes)[0]]
        prompt = f"{class_name}, industrial steel defect, realistic"

        out_img = output_dir / img_path.name
        out_txt = output_dir / f"{img_path.stem}.txt"

        Image.open(img_path).convert("RGB").save(out_img)
        with open(out_txt, "w") as f:
            f.write(prompt)


# =========================
# FIXED: generate_lora_synthetic.py (core part)
# =========================

import cv2
import numpy as np
from PIL import Image


def insert_defect(sd_generator, image, bbox, prompt, cfg):
    x1, y1, x2, y2 = bbox

    crop = image[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop)

    generated = sd_generator.generate(
        crop_pil,
        prompt=prompt,
        negative="",
        strength=np.random.uniform(0.25, 0.4),
        steps=cfg['sd_steps'],
        guidance=cfg['sd_guidance_scale']
    )

    generated = generated.astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = generated

    return result


# =========================
# FIXED: sd_generator_lora.py (bugfix)
# =========================

from pathlib import Path  # <-- FIX

# остальной код без изменений


# =========================
# IMPORTANT CONFIG FIX
# =========================

"""
Исправь config.yaml:

lora:
  learning_rate: 1e-4

"""
