# generate/testing_lora/train_lora.py
#!/usr/bin/env python3
"""LoRA fine-tuning SD на реальных дефектах"""

import logging
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def prepare_lora_dataset(real_train_dir: Path, output_dir: Path):
    """Подготавливает данные для LoRA: изображение + текстовая метка класса"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = real_train_dir / "images"
    labels_dir = real_train_dir / "labels"
    
    class_names = {0: "defect_1", 1: "defect_2", 2: "defect_3", 3: "defect_4"}
    dataset = []
    
    for img_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        
        # Читаем классы дефектов
        classes = set()
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(float(parts[0])))
        
        if classes:
            class_name = class_names.get(list(classes)[0], "defect")
            prompt = f"a {class_name} on industrial steel surface, brushed metal texture"
            
            # Копируем изображение
            img = Image.open(img_path).convert("RGB")
            out_path = output_dir / img_path.name
            img.save(out_path)
            
            dataset.append({
                "image": str(out_path),
                "prompt": prompt
            })
    
    logger.info(f"Prepared {len(dataset)} images for LoRA training")
    return dataset


def train_lora(base_model: str, dataset_dir: Path, output_dir: Path, config: dict):
    """Дообучение SD через LoRA"""
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model
    
    logger.info(f"Training LoRA: rank={config['rank']}, lr={config['learning_rate']}, steps={config['max_steps']}")
    
    # Загружаем базовую модель
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Конфиг LoRA
    lora_config = LoraConfig(
        r=config['rank'],
        lora_alpha=config['alpha'],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    
    # Применяем LoRA к UNet
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=config['learning_rate'])
    
    # Датасет
    image_files = sorted(list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg")))
    if not image_files:
        logger.error("No images for training!")
        return None
    
    pipe.unet.train()
    
    for step in tqdm(range(config['max_steps']), desc="LoRA training"):
        # Случайный батч
        batch = np.random.choice(image_files, size=config['batch_size'])
        
        for img_path in batch:
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            
            # Токенизация + генерация
            with torch.no_grad():
                latents = pipe.vae.encode(
                    pipe.image_processor.preprocess(img).unsqueeze(0).to("cuda")
                ).latent_dist.sample() * 0.18215
            
            # Шум
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,), device="cuda")
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Предикт
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=None).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (step + 1) % config['save_every'] == 0:
            save_path = output_dir / f"lora_weights_step_{step+1}.pt"
            torch.save(pipe.unet.state_dict(), save_path)
            logger.info(f"Saved: {save_path}")
    
    # Сохраняем финальные веса
    final_path = output_dir / "lora_weights_final.pt"
    torch.save(pipe.unet.state_dict(), final_path)
    logger.info(f"Final weights: {final_path}")
    
    return final_path


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    real_train = Path(cfg['paths']['real_train'])
    lora_dir = Path("lora_output")
    
    dataset = prepare_lora_dataset(real_train, lora_dir / "dataset")
    train_lora(
        "runwayml/stable-diffusion-v1-5",
        lora_dir / "dataset",
        lora_dir / "weights",
        cfg['lora']
    )