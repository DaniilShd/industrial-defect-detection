# generate/testing_lora/sd_generator_lora.py
#!/usr/bin/env python3
"""SDDefectGenerator с LoRA — генерация фона и дефекта"""

import logging
import random

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SDDefectGeneratorLoRA:
    def __init__(self, config, lora_weights_path: str = None, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        self.device = device
        self.config = config
        
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        from peft import LoraConfig, get_peft_model
        
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info("Загрузка Stable Diffusion с LoRA...")
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Загружаем LoRA веса
        if lora_weights_path and Path(lora_weights_path).exists():
            lora_config = LoraConfig(
                r=4, lora_alpha=8,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.0,
            )
            self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
            self.pipe.unet.load_state_dict(torch.load(lora_weights_path), strict=False)
            logger.info(f"✅ LoRA веса загружены: {lora_weights_path}")
        
        if device == "cuda":
            self.pipe = self.pipe.to(device)
            self.pipe.enable_attention_slicing()
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        logger.info("🚀 SD+LoRA готов")
    
    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str, negative: str,
                strength: float, steps: int, guidance: float, seed: int = None) -> np.ndarray:
        w, h = image.size
        sd_w = max(((w + 63) // 64) * 64, 512)
        sd_h = max(((h + 63) // 64) * 64, 512)
        
        crop_sd = image.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device if torch.cuda.is_available() else "cpu").manual_seed(seed)
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=crop_sd,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator
        )
        
        generated = output.images[0]
        if generated.size != (w, h):
            generated = generated.resize((w, h), Image.Resampling.LANCZOS)
        
        return np.array(generated).astype(np.float32)