# ============================================================
# generate/testing_lora/sd_generator_lora.py — ФИНАЛ
# ============================================================
#!/usr/bin/env python3
"""SD Inpainting + LoRA — xformers, noise offset, правильная загрузка"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

USE_MERGED_LORA = False


class SDDefectGeneratorLoRA:
    def __init__(self, config, lora_weights_path: str = None, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.config = config

        from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
        from peft import PeftModel

        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading SD Inpainting...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )

        try:
            self.pipe.enable_vae_tiling()
            logger.info("✅ VAE tiling enabled")
        except:
            pass

        # ✅ xformers — чище текстуры
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ xformers enabled")
        except:
            pass

        # ✅ Правильная загрузка LoRA
        # ✅ Правильная загрузка LoRA
        # sd_generator_lora.py, заменить загрузку:
        if lora_weights_path and Path(lora_weights_path).exists():
            from safetensors.torch import load_file
            
            lora_cfg = self.config['lora']
            lora_config = LoraConfig(
                r=lora_cfg['rank'], lora_alpha=lora_cfg['alpha'],
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
            self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
            
            # Загружаем веса напрямую
            state_dict = load_file(str(Path(lora_weights_path) / "adapter_model.safetensors"))
            
            # Убираем префикс base_model.model.
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k.replace('base_model.model.', '')
                cleaned[new_k] = v
            
            self.pipe.unet.load_state_dict(cleaned, strict=False)
            self.pipe.unet.eval()
            logger.info(f"✅ LoRA loaded: {len(cleaned)} params from safetensors")                                                      

        if device == "cuda":
            self.pipe = self.pipe.to(device)
            try:
                self.pipe.enable_attention_slicing(slice_size="auto")
            except:
                pass

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        logger.info("🚀 SD Inpainting ready")

    @torch.no_grad()
    def inpaint(
        self, image, mask, prompt, negative, strength, steps, guidance, seed=None
    ) -> np.ndarray:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        gen_cfg = self.config['generation']
        guidance = guidance + random.uniform(-gen_cfg['guidance_jitter'], gen_cfg['guidance_jitter'])

        # guidance = guidance + random.uniform(-0.5, 0.5)
        generator = torch.Generator(
            device=self.device if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)

        result = self.pipe(
            prompt=prompt, negative_prompt=negative,
            image=image, mask_image=mask,
            strength=strength, guidance_scale=guidance,
            num_inference_steps=steps, generator=generator,
        ).images[0]

        return np.array(result).astype(np.float32)