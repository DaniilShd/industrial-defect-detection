#!/usr/bin/env python3
"""SDDefectGenerator — генерация фона и дефекта через Stable Diffusion"""

import logging
import random

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from utils.spectral import match_spectrum, inject_high_freq
from utils.color_correction import adaptive_color_correction, color_transfer_lab

logger = logging.getLogger(__name__)


class SDDefectGenerator:
    def __init__(self, config: Config, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA недоступна, используется CPU")
        
        self.device = device
        self.config = config
        
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
        
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info("Загрузка Stable Diffusion...")
        
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            logger.info("✅ Загружен улучшенный VAE (MSE)")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить MSE VAE: {e}")
            vae = None
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            vae=vae,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if device == "cpu":
            self.pipe = self.pipe.to(device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        if hasattr(self.pipe.vae, 'disable_tiling'):
            self.pipe.vae.disable_tiling()
        
        if device == "cuda":
            self.pipe = self.pipe.to(device)
            try:
                self.pipe.enable_attention_slicing(slice_size="auto")
                logger.info("✅ Attention slicing включен (auto)")
            except Exception as e:
                logger.warning(f"Не удалось включить attention slicing: {e}")
        
        self._warmup_vae()
        logger.info("SD готов")
    
    def _warmup_vae(self):
        try:
            test_latent = torch.randn(1, 4, 64, 64, 
                                     device=self.device if self.device == "cuda" else "cpu",
                                     dtype=torch.float16 if self.device == "cuda" else torch.float32)
            with torch.no_grad():
                if hasattr(self.pipe.vae, 'decode'):
                    _ = self.pipe.vae.decode(test_latent).sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✅ VAE прогрет успешно")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось прогреть VAE: {e}")
    
    @torch.no_grad()
    def generate_defect(self, defect_crop: Image.Image, seed: int = None) -> np.ndarray:
        cfg = self.config.sd_defect
        w, h = defect_crop.size
        sd_w = max(((w + 63) // 64) * 64, 512)
        sd_h = max(((h + 63) // 64) * 64, 512)
        
        crop_sd = defect_crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device if torch.cuda.is_available() else "cpu").manual_seed(seed)
        strength = random.uniform(cfg.strength_min, cfg.strength_max)
        
        if random.random() > cfg.defect_consistency:
            return np.array(defect_crop).astype(np.float32)
        
        try:
            output = self.pipe(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                image=crop_sd,
                strength=strength,
                guidance_scale=self.config.sd_defect.guidance_scale,
                num_inference_steps=self.config.sd_defect.steps,
                generator=generator,
                height=sd_h,
                width=sd_w
            )
            
            generated = output.images[0]
            
            if generated.size != (w, h):
                generated = generated.resize((w, h), Image.Resampling.LANCZOS)
            
            generated_np = np.array(generated).astype(np.float32)
            crop_np = np.array(defect_crop).astype(np.float32)
            
            if cfg.color_correction_strength > 0:
                generated_np = adaptive_color_correction(
                    generated_np, crop_np,
                    np.ones((h, w), dtype=np.uint8),
                    cfg.color_correction_strength
                )
            
            if self.config.spectral.use_spectrum_matching:
                generated_np = match_spectrum(generated_np, crop_np)
            
            if self.config.spectral.use_high_freq_injection:
                generated_np = inject_high_freq(generated_np, crop_np, self.config.spectral.high_freq_alpha)
            
            return generated_np
        
        except Exception as e:
            logger.error(f"SD defect generation error: {e}")
            return np.array(defect_crop).astype(np.float32)

    @torch.no_grad()
    def generate_background(self, bg_crop: Image.Image, seed: int = None) -> np.ndarray:
        cfg = self.config.sd_background
        w, h = bg_crop.size
        sd_w = max(((w + 63) // 64) * 64, 512)
        sd_h = max(((h + 63) // 64) * 64, 512)
        
        crop_sd = bg_crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device if torch.cuda.is_available() else "cpu").manual_seed(seed)
        strength = random.uniform(cfg.strength_min, cfg.strength_max)
        strength = max(0.08, strength)
        
        try:
            output = self.pipe(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                image=crop_sd,
                strength=strength,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.steps,
                generator=generator,
                height=sd_h,
                width=sd_w
            )
            
            generated = output.images[0]
            
            if generated.size != (w, h):
                generated = generated.resize((w, h), Image.Resampling.LANCZOS)
            
            generated_np = np.array(generated).astype(np.float32)
            crop_np = np.array(bg_crop).astype(np.float32)
            
            if cfg.color_correction_strength > 0:
                generated_np = color_transfer_lab(crop_np, generated_np)
            
            if self.config.spectral.use_spectrum_matching:
                generated_np = match_spectrum(generated_np, crop_np).astype(np.float32)
            
            if self.config.spectral.use_high_freq_injection:
                generated_np = inject_high_freq(generated_np, crop_np, self.config.spectral.high_freq_alpha).astype(np.float32)
            
            result = cv2.addWeighted(crop_np, 0.5, generated_np, 0.5, 0)
            
            return result
        
        except Exception as e:
            logger.error(f"SD background generation error: {e}")
            return np.array(bg_crop).astype(np.float32)