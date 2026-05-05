#!/usr/bin/env python3
"""
lora_final.py — LoRA txt2img → img2img с маской
"""
import torch, numpy as np, random
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
from tqdm import tqdm

# ============================================================
# КОНФИГ
# ============================================================
LORA_DIR = Path("/app/data/results/lora_test_v3/lora")
DATASET_DIR = LORA_DIR / "dataset"
WEIGHTS_DIR = LORA_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_BG = Path("data/processed/balanced_clean_patches/train")
OUT_DIR = Path("/app/data/results/lora_test_v3/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANK = 16; ALPHA = 16; LR = 1e-4; STEPS = 500; SIZE = 512

# ============================================================
# 1. ОБУЧЕНИЕ LoRA (txt2img)
# ============================================================
print("1/3 Training LoRA...")
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet = get_peft_model(pipe.unet, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["to_q","to_k","to_v","to_out.0"]))

data = []
for p in sorted(DATASET_DIR.glob("*.png")):
    t = p.with_suffix(".txt")
    if t.exists():
        with open(t) as f: prompt = f.read().strip()
        if prompt: data.append((p, prompt))
print(f"  {len(data)} images")

opt = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)
pipe.unet.train()

for s in tqdm(range(STEPS), desc="  LoRA"):
    img_p, prompt = random.choice(data)
    if random.random() < 0.25: prompt = ""
    img = Image.open(img_p).convert("RGB").resize((SIZE, SIZE))
    inp = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                          truncation=True, return_tensors="pt")
    with torch.no_grad():
        enc = pipe.text_encoder(inp.input_ids.to("cuda"))[0]
        lat = pipe.vae.encode(pipe.image_processor.preprocess(img).to("cuda").to(torch.float16)).latent_dist.sample() * 0.18215
    noise = torch.randn_like(lat)
    t = torch.randint(0, 1000, (1,), device="cuda")
    loss = torch.nn.functional.mse_loss(pipe.unet(pipe.scheduler.add_noise(lat, noise, t), t, encoder_hidden_states=enc).sample, noise)
    opt.zero_grad(); loss.backward(); opt.step()

pipe.unet.save_pretrained(WEIGHTS_DIR)
del pipe; torch.cuda.empty_cache()
print("  Done.")

# Замените блок 2 (ГЕНЕРАЦИЯ) на этот:
print("2/3 Generating...")
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipe.unet = get_peft_model(pipe.unet, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["to_q","to_k","to_v","to_out.0"]))

from safetensors.torch import load_file
state = load_file(str(WEIGHTS_DIR / "adapter_model.safetensors"))
cleaned = {k.replace('base_model.model.',''): v for k,v in state.items()}
pipe.unet.load_state_dict(cleaned, strict=False)
pipe.unet.eval()

clean_files = list(CLEAN_BG.glob("*.png")) + list(CLEAN_BG.glob("*.jpg"))
mask_files = list(DATASET_DIR.glob("*.mask.png"))

for i in range(3):
    bg = Image.open(random.choice(clean_files)).convert("RGB").resize((SIZE, SIZE))
    mask_path = random.choice(mask_files)
    mask = Image.open(mask_path).convert("L").resize((SIZE, SIZE))
    
    # Промпт из .txt рядом с маской
    prompt = "surface cavity on steel, industrial defect"
    txt_file = mask_path.with_suffix(".txt")
    if txt_file.exists():
        with open(txt_file) as f:
            prompt = f.read().strip()
    
    # Инвертируем маску: белое = генерировать
    mask = ImageOps.invert(mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    result = pipe(
        prompt=prompt,
        negative_prompt="perfect, clean, smooth, mirror, reflection",
        image=bg, mask_image=mask,
        strength=0.85, guidance_scale=7.5, num_inference_steps=35,
        generator=torch.Generator("cuda").manual_seed(i*100)
    ).images[0]
    
    result.save(str(OUT_DIR / f"result_{i}.png"))
    print(f"  Saved result_{i}.png (prompt: {prompt})")

# ============================================================
# 3. СРАВНЕНИЕ
# ============================================================
print("3/3 Compare:")
print(f"  Output: {OUT_DIR}")
for i in range(3):
    orig = np.array(Image.open(random.choice(clean_files)).resize((SIZE,SIZE)))
    gen = np.array(Image.open(OUT_DIR / f"result_{i}.png"))
    diff = np.abs(orig.astype(float) - gen.astype(float)).mean()
    result.save(str(OUT_DIR / f"result_{i}.png"))
    mask.save(str(OUT_DIR / f"mask_{i}.png"))
    bg.save(str(OUT_DIR / f"clean_{i}.png"))
    print(f"  result_{i}.png — mean diff from clean: {diff:.1f} (should be >5)")