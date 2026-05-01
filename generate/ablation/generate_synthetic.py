#!/usr/bin/env python3
"""Генерация синтетики в рамках ablation study"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate.scripts.config import Config, load_config
from generate.scripts.generate_dataset import PoissonDefectGenerator

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    run_params: dict,
    fixed_params: dict,
    synthetic_output_dir: Path,
    real_train_dir: Path,
    rle_csv: Path,
) -> int:
    """
    Генерация синтетического датасета.
    run_params — перебираемые параметры (сетка).
    fixed_params — фиксированные для всех запусков.
    """
    config = Config()
    
    # === Перебираемые параметры ===
    config.sd_defect.strength_min = run_params['sd_defect_strength']
    config.sd_defect.strength_max = run_params['sd_defect_strength']
    
    config.sd_background.strength_min = run_params['sd_background_strength']
    config.sd_background.strength_max = run_params['sd_background_strength']
    
    config.spectral.high_freq_alpha = run_params['high_freq_alpha']
    
    config.generation.synthetic_total = run_params['synthetic_total']
    config.generation.balance_strategy = run_params['balance_strategy']
    
    # === Фиксированные параметры ===
    config.generation.variants = fixed_params['variants']
    config.generation.random_seed = fixed_params['random_seed']
    
    config.sd_defect.steps = fixed_params['sd_steps']
    config.sd_defect.guidance_scale = fixed_params['sd_guidance_scale']
    config.sd_defect.prompt = fixed_params['defect_prompt']
    config.sd_defect.negative_prompt = fixed_params['defect_negative']
    config.sd_defect.color_correction_strength = fixed_params['defect_color_correction']
    config.sd_defect.defect_consistency = fixed_params['defect_consistency']
    
    config.sd_background.steps = fixed_params['bg_sd_steps']
    config.sd_background.guidance_scale = fixed_params['bg_guidance_scale']
    config.sd_background.prompt = fixed_params['background_prompt']
    config.sd_background.negative_prompt = fixed_params['background_negative']
    config.sd_background.color_correction_strength = fixed_params['background_color_correction']
    
    config.spectral.use_spectrum_matching = fixed_params['use_spectrum_matching']
    config.spectral.use_high_freq_injection = fixed_params['use_high_freq_injection']
    
    config.scaling.factors = [tuple(f) for f in fixed_params['scale_factors']]
    
    # === Пути ===
    config.paths.input_dir = str(real_train_dir / "images")
    config.paths.rle_csv = str(rle_csv)
    config.paths.output_dir = str(synthetic_output_dir)
    
    logger.info(f"Generating {run_params['synthetic_total']} images:")
    logger.info(f"  defect_strength={run_params['sd_defect_strength']}, "
                f"bg_strength={run_params['sd_background_strength']}, "
                f"hf_alpha={run_params['high_freq_alpha']}, "
                f"balance={run_params['balance_strategy']}")
    
    generator = PoissonDefectGenerator(config)
    total = generator.generate_dataset(
        Path(config.paths.input_dir),
        Path(config.paths.rle_csv),
        Path(config.paths.output_dir),
        variants=config.generation.variants
    )
    
    return total