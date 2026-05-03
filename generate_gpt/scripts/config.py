#!/usr/bin/env python3
"""Загрузка конфигурации из YAML и аргументов командной строки"""

import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SDConfig:
    steps: int = 20
    guidance_scale: float = 1.8
    strength_min: float = 0.25
    strength_max: float = 0.45
    prompt: str = ""
    negative_prompt: str = ""
    color_correction_strength: float = 0.85
    defect_consistency: float = 0.8


@dataclass
class SpectralConfig:
    use_spectrum_matching: bool = True
    use_high_freq_injection: bool = True
    high_freq_alpha: float = 0.35


@dataclass
class ScalingConfig:
    factors: List[Tuple[float, float]] = field(default_factory=lambda: [(1.0, 1.0)])


@dataclass
class GenerationConfig:
    variants: int = 3
    limit: Optional[int] = None
    random_seed: int = 42


@dataclass
class PathsConfig:
    input_dir: str = ""
    rle_csv: str = ""
    output_dir: str = ""


@dataclass
class MLflowConfig:
    tracking_uri: str = "file:///app/mlruns"
    experiment_name: str = "synthetic_data_generation"
    log_sample_images: int = 10


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sd_defect: SDConfig = field(default_factory=SDConfig)
    sd_background: SDConfig = field(default_factory=SDConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def load_config(yaml_path: str = "config.yaml") -> Config:
    """Загрузка конфигурации из YAML"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    config = Config()
    
    if 'paths' in data:
        config.paths = PathsConfig(**data['paths'])
    
    if 'generation' in data:
        config.generation = GenerationConfig(**data['generation'])
    
    if 'sd_defect' in data:
        config.sd_defect = SDConfig(**data['sd_defect'])
    
    if 'sd_background' in data:
        config.sd_background = SDConfig(**data['sd_background'])
    
    if 'spectral' in data:
        config.spectral = SpectralConfig(**data['spectral'])
    
    if 'scaling' in data:
        factors = [tuple(f) for f in data['scaling']['factors']]
        config.scaling = ScalingConfig(factors=factors)
    
    if 'mlflow' in data:
        config.mlflow = MLflowConfig(**data['mlflow'])
    
    return config


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Обновление конфига из аргументов командной строки"""
    if hasattr(args, 'variants') and args.variants is not None:
        config.generation.variants = args.variants
    if hasattr(args, 'limit') and args.limit is not None:
        config.generation.limit = args.limit
    if hasattr(args, 'seed') and args.seed is not None:
        config.generation.random_seed = args.seed
    
    if hasattr(args, 'sd_strength') and args.sd_strength is not None:
        config.sd_defect.strength_min = config.sd_defect.strength_max = args.sd_strength
    
    return config