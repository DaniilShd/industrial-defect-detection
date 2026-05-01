# analysis/synthetic/scripts/config.py
"""Загрузка и валидация конфигурации"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch


@dataclass
class PathsConfig:
    synthetic_dir: Path
    original_dir: Path
    output_dir: Path
    subdirs: Dict[str, str] = field(default_factory=dict)


@dataclass
class DINOv2Config:
    model_name: str = "facebook/dinov2-small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_samples: int = 500
    nn_test_samples: int = 200
    random_seed: int = 42
    image_size: int = 256


@dataclass
class QualityConfig:
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_fid: bool = True
    compute_lpips: bool = False
    num_samples_fid: int = 300
    ssim_window_size: int = 11
    fid_dims: int = 2048


@dataclass
class ClassAnalysisConfig:
    class_names: Dict[int, str] = field(default_factory=dict)
    num_classes: int = 4
    compute_cooccurrence: bool = True
    compute_bbox_statistics: bool = True


@dataclass
class VisualizationConfig:
    random_samples: int = 20
    dpi: int = 150
    figsize: List[int] = field(default_factory=lambda: [14, 10])
    colormap: str = "viridis"
    style: str = "whitegrid"
    save_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    pca: Dict[str, Any] = field(default_factory=dict)
    tsne: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdsConfig:
    domain_overlap: Dict[str, float] = field(default_factory=dict)
    fidelity: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdditionalAnalysesConfig:
    compute_color_histograms: bool = True
    compute_texture_analysis: bool = True
    compute_edge_density: bool = True
    compute_frequency_analysis: bool = True


@dataclass
class AnalysisConfig:
    paths: PathsConfig
    dinov2: DINOv2Config
    quality: QualityConfig
    class_analysis: ClassAnalysisConfig
    visualization: VisualizationConfig
    thresholds: ThresholdsConfig
    additional_analyses: AdditionalAnalysesConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AnalysisConfig":
        """Загрузка конфигурации из YAML файла"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            paths=PathsConfig(
                synthetic_dir=Path(config_dict['paths']['synthetic_dir']),
                original_dir=Path(config_dict['paths']['original_dir']),
                output_dir=Path(config_dict['paths']['output_dir']),
                subdirs=config_dict['paths'].get('subdirs', {})
            ),
            dinov2=DINOv2Config(**config_dict.get('dinov2', {})),
            quality=QualityConfig(**config_dict.get('quality', {})),
            class_analysis=ClassAnalysisConfig(**config_dict.get('class_analysis', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            thresholds=ThresholdsConfig(**config_dict.get('thresholds', {})),
            additional_analyses=AdditionalAnalysesConfig(**config_dict.get('additional_analyses', {}))
        )
    
    def setup_directories(self):
        """Создание необходимых директорий с timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.paths.output_dir = Path(self.paths.output_dir) / f"analysis_{timestamp}"
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем поддиректории
        for subdir_name, subdir_path in self.paths.subdirs.items():
            full_path = self.paths.output_dir / subdir_path
            full_path.mkdir(parents=True, exist_ok=True)
            setattr(self.paths, f"{subdir_name}_path", full_path)
        
        return self.paths.output_dir