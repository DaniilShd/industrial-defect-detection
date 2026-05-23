#!/usr/bin/env python3
"""
Multi-Layer Feature Distillation: ViT учитель → ResNet18 ученик

Процесс:
  1. Загрузка учителя LTDETR (DINOv3 ViT-S/16)
  2. Создание ученика ResNet18
  3. Извлечение многоуровневых признаков
  4. Обучение через согласование признаков
  5. Сохранение предобученного бэкбона
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Добавляем путь к utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.multilayer_distiller import MultiLayerDistiller
from utils.feature_extractors import ViTFeatureExtractor, ResNetFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multilayer_distill.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class UnlabeledImageDataset(torch.utils.data.Dataset):
    """Датасет для неразмеченных изображений."""
    
    def __init__(self, image_dir: Path, img_size: tuple = (224, 224)):
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.images = [
            f for f in self.image_dir.rglob("*")
            if f.suffix.lower() in extensions
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Found {len(self.images)} images in {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # Возвращаем (image, dummy_label) для совместимости
        return image, 0


def load_teacher_model(weights_path: str, device: torch.device):
    """Загружает модель учителя."""
    
    import lightly_train
    
    logger.info(f"Loading teacher from {weights_path}")
    model = lightly_train.load_model(weights_path)
    
    # Извлекаем только бэкбон (без детекторной головы)
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    elif hasattr(model, 'model') and hasattr(model.model, 'backbone'):
        backbone = model.model.backbone
    else:
        raise ValueError("Cannot extract backbone from teacher")
    
    logger.info(f"Teacher backbone loaded: {type(backbone)}")
    return backbone.to(device)


def main():
    """Основная функция."""
    
    config_path = Path(__file__).parent.parent / "config_multilayer_distillation.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    distill_cfg = config['multilayer_distillation']
    
    # Создаём выходную директорию
    output_dir = Path(config['paths']['pretrain_output']) / "multilayer_distilled"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Загружаем учителя
    teacher_weights = config['teacher']['weights']
    teacher = load_teacher_model(teacher_weights, device)
    
    # 2. Создаём ученика
    student = resnet18(pretrained=False)
    logger.info(f"Student created: ResNet18 ({sum(p.numel() for p in student.parameters()):,} params)")
    
    # 3. Даталоадер
    data_path = Path(distill_cfg['unlabeled_data'])
    dataset = UnlabeledImageDataset(data_path, tuple(distill_cfg['image_size']))
    
    dataloader = DataLoader(
        dataset,
        batch_size=distill_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"DataLoader: {len(dataloader)} batches of size {distill_cfg['batch_size']}")
    
    # 4. Дистилляция
    distiller = MultiLayerDistiller(
        teacher_model=teacher,
        student_model=student,
        config=config,
        device=device
    )
    
    logger.info("=" * 80)
    logger.info("STARTING MULTI-LAYER FEATURE DISTILLATION")
    logger.info("=" * 80)
    logger.info(f"Epochs: {distill_cfg['epochs']}")
    logger.info(f"Teacher layers: {distill_cfg['teacher_layers']}")
    logger.info(f"Student layers: {distill_cfg['student_layers']}")
    logger.info(f"Temperature: {distill_cfg['temperature']}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    distilled_backbone = distiller.distill(
        dataloader=dataloader,
        epochs=distill_cfg['epochs'],
        output_dir=output_dir
    )
    
    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    
    logger.info(f"\n✅ Multi-Layer Distillation completed in {elapsed:.2f} hours")
    logger.info(f"Output saved to: {output_dir}")
    
    # Сохраняем путь для следующего шага
    cache_file = Path(config['paths']['pretrain_output']) / "multilayer_model_path.txt"
    cache_file.write_text(str(output_dir / 'backbone_weights.pt'))
    
    logger.info(f"Path cached to: {cache_file}")


if __name__ == "__main__":
    main()