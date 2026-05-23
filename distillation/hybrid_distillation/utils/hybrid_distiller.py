#!/usr/bin/env python3
"""
Гибридный дистиллятор: Multi-Layer + Attention + Relation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import logging
import time
from tqdm import tqdm

from .feature_extractors import ViTFeatureExtractor, ResNetFeatureExtractor
from .hybrid_losses import HybridDistillationLoss

logger = logging.getLogger(__name__)


class FeatureAdapter(nn.Module):
    """Адаптер: ResNet dim -> ViT dim (384)."""
    
    def __init__(self, input_dim: int, output_dim: int = 384, hidden_dim: int = 512):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class HybridDistiller:
    """Гибридная дистилляция знаний."""
    
    def __init__(self, teacher_model, student_model, config: dict, device: torch.device):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.config = config
        self.device = device
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        distill_cfg = config['hybrid_distillation']
        self.teacher_layers = distill_cfg['teacher_layers']
        self.student_layers = distill_cfg['student_layers']
        self.layer_mapping = distill_cfg['layer_mapping']
        self.temperature = distill_cfg['temperature']
        
        self.teacher_extractor = ViTFeatureExtractor(self.teacher, self.teacher_layers)
        self.student_extractor = ResNetFeatureExtractor(self.student)
        
        self.adapters = nn.ModuleDict()
        self._create_adapters(distill_cfg)
        self.adapters.to(device)
        
        self.criterion = HybridDistillationLoss(
            temperature=self.temperature,
            layer_weights=distill_cfg.get('layer_weights'),
            feature_weight=distill_cfg.get('feature_weight', 0.4),
            attention_weight=distill_cfg.get('attention_weight', 0.3),
            relation_weight=distill_cfg.get('relation_weight', 0.2),
            cosine_weight=distill_cfg.get('cosine_weight', 0.1),
            normalize_features=distill_cfg.get('normalize_features', True),
            use_smooth_l1=distill_cfg.get('use_smooth_l1', True),
        )
        
        trainable_params = list(self.student.parameters()) + list(self.adapters.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=distill_cfg['learning_rate'], weight_decay=distill_cfg['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=distill_cfg['epochs'], eta_min=1e-6)
        
        self.best_loss = float('inf')
    
    def _create_adapters(self, config: dict):
        student_dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        teacher_dim = 384
        hidden_dim = config.get('adapter_hidden_dim', 512)
        
        for layer_name in self.student_layers:
            if layer_name in student_dims:
                self.adapters[layer_name] = FeatureAdapter(student_dims[layer_name], teacher_dim, hidden_dim)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.student.train()
        self.adapters.train()
        
        epoch_losses = {'total': 0.0, 'feature_matching': 0.0, 'attention': 0.0, 'relation': 0.0, 'cosine': 0.0}
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for images, _ in pbar:
            images = images.to(self.device)
            
            teacher_features = self.teacher_extractor.extract_features(images)
            student_features = self.student_extractor.extract_features(images)
            adapted_features = self._adapt_features(student_features)
            
            losses = self.criterion(teacher_features, student_features, adapted_features, self.layer_mapping)
            
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(list(self.student.parameters()) + list(self.adapters.parameters()), max_norm=1.0)
            self.optimizer.step()
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            pbar.set_postfix({'loss': f"{losses['total'].item():.4f}", 'feat': f"{losses['feature_matching'].item():.3f}"})
        
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _adapt_features(self, student_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted = {}
        for layer_name, features in student_features.items():
            if layer_name in self.adapters:
                flat_features = self.criterion._flatten_features(features)
                adapted[layer_name] = self.adapters[layer_name](flat_features)
            else:
                adapted[layer_name] = self.criterion._flatten_features(features)
        return adapted
    
    def distill(self, dataloader: DataLoader, epochs: int, output_dir: Path, save_every: int = 10) -> nn.Module:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Hybrid Distillation: {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_losses = self.train_epoch(dataloader, epoch)
            
            logger.info(f"Epoch {epoch:3d}/{epochs} | Total: {epoch_losses['total']:.4f} | Feat: {epoch_losses['feature_matching']:.4f} | Attn: {epoch_losses['attention']:.4f}")
            
            if epoch_losses['total'] < self.best_loss:
                self.best_loss = epoch_losses['total']
                torch.save({'student_state_dict': self.student.state_dict(), 'adapters': {k: v.state_dict() for k, v in self.adapters.items()}, 'epoch': epoch, 'loss': epoch_losses}, output_dir / 'best_model.pt')
            
            if epoch % save_every == 0:
                torch.save({'student_state_dict': self.student.state_dict(), 'epoch': epoch}, output_dir / f'checkpoint_{epoch}.pt')
            
            self.scheduler.step()
        
        elapsed = time.time() - start_time
        logger.info(f"Hybrid Distillation completed in {elapsed/3600:.2f}h | Best loss: {self.best_loss:.4f}")
        
        torch.save(self.student.state_dict(), output_dir / 'backbone_weights.pt')
        self.teacher_extractor.remove_hooks()
        self.student_extractor.remove_hooks()
        
        return self.student