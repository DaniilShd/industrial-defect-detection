#!/usr/bin/env python3
"""
Современный дистиллятор 2025-2026
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
from .modern_losses import ModernDistillationLoss

logger = logging.getLogger(__name__)


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 384, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x): return self.adapter(x)


class ModernDistiller:
    """Современная дистилляция 2025-2026"""
    
    def __init__(self, teacher_model, student_model, config: dict, device: torch.device):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.config = config
        self.device = device
        
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        
        dcfg = config['modern_distillation']
        self.teacher_layers = dcfg['teacher_layers']
        self.student_layers = dcfg['student_layers']
        self.layer_mapping = dcfg['layer_mapping']
        
        self.teacher_extractor = ViTFeatureExtractor(self.teacher, self.teacher_layers)
        self.student_extractor = ResNetFeatureExtractor(self.student)
        
        self.adapters = nn.ModuleDict()
        self._create_adapters(dcfg)
        self.adapters.to(device)
        
        self.criterion = ModernDistillationLoss(config)
        
        params = list(self.student.parameters()) + list(self.adapters.parameters())
        self.optimizer = optim.AdamW(params, lr=dcfg['learning_rate'], weight_decay=dcfg['weight_decay'])
        
        warmup = dcfg.get('warmup_epochs', 5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=dcfg['epochs']-warmup, eta_min=1e-6)
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup)
        
        self.best_loss = float('inf')
    
    def _create_adapters(self, config: dict):
        dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        hdim = config.get('adapter_hidden_dim', 1024)
        dropout = config.get('adapter_dropout', 0.1)
        
        for ln in self.student_layers:
            if ln in dims:
                self.adapters[ln] = FeatureAdapter(dims[ln], 384, hdim, dropout)
                logger.info(f"  Adapter {ln}: {dims[ln]} → {hdim} → 384")
    
    def train_epoch(self, dataloader, epoch):
        self.student.train()
        self.adapters.train()
        
        losses = {'total': 0, 'feature': 0, 'contrastive': 0, 'structural': 0, 'masking': 0, 'attention': 0}
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for images, _ in pbar:
            images = images.to(self.device)
            
            tf = self.teacher_extractor.extract_features(images)
            sf = self.student_extractor.extract_features(images)
            af = self._adapt(sf)
            
            loss_dict = self.criterion(tf, sf, af, self.layer_mapping, epoch)
            
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(list(self.student.parameters())+list(self.adapters.parameters()), 1.0)
            self.optimizer.step()
            
            for k in losses:
                if k in loss_dict: losses[k] += loss_dict[k].item()
            
            pbar.set_postfix({'loss': f"{loss_dict['total'].item():.4f}"})
        
        n = len(dataloader)
        return {k: v/n for k, v in losses.items()}
    
    def _adapt(self, sf):
        adapted = {}
        for ln, f in sf.items():
            if ln in self.adapters:
                flat = f.mean(dim=(2,3)) if f.dim()==4 else f.mean(1)
                adapted[ln] = self.adapters[ln](flat)
            else:
                adapted[ln] = f.mean(dim=(2,3)) if f.dim()==4 else f.mean(1)
        return adapted
    
    def distill(self, dataloader, epochs, output_dir, save_every=10):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Modern Distillation: {epochs} epochs")
        t0 = time.time()
        
        for epoch in range(1, epochs+1):
            losses = self.train_epoch(dataloader, epoch)
            
            logger.info(f"Epoch {epoch:3d}/{epochs} | Total: {losses['total']:.4f} | "
                       f"Feat: {losses['feature']:.4f} | Contr: {losses['contrastive']:.4f} | "
                       f"Mask: {losses['masking']:.4f}")
            
            if epoch <= self.config['modern_distillation'].get('warmup_epochs', 5):
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                torch.save({'student_state_dict': self.student.state_dict(), 'epoch': epoch}, output_dir/'best_model.pt')
            
            if epoch % save_every == 0:
                torch.save({'student_state_dict': self.student.state_dict(), 'epoch': epoch}, output_dir/f'checkpoint_{epoch}.pt')
        
        logger.info(f"Completed in {(time.time()-t0)/3600:.2f}h | Best loss: {self.best_loss:.4f}")
        
        torch.save(self.student.state_dict(), output_dir/'backbone_weights.pt')
        self.teacher_extractor.remove_hooks()
        self.student_extractor.remove_hooks()
        
        return self.student