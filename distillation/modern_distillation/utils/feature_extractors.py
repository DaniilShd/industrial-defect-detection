#!/usr/bin/env python3
"""
Извлечение многоуровневых признаков из учителя (ViT) и ученика (ResNet18)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ViTFeatureExtractor:
    """Извлекает признаки с разных блоков ViT учителя."""
    
    def __init__(self, teacher_model, layer_indices: List[int]):
        """
        Args:
            teacher_model: DINOv3 ViT модель
            layer_indices: индексы блоков для извлечения [4, 8, 12, 16, 20, 24]
        """
        self.teacher = teacher_model
        self.layer_indices = sorted(layer_indices)
        self.features = {}
        
        # Регистрируем хуки
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Регистрирует forward hooks на указанные блоки."""
        
        def hook_fn(idx):
            def fn(module, input, output):
                self.features[f'block_{idx}'] = output
            return fn
        
        # Ищем блоки трансформера
        if hasattr(self.teacher, 'blocks'):
            blocks = self.teacher.blocks
        elif hasattr(self.teacher, 'backbone') and hasattr(self.teacher.backbone, 'blocks'):
            blocks = self.teacher.backbone.blocks
        else:
            logger.warning("Cannot find transformer blocks, trying alternative...")
            # Пробуем через named_modules
            for name, module in self.teacher.named_modules():
                for idx in self.layer_indices:
                    if f'blocks.{idx}' in name and not any(h[0] == name for h in self.hooks):
                        hook = module.register_forward_hook(hook_fn(idx))
                        self.hooks.append((name, hook))
            return
        
        for idx in self.layer_indices:
            if idx < len(blocks):
                hook = blocks[idx].register_forward_hook(hook_fn(idx))
                self.hooks.append((f'block_{idx}', hook))
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Извлекает признаки.
        
        Returns:
            Dict с ключами 'block_4', 'block_8', ...
            Каждый тензор [B, N, D] для ViT (N - патчи + CLS, D - размерность)
        """
        self.features = {}
        
        with torch.no_grad():
            _ = self.teacher(images)
        
        return self.features
    
    def remove_hooks(self):
        """Удаляет все хуки."""
        for _, hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ResNetFeatureExtractor:
    """Извлекает признаки с разных стадий ResNet18."""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: ResNet18 (без FPN)
        """
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Регистрирует хуки на стадии ResNet."""
        
        def hook_fn(name):
            def fn(module, input, output):
                self.features[name] = output
            return fn
        
        # Ищем слои в backbone.body
        target_layers = {
            'layer1': 'layer1',
            'layer2': 'layer2', 
            'layer3': 'layer3',
            'layer4': 'layer4'
        }
        
        if hasattr(self.model, 'body'):
            model_body = self.model.body
        else:
            model_body = self.model
        
        for name, module in model_body.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append((name, hook))
                logger.info(f"Registered hook on {name}")
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Извлекает признаки.
        
        Returns:
            Dict с ключами 'layer1', 'layer2', 'layer3', 'layer4'
            Каждый тензор [B, C, H, W]
        """
        self.features = {}
        
        with torch.no_grad():
            _ = self.model(images)
        
        return self.features
    
    def remove_hooks(self):
        """Удаляет все хуки."""
        for _, hook in self.hooks:
            hook.remove()
        self.hooks.clear()