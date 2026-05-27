#!/usr/bin/env python3
"""
ИСПРАВЛЕННАЯ функция потерь для Multi-Layer Feature Distillation
Ключевые изменения:
1. Нормализация L2 для ВСЕХ признаков
2. Smooth L1 вместо MSE
3. Attention через MSE вместо KL divergence
4. Пониженные веса attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MultiLayerDistillationLoss(nn.Module):
    
    def __init__(
        self,
        temperature: float = 4.0,
        layer_weights: Optional[Dict[str, float]] = None,
        use_attention_loss: bool = True,
        attention_weight: float = 0.1,        # ← УМЕНЬШЕНО с 0.3
        use_relation_loss: bool = True,
        relation_weight: float = 0.1,         # ← УМЕНЬШЕНО с 0.2
        use_cosine_loss: bool = True,
        cosine_weight: float = 0.5,
    ):
        super().__init__()
        
        self.temperature = temperature
        
        self.layer_weights = layer_weights or {
            'low_level': 0.2,
            'mid_level': 0.3,
            'high_level': 0.5,
        }
        
        self.use_attention_loss = use_attention_loss
        self.attention_weight = attention_weight
        self.use_relation_loss = use_relation_loss
        self.relation_weight = relation_weight
        self.use_cosine_loss = use_cosine_loss
        self.cosine_weight = cosine_weight
        
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()     # ← ДОБАВЛЕНО
    
    def forward(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        adapted_student_features: Dict[str, torch.Tensor],
        layer_mapping: Dict[int, str]
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # 1. Feature matching — ОСНОВНАЯ потеря
        feature_loss = self._feature_matching_loss(
            teacher_features, adapted_student_features, layer_mapping
        )
        losses['feature_matching'] = feature_loss
        
        # 2. Attention (с меньшим весом)
        if self.use_attention_loss:
            attn_loss = self._attention_transfer_loss(
                teacher_features, student_features, layer_mapping
            )
            losses['attention'] = attn_loss
        
        # 3. Relation
        if self.use_relation_loss:
            rel_loss = self._relation_loss(adapted_student_features)
            losses['relation'] = rel_loss
        
        # 4. Cosine
        if self.use_cosine_loss:
            cos_loss = self._cosine_similarity_loss(
                teacher_features, adapted_student_features, layer_mapping
            )
            losses['cosine'] = cos_loss
        
        # Взвешенная сумма
        total = feature_loss
        
        if self.use_attention_loss:
            total += self.attention_weight * losses['attention']
        if self.use_relation_loss:
            total += self.relation_weight * losses['relation']
        if self.use_cosine_loss:
            total += self.cosine_weight * losses['cosine']
        
        losses['total'] = total
        return losses
    
    def _feature_matching_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        layer_mapping: Dict[int, str]
    ) -> torch.Tensor:
        """ИСПРАВЛЕНО: L2 нормализация + Smooth L1"""
        
        total_loss = 0.0
        num_pairs = 0
        
        for teacher_block, student_layer in layer_mapping.items():
            t_key = f'block_{teacher_block}'
            
            if t_key not in teacher_features or student_layer not in student_features:
                continue
            
            t_feat = teacher_features[t_key]
            s_feat = student_features[student_layer]
            
            # Flatten
            t_flat = self._flatten_features(t_feat)
            s_flat = self._flatten_features(s_feat)
            
            # Обрезаем до одинаковой размерности
            min_dim = min(t_flat.size(-1), s_flat.size(-1))
            t_flat = t_flat[..., :min_dim]
            s_flat = s_flat[..., :min_dim]
            
            # 🔥 L2 НОРМАЛИЗАЦИЯ — КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
            t_norm = F.normalize(t_flat, dim=-1, eps=1e-8)
            s_norm = F.normalize(s_flat, dim=-1, eps=1e-8)
            
            # 🔥 Smooth L1 с температурой
            loss = self.smooth_l1(
                s_norm / self.temperature,
                t_norm / self.temperature
            )
            
            weight = self._get_layer_weight(student_layer)
            total_loss += weight * loss
            num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _attention_transfer_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        layer_mapping: Dict[int, str]
    ) -> torch.Tensor:
        """ИСПРАВЛЕНО: MSE вместо KL, нормализация"""
        
        total_loss = 0.0
        num_pairs = 0
        
        for teacher_block, student_layer in layer_mapping.items():
            t_key = f'block_{teacher_block}'
            
            if t_key not in teacher_features or student_layer not in student_features:
                continue
            
            t_feat = teacher_features[t_key]
            s_feat = student_features[student_layer]
            
            t_attn = self._compute_spatial_attention(t_feat)
            s_attn = self._compute_spatial_attention(s_feat)
            
            if t_attn is not None and s_attn is not None:
                # Приводим к 1D
                if t_attn.dim() == 1:
                    t_attn = t_attn.unsqueeze(0)
                if s_attn.dim() == 3:
                    s_attn = s_attn.view(s_attn.size(0), -1)
                elif s_attn.dim() == 1:
                    s_attn = s_attn.unsqueeze(0)
                
                # 🔥 НОРМАЛИЗАЦИЯ
                t_attn = F.normalize(t_attn, dim=1, eps=1e-8)
                s_attn = F.normalize(s_attn, dim=1, eps=1e-8)
                
                # Интерполяция к одинаковому размеру
                if t_attn.size(1) != s_attn.size(1):
                    target_size = max(t_attn.size(1), s_attn.size(1))
                    t_attn = F.interpolate(t_attn.unsqueeze(1), size=target_size, mode='linear').squeeze(1)
                    s_attn = F.interpolate(s_attn.unsqueeze(1), size=target_size, mode='linear').squeeze(1)
                
                # 🔥 MSE вместо KL divergence
                loss = self.mse(s_attn, t_attn)
                total_loss += loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _relation_loss(self, student_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ИСПРАВЛЕНО: фикс размерности тензора"""
        
        if len(student_features) < 2:
            return torch.tensor(0.0, device=next(iter(student_features.values())).device)
        
        layers = sorted(student_features.keys())
        features = [self._flatten_features(student_features[l]) for l in layers]
        features = [F.normalize(f, dim=-1) for f in features]
        
        total_loss = 0.0
        num_pairs = 0
        device = features[0].device
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                sim = F.cosine_similarity(
                    features[i].mean(0, keepdim=True),
                    features[j].mean(0, keepdim=True)
                )
                
                # 🔥 ФИКС: создаём тензор правильной размерности
                target = torch.ones_like(sim) * 0.5
                loss = F.mse_loss(sim, target)
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _cosine_similarity_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        layer_mapping: Dict[int, str]
    ) -> torch.Tensor:
        
        total_loss = 0.0
        num_pairs = 0
        
        for teacher_block, student_layer in layer_mapping.items():
            t_key = f'block_{teacher_block}'
            
            if t_key not in teacher_features or student_layer not in student_features:
                continue
            
            t_feat = self._flatten_features(teacher_features[t_key])
            s_feat = self._flatten_features(student_features[student_layer])
            
            min_dim = min(t_feat.size(-1), s_feat.size(-1))
            t_feat = F.normalize(t_feat[..., :min_dim], dim=-1)
            s_feat = F.normalize(s_feat[..., :min_dim], dim=-1)
            
            loss = 1 - F.cosine_similarity(t_feat, s_feat, dim=-1).mean()
            total_loss += loss
            num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            return F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        elif features.dim() == 3:
            return features.mean(dim=1)
        elif features.dim() == 2:
            return features
        else:
            return features.view(features.size(0), -1)
    
    def _compute_spatial_attention(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        if features.dim() == 4:
            return features.pow(2).sum(dim=1)
        elif features.dim() == 3:
            return features.norm(dim=-1)
        elif features.dim() == 2:
            return features.abs()
        return None
    
    def _get_layer_weight(self, layer_name: str) -> float:
        if layer_name == 'layer1':
            return self.layer_weights['low_level']
        elif layer_name in ['layer2', 'layer3']:
            return self.layer_weights['mid_level']
        elif layer_name == 'layer4':
            return self.layer_weights['high_level']
        return 0.25