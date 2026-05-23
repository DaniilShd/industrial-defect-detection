#!/usr/bin/env python3
"""
Комбинированные функции потерь для Multi-Layer Feature Distillation

Компоненты:
1. Feature Matching Loss - основная потеря согласования признаков
2. Attention Transfer Loss - перенос карт внимания  
3. Relation-Based Loss - сохранение взаимных отношений
4. Cosine Similarity Loss - косинусное расстояние
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MultiLayerDistillationLoss(nn.Module):
    """
    Многоуровневая функция потерь для дистилляции знаний.
    
    Согласует признаки на разных уровнях между ViT учителем и ResNet учеником.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        layer_weights: Optional[Dict[str, float]] = None,
        use_attention_loss: bool = True,
        attention_weight: float = 0.3,
        use_relation_loss: bool = True,
        relation_weight: float = 0.2,
        use_cosine_loss: bool = True,
        cosine_weight: float = 0.5,
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Веса для разных уровней признаков
        self.layer_weights = layer_weights or {
            'low_level': 0.2,   # layer1
            'mid_level': 0.3,   # layer2, layer3
            'high_level': 0.5,  # layer4
        }
        
        # Флаги и веса дополнительных потерь
        self.use_attention_loss = use_attention_loss
        self.attention_weight = attention_weight
        self.use_relation_loss = use_relation_loss
        self.relation_weight = relation_weight
        self.use_cosine_loss = use_cosine_loss
        self.cosine_weight = cosine_weight
        
        # Базовые функции потерь
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        adapted_student_features: Dict[str, torch.Tensor],
        layer_mapping: Dict[int, str]
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисляет все компоненты функции потерь.
        
        Args:
            teacher_features: {block_idx: tensor} признаки учителя
            student_features: {layer_name: tensor} оригинальные признаки ученика
            adapted_student_features: {layer_name: tensor} адаптированные признаки
            layer_mapping: {teacher_block: student_layer}
        
        Returns:
            Dict с компонентами loss и total
        """
        losses = {}
        
        # 1. Основная потеря - согласование признаков
        feature_loss = self._feature_matching_loss(
            teacher_features, adapted_student_features, layer_mapping
        )
        losses['feature_matching'] = feature_loss
        
        # 2. Перенос внимания
        if self.use_attention_loss:
            attn_loss = self._attention_transfer_loss(
                teacher_features, student_features, layer_mapping
            )
            losses['attention'] = attn_loss
        
        # 3. Сохранение отношений
        if self.use_relation_loss:
            rel_loss = self._relation_loss(adapted_student_features)
            losses['relation'] = rel_loss
        
        # 4. Косинусное расстояние
        if self.use_cosine_loss:
            cos_loss = self._cosine_similarity_loss(
                teacher_features, adapted_student_features, layer_mapping
            )
            losses['cosine'] = cos_loss
        
        # Суммируем все потери
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
        """Основная потеря согласования признаков."""
        
        total_loss = 0.0
        num_pairs = 0
        
        for teacher_block, student_layer in layer_mapping.items():
            t_key = f'block_{teacher_block}'
            
            if t_key not in teacher_features or student_layer not in student_features:
                continue
            
            t_feat = teacher_features[t_key]
            s_feat = student_features[student_layer]
            
            # Flatten признаки
            t_flat = self._flatten_features(t_feat)
            s_flat = self._flatten_features(s_feat)
            
            # Приводим к одинаковой размерности
            min_dim = min(t_flat.size(-1), s_flat.size(-1))
            t_flat = t_flat[..., :min_dim]
            s_flat = s_flat[..., :min_dim]
            
            # Нормализуем
            t_norm = F.normalize(t_flat, dim=-1)
            s_norm = F.normalize(s_flat, dim=-1)
            
            # MSE с температурой
            loss = self.mse(
                s_norm / self.temperature,
                t_norm / self.temperature
            )
            
            # Вес в зависимости от уровня
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
        """
        Перенос карт внимания от учителя к ученику.
        
        Учитель (ViT): attention из self-attention слоёв -> [B, N] где N - патчи
        Ученик (CNN): activation-based attention maps -> [B, H, W]
        
        Для сравнения приводим оба к 1D распределению вероятностей.
        """
        
        total_loss = 0.0
        num_pairs = 0
        
        for teacher_block, student_layer in layer_mapping.items():
            t_key = f'block_{teacher_block}'
            
            if t_key not in teacher_features or student_layer not in student_features:
                continue
            
            t_feat = teacher_features[t_key]
            s_feat = student_features[student_layer]
            
            # Вычисляем карты внимания
            t_attn = self._compute_spatial_attention(t_feat)  # ViT: [B, N]
            s_attn = self._compute_spatial_attention(s_feat)  # CNN: [B, H, W]
            
            if t_attn is not None and s_attn is not None:
                # ViT attention: [B, N] -> оставляем как есть
                if t_attn.dim() == 1:
                    t_attn = t_attn.unsqueeze(0)  # добавляем batch dim
                
                # CNN attention: [B, H, W] -> [B, H*W]
                if s_attn.dim() == 3:
                    s_attn = s_attn.view(s_attn.size(0), -1)
                
                # Теперь оба тензора 2D: [B, N_t] и [B, N_s]
                # Приводим к softmax распределению
                t_attn_prob = F.softmax(t_attn / self.temperature, dim=1)
                s_attn_prob = F.log_softmax(s_attn / self.temperature, dim=1)
                
                # Интерполируем меньшее к большему (если размеры разные)
                if t_attn.size(1) != s_attn.size(1):
                    # Интерполируем к среднему размеру
                    target_size = (t_attn.size(1) + s_attn.size(1)) // 2
                    
                    t_attn_prob = F.interpolate(
                        t_attn_prob.unsqueeze(1),
                        size=target_size,
                        mode='linear'
                    ).squeeze(1)
                    
                    s_attn_prob = F.interpolate(
                        s_attn_prob.unsqueeze(1),
                        size=target_size,
                        mode='linear'
                    ).squeeze(1)
                    
                    # Заново нормализуем после интерполяции
                    t_attn_prob = F.softmax(t_attn_prob, dim=1)
                    s_attn_prob = F.log_softmax(s_attn_prob, dim=1)
                
                # KL divergence
                loss = self.kl_div(s_attn_prob, t_attn_prob)
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _relation_loss(
        self,
        student_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Сохраняет взаимные отношения между разными уровнями признаков."""
        
        if len(student_features) < 2:
            return torch.tensor(0.0, device=next(iter(student_features.values())).device)
        
        layers = sorted(student_features.keys())
        features = [self._flatten_features(student_features[l]) for l in layers]
        
        # Нормализуем
        features = [F.normalize(f, dim=-1) for f in features]
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                sim = F.cosine_similarity(
                    features[i].mean(0, keepdim=True),
                    features[j].mean(0, keepdim=True)
                )
                
                target_sim = 0.5
                loss = F.mse_loss(sim, torch.tensor(target_sim, device=sim.device))
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _cosine_similarity_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        layer_mapping: Dict[int, str]
    ) -> torch.Tensor:
        """Косинусное расстояние между признаками."""
        
        total_loss = 0.0
        num_pairs = 0
        
        for teacher_block, student_layer in layer_mapping.items():
            t_key = f'block_{teacher_block}'
            
            if t_key not in teacher_features or student_layer not in student_features:
                continue
            
            t_feat = self._flatten_features(teacher_features[t_key])
            s_feat = self._flatten_features(student_features[student_layer])
            
            min_dim = min(t_feat.size(-1), s_feat.size(-1))
            t_feat = t_feat[..., :min_dim]
            s_feat = s_feat[..., :min_dim]
            
            loss = 1 - F.cosine_similarity(t_feat, s_feat, dim=-1).mean()
            
            total_loss += loss
            num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Приводит признаки любой размерности к 2D тензору [B, D].
        
        ViT: [B, N, D] -> mean pooling по патчам -> [B, D]
        CNN: [B, C, H, W] -> global average pooling -> [B, C]
        """
        
        if features.dim() == 4:
            # CNN features [B, C, H, W]
            return F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        elif features.dim() == 3:
            # ViT features [B, N, D]
            return features.mean(dim=1)
        elif features.dim() == 2:
            return features
        else:
            return features.view(features.size(0), -1)
    
    def _compute_spatial_attention(
        self, features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Вычисляет spatial attention map.
        
        CNN: суммирование квадратов по каналам -> [B, H, W]
        ViT: норма по размерности признаков -> [B, N]
        """
        
        if features.dim() == 4:
            # CNN: [B, C, H, W] -> [B, H, W]
            return features.pow(2).sum(dim=1)
        elif features.dim() == 3:
            # ViT: [B, N, D] -> [B, N]
            return features.norm(dim=-1)
        elif features.dim() == 2:
            # Уже flat: [B, D]
            return features.abs()
        
        return None
    
    def _get_layer_weight(self, layer_name: str) -> float:
        """Возвращает вес для указанного слоя."""
        
        if layer_name == 'layer1':
            return self.layer_weights['low_level']
        elif layer_name in ['layer2', 'layer3']:
            return self.layer_weights['mid_level']
        elif layer_name == 'layer4':
            return self.layer_weights['high_level']
        else:
            return 0.25