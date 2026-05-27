#!/usr/bin/env python3
"""
Современные функции потерь для дистилляции (2025-2026)
Компоненты: Multi-Scale Feature, Ranking Contrastive, 
           Structural Relation, Multi-Scale Masking, Attention Transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class ModernDistillationLoss(nn.Module):
    """Современная дистилляция с адаптивным балансированием"""
    
    def __init__(self, config: dict):
        super().__init__()
        comp = config['modern_distillation']['components']
        
        self.feature_loss = MultiScaleFeatureLoss(comp['feature_matching'])
        self.contrastive_loss = RankingContrastiveLoss(comp['contrastive'])
        self.structural_loss = StructuralRelationLoss(comp['structural'])
        self.masking_loss = MultiScaleMaskingLoss(comp['masking'])
        self.attention_loss = AttentionTransferLoss(comp['attention'])
        
        self.adaptive = comp.get('adaptive', {}).get('enabled', True)
        if self.adaptive:
            self.balancer = AdaptiveWeightBalancer(num_components=5)
        
        self.weights = {
            'feature': comp['feature_matching']['weight'],
            'contrastive': comp['contrastive']['weight'],
            'structural': comp['structural']['weight'],
            'masking': comp['masking']['weight'],
            'attention': comp['attention']['weight'],
        }
    
    def forward(self, teacher_features, student_features, 
                adapted_features, layer_mapping, epoch=1):
        losses = {}
        
        losses['feature'] = self.feature_loss(teacher_features, adapted_features, layer_mapping)
        losses['contrastive'] = self.contrastive_loss(teacher_features, adapted_features, layer_mapping)
        losses['structural'] = self.structural_loss(adapted_features)
        losses['masking'] = self.masking_loss(teacher_features, student_features, layer_mapping)
        losses['attention'] = self.attention_loss(teacher_features, student_features, layer_mapping)
        
        if self.adaptive and epoch > 3:
            weights = self.balancer(losses)
        else:
            weights = self.weights
        
        total = sum(weights.get(k, 0.1) * v for k, v in losses.items())
        losses['total'] = total
        losses['weights'] = weights
        
        return losses


class MultiScaleFeatureLoss(nn.Module):
    """Multi-Scale Feature Matching с адаптивной температурой"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.temperature = config.get('temperature', 3.0)
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, teacher_features, student_features, layer_mapping):
        total_loss = 0.0
        n = 0
        
        for tb, sl in layer_mapping.items():
            t_key = f'block_{tb}'
            if t_key not in teacher_features or sl not in student_features:
                continue
            
            t = self._flatten(teacher_features[t_key])
            s = self._flatten(student_features[sl])
            
            d = min(t.size(-1), s.size(-1))
            t, s = t[..., :d], s[..., :d]
            
            t = F.normalize(t, dim=-1, eps=1e-8)
            s = F.normalize(s, dim=-1, eps=1e-8)
            
            temp = self.temperature * (1 + math.log(d / 64 + 1))
            loss = self.smooth_l1(s / temp, t / temp)
            
            total_loss += loss
            n += 1
        
        return total_loss / max(n, 1)
    
    def _flatten(self, x):
        if x.dim() == 4: return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if x.dim() == 3: return x.mean(1)
        return x


class RankingContrastiveLoss(nn.Module):
    """Ranking-Aware Contrastive Distillation"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.temperature = config.get('temperature', 0.07)
    
    def forward(self, teacher_features, student_features, layer_mapping):
        t_global, s_global = None, None
        
        for tb, sl in layer_mapping.items():
            t_key = f'block_{tb}'
            if t_key not in teacher_features or sl not in student_features:
                continue
            
            tf = teacher_features[t_key]
            sf = student_features[sl]
            
            if tf.dim() == 3: tf = tf.mean(1)
            if sf.dim() == 4: sf = F.adaptive_avg_pool2d(sf, 1).squeeze(-1).squeeze(-1)
            
            if t_global is None:
                t_global, s_global = tf, sf
            else:
                t_global = torch.cat([t_global, tf], dim=-1)
                s_global = torch.cat([s_global, sf], dim=-1)
        
        if t_global is None:
            return torch.tensor(0.0, device=next(iter(teacher_features.values())).device)
        
        t_global = F.normalize(t_global, dim=-1)
        s_global = F.normalize(s_global, dim=-1)
        
        B = t_global.size(0)
        s_sim = torch.mm(s_global, t_global.T) / self.temperature
        
        labels = torch.arange(B, device=s_sim.device)
        loss = F.cross_entropy(s_sim, labels)
        
        return loss


class StructuralRelationLoss(nn.Module):
    """Структурные отношения между уровнями"""
    
    def __init__(self, config: dict):
        super().__init__()
    
    def forward(self, student_features):
        if len(student_features) < 2:
            return torch.tensor(0.0, device=next(iter(student_features.values())).device)
        
        layers = sorted(student_features.keys())
        feats = []
        for l in layers:
            f = student_features[l]
            if f.dim() == 4: f = F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
            if f.dim() == 3: f = f.mean(1)
            feats.append(F.normalize(f, dim=-1))
        
        loss = 0.0
        n = 0
        for i in range(len(feats)):
            for j in range(i+1, len(feats)):
                sim = (feats[i] * feats[j]).sum(-1).mean()
                loss += F.mse_loss(sim, torch.tensor(0.5, device=sim.device))
                n += 1
        
        return loss / max(n, 1)


class MultiScaleMaskingLoss(nn.Module):
    """Multi-Scale Masked Feature Reconstruction"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.mask_ratios = config.get('mask_ratios', [0.3, 0.5, 0.7])
        self.mse = nn.MSELoss()
    
    def forward(self, teacher_features, student_features, layer_mapping):
        total_loss = 0.0
        n = 0
        ratio = self.mask_ratios[torch.randint(0, len(self.mask_ratios), (1,)).item()]
        
        for tb, sl in layer_mapping.items():
            t_key = f'block_{tb}'
            if t_key not in teacher_features or sl not in student_features:
                continue
            
            s_feat = student_features[sl]
            t_feat = teacher_features[t_key]
            
            s_flat = self._flatten(s_feat)
            t_flat = self._flatten(t_feat)
            
            d = min(s_flat.size(-1), t_flat.size(-1))
            s_flat, t_flat = s_flat[..., :d], t_flat[..., :d]
            
            mask = torch.rand_like(s_flat) > ratio
            loss = self.mse(s_flat * mask, F.normalize(t_flat, dim=-1).detach() * mask)
            
            total_loss += loss
            n += 1
        
        return total_loss / max(n, 1)
    
    def _flatten(self, x):
        if x.dim() == 4: return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if x.dim() == 3: return x.mean(1)
        return x


class AttentionTransferLoss(nn.Module):
    """Attention Transfer"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, teacher_features, student_features, layer_mapping):
        total_loss = 0.0
        n = 0
        
        for tb, sl in layer_mapping.items():
            t_key = f'block_{tb}'
            if t_key not in teacher_features or sl not in student_features:
                continue
            
            t = teacher_features[t_key]
            s = student_features[sl]
            
            t_attn = F.normalize(t.norm(dim=-1) if t.dim()==3 else t.pow(2).sum(1).view(t.size(0),-1), dim=1)
            s_attn = F.normalize(s.pow(2).sum(1).view(s.size(0),-1) if s.dim()==4 else s.norm(dim=-1), dim=1)
            
            if t_attn.size(1) != s_attn.size(1):
                sz = max(t_attn.size(1), s_attn.size(1))
                t_attn = F.interpolate(t_attn.unsqueeze(1), size=sz, mode='linear').squeeze(1)
                s_attn = F.interpolate(s_attn.unsqueeze(1), size=sz, mode='linear').squeeze(1)
            
            loss = self.mse(s_attn, t_attn)
            total_loss += loss
            n += 1
        
        return total_loss / max(n, 1)


class AdaptiveWeightBalancer(nn.Module):
    """Адаптивное балансирование весов с минимальным порогом"""
    
    def __init__(self, num_components: int, momentum: float = 0.9, min_weight: float = 0.10):
        super().__init__()
        self.momentum = momentum
        self.min_weight = min_weight  # ← минимальный вес 10%
        self.register_buffer('running', torch.ones(num_components))
    
    def forward(self, losses: Dict[str, torch.Tensor]):
        keys = list(losses.keys())
        
        # Обновляем бегущие средние
        with torch.no_grad():
            for i, k in enumerate(keys):
                if i < len(self.running):
                    self.running[i] = self.momentum * self.running[i] + (1 - self.momentum) * losses[k].detach()
        
        # Вычисляем веса (обратно пропорционально лоссу)
        w = 1.0 / (self.running[:len(keys)] + 1e-8)
        
        # 🔥 Применяем минимальный порог
        w = torch.clamp(w, min=self.min_weight * w.sum())
        
        # Нормализуем
        w = w / w.sum()
        
        return {k: w[i].item() for i, k in enumerate(keys)}