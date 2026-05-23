#!/usr/bin/env python3
"""
Основной класс Multi-Layer Feature Distiller

Реализует полный цикл дистилляции:
1. Извлечение многоуровневых признаков из учителя и ученика
2. Адаптация размерностей через learnable adapters
3. Оптимизация комбинированной функции потерь
4. Сохранение чекпоинтов и мониторинг
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import logging
import time
from tqdm import tqdm

from .feature_extractors import ViTFeatureExtractor, ResNetFeatureExtractor
from .losses import MultiLayerDistillationLoss

logger = logging.getLogger(__name__)


class FeatureAdapter(nn.Module):
    """
    Адаптер для согласования размерностей признаков.
    
    ПРЕОБРАЗУЕТ: признаки ResNet -> размерность ViT
    (а не наоборот!)
    
    ResNet18 слои: 64, 128, 256, 512 каналов
    ViT DINOv3-small: 384-мерные признаки
    
    Адаптер преобразует ResNet признаки в 384-мерное пространство ViT.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 384, hidden_dim: int = 512):
        """
        Args:
            input_dim: размерность признаков ResNet (64, 128, 256, 512)
            output_dim: целевая размерность ViT (384)
            hidden_dim: скрытая размерность адаптера
        """
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),  # output_dim = 384 (ViT dim)
        )
        
        # Инициализация весов
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D_in] признаки ResNet (64, 128, 256 или 512)
        Returns:
            [B, 384] признаки в пространстве ViT
        """
        return self.adapter(x)


class MultiLayerDistiller:
    """
    Multi-Layer Feature Distillation от ViT учителя к ResNet ученику.
    
    Процесс:
    1. Учитель (ViT) извлекает признаки с блоков [4, 8, 12, 16, 20, 24]
    2. Ученик (ResNet) извлекает признаки со стадий [layer1-4]
    3. Адаптеры преобразуют ResNet признаки в размерность ViT (384)
    4. Функция потерь согласует признаки в общем пространстве
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: dict,
        device: torch.device
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.config = config
        self.device = device
        
        # Замораживаем учителя
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Конфигурация дистилляции
        distill_cfg = config['multilayer_distillation']
        self.teacher_layers = distill_cfg['teacher_layers']
        self.student_layers = distill_cfg['student_layers']
        self.layer_mapping = distill_cfg['layer_mapping']
        self.temperature = distill_cfg['temperature']
        
        # Экстракторы признаков
        logger.info("Initializing feature extractors...")
        self.teacher_extractor = ViTFeatureExtractor(
            self.teacher, self.teacher_layers
        )
        self.student_extractor = ResNetFeatureExtractor(self.student)
        
        # Адаптеры для КАЖДОГО слоя студента
        # Преобразуют: ResNet dim -> ViT dim (384)
        logger.info("Creating feature adapters...")
        self.adapters = nn.ModuleDict()
        self._create_adapters(distill_cfg)
        self.adapters.to(device)
        
        # Функция потерь
        self.criterion = MultiLayerDistillationLoss(
            temperature=self.temperature,
            layer_weights=distill_cfg.get('layer_weights'),
            use_attention_loss=distill_cfg.get('use_attention_loss', True),
            attention_weight=distill_cfg.get('attention_weight', 0.3),
            use_relation_loss=distill_cfg.get('use_relation_loss', True),
            relation_weight=distill_cfg.get('relation_weight', 0.2),
            use_cosine_loss=distill_cfg.get('use_cosine_loss', True),
            cosine_weight=distill_cfg.get('cosine_weight', 0.5),
        )
        
        # Оптимизатор (студент + адаптеры)
        trainable_params = (
            list(self.student.parameters()) + 
            list(self.adapters.parameters())
        )
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=distill_cfg['learning_rate'],
            weight_decay=distill_cfg['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=distill_cfg['epochs'],
            eta_min=1e-6
        )
        
        # Статистика
        self.best_loss = float('inf')
        self.current_epoch = 0
        
        logger.info(f"MultiLayerDistiller initialized")
        logger.info(f"  Teacher layers: {self.teacher_layers}")
        logger.info(f"  Student layers: {self.student_layers}")
        logger.info(f"  Layer mapping: {self.layer_mapping}")
        logger.info(f"  Temperature: {self.temperature}")
    
    def _create_adapters(self, config: dict):
        """Создаёт адаптеры для каждого слоя студента."""
        
        # Размерности слоёв ResNet18
        student_dims = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512,
        }
        
        # Целевая размерность ViT
        teacher_dim = 384  # DINOv3 ViT-S/16
        
        hidden_dim = config.get('adapter_hidden_dim', 512)
        
        for layer_name in self.student_layers:
            if layer_name in student_dims:
                input_dim = student_dims[layer_name]
                # Адаптер: ResNet_dim -> ViT_dim (384)
                self.adapters[layer_name] = FeatureAdapter(
                    input_dim=input_dim,      # 64, 128, 256 или 512
                    output_dim=teacher_dim,    # 384
                    hidden_dim=hidden_dim      # 512
                )
                logger.info(f"  Adapter {layer_name}: {input_dim} -> {hidden_dim} -> {teacher_dim}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Обучает одну эпоху."""
        
        self.student.train()
        self.adapters.train()
        
        epoch_losses = {
            'total': 0.0,
            'feature_matching': 0.0,
            'attention': 0.0,
            'relation': 0.0,
            'cosine': 0.0,
        }
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # 1. Извлекаем признаки учителя (ViT)
            teacher_features = self.teacher_extractor.extract_features(images)
            
            # 2. Извлекаем признаки ученика (ResNet)
            student_features = self.student_extractor.extract_features(images)
            
            # 3. Адаптируем признаки ученика в пространство учителя
            adapted_features = self._adapt_student_to_teacher(student_features)
            
            # 4. Вычисляем потери
            losses = self.criterion(
                teacher_features,
                student_features,
                adapted_features,
                self.layer_mapping
            )
            
            # 5. Обратное распространение
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.adapters.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Логируем
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Обновляем progress bar
            postfix = {'loss': f"{losses['total'].item():.4f}"}
            if 'feature_matching' in losses:
                postfix['feat'] = f"{losses['feature_matching'].item():.3f}"
            if 'attention' in losses:
                postfix['attn'] = f"{losses['attention'].item():.3f}"
            pbar.set_postfix(postfix)
        
        # Усредняем потери
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _adapt_student_to_teacher(
        self, student_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Адаптирует признаки студента в пространство учителя.
        
        CNN features [B, C, H, W] -> flatten [B, C] -> adapter -> [B, 384]
        """
        
        adapted = {}
        
        for layer_name, features in student_features.items():
            if layer_name in self.adapters:
                # Flatten CNN features: [B, C, H, W] -> [B, C]
                flat_features = self.criterion._flatten_features(features)
                
                # Применяем адаптер: [B, C] -> [B, 384]
                adapted[layer_name] = self.adapters[layer_name](flat_features)
            else:
                # Если нет адаптера, оставляем как есть (не должно случиться)
                adapted[layer_name] = self.criterion._flatten_features(features)
                logger.warning(f"No adapter for {layer_name}, using raw features")
        
        return adapted
    
    def distill(
        self,
        dataloader: DataLoader,
        epochs: int,
        output_dir: Path,
        save_every: int = 10
    ) -> nn.Module:
        """
        Полный цикл дистилляции.
        
        Args:
            dataloader: DataLoader с неразмеченными изображениями
            epochs: количество эпох
            output_dir: директория для сохранения моделей
            save_every: сохранять чекпоинт каждые N эпох
        
        Returns:
            Обученный студент (ResNet18)
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING MULTI-LAYER DISTILLATION")
        logger.info(f"{'='*60}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batches per epoch: {len(dataloader)}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        history = []
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Обучаем эпоху
            epoch_losses = self.train_epoch(dataloader, epoch)
            
            # Сохраняем в историю
            history.append({
                'epoch': epoch,
                **epoch_losses
            })
            
            # Логируем
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Total: {epoch_losses['total']:.4f} | "
                f"Feat: {epoch_losses['feature_matching']:.4f} | "
                f"Attn: {epoch_losses.get('attention', 0):.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Сохраняем лучшую модель
            if epoch_losses['total'] < self.best_loss:
                self.best_loss = epoch_losses['total']
                self._save_checkpoint(output_dir / 'best_model.pt', epoch, epoch_losses)
                logger.info(f"  ✓ Best model saved (loss: {self.best_loss:.4f})")
            
            # Периодическое сохранение
            if epoch % save_every == 0:
                self._save_checkpoint(
                    output_dir / f'checkpoint_epoch_{epoch}.pt',
                    epoch, epoch_losses
                )
            
            # Шаг scheduler
            self.scheduler.step()
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"DISTILLATION COMPLETED")
        logger.info(f"Time: {elapsed/3600:.2f} hours")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        logger.info(f"{'='*60}")
        
        # Сохраняем финальную модель
        final_path = output_dir / 'final_model.pt'
        self._save_checkpoint(final_path, epochs, epoch_losses)
        logger.info(f"Final model: {final_path}")
        
        # Сохраняем только веса бэкбона (для загрузки в детектор)
        backbone_path = output_dir / 'backbone_weights.pt'
        torch.save(self.student.state_dict(), backbone_path)
        logger.info(f"Backbone weights: {backbone_path}")
        
        # Очищаем хуки
        self.teacher_extractor.remove_hooks()
        self.student_extractor.remove_hooks()
        
        return self.student
    
    def _save_checkpoint(self, path: Path, epoch: int, losses: Dict[str, float]):
        """Сохраняет полный чекпоинт."""
        
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'adapters_state_dict': {
                name: adapter.state_dict() 
                for name, adapter in self.adapters.items()
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)