#!/usr/bin/env python3
"""Тренер для дистилляции знаний с полноценной валидацией"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .dataset import DefectDetectionDataset, collate_fn
from .models import TeacherWrapper, StudentFasterRCNN, StudentSSD
from .distillation_loss import DistillationLoss
from .model_loader import save_model_checkpoint, ModelInferenceWrapper
from .evaluate import evaluate_model

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """Тренер для дистилляции знаний с валидацией."""
    
    def __init__(self, config: Dict, student_name: str, student_config: Dict,
                 teacher_path: str, models_dir: Path):
        self.config = config
        self.student_name = student_name
        self.student_config = student_config
        self.models_dir = models_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        # Пути к данным
        dataset_path = Path(config['paths']['experiment_data']) / config['teacher']['dataset']
        self.val_images = dataset_path / "val" / "images"
        self.val_labels = dataset_path / "val" / "labels"
        
        # Учитель
        logger.info("Loading teacher...")
        self.teacher = TeacherWrapper(teacher_path, self.device)
        
        # Ученик
        logger.info(f"Creating student: {student_name}")
        self.student = self._create_student()
        self.student.to(self.device)
        
        # Loss
        self.distill_method = student_config.get('distill_method', 'baseline')
        self.distill_loss = DistillationLoss(
            temperature=student_config.get('temperature', 4.0),
            alpha=student_config.get('alpha', 0.5),
            beta=student_config.get('beta', 0.3),
            use_feature_loss=(self.distill_method in ['fitnet', 'hybrid']),
            use_kd_loss=(self.distill_method in ['kd', 'hybrid']),
        )
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=student_config['lr'],
            weight_decay=0.0005,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=student_config['epochs'],
            eta_min=1e-6,
        )
        
        # Датасеты
        self.train_loader, self.val_loader = self._create_dataloaders(dataset_path)
        
        # Early stopping
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.patience = student_config.get('patience', 15)
        
        # Для отслеживания переобучения
        self.train_losses_history = []
        self.val_maps_history = []
    
    def _create_student(self):
        """Создаёт модель ученика с правильными параметрами."""
        backbone = self.student_config['backbone']
        
        # ВАЖНО: включаем хуки в зависимости от метода дистилляции
        extract_features = self.distill_method in ['fitnet', 'hybrid']
        extract_logits = self.distill_method in ['kd', 'hybrid']
        
        logger.info(f"Student config: features={extract_features}, logits={extract_logits}")
        
        if 'ssd' in self.student_name.lower():
            return StudentSSD(
                num_classes=self.config['classes']['num_classes'],
                extract_features=extract_features,
            )
        else:
            return StudentFasterRCNN(
                num_classes=self.config['classes']['num_classes'],
                backbone=backbone,
                extract_features=extract_features,
                extract_logits=extract_logits,  # ← ТЕПЕРЬ ПЕРЕДАЁМ!
            )
    
    def _create_dataloaders(self, dataset_path: Path):
        """Создаёт DataLoader'ы для train и val."""
        
        # БЕЗ аугментации — датасет уже подготовлен!
        train_dataset = DefectDetectionDataset(
            dataset_path / "train" / "images",
            dataset_path / "train" / "labels",
            num_classes=self.config['classes']['num_classes'],
        )
        
        val_dataset = DefectDetectionDataset(
            dataset_path / "val" / "images",
            dataset_path / "val" / "labels",
            num_classes=self.config['classes']['num_classes'],
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.student_config['batch'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.student_config['batch'],
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )
        
        logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
        return train_loader, val_loader
    
    def train(self) -> Dict:
        """
        Основной цикл обучения с валидацией.
        
        Returns:
            Dict с результатами обучения
        """
        epochs = self.student_config['epochs']
        output_dir = self.models_dir / self.student_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {self.student_name}")
        logger.info(f"Method: {self.distill_method}")
        logger.info(f"Epochs: {epochs}, Batch: {self.student_config['batch']}")
        logger.info(f"Early stopping patience: {self.patience}")
        logger.info(f"{'='*60}\n")
        
        history = []
        start_time = time.time()
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            # Тренировочная эпоха
            train_losses = self._train_epoch(epoch)
            self.train_losses_history.append(train_losses)
            
            # Валидация (каждые N эпох или каждую эпоху)
            val_freq = self.student_config.get('val_freq', 1)
            if epoch % val_freq == 0 or epoch == epochs:
                val_metrics = self._validate()
                self.val_maps_history.append(val_metrics)
            else:
                val_metrics = {'mAP_50': self.val_maps_history[-1]['mAP_50'] if self.val_maps_history else 0.0}
            
            # Сохраняем информацию об эпохе
            epoch_info = {
                'epoch': epoch,
                'train_total_loss': train_losses.get('total_loss', 0.0),
                'train_detection_loss': train_losses.get('detection_loss', 0.0),
                'train_kd_loss': train_losses.get('kd_loss', 0.0),
                'train_feature_loss': train_losses.get('feature_loss', 0.0),
                'val_map50': val_metrics['mAP_50'],
                'val_map75': val_metrics.get('mAP_75', 0.0),
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            history.append(epoch_info)
            
            # Логируем прогресс
            self._log_epoch(epoch_info)
            
            # Сохраняем лучшую модель
            current_map = val_metrics['mAP_50']
            if current_map > self.best_map:
                self.best_map = current_map
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Сохраняем состояние лучшей модели
                best_model_state = {
                    'model_state_dict': {k: v.cpu().clone() for k, v in self.student.state_dict().items()},
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_map50': current_map,
                }
                
                logger.info(f"✅ New best model! mAP@50 = {current_map:.4f}")
                
                # Сохраняем на диск
                self._save_checkpoint(output_dir, epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            # Ранняя остановка
            if self.patience_counter >= self.patience:
                logger.info(f"⏹ Early stopping at epoch {epoch} (best was epoch {self.best_epoch})")
                break
            
            # Шаг планировщика
            self.scheduler.step()
        
        training_time = (time.time() - start_time) / 3600
        
        # Восстанавливаем лучшую модель
        if best_model_state is not None:
            self.student.load_state_dict(best_model_state['model_state_dict'])
            logger.info(f"Restored best model from epoch {self.best_epoch}")
        
        # Сохраняем финальную модель
        model_config = {
            'num_classes': self.config['classes']['num_classes'],
            'backbone': self.student_config['backbone'],
            'distill_method': self.distill_method,
        }
        
        final_model_path = output_dir / 'best_model.pth'
        save_model_checkpoint(
            self.student,
            final_model_path,
            model_config=model_config,
            epoch=self.best_epoch,
            metrics={'val_map50': self.best_map},
        )
        
        # Анализ переобучения
        overfitting_analysis = self._analyze_overfitting(history)
        
        # Формируем результат
        result = {
            'model_path': str(final_model_path),
            'status': 'completed',
            'best_val_map50': self.best_map,
            'best_epoch': self.best_epoch,
            'epochs_trained': len(history),
            'training_time_hours': round(training_time, 3),
            'history': history,
            'distill_method': self.distill_method,
            'overfitting': overfitting_analysis,
        }
        
        # Сохраняем информацию о тренировке
        with open(output_dir / 'training_info.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Логируем финальные результаты
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed: {self.student_name}")
        logger.info(f"Best mAP@50: {self.best_map:.4f} (epoch {self.best_epoch})")
        logger.info(f"Training time: {training_time:.2f} hours")
        logger.info(f"Overfitting detected: {overfitting_analysis['overfitting_detected']}")
        logger.info(f"{'='*60}\n")
        
        return result
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Одна эпоха обучения с дистилляцией."""
        self.student.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'detection_loss': 0.0,
            'kd_loss': 0.0,
            'feature_loss': 0.0,
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Перемещаем на устройство
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Прямой проход ученика
            student_outputs = self.student(images, targets)
            
            # Получаем знания учителя (без градиентов)
            with torch.no_grad():
                teacher_knowledge = self.teacher.generate_soft_labels(
                    images,
                    temperature=self.student_config.get('temperature', 4.0),
                )
            
            # Вычисляем комбинированный loss
            total_loss, loss_components = self.distill_loss(
                student_outputs,
                teacher_knowledge,
                targets,
            )
            
            # Обратный проход
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # Накапливаем лоссы
            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key].item()
            
            # Логируем прогресс каждые 50 батчей
            if batch_idx % 50 == 0:
                logger.debug(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {total_loss.item():.4f} "
                    f"(Det: {loss_components.get('detection', 0):.4f}, "
                    f"KD: {loss_components.get('kd', 0):.4f}, "
                    f"Feat: {loss_components.get('feature', 0):.4f})"
                )
        
        # Усредняем лоссы
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def _validate(self) -> Dict[str, float]:
        """
        Валидация модели на валидационном наборе.
        
        Использует полноценную оценку метрик через evaluate_model.
        """
        self.student.eval()
        
        # Определяем тип модели для оценки
        model_type = 'ssd' if 'ssd' in self.student_name.lower() else 'faster_rcnn'
        
        # Создаём обёртку для инференса
        wrapper = ModelInferenceWrapper(self.student, model_type)
        
        # Запускаем оценку
        try:
            metrics = evaluate_model(
                wrapper,
                model_type,
                self.val_images,
                self.val_labels,
                self.config['classes']['num_classes'],
            )
            
            return {
                'mAP_50': metrics.get('mAP_50', 0.0),
                'mAP_75': metrics.get('mAP_75', 0.0),
                'mAP_50_95': metrics.get('mAP_50_95', 0.0),
                'num_predictions': metrics.get('num_predictions', 0),
                'num_ground_truth': metrics.get('num_ground_truth', 0),
            }
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {'mAP_50': 0.0, 'mAP_75': 0.0, 'mAP_50_95': 0.0}
    
    def _save_checkpoint(self, output_dir: Path, epoch: int, metrics: Dict, is_best: bool = False):
        """Сохраняет чекпоинт модели."""
        if is_best:
            save_path = output_dir / 'best_model.pth'
        else:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        model_config = {
            'num_classes': self.config['classes']['num_classes'],
            'backbone': self.student_config['backbone'],
            'distill_method': self.distill_method,
        }
        
        save_model_checkpoint(
            self.student,
            save_path,
            model_config=model_config,
            optimizer_state=self.optimizer.state_dict(),
            epoch=epoch,
            metrics=metrics,
        )
    
    def _analyze_overfitting(self, history: list) -> Dict:
        """
        Анализирует историю обучения на признаки переобучения.
        
        Признаки:
        1. Train loss продолжает падать, а val mAP падает
        2. Val mAP достигла пика и значительно упала
        3. Большая разница между train и val метриками
        """
        if len(history) < 5:
            return {'overfitting_detected': False, 'warning': 'insufficient_data'}
        
        warning_signs = []
        
        # Извлекаем train loss и val mAP
        train_losses = [h.get('train_total_loss', 0) for h in history]
        val_maps = [h.get('val_map50', 0) for h in history]
        
        # 1. Проверяем падение val mAP после достижения пика
        best_idx = val_maps.index(max(val_maps))
        if best_idx < len(val_maps) - 5:
            recent_val_maps = val_maps[-5:]
            if max(recent_val_maps) < max(val_maps) - 0.02:  # Упала на 2% mAP
                warning_signs.append(
                    f"Val mAP dropped from {max(val_maps):.4f} to {max(recent_val_maps):.4f}"
                )
        
        # 2. Проверяем расхождение train/val трендов
        if len(train_losses) >= 10:
            train_trend = train_losses[-5:] < train_losses[-10:-5]  # Train падает
            val_trend = val_maps[-5:] < val_maps[-10:-5]  # Val падает
            if all(train_trend) and all(val_trend):
                warning_signs.append("Train loss decreasing while val mAP decreasing")
        
        # 3. Проверяем монотонный рост val (подозрительно)
        if len(val_maps) >= 10:
            if all(val_maps[i] <= val_maps[i+1] for i in range(len(val_maps)-1)):
                warning_signs.append("Monotonically increasing val mAP - possible overfitting")
        
        is_overfitting = len(warning_signs) > 0
        
        if is_overfitting:
            logger.warning(f"⚠️ Overfitting detected: {'; '.join(warning_signs)}")
        
        return {
            'overfitting_detected': is_overfitting,
            'warning_signs': warning_signs,
            'best_val_map50': max(val_maps),
            'best_epoch': val_maps.index(max(val_maps)) + 1,
        }
    
    def _log_epoch(self, info: Dict):
        """Логирует информацию об эпохе."""
        msg = (
            f"Epoch {info['epoch']:3d} | "
            f"Loss: {info['train_total_loss']:.4f} | "
            f"Det: {info['train_detection_loss']:.4f} | "
            f"KD: {info['train_kd_loss']:.4f} | "
            f"Feat: {info['train_feature_loss']:.4f} | "
            f"Val mAP@50: {info['val_map50']:.4f}"
        )
        
        if info.get('val_map75', 0) > 0:
            msg += f" | Val mAP@75: {info['val_map75']:.4f}"
        
        if info['val_map50'] == self.best_map and info['val_map50'] > 0:
            msg += " ⭐ NEW BEST!"
        
        logger.info(msg)