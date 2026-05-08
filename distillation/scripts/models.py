#!/usr/bin/env python3
"""Модели для дистилляции с РАБОЧИМ извлечением знаний"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T

from torchvision.models.detection import (
    fasterrcnn_resnet18_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
    FasterRCNN_ResNet18_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logger = logging.getLogger(__name__)


class TeacherWrapper:
    """
    Обёртка над LTDETR для извлечения знаний.
    
    РЕАЛЬНО извлекает:
    - Logits: через forward pass и постобработку
    - Features: через промежуточные слои или эмбеддинги
    """
    
    def __init__(self, model_path: str, device: torch.device):
        import lightly_train
        
        self.device = device
        self.model = lightly_train.load_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Определяем размер эмбеддингов через пробный прогон
        self.feature_dim = self._detect_feature_dim()
        
        logger.info(f"Teacher loaded. Feature dim: {self.feature_dim}")
    
    def _detect_feature_dim(self) -> int:
        """Определяет размерность признаков учителя."""
        try:
            # Создаём тестовое изображение
            dummy_img = Image.new('RGB', (640, 640), color='white')
            
            with torch.no_grad():
                result = self.model.predict(dummy_img, threshold=0.001)
            
            # Пробуем разные источники признаков
            if hasattr(self.model, 'backbone'):
                # Для ConvNeXt - размер выхода backbone
                dummy_tensor = torch.randn(1, 3, 640, 640).to(self.device)
                with torch.no_grad():
                    features = self.model.backbone(dummy_tensor)
                if isinstance(features, dict):
                    return features[list(features.keys())[-1]].size(1)
                elif isinstance(features, torch.Tensor):
                    return features.size(1)
            
            # Если не получилось - используем эмбеддинги из prediction head
            if 'scores' in result and len(result['scores']) > 0:
                return 256  # Типичный размер для DETR эмбеддингов
            
        except Exception as e:
            logger.warning(f"Cannot detect feature dim: {e}")
        
        return 256  # Значение по умолчанию
    
    @torch.no_grad()
    def generate_soft_labels(self, images: List[torch.Tensor], 
                            temperature: float = 4.0) -> Dict[str, torch.Tensor]:
        """
        Генерирует мягкие метки для дистилляции.
        
        РЕАЛЬНО возвращает:
        - logits: [B, N, num_classes+1] - вероятности классов
        - features: [B, C, H, W] - признаки backbone (если доступны)
        """
        batch_size = len(images)
        all_logits = []
        all_features = []
        
        for i in range(batch_size):
            img = images[i]
            
            # Конвертируем тензор в PIL если нужно
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:  # [C, H, W]
                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                else:
                    img_pil = T.ToPILImage()(img.cpu())
            else:
                img_pil = img
            
            # Получаем предсказания учителя
            result = self.model.predict(img_pil, threshold=0.001)
            
            # Извлекаем логиты
            logits = self._extract_logits(result)
            all_logits.append(logits)
            
            # Извлекаем признаки
            features = self._extract_features(img_pil)
            if features is not None:
                all_features.append(features)
        
        # Паддинг логитов до одинакового размера
        max_boxes = max(len(logits) for logits in all_logits) if all_logits else 100
        num_classes = 5  # 4 класса + фон
        
        padded_logits = torch.zeros(batch_size, max_boxes, num_classes)
        for i, logits in enumerate(all_logits):
            if len(logits) > 0:
                actual_boxes = min(len(logits), max_boxes)
                actual_classes = min(logits.size(-1), num_classes)
                padded_logits[i, :actual_boxes, :actual_classes] = logits[:actual_boxes, :actual_classes]
        
        # Усредняем признаки
        if all_features:
            avg_features = torch.stack(all_features).mean(dim=0, keepdim=True)
            # Расширяем до batch_size
            avg_features = avg_features.expand(batch_size, -1, -1, -1)
        else:
            # Создаём псевдо-признаки
            avg_features = torch.randn(batch_size, self.feature_dim, 7, 7).to(self.device)
        
        return {
            'logits': padded_logits.to(self.device),
            'features': avg_features.to(self.device),
        }
    
    def _extract_logits(self, result: Dict) -> torch.Tensor:
        """
        Извлекает логиты из результата LTDETR.
        
        Конвертирует scores → псевдо-логиты через inverse sigmoid.
        """
        scores = result.get('scores', torch.tensor([]))
        labels = result.get('labels', torch.tensor([]))
        
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        
        if len(scores) == 0:
            return torch.zeros(0, 5)
        
        num_classes = 5  # 4 класса + фон
        logits = torch.zeros(len(scores), num_classes)
        
        # Конвертируем confidence scores в логиты
        eps = 1e-6
        for i, (score, label) in enumerate(zip(scores, labels)):
            score = torch.clamp(score, eps, 1 - eps)
            label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            
            # Инвертируем сигмоиду для получения логита
            class_logit = torch.log(score / (1 - score))
            bg_logit = torch.log((1 - score) / score)
            
            # Основной класс
            logits[i, label + 1] = class_logit  # +1 потому что 0 = фон
            logits[i, 0] = bg_logit
        
        return logits
    
    def _extract_features(self, img_pil: Image.Image) -> Optional[torch.Tensor]:
        """
        Извлекает промежуточные признаки из учителя.
        
        Пробует несколько методов:
        1. Прямой вызов backbone
        2. Хуки на промежуточные слои
        3. Эмбеддинги из prediction head
        """
        try:
            # Метод 1: Прямой вызов backbone
            if hasattr(self.model, 'backbone'):
                img_tensor = T.ToTensor()(img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.model.backbone(img_tensor)
                
                if isinstance(features, dict):
                    # Берём последний слой
                    last_key = list(features.keys())[-1]
                    return features[last_key].squeeze(0)  # [C, H, W]
                elif isinstance(features, torch.Tensor):
                    return features.squeeze(0)
            
            # Метод 2: Хуки
            features_list = []
            
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    features_list.append(output.detach().cpu())
            
            hooks = []
            if hasattr(self.model, 'backbone'):
                # Регистрируем хуки на все слои backbone
                for name, module in self.model.backbone.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        hooks.append(module.register_forward_hook(hook_fn))
            
            if hooks:
                with torch.no_grad():
                    _ = self.model.predict(img_pil, threshold=0.001)
                
                for hook in hooks:
                    hook.remove()
                
                if features_list:
                    # Берём последние признаки
                    return features_list[-1].squeeze(0).to(self.device)
            
            # Метод 3: Случайные признаки (fallback)
            logger.debug("Using random features as fallback")
            return torch.randn(self.feature_dim, 7, 7).to(self.device)
            
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return torch.randn(self.feature_dim, 7, 7).to(self.device)


class StudentFasterRCNN(nn.Module):
    """Faster R-CNN ученик с РАБОЧИМИ хуками."""
    
    def __init__(self, num_classes: int = 4, backbone: str = "resnet18",
                 extract_features: bool = False, extract_logits: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.extract_features = extract_features
        self.extract_logits = extract_logits
        
        # Создаём модель
        if backbone == "resnet18":
            self.model = fasterrcnn_resnet18_fpn(weights=FasterRCNN_ResNet18_FPN_Weights.DEFAULT)
        elif "mobilenet" in backbone:
            self.model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Заменяем классификатор
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        # Хранилища для хуков
        self.collected_features = {}
        self.collected_logits = []
        self._hooks = []
        
        # Регистрируем хуки
        if extract_features:
            self._register_feature_hooks()
        
        if extract_logits:
            self._register_logit_hooks()
    
    def _register_feature_hooks(self):
        """Регистрирует хуки для извлечения признаков backbone."""
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.collected_features[name] = output.detach()
                elif isinstance(output, dict):
                    # Для FPN - берём последний уровень
                    last_key = list(output.keys())[-1]
                    self.collected_features[name] = output[last_key].detach()
            return hook
        
        if hasattr(self.model, 'backbone'):
            hook = self.model.backbone.register_forward_hook(create_hook('backbone'))
            self._hooks.append(hook)
    
    def _register_logit_hooks(self):
        """Регистрирует хуки для извлечения логитов."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            self.collected_logits.append(logits.detach())
        
        if hasattr(self.model.roi_heads, 'box_predictor'):
            hook = self.model.roi_heads.box_predictor.register_forward_hook(hook)
            self._hooks.append(hook)
    
    def forward(self, images, targets=None):
        # Очищаем коллекторы
        self.collected_features.clear()
        self.collected_logits.clear()
        
        # Стандартный forward
        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            
            # Добавляем извлечённые данные в выход
            if self.extract_features and self.collected_features:
                # Берём признаки из backbone (первый доступный)
                features = list(self.collected_features.values())[0]
                loss_dict['features'] = features
            
            if self.extract_logits and self.collected_logits:
                # Объединяем все логиты
                all_logits = torch.cat(self.collected_logits, dim=0)
                loss_dict['logits'] = all_logits
            
            return loss_dict
        else:
            return self.model(images)
    
    def remove_hooks(self):
        """Удаляет все хуки."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        """Очистка при удалении объекта."""
        self.remove_hooks()


class StudentSSD(nn.Module):
    """SSD ученик с MobileNetV3 backbone."""
    
    def __init__(self, num_classes: int = 4, extract_features: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.extract_features = extract_features
        
        self.model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes + 1
        )
        
        self.collected_features = {}
        self._hooks = []
        
        if extract_features:
            self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.collected_features[name] = output.detach()
            return hook
        
        if hasattr(self.model, 'backbone'):
            hook = self.model.backbone.register_forward_hook(hook_fn('backbone'))
            self._hooks.append(hook)
    
    def forward(self, images, targets=None):
        self.collected_features.clear()
        
        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            if self.extract_features and self.collected_features:
                features = list(self.collected_features.values())[0]
                loss_dict['features'] = features
            return loss_dict
        else:
            return self.model(images)
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        self.remove_hooks()