#!/usr/bin/env python3
"""Универсальная система загрузки и инференса моделей"""

import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_model(model_path: str, model_type: str = 'auto', **kwargs) -> Any:
    """Универсальный загрузчик моделей."""
    if model_type == 'auto':
        model_type = _detect_model_type(model_path)
    
    logger.info(f"Loading {model_type} model from {model_path}")
    
    if model_type == 'lightly':
        return _load_lightly_model(model_path)
    elif model_type == 'faster_rcnn':
        return _load_faster_rcnn_model(model_path, **kwargs)
    elif model_type == 'ssd':
        return _load_ssd_model(model_path, **kwargs)
    elif model_type == 'yolo':
        return _load_yolo_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _detect_model_type(model_path: str) -> str:
    """Автоопределение типа модели."""
    path = Path(model_path)
    name = path.stem.lower()
    
    if 'faster_rcnn' in name or 'frcnn' in name:
        return 'faster_rcnn'
    elif 'ssd' in name:
        return 'ssd'
    elif 'yolo' in name or 'yolov' in name:
        return 'yolo'
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return 'faster_rcnn'
    except Exception:
        pass
    
    return 'lightly'


def _load_lightly_model(model_path: str):
    """Загружает LightlyTrain модель."""
    import lightly_train
    return lightly_train.load_model(str(model_path))


def _load_faster_rcnn_model(model_path: str, **kwargs):
    """Загружает PyTorch Faster R-CNN модель."""
    from torchvision.models.detection import (
        fasterrcnn_resnet18_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_config' in checkpoint:
        num_classes = checkpoint['model_config'].get('num_classes', 4)
        backbone = checkpoint['model_config'].get('backbone', 'resnet18')
    else:
        num_classes = kwargs.get('num_classes', 4)
        backbone = kwargs.get('backbone', 'resnet18')
    
    if backbone == 'resnet18':
        model = fasterrcnn_resnet18_fpn(weights=None)
    elif 'mobilenet' in backbone:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def _load_ssd_model(model_path: str, **kwargs):
    """Загружает PyTorch SSD модель."""
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_config' in checkpoint:
        num_classes = checkpoint['model_config'].get('num_classes', 4)
    else:
        num_classes = kwargs.get('num_classes', 4)
    
    model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def _load_yolo_model(model_path: str):
    """Загружает Ultralytics YOLO модель."""
    from ultralytics import YOLO
    return YOLO(model_path)


class ModelInferenceWrapper:
    """Универсальная обёртка для инференса разных типов моделей."""
    
    def __init__(self, model: Any, model_type: str):
        self.model = model
        self.model_type = model_type
        
        if hasattr(model, 'parameters'):
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device('cpu')
    
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        """Предсказание для одного изображения."""
        if self.model_type == 'lightly':
            return self._predict_lightly(image_path, conf_threshold)
        elif self.model_type in ['faster_rcnn', 'ssd']:
            return self._predict_torch_detector(image_path, conf_threshold)
        elif self.model_type == 'yolo':
            return self._predict_yolo(image_path, conf_threshold)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _predict_lightly(self, image_path: str, conf_threshold: float) -> Dict:
        """Инференс LightlyTrain моделей."""
        with torch.no_grad():
            result = self.model.predict(image_path, threshold=conf_threshold)
        
        boxes = result.get('bboxes', torch.tensor([]))
        scores = result.get('scores', torch.tensor([]))
        labels = result.get('labels', torch.tensor([]))
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        
        return {'boxes': boxes, 'scores': scores, 'labels': labels}
    
    def _predict_torch_detector(self, image_path: str, conf_threshold: float) -> Dict:
        """Инференс PyTorch детекторов."""
        import torchvision.transforms as T
        
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        if len(predictions) == 0:
            return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}
        
        pred = predictions[0]
        keep = pred['scores'] > conf_threshold
        
        return {
            'boxes': pred['boxes'][keep].cpu(),
            'scores': pred['scores'][keep].cpu(),
            'labels': pred['labels'][keep].cpu() - 1,  # 1-indexed to 0-indexed
        }
    
    def _predict_yolo(self, image_path: str, conf_threshold: float) -> Dict:
        """Инференс YOLO моделей."""
        results = self.model.predict(image_path, conf=conf_threshold, verbose=False)
        
        if not results or len(results) == 0 or results[0].boxes is None:
            return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}
        
        result = results[0]
        return {
            'boxes': result.boxes.xyxy.cpu(),
            'scores': result.boxes.conf.cpu(),
            'labels': result.boxes.cls.cpu().int(),
        }


def save_model_checkpoint(model, save_path: Path, model_config: Dict = None,
                         optimizer_state: Dict = None, epoch: int = None,
                         metrics: Dict = None):
    """Сохраняет PyTorch модель в унифицированном формате."""
    checkpoint = {'model_state_dict': model.state_dict()}
    
    if model_config:
        checkpoint['model_config'] = model_config
    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics:
        checkpoint['metrics'] = metrics
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")