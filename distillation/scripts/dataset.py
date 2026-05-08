#!/usr/bin/env python3
"""Датасет для детекции дефектов"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class DefectDetectionDataset(Dataset):
    """Датасет для детекции дефектов с аугментациями."""
    
    def __init__(self, images_dir: Path, labels_dir: Path, 
                num_classes: int = 4, img_size: Tuple[int, int] = (640, 640)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.num_classes = num_classes
        
        self.image_files = sorted([
            f for f in images_dir.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        ])
        
        # Только ресайз и нормализация
        self.transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=1,
            label_fields=['class_labels']
        ))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes, labels = self._load_annotations(img_path)
        
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            img_tensor = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            img_tensor = transformed['image']
            boxes = []
            labels = []
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor([l + 1 for l in labels], dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]) if boxes else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }
        
        return img_tensor, target
    
    def _load_annotations(self, img_path: Path) -> Tuple[List, List]:
        boxes, labels = [], []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            return boxes, labels
        
        with Image.open(img_path) as img:
            iw, ih = img.size
        
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    if cls >= self.num_classes:
                        continue
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0, (xc - w / 2) * iw)
                    y1 = max(0, (yc - h / 2) * ih)
                    x2 = min(iw, (xc + w / 2) * iw)
                    y2 = min(ih, (yc + h / 2) * ih)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
                except (ValueError, IndexError):
                    continue
        
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))