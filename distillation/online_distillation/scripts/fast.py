#!/usr/bin/env python3
"""
Быстрая проверка Online Detection Distillation
Минимальная версия для отладки
"""

import sys, logging, time
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np, yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ПРОСТОЙ ДАТАСЕТ
# ============================================================
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_dir, num_classes=4, size=(640,640), max_samples=50):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.num_classes = num_classes
        self.size = size
        self.files = sorted([f for f in self.img_dir.glob("*") if f.suffix.lower() in {'.jpg','.jpeg','.png'}])[:max_samples]
        logger.info(f"  Loaded {len(self.files)} images from {img_dir}")
        
    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        ow, oh = img.size
        img = img.resize(self.size, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32)/255.0).permute(2,0,1)
        
        boxes, labels = [], []
        lbl_path = self.lbl_dir / f"{self.files[idx].stem}.txt"
        if lbl_path.exists():
            sx, sy = self.size[0]/ow, self.size[1]/oh
            for line in open(lbl_path):
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = int(float(parts[0]))
                if cls >= self.num_classes: continue
                xc, yc, w, h = map(float, parts[1:5])
                x1 = max(0, (xc - w/2) * ow * sx)
                y1 = max(0, (yc - h/2) * oh * sy)
                x2 = min(self.size[0], (xc + w/2) * ow * sx)
                y2 = min(self.size[1], (yc + h/2) * oh * sy)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# БЫСТРЫЙ ТРЕЙНЕР
# ============================================================
class QuickDistillationTrainer:
    def __init__(self, config_path, device='cuda'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        # Загружаем teacher
        self.teacher = None
        teacher_weights = self.config.get('teacher', {}).get('weights')
        if teacher_weights and Path(teacher_weights).exists():
            try:
                import lightly_train
                logger.info(f"Loading teacher from {teacher_weights}")
                self.teacher = lightly_train.load_model(teacher_weights)
                self.teacher.to(self.device)
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad = False
                logger.info("✅ Teacher loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load teacher: {e}")
                self.teacher = None
        else:
            logger.warning("No teacher weights found, running without distillation")
        
        # Создаем студента
        self.model = self._create_student()
        self.model.to(self.device)
        logger.info(f"✅ Student model: Faster R-CNN with ResNet18")
        
        # Данные (только для быстрой проверки)
        data_path = Path(self.config['detection']['data_path'])
        train_ds = SimpleDataset(
            data_path/'train'/'images', 
            data_path/'train'/'labels',
            max_samples=20  # Берем только 20 изображений для скорости
        )
        val_ds = SimpleDataset(
            data_path/'val'/'images', 
            data_path/'val'/'labels',
            max_samples=10
        )
        
        self.train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
    def _create_student(self):
        """Создает студент модель"""
        backbone = resnet_fpn_backbone('resnet18', pretrained=True)
        return FasterRCNN(backbone, num_classes=self.config['detection']['num_classes'] + 1)
    
    def _teacher_predict(self, img_tensor):
        """Предсказание teacher для одного изображения"""
        # Конвертируем тензор в PIL
        img_np = (img_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Получаем предсказание
        pred = self.teacher.predict(img_pil)
        
        return {
            'boxes': torch.tensor(pred['bboxes'], device=self.device).float(),
            'scores': torch.tensor(pred['scores'], device=self.device).float(),
            'labels': torch.tensor(pred['labels'], device=self.device).long(),
        }
    
    def train_step(self, images, targets, use_distill=True):
        """Один шаг обучения"""
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # Detection loss
        loss_dict = self.model(images, targets)
        det_loss = sum(loss_dict.values())
        loss = det_loss
        
        # Distillation loss
        if use_distill and self.teacher:
            distill_loss = torch.tensor(0.0, device=self.device)
            
            # Получаем предсказания teacher
            teacher_preds = [self._teacher_predict(img) for img in images]
            
            # Получаем предсказания student
            stacked = torch.stack(images)
            self.model.eval()
            with torch.no_grad():
                student_preds = self.model(stacked)
            self.model.train()
            
            # Считаем distillation loss
            d_loss = 0.0
            n_pairs = 0
            for t_pred, s_pred in zip(teacher_preds, student_preds):
                if len(t_pred['boxes']) > 0 and len(s_pred['boxes']) > 0:
                    n = min(len(t_pred['boxes']), len(s_pred['boxes']))
                    d_loss += F.mse_loss(s_pred['scores'][:n], t_pred['scores'][:n])
                    d_loss += F.smooth_l1_loss(s_pred['boxes'][:n], t_pred['boxes'][:n])
                    n_pairs += 1
            
            if n_pairs > 0:
                distill_loss = d_loss / n_pairs
                loss = 0.7 * det_loss + 0.3 * distill_loss
        else:
            distill_loss = torch.tensor(0.0, device=self.device)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'det_loss': det_loss.item(),
            'distill_loss': distill_loss.item()
        }
    
    def validate(self):
        """Быстрая валидация"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def quick_test(self, num_epochs=5):
        """Быстрый тест дистилляции"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting quick test for {num_epochs} epochs")
        logger.info(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Решаем, использовать ли дистилляцию (с 3 эпохи)
            use_distill = self.teacher is not None and epoch >= 3
            
            # Обучаем одну эпоху
            epoch_losses = {'total': 0.0, 'det': 0.0, 'distill': 0.0}
            num_batches = 0
            
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                losses = self.train_step(images, targets, use_distill)
                
                epoch_losses['total'] += losses['total_loss']
                epoch_losses['det'] += losses['det_loss']
                epoch_losses['distill'] += losses['distill_loss']
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    logger.info(f"  Epoch {epoch}, Batch {batch_idx}: Total={losses['total_loss']:.4f}, "
                               f"Det={losses['det_loss']:.4f}, Dist={losses['distill_loss']:.4f}")
            
            # Усредняем потери за эпоху
            avg_total = epoch_losses['total'] / num_batches
            avg_det = epoch_losses['det'] / num_batches
            avg_distill = epoch_losses['distill'] / num_batches
            
            # Валидация
            val_loss = self.validate()
            
            # Логируем результаты
            logger.info(f"\n📊 Epoch {epoch} Summary:")
            logger.info(f"  Train Loss: Total={avg_total:.4f}, Det={avg_det:.4f}, Distill={avg_distill:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Distillation: {'✅ ACTIVE' if use_distill else '❌ INACTIVE'}\n")
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ Quick test completed!")
        logger.info(f"{'='*60}")

# ============================================================
# ЗАПУСК
# ============================================================
def main():
    # Путь к конфигу (укажите ваш)
    config_path = Path(__file__).parent.parent / "config_online_distillation.yaml"
    
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        logger.info("Creating minimal config...")
        
        # Создаем минимальную конфигурацию если её нет
        min_config = {
            'detection': {
                'data_path': '/app/data/experiment_v3/datasets/real_baseline',
                'num_classes': 4
            },
            'teacher': {
                'weights': '/app/data/experiment_v3/models/teacher/teacher_mixed_full_ssl/exported_models/exported_best.pt'
            },
            'paths': {
                'detection_output': '/app/distillation/output'
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(min_config, f)
        logger.info(f"✅ Created minimal config at {config_path}")
    
    # Запускаем быстрый тест
    trainer = QuickDistillationTrainer(config_path)
    trainer.quick_test(num_epochs=5)

if __name__ == '__main__':
    main()