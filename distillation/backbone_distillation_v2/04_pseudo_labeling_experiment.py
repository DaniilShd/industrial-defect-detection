#!/usr/bin/env python3
"""
Эксперимент с псевдоразметкой:
1. Генерация псевдоразметки учителем LTDETR на 4000 сырых патчах
2. Обучение 3 вариантов Faster R-CNN (scratch, imagenet, distilled) 
   на исходном датасете + псевдоразмеченных данных
3. Полная оценка с COCO-метриками, FPS, анализом датасета

Все результаты сохраняются в отдельную директорию, не пересекаясь с основным экспериментом.
"""

import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

import lightly_train

# ============================================================================
# ЛОГГЕР
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pseudo_experiment.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

def load_config():
    """Загрузка конфигурации из config.yaml с дополнениями для псевдоразметки."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Конфиг не найден: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Добавляем параметры для псевдоразметки
    cfg["pseudo_labeling"] = {
        "enabled": True,
        "raw_patches_dir": "/app/data/processed/defect_patches/images/train",
        "pseudo_count": 4000,
        "pseudo_conf": 0.7,
        "seed": 42,
    }
    
    # Отдельная директория для результатов псевдо-эксперимента
    cfg["paths"]["pseudo_output"] = "results/pseudo_experiment"
    
    return cfg


# ============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ПСЕВДОРАЗМЕТКИ
# ============================================================================

class PseudoLabelGenerator:
    """Генерация псевдоразметки с помощью модели-учителя."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = None
        
    def load_teacher(self):
        """Загрузка модели учителя."""
        teacher_path = self.cfg["teacher"]["detector_path"]
        logger.info(f"Загрузка учителя: {teacher_path}")
        
        if not Path(teacher_path).exists():
            raise FileNotFoundError(f"Модель учителя не найдена: {teacher_path}")
        
        self.teacher = lightly_train.load_model(teacher_path)
        self.teacher.to(self.device)
        self.teacher.eval()
        
        # Замораживаем параметры
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        logger.info("✅ Учитель загружен")
    
    def generate(self) -> Path:
        """Генерация псевдоразметки и сохранение датасета."""
        pseudo_cfg = self.cfg["pseudo_labeling"]
        output_dir = Path(self.cfg["paths"]["pseudo_output"]) / "pseudo_dataset"
        img_dir = output_dir / "images"
        lbl_dir = output_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Собираем все патчи
        patches_dir = Path(pseudo_cfg["raw_patches_dir"])
        if not patches_dir.exists():
            logger.error(f"Директория с патчами не найдена: {patches_dir}")
            sys.exit(1)
        
        all_patches = list(patches_dir.glob("*.jpg")) + list(patches_dir.glob("*.png"))
        logger.info(f"Найдено патчей: {len(all_patches)}")
        
        # Выбираем случайные патчи
        random.seed(pseudo_cfg["seed"])
        selected = random.sample(all_patches, min(pseudo_cfg["pseudo_count"], len(all_patches)))
        logger.info(f"Выбрано для псевдоразметки: {len(selected)}")
        
        if self.teacher is None:
            self.load_teacher()
        
        # Статистика
        stats = {
            "total_processed": 0,
            "total_boxes": 0,
            "class_distribution": defaultdict(int),
            "failed_images": [],
        }
        
        logger.info("Генерация псевдоразметки...")
        for patch_path in tqdm(selected, desc="Pseudo-labeling"):
            try:
                # Получаем предсказания учителя
                results = self.teacher.predict(str(patch_path))
                
                boxes = results.get("bboxes", [])
                labels = results.get("labels", [])
                scores = results.get("scores", [])
                
                # Фильтруем по уверенности
                if len(scores) > 0:
                    mask = scores > pseudo_cfg["pseudo_conf"]
                    boxes = boxes[mask] if hasattr(boxes, '__getitem__') else [b for b, m in zip(boxes, mask) if m]
                    labels = labels[mask] if hasattr(labels, '__getitem__') else [l for l, m in zip(labels, mask) if m]
                
                if len(boxes) == 0:
                    continue
                
                # Загружаем изображение для получения размеров
                img = Image.open(patch_path).convert("RGB")
                w, h = img.size
                
                # Конвертируем в YOLO формат
                yolo_lines = []
                for box, label in zip(boxes, labels):
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    if hasattr(label, 'item'):
                        label = label.item()
                    
                    x1, y1, x2, y2 = box
                    cls_id = int(label)
                    
                    # Нормализованные координаты для YOLO
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    # Проверяем валидность
                    if 0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0 and bh > 0:
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        stats["class_distribution"][str(cls_id)] += 1
                
                if yolo_lines:
                    # Сохраняем изображение и разметку
                    img.save(img_dir / patch_path.name)
                    with open(lbl_dir / f"{patch_path.stem}.txt", "w") as f:
                        f.write("\n".join(yolo_lines))
                    
                    stats["total_processed"] += 1
                    stats["total_boxes"] += len(yolo_lines)
                    
            except Exception as e:
                stats["failed_images"].append(str(patch_path.name))
                logger.debug(f"Ошибка обработки {patch_path.name}: {e}")
        
        # Сохраняем статистику
        stats["failed_images"] = stats["failed_images"][:10]  # Только первые 10 ошибок
        stats_path = output_dir / "pseudo_generation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Псевдоразметка завершена:")
        logger.info(f"   Обработано изображений: {stats['total_processed']}")
        logger.info(f"   Всего боксов: {stats['total_boxes']}")
        logger.info(f"   Распределение по классам: {dict(stats['class_distribution'])}")
        
        return output_dir


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class YOLODataset(Dataset):
    """Датасет в YOLO формате."""
    
    def __init__(self, images_dir: Path, labels_dir: Path,
                 num_classes: int, img_size: Tuple[int, int] = (640, 640)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted([
            f for f in self.images_dir.glob("*")
            if f.suffix.lower() in exts
        ])
        
        logger.info(f"Датасет {images_dir.name}: {len(self.image_files)} изображений")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            image = image.resize(self.img_size, Image.BILINEAR)
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except Exception as e:
            logger.error(f"Ошибка загрузки {img_path}: {e}")
            return torch.zeros(3, *self.img_size), {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
        
        boxes, labels = self._parse_yolo(
            self.labels_dir / f"{img_path.stem}.txt", orig_w, orig_h
        )
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
        
        return img_tensor, target
    
    def _parse_yolo(self, label_path: Path, orig_w: int, orig_h: int):
        """Парсинг YOLO разметки."""
        boxes, labels = [], []
        
        if not label_path.exists():
            return boxes, labels
        
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
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
                    x1 = max(0.0, (xc - w/2) * orig_w * scale_x)
                    y1 = max(0.0, (yc - h/2) * orig_h * scale_y)
                    x2 = min(float(self.img_size[1]), (xc + w/2) * orig_w * scale_x)
                    y2 = min(float(self.img_size[0]), (yc + h/2) * orig_h * scale_y)
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)  # Faster R-CNN: 0 = фон
                except (ValueError, IndexError):
                    continue
        
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# ЗАГРУЗКА ВЕСОВ ИЗ LIGHTLYTRAIN
# ============================================================================

def _strip_prefix(key: str) -> str:
    """Удаление известных префиксов из ключей state_dict."""
    prefixes = [
        "model.backbone.body.", "model.backbone.",
        "student_model.backbone.body.", "student_model.backbone.",
        "backbone.body.", "backbone.", "body.",
        "student_model.", "module.", "model.",
    ]
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def load_lightly_backbone(model: FasterRCNN, ckpt_path: str) -> int:
    """Загружает веса backbone из экспортированной LightlyTrain модели."""
    logger.info(f"  Загрузка весов LightlyTrain из: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state_dict = ckpt
    
    body = model.backbone.body
    body_sd = body.state_dict()
    
    mapped = {}
    for k, v in state_dict.items():
        clean = _strip_prefix(k)
        if clean in body_sd and v.shape == body_sd[clean].shape:
            mapped[clean] = v
    
    body.load_state_dict({**body_sd, **mapped}, strict=False)
    logger.info(f"  ✓ Загружено ключей: {len(mapped)}/{len(body_sd)}")
    
    return len(mapped)


# ============================================================================
# ТРЕНЕР
# ============================================================================

class PseudoExperimentTrainer:
    """Тренер для экспериментов с псевдоразметкой."""
    
    def __init__(self, cfg: dict, name: str, group_cfg: dict,
                 train_loader: DataLoader, val_loader: DataLoader,
                 backbone_ckpt: Optional[str] = None):
        self.cfg = cfg
        self.name = name
        self.group_cfg = group_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.backbone_ckpt = backbone_ckpt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["detection"]["num_classes"]
        
        self.model = self._build_model()
        self.model.to(self.device)
        
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=group_cfg["lr"],
            weight_decay=group_cfg.get("weight_decay", 5e-4)
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=group_cfg["epochs"], eta_min=1e-6
        )
        
        self.out_dir = Path(cfg["paths"]["pseudo_output"]) / "detectors" / name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience = group_cfg.get("patience", 15)
        self._patience_cnt = 0
    
    def _build_model(self):
        """Создание модели с соответствующей инициализацией."""
        init_type = self.group_cfg["type"]
        logger.info(f"Создание модели [{self.name}], init={init_type}")
        
        backbone = resnet_fpn_backbone("resnet18", pretrained=False)
        model = FasterRCNN(backbone, num_classes=self.num_classes + 1)
        
        if init_type == "imagenet_pretrained":
            logger.info("  Загрузка ImageNet весов...")
            pretrained_bb = resnet_fpn_backbone("resnet18", pretrained=True)
            model.backbone.load_state_dict(pretrained_bb.state_dict())
        elif init_type == "lightly_pretrained":
            if self.backbone_ckpt and Path(self.backbone_ckpt).exists():
                load_lightly_backbone(model, self.backbone_ckpt)
            else:
                logger.warning(f"  ⚠️  Чекпоинт не найден: {self.backbone_ckpt}")
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Параметры: всего={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M")
        
        return model
    
    def train(self):
        """Основной цикл обучения."""
        logger.info(f"\n{'='*70}\nОБУЧЕНИЕ: {self.name}\n{'='*70}")
        
        history = []
        best_state = None
        start_time = time.time()
        
        for epoch in range(1, self.group_cfg["epochs"] + 1):
            # Тренировка
            self.model.train()
            total_loss = 0.0
            
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.group_cfg['epochs']}"):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            
            # Валидация
            metrics = self._validate()
            map50_95 = metrics["map"].item()
            map50 = metrics["map_50"].item()
            map75 = metrics["map_75"].item()
            
            marker = " ⭐" if map50_95 > self.best_map else ""
            logger.info(
                f"Epoch {epoch:3d}/{self.group_cfg['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"mAP50:95={map50_95:.4f} | "
                f"mAP50={map50:.4f} | "
                f"mAP75={map75:.4f} | "
                f"LR: {self.opt.param_groups[0]['lr']:.2e}{marker}"
            )
            
            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "map50_95": map50_95,
                "map50": map50,
                "map75": map75,
                "lr": self.opt.param_groups[0]["lr"],
            })
            
            # Сохранение лучшей модели
            if map50_95 > self.best_map:
                self.best_map = map50_95
                self.best_epoch = epoch
                self._patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save("best_model.pth")
            else:
                self._patience_cnt += 1
                if self._patience_cnt >= self.patience:
                    logger.info(f"Early stopping на эпохе {epoch}")
                    break
            
            self.sched.step()
        
        # Загрузка лучшей модели
        if best_state:
            self.model.load_state_dict(best_state)
        self._save("model_final.pth")
        
        training_time = (time.time() - start_time) / 3600
        
        result = {
            "model_name": self.name,
            "init_type": self.group_cfg["type"],
            "best_map50_95": self.best_map,
            "best_map50": max(h["map50"] for h in history),
            "best_map75": max(h["map75"] for h in history),
            "best_epoch": self.best_epoch,
            "epochs_trained": len(history),
            "training_time_hours": round(training_time, 2),
        }
        
        # Сохраняем историю
        (self.out_dir / "history.json").write_text(
            json.dumps({"history": history, "summary": result}, indent=2)
        )
        
        logger.info(
            f"✅ {self.name} обучена | "
            f"Best mAP50:95={self.best_map:.4f} (epoch {self.best_epoch}) | "
            f"Время: {training_time:.2f}ч"
        )
        
        return result
    
    def _validate(self):
        """Валидация модели."""
        metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
        self.model.eval()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [i.to(self.device) for i in images]
                outputs = self.model(images)
                
                valid_preds = []
                valid_targets = []
                
                for out, t in zip(outputs, targets):
                    pred_dict = {
                        "boxes": out["boxes"].cpu(),
                        "scores": out["scores"].cpu(),
                        "labels": (out["labels"] - 1).cpu().clamp(min=0),
                    }
                    target_dict = {
                        "boxes": t["boxes"].cpu(),
                        "labels": (t["labels"] - 1).cpu().clamp(min=0),
                    }
                    valid_preds.append(pred_dict)
                    valid_targets.append(target_dict)
                
                if len(valid_preds) == len(valid_targets):
                    metric.update(valid_preds, valid_targets)
        
        return metric.compute()
    
    def _save(self, fname: str):
        """Сохранение чекпоинта."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_map50_95": self.best_map,
            "best_epoch": self.best_epoch,
            "config": self.group_cfg,
        }, self.out_dir / fname)
    
    def get_model(self):
        """Возвращает модель для оценки."""
        return self.model


# ============================================================================
# ОЦЕНКА
# ============================================================================

class PseudoExperimentEvaluator:
    """Оценка моделей с полным набором COCO-метрик."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["detection"]["num_classes"]
        self.img_size = tuple(cfg["detection"]["img_size"])
        self.class_names = list(cfg["detection"].get("class_names", {}).values())
    
    def evaluate_all(self, trained_models: Dict[str, PseudoExperimentTrainer]):
        """Оценка всех обученных моделей и учителя."""
        # Подготовка тестовых данных
        data_path = Path(self.cfg["detection"]["data_path"])
        
        # Ищем test, если нет - используем val
        test_imgs = data_path / "test" / "images"
        test_lbls = data_path / "test" / "labels"
        if not test_imgs.exists():
            logger.warning("test/ не найден, используем val/")
            test_imgs = data_path / "val" / "images"
            test_lbls = data_path / "val" / "labels"
        
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(f for f in test_imgs.glob("*") if f.suffix.lower() in exts)
        
        logger.info(f"Оценка на {len(image_files)} изображениях")
        
        results_dir = Path(self.cfg["paths"]["pseudo_output"]) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        # Анализ датасета
        dataset_stats = self._analyze_dataset(image_files, test_lbls)
        
        # Оценка учителя
        teacher_result = self._evaluate_teacher(image_files, test_lbls)
        if teacher_result:
            all_results.append(teacher_result)
        
        # Оценка студентов
        for name, trainer in trained_models.items():
            student_result = self._evaluate_student(name, trainer, image_files, test_lbls)
            if student_result:
                all_results.append(student_result)
        
        # Сохранение результатов
        final_results = {
            "dataset_analysis": dataset_stats,
            "models": all_results,
            "evaluation_params": {
                "num_test_images": len(image_files),
                "img_size": self.img_size,
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "experiment_type": "pseudo_labeling",
            }
        }
        
        out_path = results_dir / "evaluation.json"
        with open(out_path, "w") as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\nРезультаты сохранены в: {out_path}")
        self._print_summary(all_results, dataset_stats)
        
        return final_results
    
    def _evaluate_teacher(self, image_files: List[Path], labels_dir: Path):
        """Оценка модели учителя."""
        teacher_path = self.cfg["teacher"]["detector_path"]
        if not teacher_path or not Path(teacher_path).exists():
            logger.warning("⚠️  Учитель не найден для оценки")
            return None
        
        logger.info(f"\n{'='*70}\nОЦЕНКА УЧИТЕЛЯ (LTDETR)\n{'='*70}")
        
        try:
            teacher = lightly_train.load_model(teacher_path)
            teacher.eval()
            
            metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", class_metrics=True)
            
            for img_path in tqdm(image_files, desc="Evaluating teacher"):
                try:
                    results = teacher.predict(str(img_path))
                    
                    pred = {
                        "boxes": results["bboxes"],
                        "scores": results["scores"],
                        "labels": results["labels"],
                    }
                    
                    # Загружаем ground truth
                    gt = self._load_gt(img_path, labels_dir)
                    
                    if len(pred["boxes"]) > 0 or len(gt["boxes"]) > 0:
                        metric.update([pred], [gt])
                except Exception as e:
                    logger.debug(f"Ошибка оценки {img_path.name}: {e}")
            
            result = metric.compute()
            
            model_result = {
                "model": "teacher_ltdetr",
                "type": "teacher",
                "map50_95": float(result["map"].item()),
                "map50": float(result["map_50"].item()),
                "map75": float(result["map_75"].item()),
            }
            
            # Per-class AP
            if "map_per_class" in result and result["map_per_class"].numel() > 0:
                per_cls = result["map_per_class"].tolist()
                for i, (name, ap) in enumerate(zip(self.class_names, per_cls)):
                    model_result[f"AP50_{name}"] = round(float(ap), 4)
            
            # FPS и параметры
            fps_stats = self._measure_fps(teacher, image_files[0])
            model_result.update(fps_stats)
            model_result.update(self._model_stats(teacher))
            
            logger.info(f"✅ Учитель | mAP50:95={model_result['map50_95']:.4f} | "
                       f"mAP50={model_result['map50']:.4f} | FPS={model_result['fps']:.1f}")
            
            return model_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка оценки учителя: {e}", exc_info=True)
            return None
    
    def _evaluate_student(self, name: str, trainer: PseudoExperimentTrainer,
                          image_files: List[Path], labels_dir: Path):
        """Оценка модели студента."""
        logger.info(f"\n{'='*70}\nОЦЕНКА: {name}\n{'='*70}")
        
        try:
            model = trainer.get_model()
            model.eval()
            
            metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", class_metrics=True)
            
            for img_path in tqdm(image_files, desc=f"Evaluating {name}"):
                try:
                    # Предсказание
                    img = Image.open(img_path).convert("RGB").resize(self.img_size, Image.BILINEAR)
                    tensor = torch.from_numpy(
                        np.array(img, dtype=np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        out = model(tensor)[0]
                    
                    keep = out["scores"] > 0.25
                    pred = {
                        "boxes": out["boxes"][keep].cpu(),
                        "scores": out["scores"][keep].cpu(),
                        "labels": (out["labels"][keep] - 1).cpu().clamp(min=0),
                    }
                    
                    # Ground truth
                    gt = self._load_gt(img_path, labels_dir)
                    
                    if len(pred["boxes"]) > 0 or len(gt["boxes"]) > 0:
                        metric.update([pred], [gt])
                        
                except Exception as e:
                    logger.debug(f"Ошибка оценки {img_path.name}: {e}")
            
            result = metric.compute()
            
            model_result = {
                "model": name,
                "type": trainer.group_cfg["type"],
                "map50_95": float(result["map"].item()),
                "map50": float(result["map_50"].item()),
                "map75": float(result["map_75"].item()),
            }
            
            # Per-class AP
            if "map_per_class" in result and result["map_per_class"].numel() > 0:
                per_cls = result["map_per_class"].tolist()
                for i, (cls_name, ap) in enumerate(zip(self.class_names, per_cls)):
                    model_result[f"AP50_{cls_name}"] = round(float(ap), 4)
            
            # FPS и параметры
            fps_stats = self._measure_fps(model, image_files[0])
            model_result.update(fps_stats)
            model_result.update(self._model_stats(model))
            
            logger.info(f"✅ {name} | mAP50:95={model_result['map50_95']:.4f} | "
                       f"mAP50={model_result['map50']:.4f} | FPS={model_result['fps']:.1f}")
            
            return model_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка оценки {name}: {e}", exc_info=True)
            return None
    
    def _load_gt(self, img_path: Path, labels_dir: Path) -> Dict:
        """Загрузка ground truth."""
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        
        if not lbl_path.exists():
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
        
        try:
            w0, h0 = Image.open(img_path).size
        except Exception:
            w0, h0 = self.img_size[1], self.img_size[0]
        
        sx, sy = self.img_size[1] / w0, self.img_size[0] / h0
        
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    if cls >= self.num_classes:
                        continue
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0.0, (xc - w/2) * w0 * sx)
                    y1 = max(0.0, (yc - h/2) * h0 * sy)
                    x2 = min(float(self.img_size[1]), (xc + w/2) * w0 * sx)
                    y2 = min(float(self.img_size[0]), (yc + h/2) * h0 * sy)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
                except (ValueError, IndexError):
                    continue
        
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
    
    def _measure_fps(self, model, img_path: Path, warmup: int = 30, iterations: int = 100) -> Dict:
        """Измерение FPS."""
        # Подготовка функции предсказания
        if hasattr(model, 'predict'):
            def predict_fn(p):
                return model.predict(str(p))
        else:
            def predict_fn(p):
                img = Image.open(p).convert("RGB").resize(self.img_size, Image.BILINEAR)
                tensor = torch.from_numpy(
                    np.array(img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    return model(tensor)[0]
        
        # Прогрев
        for _ in range(warmup):
            try:
                predict_fn(img_path)
            except Exception:
                pass
        
        # Измерения
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                predict_fn(img_path)
            except Exception:
                pass
            times.append(time.perf_counter() - t0)
        
        if not times:
            return {"fps": 0.0, "latency_ms": 0.0}
        
        avg_lat = np.mean(times) * 1000
        return {"fps": round(1000/avg_lat, 1), "latency_ms": round(avg_lat, 2)}
    
    def _model_stats(self, model) -> Dict:
        """Подсчёт параметров и размера модели."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        tmp = Path("/tmp/_pseudo_eval_model.pth")
        torch.save(model.state_dict(), tmp)
        size_mb = tmp.stat().st_size / (1024**2)
        tmp.unlink(missing_ok=True)
        
        return {
            "params_M": round(total/1e6, 1),
            "trainable_M": round(trainable/1e6, 1),
            "size_mb": round(size_mb, 1)
        }
    
    def _analyze_dataset(self, image_files: List[Path], labels_dir: Path) -> Dict:
        """Анализ сложности датасета."""
        stats = {
            'num_images': len(image_files),
            'total_objects': 0,
            'small': 0,
            'medium': 0,
            'large': 0,
            'objects_per_class': defaultdict(int),
        }
        
        all_areas = []
        
        for img_path in tqdm(image_files, desc="Analyzing dataset"):
            gt = self._load_gt(img_path, labels_dir)
            boxes = gt['boxes']
            labels = gt['labels']
            
            if len(boxes) == 0:
                continue
            
            stats['total_objects'] += len(boxes)
            
            for box, label in zip(boxes, labels):
                area = ((box[2] - box[0]) * (box[3] - box[1])).item()
                all_areas.append(area)
                stats['objects_per_class'][str(label.item())] += 1
                
                if area < 32**2:
                    stats['small'] += 1
                elif area < 96**2:
                    stats['medium'] += 1
                else:
                    stats['large'] += 1
        
        total = stats['total_objects']
        stats['pct_small'] = stats['small'] / max(total, 1) * 100
        stats['pct_medium'] = stats['medium'] / max(total, 1) * 100
        stats['pct_large'] = stats['large'] / max(total, 1) * 100
        stats['avg_objects_per_image'] = total / max(len(image_files), 1)
        
        if all_areas:
            stats['avg_box_area'] = np.mean(all_areas)
            stats['median_box_area'] = np.median(all_areas)
            stats['min_box_area'] = np.min(all_areas)
            stats['max_box_area'] = np.max(all_areas)
        
        # Конвертируем defaultdict в обычный dict для JSON
        stats['objects_per_class'] = dict(stats['objects_per_class'])
        
        return stats
    
    def _print_summary(self, all_results: List[Dict], dataset_stats: Dict):
        """Вывод итоговой таблицы."""
        logger.info(f"\n{'='*90}")
        logger.info("ИТОГОВОЕ СРАВНЕНИЕ С ПСЕВДОРАЗМЕТКОЙ")
        logger.info(f"{'='*90}")
        logger.info(f"{'Модель':<35} {'mAP50:95':>10} {'mAP50':>8} {'mAP75':>8} {'FPS':>7} {'Params':>8} {'Size':>7}")
        logger.info("-" * 90)
        
        for r in sorted(all_results, key=lambda x: x.get("map50_95", 0), reverse=True):
            logger.info(
                f"{r['model']:<35} {r.get('map50_95',0):>10.4f} "
                f"{r.get('map50',0):>8.4f} {r.get('map75',0):>8.4f} "
                f"{r.get('fps',0):>7.1f} {r.get('params_M',0):>7.1f}M "
                f"{r.get('size_mb',0):>6.1f}MB"
            )
        
        logger.info(f"{'='*90}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("ЭКСПЕРИМЕНТ С ПСЕВДОРАЗМЕТКОЙ")
    logger.info("=" * 70)
    
    # Загрузка конфига
    cfg = load_config()
    
    # Создание директорий
    Path(cfg["paths"]["pseudo_output"]).mkdir(parents=True, exist_ok=True)
    
    # Шаг 1: Генерация псевдоразметки
    logger.info("\n" + "=" * 70)
    logger.info("ШАГ 1: ГЕНЕРАЦИЯ ПСЕВДОРАЗМЕТКИ")
    logger.info("=" * 70)
    
    generator = PseudoLabelGenerator(cfg)
    pseudo_dir = generator.generate()
    
    # Шаг 2: Подготовка датасетов
    logger.info("\n" + "=" * 70)
    logger.info("ШАГ 2: ПОДГОТОВКА ДАТАСЕТОВ")
    logger.info("=" * 70)
    
    data_path = Path(cfg["detection"]["data_path"])
    num_classes = cfg["detection"]["num_classes"]
    img_size = tuple(cfg["detection"]["img_size"])
    
    # Оригинальные датасеты
    train_original = YOLODataset(
        data_path / "train" / "images",
        data_path / "train" / "labels",
        num_classes, img_size
    )
    
    val_dataset = YOLODataset(
        data_path / "val" / "images",
        data_path / "val" / "labels",
        num_classes, img_size
    )
    
    # Псевдоразмеченный датасет
    train_pseudo = YOLODataset(
        pseudo_dir / "images",
        pseudo_dir / "labels",
        num_classes, img_size
    )
    
    # Объединенный датасет
    train_combined = ConcatDataset([train_original, train_pseudo])
    
    logger.info(f"Оригинальный train: {len(train_original)} изображений")
    logger.info(f"Псевдоразмеченный train: {len(train_pseudo)} изображений")
    logger.info(f"Объединенный train: {len(train_combined)} изображений")
    logger.info(f"Validation: {len(val_dataset)} изображений")
    
    # DataLoaders
    batch_size = cfg["students"]["faster_rcnn_r18_scratch"]["batch"]
    
    train_loader = DataLoader(
        train_combined, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    # Шаг 3: Обучение моделей
    logger.info("\n" + "=" * 70)
    logger.info("ШАГ 3: ОБУЧЕНИЕ МОДЕЛЕЙ")
    logger.info("=" * 70)
    
    # Путь к дистиллированному бэкбону
    pt_file = Path(cfg["paths"]["pretrain_output"]) / "pretrained_path.txt"
    backbone_ckpt = None
    
    if pt_file.exists():
        backbone_ckpt = pt_file.read_text().strip()
        if not Path(backbone_ckpt).exists():
            logger.warning(f"⚠️  Файл бэкбона не найден: {backbone_ckpt}")
            backbone_ckpt = None
    
    trained_models = {}
    
    for name, group_cfg in cfg["students"].items():
        pseudo_name = f"{name}_pseudo"
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"ОБУЧЕНИЕ: {pseudo_name}")
        logger.info(f"{'#'*70}")
        
        try:
            trainer = PseudoExperimentTrainer(
                cfg, pseudo_name, group_cfg,
                train_loader, val_loader,
                backbone_ckpt
            )
            result = trainer.train()
            trained_models[pseudo_name] = trainer
            
            logger.info(f"✅ {pseudo_name}: best mAP50:95 = {result['best_map50_95']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения {pseudo_name}: {e}", exc_info=True)
    
    # Шаг 4: Оценка всех моделей
    logger.info("\n" + "=" * 70)
    logger.info("ШАГ 4: ОЦЕНКА ВСЕХ МОДЕЛЕЙ")
    logger.info("=" * 70)
    
    evaluator = PseudoExperimentEvaluator(cfg)
    final_results = evaluator.evaluate_all(trained_models)
    
    logger.info("\n✅ ЭКСПЕРИМЕНТ С ПСЕВДОРАЗМЕТКОЙ ЗАВЕРШЕН!")
    logger.info(f"Результаты сохранены в: {cfg['paths']['pseudo_output']}")


if __name__ == "__main__":
    main()