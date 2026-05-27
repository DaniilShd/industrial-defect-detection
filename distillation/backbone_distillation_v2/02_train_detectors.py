#!/usr/bin/env python3
"""
Обучение Faster R-CNN с тремя вариантами инициализации backbone.

Группы:
  scratch    — случайная инициализация (контроль)
  imagenet   — ImageNet pretrained (baseline)
  distilled  — дистилляция от LTDETR (предложенный метод)

Все группы обучаются в одинаковых условиях.
Метрики: torchmetrics.MeanAveragePrecision (COCO-совместимые)
"""

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("02_train_detectors.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ДАТАСЕТ
# ═══════════════════════════════════════════════════════════════════════════

class DefectDataset(Dataset):
    """Датасет дефектов в формате YOLO."""
    
    def __init__(self, images_dir: Path, labels_dir: Path,
                 num_classes: int, img_size: Tuple[int, int] = (640, 640)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.files = sorted(
            f for f in self.images_dir.glob("*") 
            if f.suffix.lower() in exts
        )
        
        if len(self.files) == 0:
            logger.warning(f"⚠️  Нет изображений в {images_dir}")
        
        logger.info(f"Датасет {images_dir.name}: {len(self.files)} изображений")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize(self.img_size, Image.BILINEAR)
            tensor = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
        except Exception as e:
            logger.error(f"Ошибка загрузки {img_path}: {e}")
            return torch.zeros(3, *self.img_size), {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        boxes, labels = self._load_yolo(img_path, orig_w, orig_h)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) 
                     if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) 
                      if labels else torch.zeros(0, dtype=torch.int64),
        }
        return tensor, target

    def _load_yolo(self, img_path: Path, ow: int, oh: int):
        """Загрузка разметки YOLO с конвертацией в абсолютные координаты."""
        boxes, labels = [], []
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            return boxes, labels

        sx, sy = self.img_size[1] / ow, self.img_size[0] / oh
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
                    x1 = max(0.0, (xc - w/2) * ow * sx)
                    y1 = max(0.0, (yc - h/2) * oh * sy)
                    x2 = min(float(self.img_size[1]), (xc + w/2) * ow * sx)
                    y2 = min(float(self.img_size[0]), (yc + h/2) * oh * sy)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)  # Faster R-CNN: 0 = фон
                except (ValueError, IndexError):
                    continue
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ═══════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ВЕСОВ ИЗ LIGHTLYTRAIN
# ═══════════════════════════════════════════════════════════════════════════

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
    """
    Загружает веса backbone из экспортированной LightlyTrain модели.
    
    Экспортированная модель LightlyTrain содержит state_dict обученного
    бэкбона. Функция маппит эти веса на model.backbone.body (ResNet).
    
    Returns:
        Количество успешно загруженных ключей.
    """
    logger.info(f"  Загрузка весов LightlyTrain из: {ckpt_path}")
    
    # Загружаем чекпоинт
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Извлекаем state_dict
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state_dict = ckpt
    
    logger.info(f"  Загружено ключей из чекпоинта: {len(state_dict)}")
    
    # Показываем примеры ключей для отладки
    sample_keys = list(state_dict.keys())[:5]
    logger.info(f"  Примеры ключей источника: {sample_keys}")
    
    # Целевой state_dict (backbone.body = ResNet без FPN)
    body = model.backbone.body
    body_sd = body.state_dict()
    
    logger.info(f"  Целевых ключей в backbone.body: {len(body_sd)}")
    
    # Маппинг ключей
    mapped = {}
    skipped_shape = []
    skipped_not_found = []
    
    for k, v in state_dict.items():
        clean = _strip_prefix(k)
        
        if clean in body_sd:
            if v.shape == body_sd[clean].shape:
                mapped[clean] = v
            else:
                skipped_shape.append((clean, v.shape, body_sd[clean].shape))
        else:
            skipped_not_found.append(clean)
    
    # Загружаем смаппленные веса
    body.load_state_dict({**body_sd, **mapped}, strict=False)
    
    # Логирование результатов
    logger.info(f"  ✓ Успешно загружено: {len(mapped)}/{len(body_sd)} ключей")
    
    if len(mapped) == 0:
        logger.error("  ❌ НИ ОДИН КЛЮЧ НЕ СОВПАЛ!")
        logger.error(f"  Проверьте совместимость архитектур")
        logger.error(f"  Ключи источника (первые 10): {sample_keys}")
        logger.error(f"  Ключи цели (первые 10): {list(body_sd.keys())[:10]}")
    elif len(mapped) < len(body_sd) * 0.5:
        logger.warning(f"  ⚠️  Загружено менее 50% весов ({len(mapped)}/{len(body_sd)})")
    
    if skipped_shape:
        logger.warning(f"  ⚠️  Пропущено из-за несовпадения формы: {len(skipped_shape)} ключей")
        for k, src, dst in skipped_shape[:3]:
            logger.warning(f"      {k}: источник {list(src)} -> цель {list(dst)}")
    
    return len(mapped)


# ═══════════════════════════════════════════════════════════════════════════
# ТРЕНЕР
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """Тренер для Faster R-CNN с заданной инициализацией."""
    
    def __init__(self, cfg: dict, name: str, group_cfg: dict,
                 backbone_ckpt: Optional[str] = None):
        self.cfg = cfg
        self.name = name
        self.group_cfg = group_cfg
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

        self.train_dl, self.val_dl = self._dataloaders()
        self.out_dir = Path(cfg["paths"]["detection_output"]) / name
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
            logger.info("  ✓ ImageNet веса загружены")

        elif init_type == "lightly_pretrained":
            if self.backbone_ckpt and Path(self.backbone_ckpt).exists():
                n_loaded = load_lightly_backbone(model, self.backbone_ckpt)
                if n_loaded == 0:
                    logger.error("  ❌ Дистиллированные веса не загружены! Будет случайная инициализация.")
            else:
                logger.warning(f"  ⚠️  Чекпоинт не найден: {self.backbone_ckpt}")
                logger.warning("  ⚠️  Будет использована случайная инициализация")
        else:
            logger.info("  Случайная инициализация (scratch)")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Параметры: всего={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M")
        return model

    def _dataloaders(self):
        """Создание DataLoader'ов."""
        dp = Path(self.cfg["detection"]["data_path"])
        sz = tuple(self.cfg["detection"]["img_size"])
        nc = self.num_classes
        bs = self.group_cfg["batch"]

        train_ds = DefectDataset(dp/"train"/"images", dp/"train"/"labels", nc, sz)
        val_ds = DefectDataset(dp/"val"/"images", dp/"val"/"labels", nc, sz)

        train_dl = DataLoader(
            train_ds, bs, shuffle=True, num_workers=4,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        val_dl = DataLoader(
            val_ds, bs, shuffle=False, num_workers=2,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        return train_dl, val_dl

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
            
            for batch_idx, (images, targets) in enumerate(self.train_dl):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()

                total_loss += loss.item()
                
                # Логирование каждые 50 батчей
                if (batch_idx + 1) % 50 == 0:
                    logger.debug(f"  Batch {batch_idx+1}/{len(self.train_dl)}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.train_dl)
            
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
                best_state = copy.deepcopy(self.model.state_dict())
                self._save("best_model.pth")
                logger.info(f"  ✓ Новый лучший результат! mAP50:95={map50_95:.4f}")
            else:
                self._patience_cnt += 1
                if self._patience_cnt >= self.patience:
                    logger.info(f"Early stopping на эпохе {epoch} (patience={self.patience})")
                    break

            self.sched.step()

        # Загрузка лучшей модели и финальное сохранение
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
            for images, targets in self.val_dl:
                images = [i.to(self.device) for i in images]
                outputs = self.model(images)

                # Фильтруем: оставляем только те изображения, где есть и предсказания, и цели
                valid_preds = []
                valid_targets = []
                
                for out, t in zip(outputs, targets):
                    # Всегда добавляем, даже если пустые
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
                
                # Обновляем метрику только если длины совпадают
                if len(valid_preds) == len(valid_targets):
                    metric.update(valid_preds, valid_targets)
        
        return metric.compute()
        
        return metric.compute()

    def _save(self, fname: str):
        """Сохранение чекпоинта."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_map50_95": self.best_map,
            "best_epoch": self.best_epoch,
            "config": self.group_cfg,
        }, self.out_dir / fname)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    
    if not cfg_path.exists():
        logger.error(f"Конфиг не найден: {cfg_path}")
        sys.exit(1)
    
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Путь к дистиллированному бэкбону
    pt_file = Path(cfg["paths"]["pretrain_output"]) / "pretrained_path.txt"
    backbone_ckpt = None
    
    if pt_file.exists():
        backbone_ckpt = pt_file.read_text().strip()
        if Path(backbone_ckpt).exists():
            logger.info(f"Backbone checkpoint: {backbone_ckpt}")
        else:
            logger.warning(f"⚠️  Файл из pretrained_path.txt не найден: {backbone_ckpt}")
            backbone_ckpt = None
    else:
        logger.warning("⚠️  pretrained_path.txt не найден! Distilled группа = scratch")

    summary_path = Path(cfg["paths"]["detection_output"]) / "results.json"
    all_results = []
    trained = set()
    
    # Загружаем предыдущие результаты если есть
    if summary_path.exists():
        try:
            all_results = json.loads(summary_path.read_text())
            trained = {r["model_name"] for r in all_results}
            logger.info(f"Загружены результаты для {len(trained)} моделей")
        except Exception:
            pass

    # Обучаем все группы студентов
    for name, group_cfg in cfg["students"].items():
        final_path = Path(cfg["paths"]["detection_output"]) / name / "model_final.pth"
        
        if name in trained or final_path.exists():
            logger.info(f"✅ {name} уже обучена, пропускаем")
            continue

        logger.info(f"\n{'#'*70}\nОБУЧЕНИЕ: {name}\n{'#'*70}")
        
        try:
            trainer = Trainer(cfg, name, group_cfg, backbone_ckpt)
            result = trainer.train()
            all_results.append(result)
            summary_path.write_text(json.dumps(all_results, indent=2))
        except Exception as e:
            logger.error(f"❌ Ошибка обучения {name}: {e}", exc_info=True)

    # Итоговая таблица
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("ИТОГИ ОБУЧЕНИЯ")
        logger.info(f"{'='*80}")
        logger.info(f"{'Модель':<35} {'mAP50:95':>10} {'mAP50':>8} {'mAP75':>8} {'Время':>8}")
        logger.info("-" * 80)
        
        for r in sorted(all_results, key=lambda x: x["best_map50_95"], reverse=True):
            logger.info(
                f"{r['model_name']:<35} "
                f"{r['best_map50_95']:>10.4f} "
                f"{r['best_map50']:>8.4f} "
                f"{r['best_map75']:>8.4f} "
                f"{r.get('training_time_hours', 0):>7.1f}ч"
            )
        logger.info(f"{'='*80}")
        
        # Сравнение с учителем если есть
        teacher_r = next((r for r in all_results if r.get("type") == "teacher"), None)
        distilled_r = next((r for r in all_results if r["init_type"] == "lightly_pretrained"), None)
        
        if teacher_r and distilled_r:
            ratio = distilled_r["best_map50_95"] / max(teacher_r.get("best_map50_95", 1e-6), 1e-6) * 100
            logger.info(f"\n📊 Студент (distilled) сохраняет {ratio:.1f}% точности учителя")


if __name__ == "__main__":
    main()