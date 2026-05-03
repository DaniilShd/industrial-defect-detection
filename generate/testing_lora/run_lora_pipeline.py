# generate/testing_lora/run_lora_pipeline.py
#!/usr/bin/env python3
"""Полный пайплайн: LoRA обучение → Генерация → Обучение детектора"""

import logging
import sys
from pathlib import Path
import yaml
import mlflow
import torch
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate.testing_lora.train_lora import prepare_lora_dataset, train_lora
from generate.testing_lora.generate_lora_synthetic import LoRADefectGenerator
from generate.ablation.train_ltdetr import train_ltdetr
from generate.ablation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_tensorboard_metrics(ltdetr_dir: Path):
    """Извлекает val метрики из TensorBoard-логов LightlyTrain"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        event_files = list(ltdetr_dir.glob("events.out.tfevents.*"))
        if not event_files:
            logger.warning("No TensorBoard event files found")
            return
        
        event_acc = EventAccumulator(str(event_files[0]))
        event_acc.Reload()
        
        tags = event_acc.Tags().get('scalars', [])
        logger.info(f"Found {len(tags)} scalar tags in TensorBoard log")
        
        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                mlflow.log_metric(tag.replace('/', '_'), event.value, step=event.step)
        
        logger.info(f"Logged {len(tags)} metrics from TensorBoard")
    except ImportError:
        logger.warning("tensorboard not installed — skipping TensorBoard metrics")
    except Exception as e:
        logger.warning(f"Failed to parse TensorBoard logs: {e}")


def main():
    with open(Path(__file__).parent / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    paths = cfg['paths']
    real_train = Path(paths['real_train'])
    real_val = Path(paths['real_val'])
    real_test = Path(paths['real_test'])
    results_dir = Path(paths['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("lora_finetune")
    
    with mlflow.start_run(run_name="lora_test"):
        # === 1. LoRA обучение ===
        logger.info("=" * 60)
        logger.info("Step 1/4: Training LoRA")
        
        lora_dir = results_dir / "lora"
        dataset = prepare_lora_dataset(real_train, lora_dir / "dataset")
        
        if len(dataset) < 10:
            logger.error(f"Only {len(dataset)} images for LoRA — need more data!")
            return
        
        lora_weights = train_lora(
            "runwayml/stable-diffusion-v1-5",
            lora_dir / "dataset",
            lora_dir / "weights",
            cfg['lora']
        )
        
        if lora_weights:
            mlflow.log_artifact(str(lora_weights))
        
        # === 2. Генерация синтетики ===
        logger.info("=" * 60)
        logger.info("Step 2/4: Generating synthetic data with LoRA SD")
        
        synth_dir = results_dir / "synthetic"
        generator = LoRADefectGenerator(cfg, str(lora_weights))
        
        total = generator.generate_dataset(
            real_train / "images",
            real_train / "train_rle.csv",
            synth_dir,
            variants=cfg['generation']['variants']
        )
        
        mlflow.log_metric("synthetic_images", total)
        
        # === 3. Подготовка датасета ===
        logger.info("=" * 60)
        logger.info("Step 3/4: Preparing dataset")
        
        ds_dir = results_dir / "dataset"
        for split, src_imgs, src_lbls in [
            ('train', [real_train / "images", synth_dir / "images"],
                     [real_train / "labels", synth_dir / "labels"]),
            ('val',   [real_val / "images"], [real_val / "labels"]),
            ('test',  [real_test / "images"], [real_test / "labels"])
        ]:
            (ds_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (ds_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            
            for src_i, src_l in zip(src_imgs, src_lbls):
                if not src_i.exists():
                    continue
                for img_path in src_i.glob("*"):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                        continue
                    shutil.copy2(img_path, ds_dir / split / "images" / img_path.name)
                    lbl_path = src_l / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        shutil.copy2(lbl_path, ds_dir / split / "labels" / lbl_path.name)
        
        # data.yaml — LightlyTrain ожидает структуру с images/labels на верхнем уровне
        data_yaml = ds_dir / "data.yaml"
        with open(data_yaml, 'w') as f:
            yaml.dump({
                'format': 'yolo',
                'path': str(ds_dir),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': 4,
                'names': {0: 'defect_1', 1: 'defect_2', 2: 'defect_3', 3: 'defect_4'}
            }, f)
        
        mlflow.log_artifact(str(data_yaml))
        
        # === 4. Обучение детектора ===
        logger.info("=" * 60)
        logger.info("Step 4/4: Training detector")
        
        ltdetr_dir = results_dir / "ltdetr"
        train_result = train_ltdetr(
            data_yaml, ltdetr_dir,
            max_steps=5500,
            val_every_steps=500,
            early_stopping_patience=3
        )
        
        # Логируем train.log
        train_log = ltdetr_dir / "train.log"
        if train_log.exists():
            mlflow.log_artifact(str(train_log), "training_logs")
        
        # Логируем метрики из TensorBoard
        log_tensorboard_metrics(ltdetr_dir)
        
        # Финальная оценка на тесте
        logger.info("Evaluating on test set...")
        if train_result.get('model_path'):
            metrics = evaluate_model(
                train_result['model_path'],
                real_test / "images",
                real_test / "labels"
            )
            
            mlflow.log_metrics({
                'test_mAP_50': metrics['mAP_50'],
                'test_mAP_75': metrics['mAP_75'],
                'test_mAP_50_95': metrics['mAP_50_95'],
                **{f'test_{k}': v for k, v in metrics.items() if k.startswith('cls')}
            })
            
            logger.info(f"✅ Test mAP_50={metrics['mAP_50']:.4f}")
        else:
            logger.error("No model found for evaluation!")
        
        logger.info("✅ Pipeline completed!")


if __name__ == "__main__":
    main()