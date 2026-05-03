# ============================================================
# generate/testing_lora/run_lora_pipeline.py
# ============================================================
#!/usr/bin/env python3
"""Полный пайплайн: LoRA → Inpainting → Детектор → Оценка"""

import logging
import sys
from pathlib import Path
import yaml
import mlflow
import torch
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_gpt.testing_lora_gpt.train_lora import prepare_lora_dataset, train_lora
from generate_gpt.testing_lora_gpt.generate_lora_synthetic import LoRADefectGenerator
from generate_gpt.ablation.train_ltdetr import train_ltdetr
from generate_gpt.ablation.evaluate import evaluate_model

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

    mlflow_cfg = cfg['mlflow']
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg['experiment_name'])

    with mlflow.start_run(run_name=mlflow_cfg['run_name']):
        # === 1. LoRA обучение ===
        logger.info("=" * 60)
        logger.info("Step 1/4: Training LoRA on real defects")

        mlflow.log_params({
            # Generator
            'gen_strength_small_min': cfg['generation']['strength_small_min'],
            'gen_strength_small_max': cfg['generation']['strength_small_max'],
            'gen_strength_medium_min': cfg['generation']['strength_medium_min'],
            'gen_strength_medium_max': cfg['generation']['strength_medium_max'],
            'gen_strength_large_min': cfg['generation']['strength_large_min'],
            'gen_strength_large_max': cfg['generation']['strength_large_max'],
            'gen_mask_blur_kernel': cfg['generation']['mask_blur_kernel'],
            'gen_sd_guidance_scale': cfg['generation']['sd_guidance_scale'],
            'gen_sd_steps': cfg['generation']['sd_steps'],
            'gen_color_correction': cfg['generation']['color_correction'],
            'gen_edge_blur_kernel': cfg['generation']['edge_blur_kernel'],
            'gen_high_freq_alpha': cfg['generation']['high_freq_alpha'],
            'gen_variants': cfg['generation']['variants'],
            # LoRA
            'lora_rank': cfg['lora']['rank'],
            'lora_alpha': cfg['lora']['alpha'],
            'lora_max_steps': cfg['lora']['max_steps'],
            'lora_learning_rate': cfg['lora']['learning_rate'],
            # LT-DETR
            'ltdetr_max_steps': cfg['ltdetr']['max_steps'],
            'ltdetr_lr': cfg['ltdetr']['lr'],
            'ltdetr_batch_size': cfg['ltdetr']['batch_size'],
        })

        lora_dir = results_dir / "lora"
        num_samples = prepare_lora_dataset(
            real_train, lora_dir / "dataset",
            cfg['generation']['class_labels'],
            cfg['generation']['prompt_templates'],
            rle_csv=real_train / "train_rle.csv"  # ← добавить
        )

        if num_samples < 10:
            logger.error(f"Only {num_samples} images for LoRA — need at least 10!")
            return

        mlflow.log_metric("lora_training_samples", num_samples)

        lora_weights_dir = train_lora(
            "runwayml/stable-diffusion-v1-5",
            lora_dir / "dataset",
            lora_dir / "weights",
            cfg['lora']
        )

        if lora_weights_dir:
            mlflow.log_artifact(str(lora_weights_dir))

        # === 2. Генерация синтетики ===
        logger.info("=" * 60)
        logger.info("Step 2/4: Generating synthetic data with SD Inpainting + LoRA")

        synth_dir = results_dir / "synthetic"
        generator = LoRADefectGenerator(cfg, str(lora_weights_dir))

        total = generator.generate_dataset(
            real_train / "images",
            real_train / "train_rle.csv",
            synth_dir,
            variants=cfg['generation']['variants']
        )

        mlflow.log_metric("synthetic_images", total)

        # Вместо этого — указать готовые пути:
        # lora_weights_dir = "/app/data/results/lora_test/lora/weights/lora_final"
        # synth_dir = Path("/app/data/results/lora_test/synthetic")

        # === 3. Подготовка датасета ===
        logger.info("=" * 60)
        logger.info("Step 3/4: Preparing dataset (real + synthetic)")

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
        
        nc=cfg['dataset']['nc'],
        names = {0: 'defect_1', 1: 'defect_2', 2: 'defect_3', 3: 'defect_4'}
        data_yaml = ds_dir / "data.yaml"
        with open(data_yaml, 'w') as f:
            yaml.dump({
                'format': 'yolo',
                'path': str(ds_dir),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': 4,
                'names': names
            }, f, default_flow_style=None)
        mlflow.log_artifact(str(data_yaml))

        # === 4. Обучение детектора ===
        logger.info("=" * 60)
        logger.info("Step 4/4: Training LT-DETR detector")

        ltdetr_dir = results_dir / "ltdetr"
        train_result = train_ltdetr(
            data_yaml, ltdetr_dir,
            max_steps=cfg['ltdetr']['max_steps'],
            early_stopping_patience=cfg['ltdetr']['early_stopping_patience'],
            val_every_steps=cfg['ltdetr']['val_every_steps'],
            lr=cfg['ltdetr']['lr'],
            batch_size=cfg['ltdetr']['batch_size']
        )

        # Логи
        train_log = ltdetr_dir / "train.log"
        if train_log.exists():
            mlflow.log_artifact(str(train_log), "training_logs")

        log_tensorboard_metrics(ltdetr_dir)

        # Финальная оценка
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