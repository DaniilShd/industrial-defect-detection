#!/usr/bin/env python3
"""Экстренный запуск: только LT-DETR для real_augmented_massive с готовым SSL-бэкбоном"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.scripts.train_ltdetr import train_ltdetr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Конфиг
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Параметры для проблемного датасета
    ds_name = "real_augmented_massive"
    seed = 42
    ds_cfg = config['datasets'][ds_name]
    
    # Путь к готовому SSL-бэкбону
    backbone_path = Path("/app/data/experiment_v3/models/exp3_ssl/ssl_pretrain_real_augmented_massive_seed42/exported_models/exported_last.pt")
    
    if not backbone_path.exists():
        logger.error(f"❌ Бэкбон не найден: {backbone_path}")
        logger.info("Ищу альтернативные пути...")
        candidates = list(Path("/app/data/experiment_v3/models/exp3_ssl").glob(f"*{ds_name}*/exported_models/exported_last.pt"))
        if candidates:
            backbone_path = candidates[0]
            logger.info(f"✅ Найден: {backbone_path}")
        else:
            raise FileNotFoundError(f"SSL backbone для {ds_name} не найден")
    
    # Настройка MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_exp3_ssl")
    
    run_cfg = {
        'run_name': f"{ds_name}_ssl_seed{seed}_RETRY",
        'dataset_name': ds_name,
        'data_yaml': ds_cfg['data_yaml'],
        'strategy_name': 'ssl',
        'freeze_backbone': False,
        'seed': seed,
    }
    
    models_dir = Path(config['paths']['models_dir']) / "exp3_ssl" / run_cfg['run_name']
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 ЭКСТРЕННЫЙ ЗАПУСК: {run_cfg['run_name']}")
    logger.info(f"📁 Бэкбон: {backbone_path}")
    logger.info(f"{'='*60}")
    
    with mlflow.start_run(run_name=run_cfg['run_name'], nested=False):
        mlflow.log_params(run_cfg)
        mlflow.log_param("ssl_backbone_path", str(backbone_path))
        mlflow.log_param("retry_reason", "original_run_failed_ssl_already_done")
        
        try:
            # ТОЛЬКО LT-DETR, без SSL
            logger.info("⏩ Пропускаем SSL (бэкбон готов)")
            logger.info("🔄 Запуск LT-DETR...")
            
            start = time.time()
            
            result = train_ltdetr(
                config, 
                run_cfg, 
                models_dir,
                extra_model_args={"backbone_weights": str(backbone_path)},
            )
            
            training_time = time.time() - start
            result['training_time_hours'] = round(training_time / 3600, 3)
            
            # Сохраняем результат
            with open(models_dir / "result.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Логируем метрики
            metrics = {
                'test_map50': result.get('test_map50', 0),
                'test_map75': result.get('test_map75', 0),
                'test_map50_95': result.get('test_map50_95', 0),
                'val_map50': result.get('val_map50', 0),
                'training_time_hours': result.get('training_time_hours', 0),
                'n_epochs': result.get('n_epochs', 0),
                'n_images': result.get('n_images', 0),
            }
            for k, v in result.items():
                if k.startswith('test_cls'):
                    metrics[k] = v
            
            mlflow.log_metrics(metrics)
            mlflow.set_tag("status", "completed")
            
            logger.info(f"\n✅ УСПЕХ!")
            logger.info(f"   mAP50: {result.get('test_map50', 0):.4f}")
            logger.info(f"   mAP75: {result.get('test_map75', 0):.4f}")
            logger.info(f"   mAP50-95: {result.get('test_map50_95', 0):.4f}")
            logger.info(f"   Время: {training_time/3600:.2f} часов")
            logger.info(f"   Результаты: {models_dir}/result.json")
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}", exc_info=True)
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))


if __name__ == "__main__":
    main()