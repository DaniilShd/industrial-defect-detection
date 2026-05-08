#!/usr/bin/env python3
"""Оркестратор с полным MLflow логированием"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from distillation_industrial.scripts.model_loader import load_model, ModelInferenceWrapper
from distillation_industrial.scripts.evaluate import evaluate_model
from distillation_industrial.scripts.measure_fps import measure_fps, count_parameters
from distillation_industrial.scripts.visualize import create_all_visualizations
from distillation_industrial.scripts.distillation_trainer import DistillationTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_and_measure(model_or_path, model_type: str, model_name: str,
                         config: dict, test_images: Path, test_labels: Path) -> dict:
    """Оценивает модель и возвращает все метрики."""
    if isinstance(model_or_path, (str, Path)):
        if model_type in ['faster_rcnn', 'ssd']:
            model = load_model(str(model_or_path), model_type=model_type,
                             num_classes=config['classes']['num_classes'])
        else:
            model = load_model(str(model_or_path), model_type=model_type)
    else:
        model = model_or_path
    
    wrapper = ModelInferenceWrapper(model, model_type)
    
    # Оценка метрик
    metrics = evaluate_model(wrapper, model_type, test_images, test_labels,
                            config['classes']['num_classes'])
    
    # FPS
    fps_data = measure_fps(wrapper, model_type, test_images,
                          tuple(config['fps']['img_size']),
                          config['fps']['warmup'],
                          config['fps']['iterations'],
                          config['fps']['device'])
    
    # Параметры
    params = count_parameters(model)
    
    result = {
        'model': model_name,
        'model_type': model_type,
        'map50': metrics['mAP_50'],
        'map75': metrics['mAP_75'],
        'map50_95': metrics['mAP_50_95'],
        'fps': fps_data['fps'],
        'latency_ms': fps_data['latency_ms'],
        'params_M': params['params_M'],
        'params_total': params['params_total'],
        'params_trainable': params['params_trainable'],
        'size_MB': params['size_MB'],
    }
    
    # Per-class метрики
    for c in range(config['classes']['num_classes']):
        result[f'cls{c}_AP50'] = metrics.get(f'cls{c}_AP50', 0)
    
    return result


def log_to_mlflow(result: dict, prefix: str = None):
    """
    Полное логирование метрик модели в MLflow.
    
    Логирует:
    - Основные метрики (mAP@50, mAP@75, mAP@50:95)
    - Per-class метрики
    - FPS и latency
    - Параметры модели
    - Размер модели
    """
    if prefix is None:
        prefix = result['model']
    
    # Основные метрики детекции
    mlflow.log_metrics({
        f"{prefix}/map50": result['map50'],
        f"{prefix}/map75": result['map75'],
        f"{prefix}/map50_95": result['map50_95'],
    })
    
    # Производительность
    mlflow.log_metrics({
        f"{prefix}/fps_cpu": result['fps'],
        f"{prefix}/latency_ms": result['latency_ms'],
    })
    
    # Параметры модели
    mlflow.log_metrics({
        f"{prefix}/params_M": result['params_M'],
        f"{prefix}/params_total": result['params_total'],
        f"{prefix}/params_trainable": result['params_trainable'],
        f"{prefix}/size_MB": result['size_MB'],
    })
    
    # Per-class метрики
    for c in range(4):  # 4 класса дефектов
        key = f'cls{c}_AP50'
        if key in result:
            mlflow.log_metric(f"{prefix}/cls{c}_AP50", result[key])


def log_training_params(config: dict, student_name: str, student_cfg: dict):
    """Логирует параметры обучения."""
    mlflow.log_params({
        "student_name": student_name,
        "distill_method": student_cfg.get('distill_method', 'baseline'),
        "backbone": student_cfg.get('backbone', 'resnet18'),
        "epochs": student_cfg['epochs'],
        "batch_size": student_cfg['batch'],
        "learning_rate": student_cfg['lr'],
        "temperature": student_cfg.get('temperature', 'N/A'),
        "alpha": student_cfg.get('alpha', 'N/A'),
        "beta": student_cfg.get('beta', 'N/A'),
        "teacher_model": config['teacher']['model'],
        "teacher_dataset": config['teacher']['dataset'],
    })


def log_training_artifacts(training_result: dict, models_dir: Path, student_name: str):
    """Логирует артефакты обучения."""
    output_dir = models_dir / student_name
    
    # Сохраняем training_info.json
    info_path = output_dir / 'training_info.json'
    if info_path.exists():
        mlflow.log_artifact(str(info_path), f"training_logs/{student_name}")
    
    # Сохраняем чекпоинты (если есть)
    best_model = output_dir / 'best_model.pth'
    if best_model.exists():
        mlflow.log_artifact(str(best_model), f"models/{student_name}")
    
    # Сохраняем конфигурацию
    config_path = output_dir.parent.parent / "config_distillation.yaml"
    if config_path.exists():
        mlflow.log_artifact(str(config_path), "config")


def log_training_history(history: list, student_name: str):
    """
    Логирует историю обучения как график в MLflow.
    
    Создаёт графики:
    - Train loss по эпохам
    - Val mAP по эпохам
    - KD loss и Feature loss
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if not history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = [h['epoch'] for h in history]
    
    # Train losses
    ax1 = axes[0, 0]
    ax1.plot(epochs, [h.get('train_total_loss', 0) for h in history], 'b-', label='Total Loss', linewidth=2)
    ax1.plot(epochs, [h.get('train_detection_loss', 0) for h in history], 'g--', label='Detection Loss')
    ax1.plot(epochs, [h.get('train_kd_loss', 0) for h in history], 'r--', label='KD Loss')
    ax1.plot(epochs, [h.get('train_feature_loss', 0) for h in history], 'y--', label='Feature Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{student_name} - Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Val mAP
    ax2 = axes[0, 1]
    val_maps = [h.get('val_map50', 0) for h in history]
    ax2.plot(epochs, val_maps, 'g-', linewidth=2, marker='o', markersize=3)
    ax2.fill_between(epochs, 0, val_maps, alpha=0.2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP@50')
    ax2.set_title(f'{student_name} - Validation mAP@50')
    ax2.grid(True, alpha=0.3)
    
    # Best epoch marker
    if val_maps:
        best_idx = val_maps.index(max(val_maps))
        ax2.axvline(x=epochs[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best: {max(val_maps):.4f}')
        ax2.legend()
    
    # Loss vs mAP correlation
    ax3 = axes[1, 0]
    train_losses = [h.get('train_total_loss', 0) for h in history]
    ax3.scatter(train_losses, val_maps, c=epochs, cmap='viridis', alpha=0.7, s=30)
    ax3.set_xlabel('Train Loss')
    ax3.set_ylabel('Val mAP@50')
    ax3.set_title('Loss vs Performance')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(ax3.collections[0], ax=ax3, label='Epoch')
    
    # LR schedule
    ax4 = axes[1, 1]
    lrs = [h.get('lr', 0) for h in history]
    ax4.plot(epochs, lrs, 'b-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем и логируем
    save_path = f"/tmp/{student_name}_training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    mlflow.log_artifact(save_path, f"training_curves/{student_name}")
    plt.close()


def log_comparison_table(all_results: list):
    """Логирует сравнительную таблицу как артефакт."""
    import pandas as pd
    
    df = pd.DataFrame(all_results)
    
    # Выбираем ключевые колонки
    columns = ['model', 'map50', 'map75', 'map50_95', 'fps', 'params_M', 'size_MB']
    available_cols = [c for c in columns if c in df.columns]
    
    comparison_df = df[available_cols].sort_values('map50', ascending=False)
    
    # Сохраняем как CSV
    csv_path = "/tmp/model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, "comparison")
    
    # Сохраняем как HTML таблицу для MLflow
    html_path = "/tmp/model_comparison.html"
    comparison_df.to_html(html_path, index=False, float_format='%.4f')
    mlflow.log_artifact(html_path, "comparison")
    
    # Логируем как отдельные метрики для удобного сравнения в UI
    for _, row in comparison_df.iterrows():
        model_name = row['model']
        mlflow.log_metric(f"comparison/{model_name}_rank", 
                         comparison_df.index.get_loc(_) + 1)


def main():
    config_path = Path(__file__).parent.parent / "config_distillation.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    results_dir = Path(config['paths']['results_dir'])
    models_dir = Path(config['paths']['models_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = config['teacher']['dataset']
    test_images = Path(config['paths']['experiment_data']) / dataset_name / "test" / "images"
    test_labels = Path(config['paths']['experiment_data']) / dataset_name / "test" / "labels"
    
    if not test_images.exists():
        logger.error(f"❌ Test images not found: {test_images}")
        return
    
    all_results = []
    
    with mlflow.start_run(run_name=config['experiment']['name']) as parent_run:
        # Логируем полный конфиг
        mlflow.log_dict(config, "experiment_config.yaml")
        
        # ==========================================
        # 1. УЧИТЕЛЬ
        # ==========================================
        logger.info("\n" + "="*60)
        logger.info("1/7: Loading Teacher (LTDETR)")
        logger.info("="*60)
        
        teacher_path = config['teacher']['model_path']
        if not Path(teacher_path).exists():
            logger.error(f"❌ Teacher not found: {teacher_path}")
            return
        
        teacher_result = evaluate_and_measure(
            teacher_path, 'lightly', 'teacher_ltdetr',
            config, test_images, test_labels
        )
        all_results.append(teacher_result)
        
        # Логируем учителя
        log_to_mlflow(teacher_result, 'teacher')
        mlflow.log_param("teacher_model", config['teacher']['model'])
        mlflow.log_param("teacher_path", teacher_path)
        
        # ==========================================
        # 2-7. УЧЕНИКИ
        # ==========================================
        for student_name, student_cfg in config['students'].items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {student_name}")
            logger.info(f"Method: {student_cfg.get('distill_method', 'baseline')}")
            logger.info(f"Backbone: {student_cfg.get('backbone', 'resnet18')}")
            logger.info(f"{'='*60}")
            
            with mlflow.start_run(run_name=student_name, nested=True) as child_run:
                # Логируем параметры
                log_training_params(config, student_name, student_cfg)
                
                try:
                    # Обучаем
                    trainer = DistillationTrainer(
                        config, student_name, student_cfg,
                        teacher_path, models_dir
                    )
                    
                    training_result = trainer.train()
                    
                    if training_result['status'] != 'completed':
                        mlflow.set_tag("status", "failed")
                        logger.error(f"❌ Training failed for {student_name}")
                        continue
                    
                    # Логируем историю обучения
                    log_training_history(training_result['history'], student_name)
                    
                    # Логируем артефакты
                    log_training_artifacts(training_result, models_dir, student_name)
                    
                    # Оцениваем
                    model_type = 'ssd' if 'ssd' in student_name.lower() else 'faster_rcnn'
                    student_result = evaluate_and_measure(
                        training_result['model_path'],
                        model_type, student_name,
                        config, test_images, test_labels
                    )
                    
                    all_results.append(student_result)
                    
                    # Логируем все метрики
                    log_to_mlflow(student_result, student_name)
                    
                    # Дополнительные метаданные
                    mlflow.set_tag("status", "completed")
                    mlflow.set_tag("distill_method", student_cfg.get('distill_method', 'baseline'))
                    mlflow.set_tag("backbone", student_cfg.get('backbone', 'resnet18'))
                    mlflow.log_metric(f"{student_name}/best_val_map50", training_result['best_val_map50'])
                    mlflow.log_metric(f"{student_name}/training_hours", training_result['training_time_hours'])
                    mlflow.log_metric(f"{student_name}/epochs_trained", training_result['epochs_trained'])
                    
                    # Логируем информацию о переобучении
                    overfitting = training_result.get('overfitting', {})
                    if overfitting.get('overfitting_detected'):
                        mlflow.set_tag(f"{student_name}_overfitting", "detected")
                        mlflow.log_param(f"{student_name}_overfitting_warnings", 
                                       str(overfitting.get('warning_signs', [])))
                    
                except Exception as e:
                    logger.error(f"❌ Failed: {e}", exc_info=True)
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error", str(e))
        
        # ==========================================
        # ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ
        # ==========================================
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Сохраняем JSON
        results_path = results_dir / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path), "results")
        
        # Визуализации
        create_all_visualizations(all_results, results_dir)
        for p in results_dir.glob('*.png'):
            mlflow.log_artifact(str(p), "plots")
        
        # Сравнительная таблица
        log_comparison_table(all_results)
        
        # Логируем сводные метрики
        for r in all_results:
            mlflow.log_metric(f"final_ranking/{r['model']}_map50", r['map50'])
        
        # Итоговая таблица в логах
        logger.info("\n" + "="*100)
        logger.info(f"{'Model':<35} {'mAP@50':<10} {'mAP@75':<10} {'FPS':<10} {'Params(M)':<12} {'Method':<15} {'Status':<12}")
        logger.info("-"*100)
        
        for r in sorted(all_results, key=lambda x: x['map50'], reverse=True):
            logger.info(
                f"{r['model']:<35} {r['map50']:<10.4f} {r['map75']:<10.4f} "
                f"{r['fps']:<10.1f} {r['params_M']:<12.1f} "
                f"{r.get('distill_method', 'teacher'):<15} {'✅' if r['map50'] > 0 else '❌'}"
            )
        logger.info("="*100)


if __name__ == "__main__":
    main()