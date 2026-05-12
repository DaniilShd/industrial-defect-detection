#!/usr/bin/env python3
"""
Главный скрипт анализа синтетических данных
Запускает все аналитические модули
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np

# Добавляем пути
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from config import AnalysisConfig
from domain_gap import run_domain_gap_analysis
from class_analysis import run_class_analysis


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder для numpy типов"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Комплексный анализ синтетических данных"
    )
    parser.add_argument(
        "--config", type=str, 
        default="analysis/synthetic/config.yaml",
        help="Путь к конфигурационному файлу"
    )
    parser.add_argument(
        "--original_dir", type=str, default=None,
        help="Путь к оригинальным данным (переопределяет config)"
    )
    parser.add_argument(
        "--synthetic_dir", type=str, default=None,
        help="Путь к синтетическим данным (переопределяет config)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Путь для сохранения результатов"
    )
    parser.add_argument(
        "--skip_domain_gap", action="store_true",
        help="Пропустить domain gap анализ"
    )
    parser.add_argument(
        "--skip_class_analysis", action="store_true",
        help="Пропустить анализ классов"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Количество семплов для анализа"
    )
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    config = AnalysisConfig.from_yaml(config_path)
    
    # Переопределение параметров из командной строки
    if args.original_dir:
        config.paths.original_dir = Path(args.original_dir)
    if args.synthetic_dir:
        config.paths.synthetic_dir = Path(args.synthetic_dir)
    if args.num_samples:
        config.dinov2.num_samples = args.num_samples
    
    # Создание выходной директории с timestamp
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = config.setup_directories()
    
    print("=" * 80)
    print("🔬 АНАЛИЗ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("=" * 80)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Конфиг: {config_path}")
    print(f"Оригинал: {config.paths.original_dir}")
    print(f"Синтетика: {config.paths.synthetic_dir}")
    print(f"Результаты: {output_dir}")
    
    # Проверка директорий
    if not (config.paths.original_dir / "images").exists():
        print(f"❌ Оригинальные изображения не найдены: {config.paths.original_dir / 'images'}")
        sys.exit(1)
    
    if not (config.paths.synthetic_dir / "images").exists():
        print(f"❌ Синтетические изображения не найдены: {config.paths.synthetic_dir / 'images'}")
        sys.exit(1)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config_file": str(config_path),
        "original_dir": str(config.paths.original_dir),
        "synthetic_dir": str(config.paths.synthetic_dir),
        "analyses": {}
    }
    
    # 1. Domain Gap Analysis
    if not args.skip_domain_gap:
        print("\n" + "=" * 80)
        print("📊 ШАГ 1/3: АНАЛИЗ РАЗРЫВА ДОМЕНОВ")
        print("=" * 80)
        
        try:
            domain_gap_results = run_domain_gap_analysis(config)
            
            # Извлекаем ключевые метрики (используем английские ключи из JSON)
            dg = domain_gap_results.get('domain_gap', {})
            emd = domain_gap_results.get('emd_analysis', {})
            
            all_results['analyses']['domain_gap'] = {
                "status": "completed",
                "output_dir": str(output_dir),
                "summary": {
                    "domain_overlap_score": dg.get('overlap_score'),
                    "1nn_accuracy": dg.get('nn_accuracy'),
                    "cosine_similarity": dg.get('centroid_cosine_similarity'),
                    "gap_ratio": dg.get('gap_ratio'),
                    "mean_emd": emd.get('mean')
                }
            }
        except Exception as e:
            print(f"❌ Ошибка анализа разрыва доменов: {e}")
            import traceback
            traceback.print_exc()
            all_results['analyses']['domain_gap'] = {
                "status": "failed", 
                "error": str(e)
            }
    
    # 2. Class Distribution Analysis
    if not args.skip_class_analysis:
        print("\n" + "=" * 80)
        print("📊 ШАГ 2/3: АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
        print("=" * 80)
        
        try:
            class_results = run_class_analysis(config)
            
            # Извлекаем сводку (используем английские ключи из JSON)
            cs = class_results.get('summary', {})
            
            all_results['analyses']['class_distribution'] = {
                "status": "completed",
                "output_dir": str(output_dir),
                "summary": {
                    "total_bboxes": cs.get('total_bboxes'),
                    "images_with_defects": cs.get('images_with_defects'),
                    "empty_images": cs.get('empty_images'),
                    "total_images": cs.get('total_images'),
                    "class_distribution": class_results.get('class_stats', {})
                }
            }
        except Exception as e:
            print(f"❌ Ошибка анализа классов: {e}")
            import traceback
            traceback.print_exc()
            all_results['analyses']['class_distribution'] = {
                "status": "failed", 
                "error": str(e)
            }
    
    # 3. Сохранение финального отчёта
    print("\n" + "=" * 80)
    print("📊 ШАГ 3/3: СОХРАНЕНИЕ ОТЧЁТА")
    print("=" * 80)
    
    # Сохраняем сводный JSON
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"✅ Сводка сохранена: {summary_path}")
    
    # Создаем русский текстовый отчёт
    report_path = output_dir / "analysis_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("СВОДКА АНАЛИЗА СИНТЕТИЧЕСКИХ ДАННЫХ\n")
        f.write("=" * 80 + "\n")
        f.write(f"Создан: {all_results['timestamp']}\n")
        f.write(f"Оригинал: {all_results['original_dir']}\n")
        f.write(f"Синтетика: {all_results['synthetic_dir']}\n\n")
        
        # Domain Gap summary
        if 'domain_gap' in all_results['analyses']:
            dg = all_results['analyses']['domain_gap']
            f.write("-" * 80 + "\n")
            f.write("АНАЛИЗ РАЗРЫВА ДОМЕНОВ\n")
            f.write("-" * 80 + "\n")
            if dg['status'] == 'completed':
                s = dg['summary']
                f.write(f"  Оценка перекрытия доменов: {s.get('domain_overlap_score', 'N/A')}\n")
                f.write(f"  Точность 1-NN: {s.get('1nn_accuracy', 'N/A')}\n")
                f.write(f"  Косинусное сходство: {s.get('cosine_similarity', 'N/A')}\n")
                f.write(f"  Коэффициент разрыва: {s.get('gap_ratio', 'N/A')}\n")
                f.write(f"  Среднее EMD: {s.get('mean_emd', 'N/A')}\n")
            else:
                f.write(f"  Статус: ОШИБКА - {dg.get('error', 'Неизвестная ошибка')}\n")
        
        # Class Distribution summary
        if 'class_distribution' in all_results['analyses']:
            cd = all_results['analyses']['class_distribution']
            f.write("\n" + "-" * 80 + "\n")
            f.write("АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ\n")
            f.write("-" * 80 + "\n")
            if cd['status'] == 'completed':
                s = cd['summary']
                f.write(f"  Всего рамок: {s.get('total_bboxes', 'N/A')}\n")
                f.write(f"  Изображений с дефектами: {s.get('images_with_defects', 'N/A')}\n")
                f.write(f"  Пустых изображений: {s.get('empty_images', 'N/A')}\n\n")
                
                for cls_name, cls_stats in s.get('class_distribution', {}).items():
                    f.write(f"  - {cls_name}: {cls_stats.get('bbox_count', 0)} рамок "
                           f"({cls_stats.get('percentage', 0)}%)\n")
            else:
                f.write(f"  Статус: ОШИБКА - {cd.get('error', 'Неизвестная ошибка')}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Подробные результаты: {output_dir}\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Отчёт сохранён: {report_path}")
    
    # Выводим сводку в консоль
    print("\n" + open(report_path, 'r', encoding='utf-8').read())
    
    print(f"\n✅ Анализ завершён! Результаты: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())