# analysis/synthetic/scripts/main.py
#!/usr/bin/env python3
"""
Главный скрипт анализа synthetic данных
Запускает все аналитические модули
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Добавляем пути
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from config import AnalysisConfig
from domain_gap import run_domain_gap_analysis
from class_analysis import run_class_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Комплексный анализ synthetic данных"
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
    print("🔬 SYNTHETIC DATA ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print(f"Original dir: {config.paths.original_dir}")
    print(f"Synthetic dir: {config.paths.synthetic_dir}")
    print(f"Output dir: {output_dir}")
    
    # Проверка директорий
    if not (config.paths.original_dir / "images").exists():
        print(f"❌ Original images not found: {config.paths.original_dir / 'images'}")
        sys.exit(1)
    
    if not (config.paths.synthetic_dir / "images").exists():
        print(f"❌ Synthetic images not found: {config.paths.synthetic_dir / 'images'}")
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
        print("📊 STEP 1/3: DOMAIN GAP ANALYSIS")
        print("=" * 80)
        
        try:
            domain_gap_results = run_domain_gap_analysis(config)
            all_results['analyses']['domain_gap'] = {
                "status": "completed",
                "output_dir": str(output_dir),
                "summary": {
                    "domain_overlap_score": domain_gap_results.get('similarity_metrics', {}).get('domain_overlap_score'),
                    "1nn_accuracy": domain_gap_results.get('similarity_metrics', {}).get('1nn_domain_accuracy'),
                    "mean_emd": domain_gap_results.get('emd_metrics', {}).get('mean_emd')
                }
            }
        except Exception as e:
            print(f"❌ Domain gap analysis failed: {e}")
            import traceback
            traceback.print_exc()
            all_results['analyses']['domain_gap'] = {"status": "failed", "error": str(e)}
    
    # 2. Class Distribution Analysis
    if not args.skip_class_analysis:
        print("\n" + "=" * 80)
        print("📊 STEP 2/3: CLASS DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        try:
            class_results = run_class_analysis(config)
            
            # Извлекаем сводку
            class_summary = {
                "total_bboxes": class_results.get('total_bboxes'),
                "images_with_defects": class_results.get('images_with_defects'),
                "empty_images": class_results.get('empty_images'),
                "class_distribution": class_results.get('class_distribution', {})
            }
            
            all_results['analyses']['class_distribution'] = {
                "status": "completed",
                "output_dir": str(output_dir),
                "summary": class_summary
            }
        except Exception as e:
            print(f"❌ Class analysis failed: {e}")
            import traceback
            traceback.print_exc()
            all_results['analyses']['class_distribution'] = {"status": "failed", "error": str(e)}
    
    # 3. Сохранение финального отчёта
    print("\n" + "=" * 80)
    print("📊 STEP 3/3: SAVING FINAL REPORT")
    print("=" * 80)
    
    # Сохраняем сводный JSON
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✅ Summary saved: {summary_path}")
    
    # Создаем сводный текстовый отчёт
    report_path = output_dir / "analysis_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SYNTHETIC DATA ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {all_results['timestamp']}\n")
        f.write(f"Original: {all_results['original_dir']}\n")
        f.write(f"Synthetic: {all_results['synthetic_dir']}\n\n")
        
        # Domain Gap summary
        if 'domain_gap' in all_results['analyses']:
            dg = all_results['analyses']['domain_gap']
            f.write("-" * 80 + "\n")
            f.write("DOMAIN GAP ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if dg['status'] == 'completed':
                s = dg['summary']
                f.write(f"  Domain Overlap Score: {s.get('domain_overlap_score', 'N/A')}\n")
                f.write(f"  1-NN Accuracy: {s.get('1nn_accuracy', 'N/A')}\n")
                f.write(f"  Mean EMD: {s.get('mean_emd', 'N/A')}\n")
            else:
                f.write(f"  Status: FAILED - {dg.get('error', 'Unknown error')}\n")
        
        # Class Distribution summary
        if 'class_distribution' in all_results['analyses']:
            cd = all_results['analyses']['class_distribution']
            f.write("\n" + "-" * 80 + "\n")
            f.write("CLASS DISTRIBUTION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if cd['status'] == 'completed':
                s = cd['summary']
                f.write(f"  Total BBoxes: {s.get('total_bboxes', 'N/A')}\n")
                f.write(f"  Images with defects: {s.get('images_with_defects', 'N/A')}\n")
                f.write(f"  Empty images: {s.get('empty_images', 'N/A')}\n")
                
                for cls_name, cls_stats in s.get('class_distribution', {}).items():
                    f.write(f"  - {cls_name}: {cls_stats.get('bbox_count', 0)} bboxes "
                           f"({cls_stats.get('percentage', 0)}%)\n")
            else:
                f.write(f"  Status: FAILED - {cd.get('error', 'Unknown error')}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Full details: {output_dir}\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Report saved: {report_path}")
    
    # Выводим сводку в консоль
    print("\n" + open(report_path, 'r').read())
    
    print(f"\n✅ Analysis complete! Results: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())