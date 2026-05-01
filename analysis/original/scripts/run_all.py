#!/usr/bin/env python3
"""Поочерёдный запуск всех скриптов анализа."""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SCRIPTS = [
    "01_load_and_check.py",
    "02_texture_analysis.py",
    "03_color_analysis.py",
    "04_brightness_analysis.py",
    "05_spectral_analysis.py",
    "06_cluster_analysis.py",
    "07_generate_report.py",
]


def run_script(script_name: str) -> bool:
    """Запуск одного скрипта."""
    script_path = SCRIPT_DIR / "scripts" / script_name
    print(f"\n{'=' * 60}")
    print(f"Запуск: {script_name}")
    print(f"{'=' * 60}")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        capture_output=False
    )
    return result.returncode == 0


def main():
    print("=" * 60)
    print("ПОЛНЫЙ АНАЛИЗ ОРИГИНАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    total = len(SCRIPTS)
    passed = 0
    failed = 0
    
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{total}] {script}")
        
        if run_script(script):
            print(f"✅ [{i}/{total}] {script} — УСПЕШНО")
            passed += 1
        else:
            print(f"❌ [{i}/{total}] {script} — ОШИБКА")
            failed += 1
            
            reply = input("Продолжить? (y/n): ").strip().lower()
            if reply != 'y':
                break
    
    print(f"\n{'=' * 60}")
    print(f"ИТОГО: {passed} успешно, {failed} с ошибками из {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()