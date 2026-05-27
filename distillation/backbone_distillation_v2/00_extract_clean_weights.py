#!/usr/bin/env python3
"""
Извлечение чистого state_dict DINOv3 из чекпоинта LTDETR детектора.
Убирает префикс 'model.backbone.dinov3.' из ключей.
Сохраняет в ту же директорию, где находится скрипт.
"""

import sys
from pathlib import Path
import torch
import yaml
from collections import OrderedDict


def extract_clean_state_dict(checkpoint_path: str, output_dir: str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка чекпоинта: {checkpoint_path}")
    
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    
    if "train_model" not in ckpt:
        print("❌ Ключ 'train_model' не найден")
        return None
    
    model_state = ckpt["train_model"]
    
    # Убираем префикс 'model.backbone.dinov3.' из ключей
    clean_state_dict = OrderedDict()
    prefix = "model.backbone.dinov3."
    
    for key, value in model_state.items():
        if key.startswith(prefix):
            clean_key = key[len(prefix):]
            clean_state_dict[clean_key] = value
    
    print(f"Исходных ключей: {len(model_state)}")
    print(f"После удаления префикса: {len(clean_state_dict)}")
    
    if len(clean_state_dict) == 0:
        print("❌ Не найдено ключей с префиксом 'model.backbone.dinov3.'")
        return None
    
    # Показываем примеры
    keys = list(clean_state_dict.keys())
    print("Первые 5 ключей после очистки:")
    for k in keys[:5]:
        print(f"  {k}: shape={clean_state_dict[k].shape}")
    
    # Сохраняем в ту же директорию, где скрипт
    output_path = output_dir / "teacher_clean_state_dict.pt"
    torch.save(clean_state_dict, str(output_path))
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"\n✅ Чистый state_dict сохранён: {output_path}")
    print(f"   Размер: {size_mb:.1f} MB")
    print(f"   Ключей: {len(clean_state_dict)}")
    
    return output_path


def main():
    # Директория, где находится этот скрипт
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        print("config.yaml не найден")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    checkpoint_path = cfg["teacher"]["teacher_weights"]
    
    print("=" * 70)
    print("ИЗВЛЕЧЕНИЕ ЧИСТОГО STATE_DICT (БЕЗ ПРЕФИКСА)")
    print("=" * 70)
    print(f"Чекпоинт:    {checkpoint_path}")
    print(f"Сохранение в: {script_dir}")
    print("=" * 70)
    
    output_path = extract_clean_state_dict(checkpoint_path, script_dir)
    
    if output_path:
        print(f"\n✅ ГОТОВО!")
        print(f"Обновите config.yaml:")
        print(f'  teacher_weights: "teacher_clean_state_dict.pt"')
    else:
        print("\n❌ Не удалось извлечь веса")
        sys.exit(1)


if __name__ == "__main__":
    main()