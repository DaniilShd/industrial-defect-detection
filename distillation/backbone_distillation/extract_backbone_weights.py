# extract_backbone_weights.py
import torch
import lightly_train
from pathlib import Path

# Путь к вашему чекпоинту детектора
detector_path = "/app/data/experiment_v2/models/exp3_ssl/mixed_full_ssl_seed777/exported_models/exported_best.pt"

# Загружаем детектор через load_model
detector = lightly_train.load_model(detector_path)

print(f"Тип модели: {type(detector)}")

# Извлекаем бэкбон
if hasattr(detector, 'backbone'):
    backbone = detector.backbone
    print(f"✅ Бэкбон найден: {type(backbone)}")
    
    # Получаем state_dict бэкбона
    backbone_state_dict = backbone.state_dict()
    
    print(f"Всего ключей в state_dict бэкбона: {len(backbone_state_dict)}")
    
    # Фильтруем: оставляем только ключи для DINOv3 (исключаем sta., convs., norms.)
    filtered_state_dict = {}
    excluded_keys = []
    
    for key, value in backbone_state_dict.items():
        # Пропускаем ключи, которые не относятся к DINOv3 бэкбону
        if key.startswith(('sta.', 'convs.', 'norms.', 'stem.')):
            excluded_keys.append(key)
            continue
        
        # Удаляем префикс "dinov3." если есть
        if key.startswith("dinov3."):
            new_key = key[7:]  # удаляем "dinov3."
        else:
            new_key = key
        
        filtered_state_dict[new_key] = value
    
    print(f"Исключено ключей (не backbone): {len(excluded_keys)}")
    print(f"Оставлено ключей (чистый backbone): {len(filtered_state_dict)}")
    
    if excluded_keys:
        print(f"\nПримеры исключенных ключей:")
        for key in excluded_keys[:5]:
            print(f"  - {key}")
    
    # Сохраняем отфильтрованные веса
    output_path = "/app/backbone_distillation/teacher_backbone_weights.pt"
    torch.save(filtered_state_dict, output_path)
    
    print(f"\n✅ Веса чистого бэкбона сохранены в {output_path}")
    print(f"   Размер файла: {filtered_state_dict.__sizeof__() / (1024**2):.1f} MB")
    
    # Проверка
    print("\nПроверка сохраненных ключей:")
    test_keys = list(filtered_state_dict.keys())
    print(f"Первые 5 ключей: {test_keys[:5]}")
    print(f"Последние 5 ключей: {test_keys[-5:]}")
    
    # Проверяем, что нет запрещенных ключей
    has_invalid = any(k.startswith(('sta.', 'convs.', 'norms.')) for k in test_keys)
    if not has_invalid:
        print("✅ Фильтрация успешна: запрещенные ключи отсутствуют")
    else:
        print("⚠️ ВНИМАНИЕ: Все еще есть запрещенные ключи")
    
elif hasattr(detector, 'model') and hasattr(detector.model, 'backbone'):
    backbone = detector.model.backbone
    backbone_state_dict = backbone.state_dict()
    
    # Фильтруем
    filtered_state_dict = {}
    for key, value in backbone_state_dict.items():
        if key.startswith(('sta.', 'convs.', 'norms.', 'stem.')):
            continue
        if key.startswith("dinov3."):
            filtered_state_dict[key[7:]] = value
        else:
            filtered_state_dict[key] = value
    
    output_path = "/app/backbone_distillation/teacher_backbone_weights.pt"
    torch.save(filtered_state_dict, output_path)
    print(f"✅ Веса бэкбона сохранены и отфильтрованы")
    
else:
    print("❌ Не удалось найти бэкбон")

print("\n" + "="*50)
print("ГОТОВО! Теперь используйте в конфиге:")
print("  weights: /app/backbone_distillation/teacher_backbone_weights.pt")