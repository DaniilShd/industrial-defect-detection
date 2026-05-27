#!/usr/bin/env bash
# run_full_experiment.sh
# Полный пайплайн: дистилляция → обучение детекторов → оценка
# Запуск: bash run_full_experiment.sh

set -e  # Остановка при ошибке

echo "========================================"
echo "ЗАПУСК ПОЛНОГО ЭКСПЕРИМЕНТА"
echo "========================================"
echo ""

# Шаг 1: Дистилляция DINOv3 → ResNet18
echo "========================================"
echo "ШАГ 1/3: ДИСТИЛЛЯЦИЯ БЭКБОНА"
echo "========================================"
python 01_pretrain_backbone.py
if [ $? -eq 0 ]; then
    echo "✅ Дистилляция завершена успешно"
else
    echo "❌ Ошибка дистилляции"
    exit 1
fi
echo ""

# Шаг 2: Обучение Faster R-CNN
echo "========================================"
echo "ШАГ 2/3: ОБУЧЕНИЕ ДЕТЕКТОРОВ"
echo "========================================"
python 02_train_detectors.py
if [ $? -eq 0 ]; then
    echo "✅ Обучение завершено успешно"
else
    echo "❌ Ошибка обучения"
    exit 1
fi
echo ""

# Шаг 3: Оценка моделей
echo "========================================"
echo "ШАГ 3/3: ОЦЕНКА МОДЕЛЕЙ"
echo "========================================"
python 03_evaluate.py
if [ $? -eq 0 ]; then
    echo "✅ Оценка завершена успешно"
else
    echo "❌ Ошибка оценки"
    exit 1
fi
echo ""

# Итоги
echo "========================================"
echo "ЭКСПЕРИМЕНТ ЗАВЕРШЁН"
echo "========================================"
echo ""
echo "Результаты:"
echo "  - Дистиллированный бэкбон: outputs/pretrain/resnet18_distilled/exported_models/exported_last.pt"
echo "  - Обученные детекторы:     outputs/detection/"
echo "  - Метрики:                 outputs/results/evaluation.json"
echo "  - Лог оценки:              03_evaluate.log"
echo ""

# Показать краткие результаты
if [ -f "outputs/results/evaluation.json" ]; then
    echo "Краткие результаты:"
    python -c "
import json
data = json.load(open('outputs/results/evaluation.json'))
for r in data:
    print(f\"  {r['model']:<35} mAP50:95={r.get('map50_95',0):.4f}  FPS={r.get('fps',0):.1f}\")
"
fi