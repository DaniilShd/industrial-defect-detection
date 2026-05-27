#!/bin/bash
# check_modern.sh — полная проверка пайплайна modern_distillation

echo "============================================"
echo "  ПРОВЕРКА MODERN DISTILLATION PIPELINE"
echo "============================================"
echo ""

PASS=0
FAIL=0
WARN=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "  ✅ $1"
        PASS=$((PASS+1))
    else
        echo -e "  ❌ $1"
        FAIL=$((FAIL+1))
    fi
}

warn() {
    echo -e "  ⚠️  $1"
    WARN=$((WARN+1))
}

ROOT="/app/distillation/modern_distillation"

echo "=== 1. СТРУКТУРА ПРОЕКТА ==="
for dir in scripts utils pretrain detectors reports logs; do
    [ -d "$ROOT/$dir" ] && check "Папка $dir" || { echo "  ❌ Нет папки $dir"; FAIL=$((FAIL+1)); }
done

echo ""
echo "=== 2. КОНФИГ ==="
CONFIG="$ROOT/config_modern_distillation.yaml"
[ -f "$CONFIG" ] && check "config_modern_distillation.yaml существует" || { echo "  ❌ Нет конфига!"; FAIL=$((FAIL+1)); }

python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print('OK')" 2>/dev/null && check "YAML валидный" || { echo "  ❌ YAML невалидный!"; FAIL=$((FAIL+1)); }

python3 -c "
import yaml
c=yaml.safe_load(open('$CONFIG'))
required = ['experiment','teacher','modern_distillation','students','detection','paths','fps']
for k in required:
    assert k in c, f'Missing: {k}'
print('OK')
" 2>/dev/null && check "Все секции конфига" || { echo "  ❌ Не хватает секций!"; FAIL=$((FAIL+1)); }

python3 -c "
import yaml
c=yaml.safe_load(open('$CONFIG'))
comp = c['modern_distillation']['components']
for k in ['feature_matching','contrastive','structural','masking','attention']:
    assert comp[k]['enabled'] == True, f'{k} not enabled'
print('OK')
" 2>/dev/null && check "Все 5 компонент включены" || warn "Не все компоненты включены"

echo ""
echo "=== 3. СКРИПТЫ ==="
for f in 01_modern_distill.py 02_train_detectors.py 03_evaluate_models.py 04_visualize_results.py 05_generate_report.py; do
    [ -f "$ROOT/scripts/$f" ] && check "$f" || { echo "  ❌ Нет $f"; FAIL=$((FAIL+1)); }
done

echo ""
echo "=== 4. UTILS ==="
for f in __init__.py feature_extractors.py modern_losses.py modern_distiller.py; do
    [ -f "$ROOT/utils/$f" ] && check "utils/$f" || { echo "  ❌ Нет utils/$f"; FAIL=$((FAIL+1)); }
done

echo ""
echo "=== 5. ПУТИ В СКРИПТАХ ==="
cd "$ROOT/scripts/"

# Проверка конфигов
for f in *.py; do
    grep -q "config_modern_distillation.yaml" "$f" && check "$f → правильный конфиг" || warn "$f → нет ссылки на modern конфиг"
done

# Проверка modern_distilled
grep -q "modern_distilled" 02_train_detectors.py && check "02 → modern_distilled init" || { echo "  ❌ Нет modern_distilled в 02!"; FAIL=$((FAIL+1)); }
grep -q "modern_model_path.txt" 02_train_detectors.py && check "02 → modern_model_path.txt" || warn "02 → нет modern_model_path.txt"

# Проверка modern_distilled в evaluator
grep -q "modern_distilled" 03_evaluate_models.py && check "03 → modern_distilled" || warn "03 → нет modern_distilled"

# Проверка цветов в визуализаторе
grep -q "modern_distilled.*#E67E22" 04_visualize_results.py && check "04 → цвет modern_distilled" || warn "04 → нет цвета modern_distilled"
grep -q "faster_rcnn_r18_modern_distilled" 04_visualize_results.py && check "04 → метка modern" || warn "04 → нет метки modern"

# Проверка меток в отчёте
grep -q "modern_distilled" 05_generate_report.py && check "05 → метка modern" || warn "05 → нет метки modern"

echo ""
echo "=== 6. ПРОВЕРКА IMPORTS ==="
cd "$ROOT"
python3 -c "
import sys
sys.path.insert(0, 'utils')
from feature_extractors import ViTFeatureExtractor, ResNetFeatureExtractor
from modern_losses import ModernDistillationLoss
from modern_distiller import ModernDistiller
print('OK')
" 2>/dev/null && check "Все imports работают" || { echo "  ❌ Ошибка импорта!"; FAIL=$((FAIL+1)); }

echo ""
echo "=== 7. ДАННЫЕ ==="
DATA_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['modern_distillation']['unlabeled_data'])" 2>/dev/null)
[ -d "$DATA_PATH" ] && check "Датасет: $DATA_PATH" || { echo "  ❌ Нет датасета: $DATA_PATH"; FAIL=$((FAIL+1)); }

TEACHER_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['teacher']['weights'])" 2>/dev/null)
[ -f "$TEACHER_PATH" ] && check "Учитель: $TEACHER_PATH" || { echo "  ❌ Нет учителя: $TEACHER_PATH"; FAIL=$((FAIL+1)); }

DETECTION_DATA=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['detection']['data_path'])" 2>/dev/null)
[ -d "$DETECTION_DATA" ] && check "Данные детекции: $DETECTION_DATA" || warn "Нет данных детекции: $DETECTION_DATA"

echo ""
echo "=== 8. RUN SCRIPT ==="
[ -f "$ROOT/run_modern.sh" ] && check "run_modern.sh" || { echo "  ❌ Нет run_modern.sh"; FAIL=$((FAIL+1)); }
[ -x "$ROOT/run_modern.sh" ] && check "run_modern.sh исполняемый" || { chmod +x "$ROOT/run_modern.sh"; check "run_modern.sh сделан исполняемым"; }

echo ""
echo "=== 9. ПРОВЕРКА КОНФЛИКТОВ ==="
# Проверяем что нет старых путей
grep -r "multilayer_distillation" "$ROOT/scripts/" 2>/dev/null && warn "Найдены старые multilayer_distillation пути!" || check "Нет старых multilayer путей"
grep -r "config_pretrain_comparison" "$ROOT/scripts/" 2>/dev/null && warn "Найдены старые config_pretrain_comparison!" || check "Нет старых pretrain_comparison путей"
grep -r "/app/multilayer_distillation" "$ROOT/scripts/" 2>/dev/null && warn "Найдены /app/multilayer_distillation!" || check "Нет старых /app/multilayer_distillation"

echo ""
echo "============================================"
echo "  ИТОГИ ПРОВЕРКИ"
echo "============================================"
echo -e "  ✅ Пройдено: $PASS"
echo -e "  ⚠️  Предупреждений: $WARN"
echo -e "  ❌ Ошибок: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ Пайплайн готов к запуску!"
    echo ""
    echo "Запуск:"
    echo "  cd /app && ./distillation/modern_distillation/run_modern.sh"
else
    echo "❌ Есть ошибки, исправьте перед запуском!"
fi