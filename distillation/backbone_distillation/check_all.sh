#!/bin/bash
# ============================================================================
# ЧЕК-СКРИПТ ДЛЯ ПРОВЕРКИ ВСЕХ КОМПОНЕНТОВ ЭКСПЕРИМЕНТА
# ============================================================================

# НЕ используем set -e - скрипт продолжает работу даже при ошибках

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Счётчики
PASSED=0
FAILED=0
WARNINGS=0

# ============================================================================
# Функции
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}============================================================================${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}============================================================================${NC}"
}

print_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
    ((WARNINGS++))
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

check_file() {
    if [ -f "$1" ]; then
        print_ok "Файл найден: $1"
        return 0
    else
        print_fail "Файл не найден: $1"
        return 1
    fi
}

check_directory() {
    if [ -d "$1" ]; then
        print_ok "Директория найдена: $1"
        return 0
    else
        print_fail "Директория не найдена: $1"
        return 1
    fi
}

check_python_import() {
    if python3 -c "import $1" 2>/dev/null; then
        print_ok "Python import: $1"
        return 0
    else
        print_fail "Python import: $1 (не установлен)"
        return 1
    fi
}

check_python_import_version() {
    VERSION=$(python3 -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
    if [ "$VERSION" != "unknown" ]; then
        print_ok "Python import: $1 (версия $VERSION)"
        return 0
    else
        print_fail "Python import: $1 (не установлен или нет __version__)"
        return 1
    fi
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_ok "Команда найдена: $1"
        return 0
    else
        print_warning "Команда не найдена: $1"
        return 1
    fi
}

# ============================================================================
# Основная проверка
# ============================================================================

echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║                    ЧЕК-СКРИПТ ЭКСПЕРИМЕНТА                              ║${NC}"
echo -e "${BOLD}${BLUE}║                 Дистилляция бэкбона для детекции дефектов               ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# 1. Проверка Python и базовых инструментов
# ============================================================================

print_header "1. ПРОВЕРКА PYTHON И ИНСТРУМЕНТОВ"

# Python версия
PYTHON_VERSION=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    print_ok "Python: $PYTHON_VERSION"
else
    print_fail "Python не найден"
fi

# pip
check_command pip3

# nvidia-smi (если есть GPU)
if command -v nvidia-smi &> /dev/null; then
    print_ok "nvidia-smi: доступен"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        print_info "GPU: $GPU_INFO"
    fi
else
    print_info "nvidia-smi не найден (возможно, нет GPU)"
fi

# ============================================================================
# 2. Проверка зависимостей Python
# ============================================================================

print_header "2. ПРОВЕРКА ЗАВИСИМОСТЕЙ PYTHON"

# Основные зависимости
check_python_import_version torch || print_warning "torch не установлен"
check_python_import_version torchvision || print_warning "torchvision не установлен"
check_python_import_version numpy || print_warning "numpy не установлен"
check_python_import PIL || print_warning "PIL не установлен"
check_python_import yaml || print_warning "yaml (PyYAML) не установлен"
check_python_import_version lightly_train || print_warning "lightly-train не установлен"

# Дополнительные зависимости
check_python_import matplotlib || print_info "matplotlib не установлен (необходим для визуализации)"
check_python_import scipy || print_info "scipy не установлен (необходим для стат. тестов)"
check_python_import mlflow || print_info "mlflow не установлен (опционально)"

# ============================================================================
# 3. Проверка наличия всех скриптов
# ============================================================================

print_header "3. ПРОВЕРКА СКРИПТОВ ЭКСПЕРИМЕНТА"

SCRIPT_DIR="/app/backbone_distillation"

check_file "$SCRIPT_DIR/01_pretrain_backbone.py"
check_file "$SCRIPT_DIR/02_train_detectors.py"
check_file "$SCRIPT_DIR/03_evaluate_all_models.py"
check_file "$SCRIPT_DIR/04_visualize_results.py"
check_file "$SCRIPT_DIR/05_generate_report.py"
check_file "$SCRIPT_DIR/run_all.sh"
check_file "$SCRIPT_DIR/config_pretrain_comparison.yaml"

# ============================================================================
# 4. Проверка конфигурации
# ============================================================================

print_header "4. ПРОВЕРКА КОНФИГУРАЦИИ"

if [ -f "$SCRIPT_DIR/config_pretrain_comparison.yaml" ]; then
    print_ok "Конфиг найден"
    
    print_info "Содержимое конфига:"
    
    TEACHER_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/config_pretrain_comparison.yaml'))['teacher']['model'])" 2>/dev/null || echo "ошибка")
    print_info "  teacher.model: $TEACHER_MODEL"
    
    TEACHER_WEIGHTS=$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/config_pretrain_comparison.yaml'))['teacher'].get('weights', 'не указан'))" 2>/dev/null || echo "ошибка")
    print_info "  teacher.weights: $TEACHER_WEIGHTS"
    
    DATA_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/config_pretrain_comparison.yaml'))['detection']['data_path'])" 2>/dev/null || echo "ошибка")
    print_info "  detection.data_path: $DATA_PATH"
    
    NUM_CLASSES=$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/config_pretrain_comparison.yaml'))['detection']['num_classes'])" 2>/dev/null || echo "ошибка")
    print_info "  detection.num_classes: $NUM_CLASSES"
else
    print_fail "Конфиг не найден"
fi

# ============================================================================
# 5. Проверка путей к данным
# ============================================================================

print_header "5. ПРОВЕРКА ПУТЕЙ К ДАННЫМ"

# Датасет
if [ -n "$DATA_PATH" ] && [ "$DATA_PATH" != "ошибка" ]; then
    if [ -d "$DATA_PATH" ]; then
        print_ok "Директория датасета: $DATA_PATH"
        
        if [ -d "$DATA_PATH/train/images" ]; then
            TRAIN_COUNT=$(find "$DATA_PATH/train/images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
            print_info "  Train images: $TRAIN_COUNT"
        else
            print_warning "  Нет папки train/images"
        fi
        
        if [ -d "$DATA_PATH/val/images" ]; then
            VAL_COUNT=$(find "$DATA_PATH/val/images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
            print_info "  Val images: $VAL_COUNT"
        else
            print_warning "  Нет папки val/images"
        fi
    else
        print_warning "Директория датасета не найдена: $DATA_PATH"
    fi
fi

# Путь к весам учителя
if [ -n "$TEACHER_WEIGHTS" ] && [ "$TEACHER_WEIGHTS" != "не указан" ] && [ "$TEACHER_WEIGHTS" != "ошибка" ]; then
    if [ -f "$TEACHER_WEIGHTS" ]; then
        SIZE=$(du -h "$TEACHER_WEIGHTS" 2>/dev/null | cut -f1)
        print_ok "Веса учителя: $TEACHER_WEIGHTS ($SIZE)"
    else
        print_warning "Веса учителя не найдены: $TEACHER_WEIGHTS"
    fi
fi

# ============================================================================
# 6. Проверка импорта fasterrcnn (критично)
# ============================================================================

print_header "6. ПРОВЕРКА КРИТИЧЕСКОГО ИМПОРТА"

print_info "Проверка импорта fasterrcnn_resnet18_fpn..."
if python3 -c "from torchvision.models.detection import fasterrcnn_resnet18_fpn" 2>/dev/null; then
    print_ok "  fasterrcnn_resnet18_fpn: импорт работает"
else
    print_fail "  fasterrcnn_resnet18_fpn: импорт НЕ РАБОТАЕТ"
    print_info "    Решение: обновите torchvision: pip install --upgrade torchvision"
fi

# ============================================================================
# 7. Проверка GPU/CPU
# ============================================================================

print_header "7. ПРОВЕРКА GPU/CPU"

python3 << 'EOF' 2>/dev/null
import torch
import sys
print(f"  PyTorch версия: {torch.__version__}")
print(f"  CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA версия: {torch.version.cuda}")
    print(f"  GPU устройств: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print(f"  CPU потоков: {torch.get_num_threads()}")
    print(f"  MPS доступен: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
EOF

if [ $? -eq 0 ]; then
    print_ok "GPU/CPU проверка пройдена"
else
    print_warning "Ошибка при проверке GPU/CPU"
fi

# ============================================================================
# 8. Проверка синтаксиса скриптов
# ============================================================================

print_header "8. ПРОВЕРКА СИНТАКСИСА PYTHON СКРИПТОВ"

check_syntax() {
    if python3 -m py_compile "$1" 2>/dev/null; then
        print_ok "Синтаксис верен: $(basename $1)"
        return 0
    else
        print_fail "Синтаксическая ошибка: $(basename $1)"
        return 1
    fi
}

check_syntax "$SCRIPT_DIR/01_pretrain_backbone.py" 2>/dev/null || true
check_syntax "$SCRIPT_DIR/02_train_detectors.py" 2>/dev/null || true
check_syntax "$SCRIPT_DIR/03_evaluate_all_models.py" 2>/dev/null || true

# ============================================================================
# 9. Проверка обученных моделей
# ============================================================================

print_header "9. ПРОВЕРКА ОБУЧЕННЫХ МОДЕЛЕЙ"

DETECTION_OUTPUT="/app/backbone_distillation/detectors"

for student in faster_rcnn_r18_scratch faster_rcnn_r18_imagenet faster_rcnn_r18_distilled; do
    MODEL_PATH="$DETECTION_OUTPUT/$student/model_final.pth"
    if [ -f "$MODEL_PATH" ]; then
        SIZE=$(du -h "$MODEL_PATH" 2>/dev/null | cut -f1)
        print_ok "  $student: $SIZE"
    else
        print_info "  $student: не найдена (будет обучена)"
    fi
done

# ============================================================================
# ИТОГИ
# ============================================================================

print_header "ИТОГИ ПРОВЕРКИ"

echo ""
echo -e "${GREEN}✓ Пройдено: $PASSED${NC}"
echo -e "${RED}✗ Ошибок: $FAILED${NC}"
echo -e "${YELLOW}⚠ Предупреждений: $WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ Все проверки пройдены! Эксперимент готов к запуску.${NC}"
    echo ""
    echo -e "Запустите команду:"
    echo -e "  ${BOLD}cd /app/backbone_distillation && ./run_all.sh${NC}"
else
    echo -e "${RED}${BOLD}❌ Обнаружены ошибки. Исправьте их перед запуском.${NC}"
    echo ""
    echo -e "Основные проблемы:"
    
    if ! python3 -c "from torchvision.models.detection import fasterrcnn_resnet18_fpn" 2>/dev/null; then
        echo -e "  ${RED}•${NC} Проблема с импортом fasterrcnn_resnet18_fpn"
        echo -e "    Решение: обновите torchvision или исправьте импорт в 02_train_detectors.py"
        echo -e "    Команда: pip install --upgrade torchvision"
    fi
    
    echo ""
    echo -e "Проверьте также:"
    echo -e "  • Все ли пути в config_pretrain_comparison.yaml верны"
    echo -e "  • Доступны ли данные по указанным путям"
    echo -e "  • Установлены ли все зависимости"
fi

echo ""
exit $FAILED