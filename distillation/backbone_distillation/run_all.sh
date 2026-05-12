#!/bin/bash
# ============================================================================
# Полный пайплайн эксперимента: Дистилляция бэкбона
# ============================================================================
# 
# Этапы:
#   1. Предобучение ResNet18 через дистилляцию от LTDETR (LightlyTrain)
#   2. Обучение Faster R-CNN с разной инициализацией
#   3. Комплексная оценка всех моделей
#   4. Визуализация результатов
#   5. Генерация LaTeX-таблиц для диссертации
#
# Использование:
#   chmod +x run_all.sh
#   ./run_all.sh
#   ./run_all.sh --skip-pretrain  # пропустить предобучение если уже готово
#   ./run_all.sh --only-evaluate   # только оценка и визуализация
# ============================================================================

set -e  # Выход при ошибке
set -o pipefail

# ============================================================================
# Конфигурация
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Флаги
# ============================================================================

SKIP_PRETRAIN=false
ONLY_EVALUATE=false

for arg in "$@"; do
    case $arg in
        --skip-pretrain)
            SKIP_PRETRAIN=true
            shift
            ;;
        --only-evaluate)
            ONLY_EVALUATE=true
            shift
            ;;
        --help|-h)
            echo "Использование: $0 [OPTIONS]"
            echo ""
            echo "Опции:"
            echo "  --skip-pretrain    Пропустить этап предобучения (если уже выполнено)"
            echo "  --only-evaluate    Только оценка и визуализация (без обучения)"
            echo "  --help, -h         Показать эту справку"
            exit 0
            ;;
    esac
done

# ============================================================================
# Функции
# ============================================================================

log_section() {
    echo -e "\n${BLUE}${BOLD}============================================================================${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}============================================================================${NC}\n"
}

log_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗] ОШИБКА:${NC} $1"
    exit 1
}

log_warning() {
    echo -e "${YELLOW}[!] ВНИМАНИЕ:${NC} $1"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 не найден"
    fi
    echo -e "${GREEN}[✓]${NC} Python: $(python3 --version)"
}

check_dependencies() {
    log_step "Проверка зависимостей..."
    
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" || log_error "PyTorch не установлен"
    python3 -c "import lightly_train; print(f'  LightlyTrain: {lightly_train.__version__}')" || log_error "LightlyTrain не установлен"
    python3 -c "import yaml; print('  PyYAML: OK')" || log_error "PyYAML не установлен"
    python3 -c "import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')" || log_error "Matplotlib не установлен"
    python3 -c "import scipy; print(f'  SciPy: {scipy.__version__}')" || log_error "SciPy не установлен"
}

check_data() {
    log_step "Проверка данных..."
    
    # Проверяем конфиг
    if [ ! -f "config_pretrain_comparison.yaml" ]; then
        log_error "Конфиг не найден: config_pretrain_comparison.yaml"
    fi
    
    # Проверяем наличие датасета
    DATA_PATH=$(python3 -c "
import yaml
with open('config_pretrain_comparison.yaml') as f:
    config = yaml.safe_load(f)
print(config['detection']['data_path'])
")
    
    if [ ! -d "$DATA_PATH" ]; then
        log_warning "Датасет не найден: $DATA_PATH"
        log_warning "Убедитесь, что данные доступны"
    else
        log_step "Датасет найден: $DATA_PATH"
    fi
}

check_gpu() {
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        echo -e "${GREEN}[✓]${NC} GPU: $GPU_NAME"
    else
        log_warning "GPU не обнаружен, используется CPU"
    fi
}

# ============================================================================
# Основной пайплайн
# ============================================================================

{
    echo "============================================"
    echo "  Пайплайн дистилляции бэкбона"
    echo "  Запуск: $TIMESTAMP"
    echo "============================================"
    echo ""
    
    check_python
    check_gpu
    check_dependencies
    check_data
    
    # ========================================================================
    # Шаг 1: Предобучение бэкбона
    # ========================================================================
    
    if [ "$ONLY_EVALUATE" = false ] && [ "$SKIP_PRETRAIN" = false ]; then
        log_section "ШАГ 1/5: Предобучение ResNet18 через дистилляцию"
        
        python3 01_pretrain_backbone.py 2>&1 | tee -a "$LOG_DIR/01_pretrain_${TIMESTAMP}.log"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_error "Предобучение завершилось с ошибкой"
        fi
        
        log_step "Предобучение завершено успешно"
    elif [ "$SKIP_PRETRAIN" = true ]; then
        log_section "ШАГ 1/5: Предобучение ПРОПУЩЕНО (--skip-pretrain)"
    fi
    
    # ========================================================================
    # Шаг 2: Обучение детекторов
    # ========================================================================
    
    if [ "$ONLY_EVALUATE" = false ]; then
        log_section "ШАГ 2/5: Обучение Faster R-CNN с разной инициализацией"
        
        python3 02_train_detectors.py 2>&1 | tee -a "$LOG_DIR/02_train_${TIMESTAMP}.log"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_error "Обучение детекторов завершилось с ошибкой"
        fi
        
        log_step "Обучение детекторов завершено"
    else
        log_section "ШАГ 2/5: Обучение ПРОПУЩЕНО (--only-evaluate)"
    fi
    
    # ========================================================================
    # Шаг 3: Оценка моделей
    # ========================================================================
    
    log_section "ШАГ 3/5: Комплексная оценка всех моделей"
    
    python3 03_evaluate_all_models.py 2>&1 | tee -a "$LOG_DIR/03_evaluate_${TIMESTAMP}.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Оценка моделей завершилась с ошибкой"
    fi
    
    log_step "Оценка завершена"
    
    # ========================================================================
    # Шаг 4: Визуализация
    # ========================================================================
    
    log_section "ШАГ 4/5: Визуализация результатов"
    
    python3 04_visualize_results.py 2>&1 | tee -a "$LOG_DIR/04_visualize_${TIMESTAMP}.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Визуализация завершилась с ошибкой"
    fi
    
    log_step "Визуализация завершена"
    
    # ========================================================================
    # Шаг 5: Генерация отчёта
    # ========================================================================
    
    log_section "ШАГ 5/5: Генерация LaTeX-отчёта"
    
    python3 05_generate_report.py 2>&1 | tee -a "$LOG_DIR/05_report_${TIMESTAMP}.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Генерация отчёта завершилась с ошибкой"
    fi
    
    log_step "Отчёт сгенерирован"
    
    # ========================================================================
    # Итоги
    # ========================================================================
    
    log_section "ПАЙПЛАЙН ЗАВЕРШЁН"
    
    RESULTS_DIR=$(python3 -c "
import yaml
with open('config_pretrain_comparison.yaml') as f:
    config = yaml.safe_load(f)
print(config['paths']['results_dir'])
")
    
    echo ""
    echo -e "${GREEN}${BOLD}Результаты сохранены в:${NC}"
    echo -e "  📁 Результаты:     ${RESULTS_DIR}"
    echo -e "  📊 Визуализации:   ${RESULTS_DIR}/*.png"
    echo -e "  📄 LaTeX таблицы:  ${RESULTS_DIR}/dissertation_tables.tex"
    echo -e "  📝 Текстовый отчёт: ${RESULTS_DIR}/report_summary.txt"
    echo -e "  📋 Логи:           ${LOG_DIR}"
    echo ""
    
    # Выводим краткие результаты, если есть
    EVAL_FILE="${RESULTS_DIR}/evaluation_results.json"
    if [ -f "$EVAL_FILE" ]; then
        echo -e "${BOLD}Краткие результаты:${NC}"
        python3 -c "
import json
with open('${EVAL_FILE}') as f:
    results = json.load(f)
    
print(f\"{'Модель':<35} {'mAP@50':<10} {'FPS':<8}\")
print('-' * 55)
for r in sorted(results, key=lambda x: x.get('mAP_50', 0), reverse=True):
    print(f\"{r['model']:<35} {r.get('mAP_50', 0):<10.4f} {r.get('fps', 0):<8.1f}\")
"
    fi
    
    echo ""
    echo -e "${GREEN}${BOLD}✓ Эксперимент успешно завершён${NC}"
    
} 2>&1 | tee "$MAIN_LOG"

exit ${PIPESTATUS[0]}