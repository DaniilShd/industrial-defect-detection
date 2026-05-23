#!/bin/bash
# ============================================================================
# Пайплайн Multi-Layer Feature Distillation
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RED='\033[0;31m'
NC='\033[0m'

log_section() {
    echo -e "\n${BLUE}${BOLD}======================================${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}======================================${NC}\n"
}

# Шаг 1: Multi-Layer Distillation
log_section "ШАГ 1/3: Multi-Layer Feature Distillation"
python3 scripts/01_multilayer_distill.py 2>&1 | tee "$LOG_DIR/distill_${TIMESTAMP}.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Ошибка на шаге дистилляции${NC}"
    exit 1
fi

# Шаг 2: Обучение детекторов
log_section "ШАГ 2/3: Обучение Faster R-CNN детекторов"
python3 scripts/02_train_detectors.py 2>&1 | tee "$LOG_DIR/train_${TIMESTAMP}.log"

# Шаг 3: Оценка моделей
log_section "ШАГ 3/3: Оценка и визуализация"
python3 scripts/03_evaluate_models.py 2>&1 | tee "$LOG_DIR/eval_${TIMESTAMP}.log"
python3 scripts/04_visualize_results.py 2>&1 | tee "$LOG_DIR/viz_${TIMESTAMP}.log"
python3 scripts/05_generate_report.py 2>&1 | tee "$LOG_DIR/report_${TIMESTAMP}.log"

echo -e "\n${GREEN}${BOLD}✓ Multi-Layer Distillation Pipeline Completed${NC}"
echo -e "Results: ${SCRIPT_DIR}/reports/"