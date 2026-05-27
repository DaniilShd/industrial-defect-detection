#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs pretrain detectors reports

GREEN='\033[0;32m'; BLUE='\033[0;34m'; BOLD='\033[1m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_section() { echo -e "\n${BLUE}${BOLD}======================================${NC}"; echo -e "${BLUE}${BOLD}  $1${NC}"; echo -e "${BLUE}${BOLD}======================================${NC}\n"; }

# Проверяем наличие скриптов
for f in 02_train_detectors.py 03_evaluate_models.py 04_visualize_results.py 05_generate_report.py; do
    [ -f "scripts/$f" ] || cp "../multilayer_distillation/scripts/$f" "scripts/"
    sed -i 's|config_multilayer_distillation.yaml|../config_modern_distillation.yaml|g' "scripts/$f"
    sed -i 's|multilayer_model_path.txt|modern_model_path.txt|g' "scripts/$f"
    sed -i 's|multilayer_distilled|modern_distilled|g' "scripts/$f"
    sed -i 's|/app/multilayer_distillation|/app/modern_distillation|g' "scripts/$f"
done

# Шаг 1: Дистилляция (только если нет весов)
BACKBONE="pretrain/modern_distilled/backbone_weights.pt"
if [ -f "$BACKBONE" ]; then
    SIZE=$(du -sh "$BACKBONE" | cut -f1)
    echo -e "${YELLOW}[!] Веса уже есть: $BACKBONE ($SIZE)${NC}"
    echo -e "${YELLOW}[!] Пропускаем дистилляцию. Для переобучения удалите $BACKBONE${NC}"
else
    log_section "ШАГ 1/3: Modern Distillation 2025"
    python3 scripts/01_modern_distill.py 2>&1 | tee "logs/distill_$(date +%Y%m%d_%H%M%S).log"
fi

# Шаг 2: Обучение детекторов
log_section "ШАГ 2/3: Training Detectors"
python3 scripts/02_train_detectors.py 2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"

# Шаг 3: Оценка
log_section "ШАГ 3/3: Evaluation & Reports"
python3 scripts/03_evaluate_models.py 2>&1 | tee "logs/eval_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/04_visualize_results.py 2>&1 | tee "logs/viz_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/05_generate_report.py 2>&1 | tee "logs/report_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${GREEN}${BOLD}✓ Modern Distillation Pipeline Completed${NC}"
