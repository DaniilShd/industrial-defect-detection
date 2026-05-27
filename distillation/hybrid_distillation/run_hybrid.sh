#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Создаём папки
mkdir -p logs pretrain detectors reports

GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RED='\033[0;31m'
NC='\033[0m'

log_section() { echo -e "\n${BLUE}${BOLD}======================================${NC}"; echo -e "${BLUE}${BOLD}  $1${NC}"; echo -e "${BLUE}${BOLD}======================================${NC}\n"; }

# Копируем скрипты из multilayer если их нет
for f in 02_train_detectors.py 03_evaluate_models.py 04_visualize_results.py 05_generate_report.py; do
    if [ ! -f "scripts/$f" ]; then
        echo "Copying $f from multilayer_distillation..."
        cp "../multilayer_distillation/scripts/$f" "scripts/"
    fi
done

# Применяем все правки
echo "Applying fixes to scripts..."
cd scripts/

for f in 02_train_detectors.py 03_evaluate_models.py 04_visualize_results.py 05_generate_report.py; do
    # Путь к конфигу
    sed -i 's|config_pretrain_comparison.yaml|../config_hybrid_distillation.yaml|g' "$f"
    sed -i 's|config_multilayer_distillation.yaml|../config_hybrid_distillation.yaml|g' "$f"
    echo "  Fixed config path in $f"
done

# Специфичные правки для 02
sed -i 's|multilayer_model_path.txt|hybrid_model_path.txt|g' 02_train_detectors.py
sed -i 's|pretrained_model_path.txt|hybrid_model_path.txt|g' 02_train_detectors.py

# Специфичные правки для 04
sed -i "s/'multilayer_distilled': '#F39C12'/'multilayer_distilled': '#F39C12',\n            'hybrid_distilled': '#9B59B6'/g" 04_visualize_results.py
sed -i "s/'faster_rcnn_r18_multilayer_distilled': 'Multi-Layer Дистилляция'/'faster_rcnn_r18_multilayer_distilled': 'Multi-Layer Дистилляция',\n            'faster_rcnn_r18_hybrid_distilled': 'Гибридная дистилляция'/g" 04_visualize_results.py

# Специфичные правки для 05
sed -i "s/'lightly_pretrained': 'Дистилляция (предложенный)'/'lightly_pretrained': 'Дистилляция (предложенный)',\n            'multilayer_distilled': 'Multi-Layer Дистилляция',\n            'hybrid_distilled': 'Гибридная дистилляция'/g" 05_generate_report.py
sed -i "s/'lightly_pretrained': 'Дистилл.'/'lightly_pretrained': 'Дистилл.',\n            'multilayer_distilled': 'Multi-Layer',\n            'hybrid_distilled': 'Гибрид'/g" 05_generate_report.py

cd "$SCRIPT_DIR"
echo "All fixes applied!"

# Шаг 1: Дистилляция
log_section "ШАГ 1/3: Hybrid Distillation"
python3 scripts/01_hybrid_distill.py 2>&1 | tee "logs/distill_$(date +%Y%m%d_%H%M%S).log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo -e "${RED}Error in distillation${NC}"; exit 1; fi

# Шаг 2: Обучение детекторов
log_section "ШАГ 2/3: Training Detectors"
python3 scripts/02_train_detectors.py 2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"

# Шаг 3: Оценка
log_section "ШАГ 3/3: Evaluation & Reports"
python3 scripts/03_evaluate_models.py 2>&1 | tee "logs/eval_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/04_visualize_results.py 2>&1 | tee "logs/viz_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/05_generate_report.py 2>&1 | tee "logs/report_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${GREEN}${BOLD}✓ Hybrid pipeline completed!${NC}"
echo "Results: $SCRIPT_DIR/reports/"