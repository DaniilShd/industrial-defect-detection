#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs detectors reports

GREEN='\033[0;32m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log_section() { echo -e "\n${BLUE}${BOLD}======================================${NC}"; echo -e "${BLUE}${BOLD}  $1${NC}"; echo -e "${BLUE}${BOLD}======================================${NC}\n"; }

log_section "Online Detection Distillation (50 epochs total)"
python3 scripts/01_online_distill_train.py 2>&1 | tee "logs/online_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${GREEN}${BOLD}✓ Done! Results: detectors/${NC}"