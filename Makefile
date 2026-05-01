# Makefile для управления Docker-пайплайном

.PHONY: build up down shell generate insert validate viz all clean

# Сборка образа
build:
	docker compose build

# Запуск контейнера
up:
	docker compose up -d

# Остановка контейнера
down:
	docker compose down

# Интерактивная оболочка
shell-processed:
	docker exec -it processed /bin/bash

shell-analysis:
	docker exec -it analysis /bin/bash

shell-generate:
	docker exec -it generate /bin/bash

shell-prepare_dataset:
	docker exec -it prepare_dataset /bin/bash


# # Дефектные патчи (по умолчанию)
# python generation/scripts/05_visualize_bboxes.py --samples 20

# # Сбалансированные дефектные
# python generation/scripts/05_visualize_bboxes.py \
#     --images data/processed/balanced_defect_patches/train/images \
#     --labels data/processed/balanced_defect_patches/train/labels \
#     --output generation/reports/patches/balanced_vis \
#     --samples 50