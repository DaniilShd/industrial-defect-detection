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
shell_generate:
	docker exec -it synthetic_generator /bin/bash
