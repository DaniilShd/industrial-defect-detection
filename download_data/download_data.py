import subprocess
import zipfile
import os
import yaml

# Читаем конфиг из YAML файла
with open("./download_data/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    download_path = config["download_path"]

# 1. Создаём папку из конфига
os.makedirs(download_path, exist_ok=True)

# 2. Скачиваем датасет с визуализацией
print("Скачивание датасета...")
process = subprocess.Popen(["kaggle", "competitions", "download", "-c", "severstal-steel-defect-detection"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# Показываем вывод в реальном времени
for line in process.stdout:
    print(line.strip())

process.wait()

# 3. Распаковываем в папку из конфига
print("Распаковка архива...")
with zipfile.ZipFile("severstal-steel-defect-detection.zip", 'r') as zip_ref:
    zip_ref.extractall(download_path)

# 4. Удаляем архив
os.remove("severstal-steel-defect-detection.zip")
print(f"Готово! Данные сохранены в: {download_path}")