"""Утилиты ввода-вывода"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List


class NumpyEncoder(json.JSONEncoder):
    """JSON кодировщик с поддержкой numpy типов"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict, path: Path, indent: int = 2):
    """Сохранить словарь в JSON файл"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=NumpyEncoder)
    print(f"  ✓ Сохранено: {path}")


def load_json(path: Path) -> Dict:
    """Загрузить JSON файл"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)