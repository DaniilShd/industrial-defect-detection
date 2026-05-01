"""Ввод-вывод, конфиг, логирование."""
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]:
    """Загрузка YAML конфига."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Создать директорию."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_section(title: str, width: int = 80):
    """Заголовок секции."""
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}")