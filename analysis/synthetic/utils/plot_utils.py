"""Утилиты для построения графиков"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple

# Настройка русского шрифта
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def setup_style(style: str = "whitegrid"):
    """Настройка стиля matplotlib"""
    sns.set_style(style)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150, 
                formats: List[str] = None, close: bool = True):
    """Сохранить график в нескольких форматах"""
    if formats is None:
        formats = ['png']
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    stem = path.stem
    for fmt in formats:
        save_path = path.parent / f"{stem}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Сохранено: {save_path}")
    
    if close:
        plt.close(fig)


def add_value_labels(ax, bars, fmt: str = '{:.0f}', offset: float = 0.02):
    """Добавить подписи значений над столбцами"""
    max_height = max(bar.get_height() for bar in bars)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + max_height * offset,
                fmt.format(height),
                ha='center', va='bottom', fontweight='bold', fontsize=9)