"""Сохранение графиков и отчётов."""
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_figure(fig, filename: str, report_dir: Path, dpi: int = 150) -> Path:
    """Сохранить matplotlib figure."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    filepath = report_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Сохранено: {filepath}")
    return filepath


def create_summary_table(data: dict, title: str) -> str:
    """Форматированная таблица для текстового отчёта."""
    lines = [f"\n{title}", "-" * 60]
    for key, value in data.items():
        if isinstance(value, float):
            lines.append(f"  {key:<30} {value:>12.4f}")
        elif isinstance(value, tuple):
            lines.append(f"  {key:<30} {str(value):>12}")
        else:
            lines.append(f"  {key:<30} {value:>12}")
    lines.append("-" * 60)
    return '\n'.join(lines)