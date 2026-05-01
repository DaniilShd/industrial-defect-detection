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
    logger.info(f"График сохранён: {filepath}")
    return filepath