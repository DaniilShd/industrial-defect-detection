# analysis/synthetic/utils/io_utils.py
"""IO utility functions"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
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
    """Save dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
    print(f"  ✓ Saved: {path}")


def load_json(path: Path) -> Dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)