import numpy as np
import pandas as pd
from typing import Optional, Dict


def rle_to_mask(rle_string: Optional[str], height: int = 256, width: int = 1600) -> np.ndarray:
    """RLE → бинарная маска (column-major, Severstal)."""
    if rle_string is None or pd.isna(rle_string) or str(rle_string).strip() in ('', 'nan'):
        return np.zeros((height, width), dtype=np.uint8)
    
    try:
        numbers = list(map(int, str(rle_string).split()))
    except ValueError:
        return np.zeros((height, width), dtype=np.uint8)
    
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        end = min(start + length, len(flat_mask))
        flat_mask[start:end] = 1
    
    return flat_mask.reshape(width, height).T


def mask_to_rle(mask: np.ndarray) -> str:
    """Бинарная маска → RLE строка."""
    if mask.sum() == 0:
        return ''
    
    flat_mask = mask.flatten()
    rle_parts = []
    prev_val = 0
    run_start = 0
    
    for i, val in enumerate(flat_mask):
        if val != prev_val:
            if prev_val == 1:
                rle_parts.append(str(run_start + 1))
                rle_parts.append(str(i - run_start))
            run_start = i
            prev_val = val
    
    if prev_val == 1:
        rle_parts.append(str(run_start + 1))
        rle_parts.append(str(len(flat_mask) - run_start))
    
    return ' '.join(rle_parts)


def create_masks_by_class(df_group: pd.DataFrame, height: int = 256, width: int = 1600) -> Dict[int, np.ndarray]:
    """Маски по классам без перезаписи."""
    masks = {}
    
    for _, row in df_group.iterrows():
        class_id = int(row['ClassId'])
        rle = row['EncodedPixels']
        
        if pd.isna(rle) or str(rle).strip() == '':
            continue
        
        mask = rle_to_mask(rle, height, width)
        masks[class_id] = np.logical_or(masks.get(class_id, np.zeros_like(mask)), mask).astype(np.uint8)
    
    return masks