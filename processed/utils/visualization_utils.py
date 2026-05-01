"""Визуализация YOLO bbox на изображениях. Цвета из config.yaml."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def hex_to_bgr(hex_color: str) -> tuple:
    """Конвертирует HEX (#FF6B6B) в BGR для OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def get_class_colors(classes_config: Dict) -> Dict[int, tuple]:
    """
    Создаёт словарь {yolo_class_id: (B, G, R)} из конфига.
    YOLO class_id = ClassId - 1 (0-based).
    """
    colors = {}
    for class_id, hex_color in classes_config['colors'].items():
        yolo_id = int(class_id) - 1
        colors[yolo_id] = hex_to_bgr(hex_color)
    return colors


def draw_yolo_bbox(
    image_path: Path,
    label_path: Path,
    output_path: Optional[Path] = None,
    class_names: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[int, tuple]] = None,
    line_thickness: int = 2
) -> Optional[np.ndarray]:
    """
    Отрисовка YOLO bbox на изображении.
    
    Args:
        image_path: путь к изображению
        label_path: путь к .txt с YOLO разметкой
        output_path: путь для сохранения
        class_names: словарь {yolo_class_id: имя}
        class_colors: словарь {yolo_class_id: (B, G, R)}
        line_thickness: толщина линий
    
    Returns:
        Изображение с bbox или None
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Не удалось загрузить: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    if not label_path.exists():
        logger.warning(f"Нет разметки: {label_path}")
        return img
    
    # Цвета по умолчанию
    default_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            yolo_class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h
            
            # Угловые координаты
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Цвет из конфига или дефолтный
            if class_colors and yolo_class_id in class_colors:
                color = class_colors[yolo_class_id]
            else:
                color = default_colors[yolo_class_id % len(default_colors)]
            
            # Отрисовка
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            # Метка класса
            if class_names:
                label = class_names.get(yolo_class_id, f"Class {yolo_class_id}")
            else:
                label = f"Class {yolo_class_id}"
            
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
    
    return img


def visualize_batch(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    num_samples: int = 100,
    class_names: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[int, tuple]] = None,
    show: bool = False
) -> List[Path]:
    """
    Пакетная визуализация YOLO bbox.
    
    Args:
        images_dir: директория с изображениями
        labels_dir: директория с разметкой
        output_dir: директория для сохранения
        num_samples: сколько примеров (0 = все)
        class_names: словарь {yolo_class_id: имя}
        class_colors: словарь {yolo_class_id: (B, G, R)}
        show: показывать ли изображения
    
    Returns:
        Список сохранённых файлов
    """
    import random
    from tqdm import tqdm
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    if 0 < num_samples < len(image_files):
        image_files = random.sample(image_files, num_samples)
    
    logger.info(f"Обработка {len(image_files)} изображений...")
    
    saved = []
    for img_path in tqdm(image_files, desc="Отрисовка"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        out_path = output_dir / f"vis_{img_path.name}"
        
        img = draw_yolo_bbox(img_path, label_path, out_path, class_names, class_colors)
        
        if show and img is not None:
            cv2.imshow("BBox", img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
        if img is not None:
            saved.append(out_path)
    
    if show:
        cv2.destroyAllWindows()
    
    logger.info(f"Готово: {len(saved)} файлов в {output_dir}")
    return saved