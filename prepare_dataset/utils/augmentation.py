#!/usr/bin/env python3
"""Аугментации для металлического проката с сохранением YOLO-разметки"""

import albumentations as A


def get_metal_augmentation(params: dict) -> A.Compose:
    """
    Аугментации для металлического проката.
    Все преобразования совместимы с YOLO-разметкой (bbox_params).
    
    Виды аугментаций:
    - HorizontalFlip: прокат симметричен, дефекты могут быть с любой стороны
    - BrightnessContrast: разное освещение в цехе
    - HueSaturationValue: оттенки металла разных партий
    - Affine: лёгкие геометрические искажения (вибрация, скорость проката)
    - GaussNoise: шум матрицы камеры
    - MotionBlur: смаз от движения
    
    Args:
        params: словарь с параметрами аугментации
    """
    bc = params.get('brightness_contrast', {})
    hs = params.get('hue_saturation', {})
    af = params.get('affine', {})
    gn = params.get('gauss_noise', {})
    mb = params.get('motion_blur', {})
    
    return A.Compose([
        A.HorizontalFlip(p=params.get('horizontal_flip', 0.5)),
        
        A.RandomBrightnessContrast(
            brightness_limit=bc.get('brightness', 0.15),
            contrast_limit=bc.get('contrast', 0.15),
            p=bc.get('p', 0.6)
        ),
        
        A.HueSaturationValue(
            hue_shift_limit=hs.get('hue', 5),
            sat_shift_limit=hs.get('saturation', 10),
            val_shift_limit=hs.get('value', 10),
            p=hs.get('p', 0.3)
        ),
        
        A.Affine(
            scale=tuple(af.get('scale', [0.98, 1.02])),
            translate_percent=tuple(af.get('translate', [-0.02, 0.02])),
            rotate=tuple(af.get('rotate', [-1, 1])),
            fit_output=True,
            p=af.get('p', 0.2)
        ),
        
        A.GaussNoise(
            var_limit=(gn.get('var_min', 5.0), gn.get('var_max', 15.0)),
            p=gn.get('p', 0.2)
        ),
        
        A.MotionBlur(
            blur_limit=mb.get('blur_limit', 3),
            p=mb.get('p', 0.15)
        ),
        
    ], bbox_params=A.BboxParams(
        format='yolo',              # YOLO формат: [x_center, y_center, width, height] (нормализованные)
        label_fields=['class_labels'],
        min_visibility=0.7,         # Минимальная видимость bbox после аугментации
        min_area=25                 # Минимальная площадь bbox в пикселях
    ))