"""Анализ текстур: GLCM, LBP, Gabor."""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_glcm_features(gray: np.ndarray, distances: List[int], angles: List[int]) -> Dict[str, float]:
    """
    Вычисление признаков GLCM (матрица смежности).
    Возвращает: contrast, dissimilarity, homogeneity, energy, correlation, ASM.
    """
    glcm = graycomatrix(gray, distances=distances, angles=[np.deg2rad(a) for a in angles],
                        levels=256, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = {}
    
    for prop in props:
        values = graycoprops(glcm, prop).flatten()
        features[f'glcm_{prop}_mean'] = values.mean()
        features[f'glcm_{prop}_std'] = values.std()
    
    features['glcm_asm'] = graycoprops(glcm, 'ASM').mean()
    
    return features


def compute_lbp_features(gray: np.ndarray, radius: int = 2, n_points: int = 16) -> Dict[str, float]:
    """
    Вычисление признаков LBP (Local Binary Patterns).
    Возвращает: среднее, std, энтропию гистограммы LBP.
    """
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
    
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return {
        'lbp_mean': lbp.mean(),
        'lbp_std': lbp.std(),
        'lbp_entropy': entropy,
        'lbp_uniform_ratio': np.sum(hist[hist > 0.01]) / len(hist)
    }


def compute_gabor_features(gray: np.ndarray, frequencies: List[float], angles: List[int]) -> Dict[str, float]:
    """
    Вычисление признаков Gabor фильтров.
    Возвращает: среднюю энергию отклика по частотам и углам.
    """
    features = {}
    
    for theta in angles:
        for freq in frequencies:
            kernel = cv2.getGaborKernel((21, 21), 4.0, np.deg2rad(theta), freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            features[f'gabor_t{theta}_f{freq}_mean'] = filtered.mean()
            features[f'gabor_t{theta}_f{freq}_std'] = filtered.std()
    
    return features