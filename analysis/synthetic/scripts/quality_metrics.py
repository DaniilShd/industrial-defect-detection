#!/usr/bin/env python3
"""Метрики качества изображений: PSNR, SSIM, FID, попиксельные сравнения"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from config import AnalysisConfig

logger = logging.getLogger(__name__)


class QualityMetricsAnalyzer:
    """Анализатор качества синтетических изображений"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = {
            "psnr": {},
            "ssim": {},
            "fid": None,
            "per_image": []
        }
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Вычисление PSNR между двумя изображениями"""
        return psnr(img1, img2, data_range=255)
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Вычисление SSIM между двумя изображениями"""
        win_size = min(self.config.quality.ssim_window_size, 
                      min(img1.shape[0], img1.shape[1]) // 2 * 2 + 1)
        return ssim(img1, img2, data_range=255, channel_axis=2, win_size=win_size)
    
    def compute_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Сравнение цветовых гистограмм"""
        similarities = {}
        
        for channel, name in enumerate(['blue', 'green', 'red']):
            hist1 = cv2.calcHist([img1], [channel], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [channel], None, [256], [0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            
            similarities[name] = {
                "correlation": float(correlation),
                "bhattacharyya": float(bhattacharyya),
                "chi_square": float(chi_square)
            }
        
        return similarities
    
    def compute_edge_density(self, img: np.ndarray) -> Dict:
        """Вычисление плотности границ"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        
        total_pixels = gray.size
        
        return {
            "canny_density": float(np.sum(edges_canny > 0) / total_pixels),
            "sobel_mean": float(np.mean(edges_sobel)),
            "sobel_std": float(np.std(edges_sobel))
        }
    
    def compute_texture_features(self, img: np.ndarray) -> Dict:
        """Вычисление текстурных признаков через GLCM"""
        from skimage.feature import graycomatrix, graycoprops
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_uint8 = (gray / 256).astype(np.uint8)
        
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(gray_uint8, distances, angles, levels=256, symmetric=True, normed=True)
            
            contrast = float(np.mean(graycoprops(glcm, 'contrast')))
            homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
            energy = float(np.mean(graycoprops(glcm, 'energy')))
            correlation = float(np.mean(graycoprops(glcm, 'correlation')))
            
            return {
                "contrast": contrast,
                "homogeneity": homogeneity,
                "energy": energy,
                "correlation": correlation
            }
        except Exception:
            return {"contrast": 0, "homogeneity": 0, "energy": 0, "correlation": 0}
    
    def compute_frequency_spectrum(self, img: np.ndarray) -> Dict:
        """Анализ частотного спектра"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        low_freq_radius = min(h, w) // 8
        high_freq_radius = min(h, w) // 3
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        low_mask = dist <= low_freq_radius
        mid_mask = (dist > low_freq_radius) & (dist <= high_freq_radius)
        high_mask = dist > high_freq_radius
        
        total_energy = np.sum(magnitude)
        
        return {
            "low_freq_energy_ratio": float(np.sum(magnitude[low_mask]) / total_energy),
            "mid_freq_energy_ratio": float(np.sum(magnitude[mid_mask]) / total_energy),
            "high_freq_energy_ratio": float(np.sum(magnitude[high_mask]) / total_energy),
            "spectral_centroid": float(np.sum(dist * magnitude) / total_energy)
        }
    
    def analyze(self, original_dir: Path, synthetic_dir: Path, output_dir: Path) -> Dict:
        """Полный анализ качества изображений"""
        logger.info("=" * 80)
        logger.info("📊 QUALITY METRICS ANALYSIS")
        logger.info("=" * 80)
        
        cfg = self.config.quality
        
        # Подбор пар изображений
        pairs = self._find_image_pairs(original_dir, synthetic_dir, 
                                       min(cfg.num_samples_fid, 100))
        
        logger.info(f"Found {len(pairs)} image pairs for comparison")
        
        psnr_values = []
        ssim_values = []
        edge_density_orig = []
        edge_density_synth = []
        texture_orig = defaultdict(list)
        texture_synth = defaultdict(list)
        hist_similarities = defaultdict(list)
        freq_orig = defaultdict(list)
        freq_synth = defaultdict(list)
        
        # Попарное сравнение
        for orig_path, synth_path, name in tqdm(pairs, desc="Computing metrics"):
            orig_img = cv2.imread(str(orig_path))
            synth_img = cv2.imread(str(synth_path))
            
            if orig_img is None or synth_img is None:
                continue
            
            orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            synth_rgb = cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB)
            
            # Ресайз если нужно
            if orig_rgb.shape != synth_rgb.shape:
                synth_rgb = cv2.resize(synth_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
            
            # PSNR
            if cfg.compute_psnr:
                p_val = self.compute_psnr(orig_rgb, synth_rgb)
                psnr_values.append(p_val)
            
            # SSIM
            if cfg.compute_ssim:
                s_val = self.compute_ssim(orig_rgb, synth_rgb)
                ssim_values.append(s_val)
            
            # Гистограммы
            if self.config.additional_analyses.compute_color_histograms:
                hist_sim = self.compute_histogram_similarity(orig_rgb, synth_rgb)
                for ch, metrics in hist_sim.items():
                    for metric_name, value in metrics.items():
                        hist_similarities[f"{ch}_{metric_name}"].append(value)
            
            # Edge density
            if self.config.additional_analyses.compute_edge_density:
                edge_orig = self.compute_edge_density(orig_rgb)
                edge_synth = self.compute_edge_density(synth_rgb)
                for key in edge_orig:
                    edge_density_orig.append(edge_orig[key])
                    edge_density_synth.append(edge_synth[key])
            
            # Текстуры
            if self.config.additional_analyses.compute_texture_analysis:
                tex_orig = self.compute_texture_features(orig_rgb)
                tex_synth = self.compute_texture_features(synth_rgb)
                for key in tex_orig:
                    texture_orig[key].append(tex_orig[key])
                    texture_synth[key].append(tex_synth[key])
            
            # Частотный анализ
            if self.config.additional_analyses.compute_frequency_analysis:
                f_orig = self.compute_frequency_spectrum(orig_rgb)
                f_synth = self.compute_frequency_spectrum(synth_rgb)
                for key in f_orig:
                    freq_orig[key].append(f_orig[key])
                    freq_synth[key].append(f_synth[key])
            
            # Per-image stats
            self.results['per_image'].append({
                "name": name,
                "psnr": float(psnr_values[-1]) if psnr_values else None,
                "ssim": float(ssim_values[-1]) if ssim_values else None
            })
        
        # Агрегация результатов
        if psnr_values:
            self.results['psnr'] = {
                "mean": float(np.mean(psnr_values)),
                "std": float(np.std(psnr_values)),
                "min": float(np.min(psnr_values)),
                "max": float(np.max(psnr_values)),
                "median": float(np.median(psnr_values)),
                "values": [float(v) for v in psnr_values]
            }
        
        if ssim_values:
            self.results['ssim'] = {
                "mean": float(np.mean(ssim_values)),
                "std": float(np.std(ssim_values)),
                "min": float(np.min(ssim_values)),
                "max": float(np.max(ssim_values)),
                "median": float(np.median(ssim_values)),
                "values": [float(v) for v in ssim_values]
            }
        
        # Сохранение результатов
        self._save_results(output_dir)
        self._create_visualizations(output_dir, psnr_values, ssim_values)
        
        return self.results
    
    def _find_image_pairs(self, original_dir: Path, synthetic_dir: Path, 
                          max_pairs: int) -> List[Tuple[Path, Path, str]]:
        """Поиск пар оригинал-синтетика"""
        orig_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in (original_dir / "images").glob(ext):
                orig_images[img.stem] = img
        
        synth_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in (synthetic_dir / "images").glob(ext):
                synth_images[img.stem] = img
        
        pairs = []
        
        # Ищем прямые соответствия по имени
        for orig_stem, orig_path in orig_images.items():
            # Ищем синтетику, содержащую имя оригинала
            for synth_stem, synth_path in synth_images.items():
                if orig_stem in synth_stem:
                    pairs.append((orig_path, synth_path, f"{orig_stem} ↔ {synth_stem}"))
                    break
        
        # Если пар мало, добавляем случайные
        if len(pairs) < max_pairs and len(synth_images) > 0:
            import random
            random.seed(42)
            synth_paths = list(synth_images.values())
            orig_paths = list(orig_images.values())
            
            for _ in range(max_pairs - len(pairs)):
                if not orig_paths or not synth_paths:
                    break
                pairs.append((
                    random.choice(orig_paths),
                    random.choice(synth_paths),
                    "random_pair"
                ))
        
        return pairs[:max_pairs]
    
    def _save_results(self, output_dir: Path):
        """Сохранение результатов в JSON"""
        import json
        
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Конвертируем numpy
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        
        with open(metrics_dir / "quality_metrics.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=convert)
        
        logger.info(f"Results saved: {metrics_dir / 'quality_metrics.json'}")
    
    def _create_visualizations(self, output_dir: Path, 
                               psnr_values: List[float],
                               ssim_values: List[float]):
        """Создание визуализаций метрик качества"""
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PSNR histogram
        if psnr_values:
            axes[0, 0].hist(psnr_values, bins=30, color='#2196F3', alpha=0.8, edgecolor='black')
            axes[0, 0].axvline(np.mean(psnr_values), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(psnr_values):.2f} dB')
            axes[0, 0].set_xlabel('PSNR (dB)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('PSNR Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM histogram
        if ssim_values:
            axes[0, 1].hist(ssim_values, bins=30, color='#4CAF50', alpha=0.8, edgecolor='black')
            axes[0, 1].axvline(np.mean(ssim_values), color='r', linestyle='--',
                              label=f'Mean: {np.mean(ssim_values):.4f}')
            axes[0, 1].set_xlabel('SSIM')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('SSIM Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # PSNR vs SSIM scatter
        if psnr_values and ssim_values and len(psnr_values) == len(ssim_values):
            axes[1, 0].scatter(psnr_values, ssim_values, alpha=0.5, c='#673AB7', s=20)
            axes[1, 0].set_xlabel('PSNR (dB)')
            axes[1, 0].set_ylabel('SSIM')
            axes[1, 0].set_title('PSNR vs SSIM')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary table
        axes[1, 1].axis('off')
        summary_text = "QUALITY METRICS SUMMARY\n\n"
        
        if psnr_values:
            summary_text += f"PSNR:\n"
            summary_text += f"  Mean: {np.mean(psnr_values):.2f} dB\n"
            summary_text += f"  Std:  {np.std(psnr_values):.2f} dB\n"
            summary_text += f"  Min:  {np.min(psnr_values):.2f} dB\n"
            summary_text += f"  Max:  {np.max(psnr_values):.2f} dB\n\n"
        
        if ssim_values:
            summary_text += f"SSIM:\n"
            summary_text += f"  Mean: {np.mean(ssim_values):.4f}\n"
            summary_text += f"  Std:  {np.std(ssim_values):.4f}\n"
            summary_text += f"  Min:  {np.min(ssim_values):.4f}\n"
            summary_text += f"  Max:  {np.max(ssim_values):.4f}\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Image Quality Metrics: Original vs Synthetic', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        for fmt in self.config.visualization.save_formats:
            plt.savefig(viz_dir / f'quality_metrics.{fmt}', 
                       dpi=self.config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved: {viz_dir}")


def run_quality_analysis(config: AnalysisConfig) -> Dict:
    """Запуск анализа качества изображений"""
    analyzer = QualityMetricsAnalyzer(config)
    
    results = analyzer.analyze(
        original_dir=config.paths.original_dir,
        synthetic_dir=config.paths.synthetic_dir,
        output_dir=config.paths.output_dir
    )
    
    return results