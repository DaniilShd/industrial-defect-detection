# analysis/synthetic/scripts/domain_gap.py
"""Domain gap analysis using DINOv2 features"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from config import AnalysisConfig


class ImageFeatureDataset(Dataset):
    """Датасет для извлечения признаков"""
    
    def __init__(self, image_paths: List[Path], processor, resize_to: int = 256):
        self.image_paths = image_paths
        self.processor = processor
        self.resize_to = resize_to
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.resize_to and img.size != (self.resize_to, self.resize_to):
                img = img.resize((self.resize_to, self.resize_to), Image.Resampling.LANCZOS)
            
            inputs = self.processor(images=img, return_tensors="pt")
            return inputs.pixel_values.squeeze(0), path.name
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            # Return blank image on error
            blank = Image.new('RGB', (self.resize_to, self.resize_to), (0, 0, 0))
            inputs = self.processor(images=blank, return_tensors="pt")
            return inputs.pixel_values.squeeze(0), path.name


class DomainGapAnalyzer:
    """Анализатор domain gap с использованием DINOv2"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = config.dinov2.device
        self.processor = None
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": config.dinov2.model_name,
                "device": config.dinov2.device,
                "num_samples": config.dinov2.num_samples
            }
        }
        
    def load_model(self):
        """Загрузка модели DINOv2"""
        print(f"\n🔄 Загрузка {self.config.dinov2.model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(self.config.dinov2.model_name)
        self.model = AutoModel.from_pretrained(self.config.dinov2.model_name).to(self.device)
        self.model.eval()
        print("✅ Модель загружена")
        
    def extract_features(self, image_dir: Path, dataset_name: str) -> Tuple[np.ndarray, List[str]]:
        """Извлечение признаков из директории с изображениями"""
        images_dir = image_dir / "images"
        
        # Поиск изображений
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_paths.extend(list(images_dir.glob(ext)))
        
        image_paths = sorted(image_paths)
        
        # Ограничение количества семплов
        if len(image_paths) > self.config.dinov2.num_samples:
            np.random.seed(self.config.dinov2.random_seed)
            indices = np.random.choice(len(image_paths), self.config.dinov2.num_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
        
        print(f"\n📊 Извлечение признаков из {dataset_name} ({len(image_paths)} изображений)...")
        
        dataset = ImageFeatureDataset(image_paths, self.processor, self.config.dinov2.image_size)
        loader = DataLoader(dataset, batch_size=self.config.dinov2.batch_size, 
                          shuffle=False, num_workers=4)
        
        features_list = []
        filenames = []
        
        with torch.no_grad():
            for batch, names in tqdm(loader, desc=f"  {dataset_name}"):
                batch = batch.to(self.device)
                output = self.model(batch)
                features = output.pooler_output.cpu().numpy()
                
                features_list.append(features)
                filenames.extend(names)
        
        features = np.concatenate(features_list, axis=0)
        print(f"  ✓ Извлечено признаков: {features.shape}")
        
        return features, filenames
    
    def compute_statistics(self, features: np.ndarray) -> Dict:
        """Вычисление статистик распределения признаков"""
        l2_norms = np.linalg.norm(features, axis=1)
        
        return {
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "median": float(np.median(features)),
            "skewness": float(np.mean((features - np.mean(features))**3) / (np.std(features)**3 + 1e-8)),
            "kurtosis": float(np.mean((features - np.mean(features))**4) / (np.std(features)**4 + 1e-8)),
            "variance": float(np.var(features)),
            "l2_norm_mean": float(np.mean(l2_norms)),
            "l2_norm_std": float(np.std(l2_norms)),
            "feature_dim": features.shape[1],
            "num_samples": features.shape[0]
        }
    
    def compute_per_channel_emd(self, features1: np.ndarray, features2: np.ndarray) -> Dict:
        """Вычисление Earth Mover's Distance поканально"""
        print("\n📊 Вычисление EMD...")
        
        emd_values = []
        for i in tqdm(range(features1.shape[1]), desc="  EMD per channel"):
            emd = wasserstein_distance(features1[:, i], features2[:, i])
            emd_values.append(emd)
        
        emd_values = np.array(emd_values)
        
        return {
            "mean_emd": float(np.mean(emd_values)),
            "std_emd": float(np.std(emd_values)),
            "max_emd": float(np.max(emd_values)),
            "min_emd": float(np.min(emd_values)),
            "median_emd": float(np.median(emd_values)),
            "top10_unstable_channels": np.argsort(emd_values)[-10:].tolist(),
            "bottom10_stable_channels": np.argsort(emd_values)[:10].tolist(),
            "percentile_95_emd": float(np.percentile(emd_values, 95)),
            "percentile_75_emd": float(np.percentile(emd_values, 75))
        }
    
    def compute_similarity_metrics(self, features1: np.ndarray, features2: np.ndarray) -> Dict:
        """
        ИСПРАВЛЕННЫЕ метрики схожести доменов
        """
        print("\n📊 Вычисление метрик схожести...")
        
        n_test = min(self.config.dinov2.nn_test_samples, len(features1), len(features2))
        np.random.seed(self.config.dinov2.random_seed)
        
        # Центроиды
        centroid1 = np.mean(features1, axis=0)
        centroid2 = np.mean(features2, axis=0)
        centroid_distance = np.linalg.norm(centroid1 - centroid2)
        
        # Cosine similarity между центроидами
        cosine_sim = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-8)
        
        # 1-NN тест
        indices1 = np.random.choice(len(features1), n_test, replace=False)
        indices2 = np.random.choice(len(features2), n_test, replace=False)
        
        feats1 = features1[indices1]
        feats2 = features2[indices2]
        
        combined = np.concatenate([feats1, feats2])
        labels = np.array([0] * n_test + [1] * n_test)
        
        correct_same_domain = 0
        total = 0
        
        for i in range(2 * n_test):
            query = combined[i:i+1]
            distances = cdist(query, combined, metric='cosine')[0]
            distances[i] = np.inf
            nn_idx = np.argmin(distances)
            
            if labels[i] == labels[nn_idx]:
                correct_same_domain += 1
            total += 1
        
        nn_accuracy = correct_same_domain / total if total > 0 else 0.5
        
        # Intra/inter domain distances
        distances_a_to_b = cdist(feats1, feats2, metric='cosine')
        distances_a_to_a = cdist(feats1, feats1, metric='cosine')
        distances_b_to_b = cdist(feats2, feats2, metric='cosine')
        
        np.fill_diagonal(distances_a_to_a, np.inf)
        np.fill_diagonal(distances_b_to_b, np.inf)
        
        mean_intra_a = np.mean(np.min(distances_a_to_a, axis=1))
        mean_intra_b = np.mean(np.min(distances_b_to_b, axis=1))
        mean_inter_a_to_b = np.mean(np.min(distances_a_to_b, axis=1))
        mean_inter_b_to_a = np.mean(np.min(distances_a_to_b, axis=0))
        
        intra_avg = (mean_intra_a + mean_intra_b) / 2
        inter_avg = (mean_inter_a_to_b + mean_inter_b_to_a) / 2
        gap_ratio = inter_avg / (intra_avg + 1e-8)
        
        # Domain Overlap Score
        overlap_score = 2.0 * (1.0 - nn_accuracy)
        overlap_score = np.clip(overlap_score, 0.0, 1.0)
        
        # Silhouette-like score
        # Насколько меж-доменное расстояние больше внутри-доменного
        silhouette_score = np.mean(
            (np.min(distances_a_to_b, axis=1) - np.min(distances_a_to_a, axis=1)) / 
            (np.maximum(np.min(distances_a_to_b, axis=1), np.min(distances_a_to_a, axis=1)) + 1e-8)
        )
        
        return {
            "centroid_distance": float(centroid_distance),
            "centroid_cosine_similarity": float(cosine_sim),
            "mean_intra_domain_distance_A": float(mean_intra_a),
            "mean_intra_domain_distance_B": float(mean_intra_b),
            "mean_inter_domain_distance": float(inter_avg),
            "domain_gap_ratio": float(gap_ratio),
            "1nn_domain_accuracy": float(nn_accuracy),
            "1nn_expected_for_identical": 0.5,
            "domain_overlap_score": float(overlap_score),
            "domain_silhouette_score": float(silhouette_score),
            "intra_domain_variance_A": float(np.var(mean_intra_a)),
            "intra_domain_variance_B": float(np.var(mean_intra_b))
        }
    
    def visualize_embeddings(self, features_list: List[Tuple[np.ndarray, str]], 
                            output_dir: Path) -> Dict:
        """
        Визуализация эмбеддингов: PCA и t-SNE
        """
        print("\n📊 Создание визуализаций эмбеддингов...")
        
        viz_config = self.config.visualization
        results = {}
        
        # Подготовка данных
        all_features = []
        all_names = []
        
        for features, name in features_list:
            n = min(len(features), viz_config.pca.get('max_points', 500))
            np.random.seed(self.config.dinov2.random_seed)
            indices = np.random.choice(len(features), n, replace=False)
            
            all_features.append(features[indices])
            all_names.extend([name] * n)
        
        all_features = np.concatenate(all_features, axis=0)
        
        # Стандартизация
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # --- PCA Visualization ---
        pca = PCA(n_components=2)
        features_2d_pca = pca.fit_transform(features_scaled)
        explained_var = pca.explained_variance_ratio_ * 100
        
        colors = {'original': '#2196F3', 'synthetic': '#FF9800'}
        
        fig, axes = plt.subplots(1, 2, figsize=viz_config.figsize)
        
        # PCA scatter с эллипсами
        ax1 = axes[0]
        for name in set(all_names):
            mask = np.array([n == name for n in all_names])
            data = features_2d_pca[mask]
            color = colors.get(name, '#999999')
            
            ax1.scatter(data[:, 0], data[:, 1], c=color, label=name, 
                       alpha=0.5, s=10, edgecolors='none')
            
            # Эллипс 95% доверительного интервала
            if len(data) > 2 and viz_config.pca.get('show_ellipses', True):
                try:
                    from matplotlib.patches import Ellipse
                    import matplotlib.transforms as transforms
                    
                    cov = np.cov(data.T)
                    mean = np.mean(data, axis=0)
                    
                    # Собственные значения и векторы
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    
                    # 95% доверительный интервал (2.447 std для 2D)
                    width, height = 2 * np.sqrt(eigenvalues) * 2.447
                    
                    ellipse = Ellipse(xy=mean, width=width, height=height,
                                    angle=angle, facecolor='none',
                                    edgecolor=color, linewidth=2, 
                                    linestyle='--', alpha=0.8)
                    ax1.add_patch(ellipse)
                except Exception as e:
                    print(f"  ⚠️ Не удалось построить эллипс для {name}: {e}")
        
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}% variance)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}% variance)', fontsize=12)
        ax1.set_title('DINOv2 Feature Distribution (PCA)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Расстояния между доменами
        ax2 = axes[1]
        names_list = list(set(all_names))
        if len(names_list) >= 2:
            data_a = features_2d_pca[np.array([n == names_list[0] for n in all_names])][:100]
            data_b = features_2d_pca[np.array([n == names_list[1] for n in all_names])][:100]
            
            min_len = min(len(data_a), len(data_b))
            data_a = data_a[:min_len]
            data_b = data_b[:min_len]
            
            diff = data_a - data_b
            distances = np.linalg.norm(diff, axis=1)
            
            ax2.hist(distances, bins=30, alpha=0.7, color='#673AB7', 
                    edgecolor='black', linewidth=0.5)
            ax2.axvline(x=np.mean(distances), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(distances):.3f}')
            ax2.axvline(x=np.median(distances), color='g', linestyle=':', 
                       linewidth=2, label=f'Median: {np.median(distances):.3f}')
            
            ax2.set_xlabel('Pairwise distance in PCA space', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'Distances: {names_list[0]} vs {names_list[1]}', 
                         fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        for fmt in viz_config.save_formats:
            save_path = output_dir / f'pca_visualization.{fmt}'
            plt.savefig(save_path, dpi=viz_config.dpi, bbox_inches='tight')
        
        plt.close()
        
        results['pca'] = {
            "explained_variance_ratio": explained_var.tolist(),
            "total_explained_variance": float(np.sum(explained_var)),
            "saved_path": str(output_dir / 'pca_visualization.png')
        }
        
        # --- t-SNE Visualization ---
        if viz_config.tsne.get('enabled', False):
            print("  Создание t-SNE визуализации...")
            
            n_tsne = min(len(features_scaled), viz_config.tsne.get('max_points', 300))
            indices = np.random.choice(len(features_scaled), n_tsne, replace=False)
            
            tsne = TSNE(
                n_components=2, 
                perplexity=viz_config.tsne.get('perplexity', 30),
                n_iter=viz_config.tsne.get('n_iter', 1000),
                random_state=self.config.dinov2.random_seed
            )
            
            features_2d_tsne = tsne.fit_transform(features_scaled[indices])
            names_tsne = [all_names[i] for i in indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for name in set(names_tsne):
                mask = np.array([n == name for n in names_tsne])
                color = colors.get(name, '#999999')
                ax.scatter(features_2d_tsne[mask, 0], features_2d_tsne[mask, 1],
                          c=color, label=name, alpha=0.6, s=15, edgecolors='none')
            
            ax.set_xlabel('t-SNE Component 1', fontsize=12)
            ax.set_ylabel('t-SNE Component 2', fontsize=12)
            ax.set_title('DINOv2 Feature Distribution (t-SNE)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            for fmt in viz_config.save_formats:
                plt.savefig(output_dir / f'tsne_visualization.{fmt}', 
                          dpi=viz_config.dpi, bbox_inches='tight')
            
            plt.close()
            
            results['tsne'] = {
                "perplexity": viz_config.tsne.get('perplexity', 30),
                "saved_path": str(output_dir / 'tsne_visualization.png')
            }
        
        print("✅ Визуализации сохранены")
        return results
    
    def generate_report(self, output_dir: Path) -> str:
        """Генерация человекочитаемого отчёта"""
        thresholds = self.config.thresholds
        
        original_stats = self.results.get('original_statistics', {})
        synthetic_stats = self.results.get('synthetic_statistics', {})
        similarity = self.results.get('similarity_metrics', {})
        emd_data = self.results.get('emd_metrics', {})
        
        report = []
        report.append("=" * 80)
        report.append("🔬 DOMAIN GAP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['timestamp']}")
        report.append(f"Model: {self.config.dinov2.model_name}")
        report.append(f"Device: {self.config.dinov2.device}")
        report.append("")
        
        # Секция 1: Статистики признаков
        report.append("-" * 80)
        report.append("1. FEATURE DISTRIBUTION STATISTICS")
        report.append("-" * 80)
        report.append(f"{'Metric':<30} {'Original':<15} {'Synthetic':<15} {'Diff %':<15}")
        report.append("-" * 80)
        
        for key in ['mean', 'std', 'variance', 'l2_norm_mean']:
            orig_val = original_stats.get(key, 0)
            syn_val = synthetic_stats.get(key, 0)
            diff_pct = abs(syn_val - orig_val) / (abs(orig_val) + 1e-8) * 100
            report.append(f"{key:<30} {orig_val:<15.6f} {syn_val:<15.6f} {diff_pct:<15.2f}%")
        
        report.append("")
        
        # Секция 2: Метрики схожести
        report.append("-" * 80)
        report.append("2. DOMAIN SIMILARITY METRICS")
        report.append("-" * 80)
        
        overlap = similarity.get('domain_overlap_score', 0)
        nn_acc = similarity.get('1nn_domain_accuracy', 1.0)
        gap_ratio = similarity.get('domain_gap_ratio', 999)
        silhouette = similarity.get('domain_silhouette_score', 0)
        
        report.append(f"  {'Centroid Cosine Similarity:':<40} {similarity.get('centroid_cosine_similarity', 0):.4f}")
        report.append(f"  {'1-NN Domain Accuracy:':<40} {nn_acc:.4f} (0.50 = indistinguishable)")
        report.append(f"  {'Domain Overlap Score:':<40} {overlap:.4f} (1.00 = complete overlap)")
        report.append(f"  {'Domain Gap Ratio:':<40} {gap_ratio:.4f} (1.00 = identical)")
        report.append(f"  {'Domain Silhouette Score:':<40} {silhouette:.4f} (<0 = overlapped)")
        report.append(f"  {'Intra-domain Distance (A):':<40} {similarity.get('mean_intra_domain_distance_A', 0):.6f}")
        report.append(f"  {'Intra-domain Distance (B):':<40} {similarity.get('mean_intra_domain_distance_B', 0):.6f}")
        report.append(f"  {'Inter-domain Distance:':<40} {similarity.get('mean_inter_domain_distance', 0):.6f}")
        
        report.append("")
        
        # Секция 3: EMD
        report.append("-" * 80)
        report.append("3. EARTH MOVER'S DISTANCE (PER CHANNEL)")
        report.append("-" * 80)
        report.append(f"  {'Mean EMD:':<30} {emd_data.get('mean_emd', 0):.6f}")
        report.append(f"  {'Median EMD:':<30} {emd_data.get('median_emd', 0):.6f}")
        report.append(f"  {'95th Percentile EMD:':<30} {emd_data.get('percentile_95_emd', 0):.6f}")
        report.append(f"  {'Max EMD:':<30} {emd_data.get('max_emd', 0):.6f}")
        report.append(f"  {'Top 10 Unstable Channels:':<30} {emd_data.get('top10_unstable_channels', [])}")
        
        report.append("")
        
        # Секция 4: Интерпретация
        report.append("-" * 80)
        report.append("4. QUALITY INTERPRETATION")
        report.append("-" * 80)
        
        # Используем пороги из конфигурации
        t_overlap = thresholds.domain_overlap
        
        if overlap > t_overlap['excellent'] and nn_acc < 0.55:
            quality = "✅ EXCELLENT"
            details = [
                "Synthetic data is nearly indistinguishable from original",
                "Feature distributions heavily overlap",
                "1-NN classifier cannot separate domains (< 55% accuracy)",
                "Recommendation: Use synthetic data without restrictions"
            ]
        elif overlap > t_overlap['good'] and nn_acc < 0.65:
            quality = "🟡 GOOD"
            details = [
                "Synthetic data is close to original with minor shift",
                "Distributions mostly overlap",
                "1-NN classifier shows moderate separability (55-65%)",
                "Recommendation: Use synthetic with weight 0.3-0.5"
            ]
        elif overlap > t_overlap['satisfactory'] and nn_acc < 0.75:
            quality = "🟠 SATISFACTORY"
            details = [
                "Noticeable domain gap between synthetic and original",
                "Distributions partially overlap",
                "1-NN classifier separates domains well (65-75%)",
                "Recommendation: Reduce strength to 0.05-0.10"
            ]
        elif overlap > t_overlap['poor']:
            quality = "🔴 POOR"
            details = [
                "Strong domain gap",
                "Distributions significantly differ",
                "1-NN classifier easily separates domains (> 75%)",
                "Recommendation: Reduce strength to 0.02-0.05"
            ]
        else:
            quality = "⛔ CRITICAL"
            details = [
                "Synthetic completely misaligned with original",
                "Distributions barely overlap",
                "1-NN classifier achieves 100% accuracy",
                "Recommendation: Review generation parameters entirely"
            ]
        
        report.append(f"\n  Synthetic Quality: {quality}")
        report.append(f"  Domain Overlap Score: {overlap:.3f}/1.000")
        report.append(f"  1-NN accuracy (closer to 0.5 is better): {nn_acc:.3f}")
        report.append("")
        for detail in details:
            report.append(f"  • {detail}")
        
        report.append("")
        report.append("=" * 80)
        report.append("HOW TO READ METRICS:")
        report.append("  Domain Overlap Score: 1.0 = perfect, 0.0 = completely different domains")
        report.append("  1-NN accuracy: 0.50 = domains indistinguishable (good)")
        report.append("               0.75+ = domains easily separable (bad)")
        report.append("  Cosine similarity: >0.95 = excellent, <0.85 = poor")
        report.append("  Domain Gap Ratio: ~1.0 = identical domains, >2.0 = strong gap")
        report.append("  Silhouette Score: <0 = domains overlap, >0.5 = well separated")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Сохраняем отчёт
        report_path = output_dir / "domain_gap_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        return report_text
    
    def analyze(self, original_dir: Path, synthetic_dir: Path, output_dir: Path) -> Dict:
        """Полный анализ domain gap"""
        print("=" * 80)
        print("🔬 DOMAIN GAP ANALYSIS")
        print("=" * 80)
        
        # Извлечение признаков
        original_features, original_names = self.extract_features(original_dir, "original")
        synthetic_features, synthetic_names = self.extract_features(synthetic_dir, "synthetic")
        
        # Статистики
        self.results['original_statistics'] = self.compute_statistics(original_features)
        self.results['synthetic_statistics'] = self.compute_statistics(synthetic_features)
        
        # Метрики схожести
        self.results['similarity_metrics'] = self.compute_similarity_metrics(
            original_features, synthetic_features
        )
        
        # EMD
        self.results['emd_metrics'] = self.compute_per_channel_emd(
            original_features, synthetic_features
        )
        
        # Визуализации
        viz_dir = output_dir / self.config.paths.subdirs.get('visualizations', 'visualizations')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        viz_results = self.visualize_embeddings(
            [(original_features, 'original'), (synthetic_features, 'synthetic')],
            viz_dir
        )
        self.results['visualization'] = viz_results
        
        # Сохраняем JSON результаты
        json_path = output_dir / "domain_gap_results.json"
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        print(f"\n✅ JSON results saved: {json_path}")
        
        # Генерируем отчёт
        self.generate_report(output_dir)
        
        print(f"\n✅ All results saved to: {output_dir}")
        print("=" * 80)
        
        return self.results


def run_domain_gap_analysis(config: AnalysisConfig) -> Dict:
    """Запуск анализа domain gap"""
    analyzer = DomainGapAnalyzer(config)
    analyzer.load_model()
    
    output_dir = config.paths.output_dir
    
    results = analyzer.analyze(
        original_dir=config.paths.original_dir,
        synthetic_dir=config.paths.synthetic_dir,
        output_dir=output_dir
    )
    
    return results