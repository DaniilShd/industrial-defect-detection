"""Анализ разрыва доменов с использованием признаков DINOv2"""

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

# Настройка русского шрифта для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
            print(f"⚠️ Ошибка загрузки {path}: {e}")
            blank = Image.new('RGB', (self.resize_to, self.resize_to), (0, 0, 0))
            inputs = self.processor(images=blank, return_tensors="pt")
            return inputs.pixel_values.squeeze(0), path.name


class DomainGapAnalyzer:
    """Анализатор разрыва доменов с использованием DINOv2"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = config.dinov2.device
        self.processor = None
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": config.dinov2.model_name,
            "device": config.dinov2.device,
            "num_samples": config.dinov2.num_samples,
            "feature_statistics": {},
            "domain_gap": {},
            "emd_analysis": {},
            "visualization": {}
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
        
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_paths.extend(list(images_dir.glob(ext)))
        
        image_paths = sorted(image_paths)
        
        if len(image_paths) > self.config.dinov2.num_samples:
            np.random.seed(self.config.dinov2.random_seed)
            indices = np.random.choice(len(image_paths), self.config.dinov2.num_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
        
        print(f"\n📊 Извлечение признаков из '{dataset_name}' ({len(image_paths)} изображений)...")
        
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
    
    def compute_statistics(self, features: np.ndarray, domain_name: str) -> Dict:
        """Вычисление статистик распределения признаков"""
        l2_norms = np.linalg.norm(features, axis=1)
        
        return {
            "domain": domain_name,
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
            "num_samples": features.shape[0]
        }
    
    def compute_per_channel_emd(self, features1: np.ndarray, features2: np.ndarray) -> Dict:
        """Вычисление Earth Mover's Distance поканально"""
        print("\n📊 Вычисление EMD...")
        
        emd_values = []
        for i in tqdm(range(features1.shape[1]), desc="  EMD по каналам"):
            emd = wasserstein_distance(features1[:, i], features2[:, i])
            emd_values.append(emd)
        
        emd_values = np.array(emd_values)
        
        return {
            "mean": float(np.mean(emd_values)),
            "std": float(np.std(emd_values)),
            "max": float(np.max(emd_values)),
            "min": float(np.min(emd_values)),
            "median": float(np.median(emd_values)),
            "percentile_95": float(np.percentile(emd_values, 95)),
            "percentile_75": float(np.percentile(emd_values, 75)),
            "top10_channels": np.argsort(emd_values)[-10:].tolist(),
            "bottom10_channels": np.argsort(emd_values)[:10].tolist()
        }
    
    def compute_similarity_metrics(self, features1: np.ndarray, features2: np.ndarray) -> Dict:
        """Вычисление метрик схожести доменов"""
        print("\n📊 Вычисление метрик схожести...")
        
        n_test = min(self.config.dinov2.nn_test_samples, len(features1), len(features2))
        np.random.seed(self.config.dinov2.random_seed)
        
        # Центроиды
        centroid1 = np.mean(features1, axis=0)
        centroid2 = np.mean(features2, axis=0)
        centroid_distance = np.linalg.norm(centroid1 - centroid2)
        
        # Косинусное сходство
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
        
        # Внутри- и меж-доменные расстояния
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
        
        # Оценка перекрытия доменов
        overlap_score = 2.0 * (1.0 - nn_accuracy)
        overlap_score = np.clip(overlap_score, 0.0, 1.0)
        
        # Силуэт-оценка
        silhouette_score = np.mean(
            (np.min(distances_a_to_b, axis=1) - np.min(distances_a_to_a, axis=1)) / 
            (np.maximum(np.min(distances_a_to_b, axis=1), np.min(distances_a_to_a, axis=1)) + 1e-8)
        )
        
        return {
            "centroid_distance": float(centroid_distance),
            "centroid_cosine_similarity": float(cosine_sim),
            "intra_distance_original": float(mean_intra_a),
            "intra_distance_synthetic": float(mean_intra_b),
            "inter_distance": float(inter_avg),
            "gap_ratio": float(gap_ratio),
            "nn_accuracy": float(nn_accuracy),
            "overlap_score": float(overlap_score),
            "silhouette_score": float(silhouette_score)
        }
    
    def visualize_embeddings(self, features_list: List[Tuple[np.ndarray, str]], 
                            output_dir: Path) -> Dict:
        """Визуализация эмбеддингов: PCA и t-SNE"""
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
        
        # Словарь цветов и названий для графиков (русские подписи)
        domain_names = {
            'original': 'Оригинал',
            'synthetic': 'Синтетика'
        }
        colors = {'original': '#2196F3', 'synthetic': '#FF9800'}
        
        # --- PCA Визуализация ---
        pca = PCA(n_components=2)
        features_2d_pca = pca.fit_transform(features_scaled)
        explained_var = pca.explained_variance_ratio_ * 100
        
        fig, axes = plt.subplots(1, 2, figsize=viz_config.figsize)
        
        # PCA диаграмма рассеяния с эллипсами
        ax1 = axes[0]
        for name in set(all_names):
            mask = np.array([n == name for n in all_names])
            data = features_2d_pca[mask]
            color = colors.get(name, '#999999')
            display_name = domain_names.get(name, name)
            
            ax1.scatter(data[:, 0], data[:, 1], c=color, label=display_name, 
                       alpha=0.5, s=10, edgecolors='none')
            
            # Эллипс 95% доверительного интервала
            if len(data) > 2 and viz_config.pca.get('show_ellipses', True):
                try:
                    cov = np.cov(data.T)
                    mean = np.mean(data, axis=0)
                    
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    
                    width, height = 2 * np.sqrt(eigenvalues) * 2.447
                    
                    ellipse = Ellipse(xy=mean, width=width, height=height,
                                    angle=angle, facecolor='none',
                                    edgecolor=color, linewidth=2, 
                                    linestyle='--', alpha=0.8)
                    ax1.add_patch(ellipse)
                except Exception as e:
                    print(f"  ⚠️ Не удалось построить эллипс для {name}: {e}")
        
        ax1.set_xlabel(f'Главная компонента 1 ({explained_var[0]:.1f}% дисперсии)', fontsize=12)
        ax1.set_ylabel(f'Главная компонента 2 ({explained_var[1]:.1f}% дисперсии)', fontsize=12)
        ax1.set_title('Распределение признаков DINOv2 (PCA)', fontsize=14, fontweight='bold')
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
                       linewidth=2, label=f'Среднее: {np.mean(distances):.3f}')
            ax2.axvline(x=np.median(distances), color='g', linestyle=':', 
                       linewidth=2, label=f'Медиана: {np.median(distances):.3f}')
            
            display_names = [domain_names.get(n, n) for n in names_list]
            ax2.set_xlabel('Попарное расстояние в пространстве PCA', fontsize=12)
            ax2.set_ylabel('Частота', fontsize=12)
            ax2.set_title(f'Расстояния: {display_names[0]} vs {display_names[1]}', 
                         fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        for fmt in viz_config.save_formats:
            save_path = output_dir / f'pca_визуализация.{fmt}'
            plt.savefig(save_path, dpi=viz_config.dpi, bbox_inches='tight')
        
        plt.close()
        
        results['pca'] = {
            "explained_variance": explained_var.tolist(),
            "total_variance": float(np.sum(explained_var)),
            "path": str(output_dir / 'pca_визуализация.png')
        }
        
        # --- t-SNE Визуализация ---
        if viz_config.tsne.get('enabled', False):
            print("  Создание t-SNE визуализации...")
            
            n_tsne = min(len(features_scaled), viz_config.tsne.get('max_points', 300))
            indices = np.random.choice(len(features_scaled), n_tsne, replace=False)
            
            tsne = TSNE(
                n_components=2, 
                perplexity=viz_config.tsne.get('perplexity', 30),
                max_iter=viz_config.tsne.get('max_iter', 1000),
                random_state=self.config.dinov2.random_seed
            )
            
            features_2d_tsne = tsne.fit_transform(features_scaled[indices])
            names_tsne = [all_names[i] for i in indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for name in set(names_tsne):
                mask = np.array([n == name for n in names_tsne])
                color = colors.get(name, '#999999')
                display_name = domain_names.get(name, name)
                ax.scatter(features_2d_tsne[mask, 0], features_2d_tsne[mask, 1],
                          c=color, label=display_name, alpha=0.6, s=15, edgecolors='none')
            
            ax.set_xlabel('t-SNE Компонента 1', fontsize=12)
            ax.set_ylabel('t-SNE Компонента 2', fontsize=12)
            ax.set_title('Распределение признаков DINOv2 (t-SNE)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            for fmt in viz_config.save_formats:
                plt.savefig(output_dir / f'tsne_визуализация.{fmt}', 
                          dpi=viz_config.dpi, bbox_inches='tight')
            
            plt.close()
            
            results['tsne'] = {
                "perplexity": viz_config.tsne.get('perplexity', 30),
                "path": str(output_dir / 'tsne_визуализация.png')
            }
        
        print("✅ Визуализации сохранены")
        return results
    
    def generate_report(self, output_dir: Path) -> str:
        """Генерация текстового отчёта на русском языке"""
        thresholds = self.config.thresholds
        
        orig_stats = self.results.get('feature_statistics', {}).get('original', {})
        synth_stats = self.results.get('feature_statistics', {}).get('synthetic', {})
        similarity = self.results.get('domain_gap', {})
        emd_data = self.results.get('emd_analysis', {})
        
        report = []
        report.append("=" * 80)
        report.append("🔬 АНАЛИЗ РАЗРЫВА ДОМЕНОВ (DOMAIN GAP)")
        report.append("=" * 80)
        report.append(f"Создан: {self.results.get('timestamp', 'N/A')}")
        report.append(f"Модель: {self.config.dinov2.model_name}")
        report.append(f"Устройство: {self.config.dinov2.device}")
        report.append("")
        
        # Секция 1: Статистики признаков
        report.append("-" * 80)
        report.append("1. СТАТИСТИКИ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ")
        report.append("-" * 80)
        report.append(f"{'Метрика':<30} {'Оригинал':<15} {'Синтетика':<15} {'Разница %':<15}")
        report.append("-" * 80)
        
        for key, display_name in [('mean', 'Среднее'), ('std', 'Станд. откл.'), 
                                   ('variance', 'Дисперсия'), ('l2_norm_mean', 'L2-норма (сред.)')]:
            orig_val = orig_stats.get(key, 0)
            synth_val = synth_stats.get(key, 0)
            diff_pct = abs(synth_val - orig_val) / (abs(orig_val) + 1e-8) * 100
            report.append(f"{display_name:<30} {orig_val:<15.6f} {synth_val:<15.6f} {diff_pct:<15.2f}%")
        
        report.append("")
        
        # Секция 2: Метрики схожести
        report.append("-" * 80)
        report.append("2. МЕТРИКИ СХОЖЕСТИ ДОМЕНОВ")
        report.append("-" * 80)
        
        overlap = similarity.get('overlap_score', 0)
        nn_acc = similarity.get('nn_accuracy', 1.0)
        gap_ratio = similarity.get('gap_ratio', 999)
        cosine_sim = similarity.get('centroid_cosine_similarity', 0)
        
        report.append(f"  {'Косинусное сходство центроидов:':<40} {cosine_sim:.4f}")
        report.append(f"  {'Точность 1-NN классификации:':<40} {nn_acc:.4f} (0.50 = неразличимы)")
        report.append(f"  {'Оценка перекрытия доменов:':<40} {overlap:.4f} (1.00 = полное перекрытие)")
        report.append(f"  {'Коэффициент разрыва доменов:':<40} {gap_ratio:.4f} (1.00 = идентичны)")
        report.append(f"  {'Внутридоменное расстояние (Оригинал):':<40} {similarity.get('intra_distance_original', 0):.6f}")
        report.append(f"  {'Внутридоменное расстояние (Синтетика):':<40} {similarity.get('intra_distance_synthetic', 0):.6f}")
        report.append(f"  {'Междоменное расстояние:':<40} {similarity.get('inter_distance', 0):.6f}")
        
        report.append("")
        
        # Секция 3: EMD
        report.append("-" * 80)
        report.append("3. РАССТОЯНИЕ EARTH MOVER'S (ПОКАНАЛЬНО)")
        report.append("-" * 80)
        report.append(f"  {'Среднее EMD:':<30} {emd_data.get('mean', 0):.6f}")
        report.append(f"  {'Медиана EMD:':<30} {emd_data.get('median', 0):.6f}")
        report.append(f"  {'95-й процентиль EMD:':<30} {emd_data.get('percentile_95', 0):.6f}")
        report.append(f"  {'Максимум EMD:':<30} {emd_data.get('max', 0):.6f}")
        
        report.append("")
        
        # Секция 4: Интерпретация
        report.append("-" * 80)
        report.append("4. ИНТЕРПРЕТАЦИЯ КАЧЕСТВА")
        report.append("-" * 80)
        
        t_overlap = thresholds.domain_overlap
        
        if overlap > t_overlap['excellent'] and nn_acc < 0.55:
            quality = "✅ ОТЛИЧНО"
            details = [
                "Синтетические данные практически неотличимы от оригинальных",
                "Распределения признаков значительно перекрываются",
                "1-NN классификатор не может разделить домены (< 55% точности)",
                "Рекомендация: Использовать синтетические данные без ограничений"
            ]
        elif overlap > t_overlap['good'] and nn_acc < 0.65:
            quality = "🟡 ХОРОШО"
            details = [
                "Синтетические данные близки к оригиналам с небольшим сдвигом",
                "Распределения в основном перекрываются",
                "1-NN классификатор показывает умеренную разделимость (55-65%)",
                "Рекомендация: Использовать синтетику с весом 0.3-0.5"
            ]
        elif overlap > t_overlap['satisfactory'] and nn_acc < 0.75:
            quality = "🟠 УДОВЛЕТВОРИТЕЛЬНО"
            details = [
                "Заметный разрыв между синтетикой и оригиналами",
                "Распределения частично перекрываются",
                "1-NN классификатор хорошо разделяет домены (65-75%)",
                "Рекомендация: Уменьшить силу аугментации до 0.05-0.10"
            ]
        elif overlap > t_overlap['poor']:
            quality = "🔴 ПЛОХО"
            details = [
                "Сильный разрыв доменов",
                "Распределения значительно различаются",
                "1-NN классификатор легко разделяет домены (> 75%)",
                "Рекомендация: Уменьшить силу аугментации до 0.02-0.05"
            ]
        else:
            quality = "⛔ КРИТИЧЕСКИ"
            details = [
                "Синтетика полностью не совпадает с оригиналами",
                "Распределения практически не перекрываются",
                "1-NN классификатор достигает 100% точности",
                "Рекомендация: Полностью пересмотреть параметры генерации"
            ]
        
        report.append(f"\n  Качество синтетики: {quality}")
        report.append(f"  Оценка перекрытия доменов: {overlap:.3f}/1.000")
        report.append(f"  Точность 1-NN (чем ближе к 0.5, тем лучше): {nn_acc:.3f}")
        report.append("")
        for detail in details:
            report.append(f"  • {detail}")
        
        report.append("")
        report.append("=" * 80)
        report.append("КАК ЧИТАТЬ МЕТРИКИ:")
        report.append("  Оценка перекрытия доменов: 1.0 = идеально, 0.0 = полностью разные домены")
        report.append("  Точность 1-NN: 0.50 = домены неразличимы (хорошо)")
        report.append("                 0.75+ = домены легко разделимы (плохо)")
        report.append("  Косинусное сходство: >0.95 = отлично, <0.85 = плохо")
        report.append("  Коэффициент разрыва доменов: ~1.0 = идентичны, >2.0 = сильный разрыв")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Сохраняем отчёт
        report_path = output_dir / "отчёт_разрыва_доменов.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        return report_text
    
    def analyze(self, original_dir: Path, synthetic_dir: Path, output_dir: Path) -> Dict:
        """Полный анализ разрыва доменов"""
        print("=" * 80)
        print("🔬 АНАЛИЗ РАЗРЫВА ДОМЕНОВ")
        print("=" * 80)
        
        # Извлечение признаков
        original_features, _ = self.extract_features(original_dir, "оригинал")
        synthetic_features, _ = self.extract_features(synthetic_dir, "синтетика")
        
        # Статистики
        self.results['feature_statistics'] = {
            'original': self.compute_statistics(original_features, "original"),
            'synthetic': self.compute_statistics(synthetic_features, "synthetic")
        }
        
        # Метрики схожести
        self.results['domain_gap'] = self.compute_similarity_metrics(
            original_features, synthetic_features
        )
        
        # EMD
        self.results['emd_analysis'] = self.compute_per_channel_emd(
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
        
        # Сохраняем JSON результаты (только английские ключи)
        json_path = output_dir / "domain_gap_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        print(f"\n✅ JSON результаты сохранены: {json_path}")
        
        # Генерируем русский текстовый отчёт
        self.generate_report(output_dir)
        
        print(f"\n✅ Все результаты сохранены в: {output_dir}")
        print("=" * 80)
        
        return self.results


def run_domain_gap_analysis(config: AnalysisConfig) -> Dict:
    """Запуск анализа разрыва доменов"""
    analyzer = DomainGapAnalyzer(config)
    analyzer.load_model()
    
    output_dir = config.paths.output_dir
    
    results = analyzer.analyze(
        original_dir=config.paths.original_dir,
        synthetic_dir=config.paths.synthetic_dir,
        output_dir=output_dir
    )
    
    return results