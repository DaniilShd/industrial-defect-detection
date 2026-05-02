#!/usr/bin/env python3
"""Статистический анализ: ANOVA, t-тесты, Cohen's d, Hedges' g"""

import logging
from collections import defaultdict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def run_statistical_analysis(all_results: list, config: dict) -> dict:
    completed = [r for r in all_results if r.get('status') == 'completed']
    groups = defaultdict(list)
    for r in completed:
        key = f"{r['dataset_name']}_{r['strategy_name']}"
        groups[key].append(r['test_map50'])

    if len(groups) < 2:
        return {'error': 'insufficient_data'}

    analysis = {
        'descriptive': {}, 'anova': {}, 'pairwise': [], 'significant_pairs': [],
    }

    for name, values in groups.items():
        n = len(values)
        analysis['descriptive'][name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),
            'sem': float(np.std(values, ddof=1) / np.sqrt(n)),
            'min': float(np.min(values)), 'max': float(np.max(values)), 'n': n,
        }

    group_values = [v for v in groups.values() if len(v) > 1]
    if len(group_values) >= 2:
        f_stat, p_val = stats.f_oneway(*group_values)
        analysis['anova'] = {
            'f_statistic': float(f_stat), 'p_value': float(p_val),
            'significant': p_val < 0.05,
        }

    group_names = list(groups.keys())
    baseline_key = next((n for n in group_names if 'baseline' in n or 'real_baseline' in n), group_names[0])
    baseline_vals = groups[baseline_key]

    for name in group_names:
        if name == baseline_key: continue
        vals = groups[name]
        if len(vals) < 2: continue

        t_stat, p_val = stats.ttest_ind(vals, baseline_vals, equal_var=False)
        mean_diff = np.mean(vals) - np.mean(baseline_vals)
        n1, n2 = len(vals), len(baseline_vals)
        s1, s2 = np.std(vals, ddof=1), np.std(baseline_vals, ddof=1)
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        hedges_g = cohens_d * (1 - 3/(4*(n1+n2) - 9))

        pair = {
            'comparison': f"{baseline_key} vs {name}",
            'mean_difference': float(mean_diff), 't_statistic': float(t_stat),
            'p_value': float(p_val), 'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g),
            'effect_size': interpret_effect(abs(hedges_g)),
        }
        analysis['pairwise'].append(pair)
        if p_val < config['statistics']['alpha']:
            analysis['significant_pairs'].append(pair['comparison'])

    return analysis


def interpret_effect(d: float) -> str:
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"