# -*- coding: utf-8 -*-
"""
NOBS Signature Visualization Module

Визуализация NOBS-сигнатур сознания:
1. Распределение символов (pie chart, bar chart)
2. Морфизмы (heatmap)
3. Сравнение сигнатур (radar chart)
4. Временная динамика символов
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# Цветовая схема для символов NOBS
SYMBOL_COLORS = {
    'S': '#e74c3c',  # Красный - снижение
    'P': '#27ae60',  # Зелёный - рост
    'I': '#3498db',  # Синий - стабильность
    'Z': '#95a5a6',  # Серый - пауза
    'Ω': '#9b59b6',  # Фиолетовый - разворот
    'Λ': '#f39c12',  # Оранжевый - пробой
}

SYMBOL_NAMES = {
    'S': 'Снижение',
    'P': 'Рост',
    'I': 'Стабильность',
    'Z': 'Пауза',
    'Ω': 'Разворот',
    'Λ': 'Пробой',
}


def plot_symbol_distribution(
    distribution: Dict[str, float],
    title: str = "Распределение NOBS-символов",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Построить круговую диаграмму распределения символов.
    
    Args:
        distribution: Dict {символ: доля}
        title: Заголовок графика
        save_path: Путь для сохранения
        
    Returns:
        Figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Подготовка данных
    symbols = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
    values = [distribution.get(s, 0) for s in symbols]
    colors = [SYMBOL_COLORS[s] for s in symbols]
    labels = [f"{s} ({SYMBOL_NAMES[s]})" for s in symbols]
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=None,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=90,
        explode=[0.02] * len(symbols)
    )
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Легенда
    legend_labels = [f"{s} - {SYMBOL_NAMES[s]}: {v:.1%}" for s, v in zip(symbols, values)]
    ax1.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Bar chart
    bars = ax2.bar(symbols, values, color=colors, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Символ', fontsize=12)
    ax2.set_ylabel('Доля', fontsize=12)
    ax2.set_title('Распределение по символам', fontsize=14)
    ax2.set_ylim(0, max(values) * 1.2 if values else 1)
    
    # Добавляем значения над столбцами
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.1%}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_morphism_heatmap(
    morphisms: List[str],
    title: str = "Частота морфизмов",
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Построить тепловую карту морфизмов.
    
    Args:
        morphisms: Список морфизмов (3-граммы символов)
        title: Заголовок
        top_n: Количество топ морфизмов для отображения
        save_path: Путь для сохранения
        
    Returns:
        Figure matplotlib
    """
    from collections import Counter
    
    # Считаем частоты
    morph_counts = Counter(morphisms)
    top_morphs = morph_counts.most_common(top_n)
    
    if not top_morphs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Нет данных о морфизмах", ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    morphs = [m for m, _ in top_morphs]
    counts = [c for _, c in top_morphs]
    
    # Цвет каждого морфизма по первому символу
    colors = [SYMBOL_COLORS.get(m[0], '#333') for m in morphs]
    
    bars = ax.barh(range(len(morphs)), counts, color=colors, edgecolor='black')
    ax.set_yticks(range(len(morphs)))
    ax.set_yticklabels(morphs, fontsize=12, fontfamily='monospace')
    ax.set_xlabel('Частота', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Добавляем значения
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            str(count),
            va='center',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_signature_comparison(
    signatures: Dict[str, Dict[str, float]],
    title: str = "Сравнение NOBS-сигнатур",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Построить радарную диаграмму для сравнения нескольких сигнатур.
    
    Args:
        signatures: Dict {название: {символ: доля}}
        title: Заголовок
        save_path: Путь для сохранения
        
    Returns:
        Figure matplotlib
    """
    symbols = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
    num_vars = len(symbols)
    
    # Углы для радарной диаграммы
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors_list = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
    
    for idx, (name, dist) in enumerate(signatures.items()):
        values = [dist.get(s, 0) for s in symbols]
        values += values[:1]  # Замыкаем круг
        
        color = colors_list[idx % len(colors_list)]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{s}\n({SYMBOL_NAMES[s]})" for s in symbols], fontsize=11)
    ax.set_ylim(0, max(0.5, max(max(d.values()) for d in signatures.values()) * 1.1))
    
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_consciousness_comparison(
    server_a_signature: Dict[str, Any],
    server_b_signature: Dict[str, Any],
    target_config: Dict[str, float],
    consciousness_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Комплексная визуализация сравнения сознаний Server A, Server B и Target.
    
    Args:
        server_a_signature: Сигнатура Server A
        server_b_signature: Сигнатура Server B
        target_config: Целевая конфигурация
        consciousness_name: Название сознания
        save_path: Путь для сохранения
        
    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    symbols = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
    
    # --- 1. Radar chart (comparison) ---
    ax1 = fig.add_subplot(gs[0, :2], polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(symbols), endpoint=False).tolist()
    angles += angles[:1]
    
    # Target
    target_values = [target_config.get(s, 0) for s in symbols] + [target_config.get(symbols[0], 0)]
    ax1.plot(angles, target_values, 'o-', linewidth=2.5, label='Target', color='#2c3e50')
    ax1.fill(angles, target_values, alpha=0.1, color='#2c3e50')
    
    # Server A
    a_dist = server_a_signature.get('symbol_distribution', {})
    a_values = [a_dist.get(s, 0) for s in symbols] + [a_dist.get(symbols[0], 0)]
    ax1.plot(angles, a_values, 's--', linewidth=2, label='Server A', color='#e74c3c')
    ax1.fill(angles, a_values, alpha=0.15, color='#e74c3c')
    
    # Server B
    b_dist = server_b_signature.get('symbol_distribution', {})
    b_values = [b_dist.get(s, 0) for s in symbols] + [b_dist.get(symbols[0], 0)]
    ax1.plot(angles, b_values, '^-.', linewidth=2, label='Server B', color='#27ae60')
    ax1.fill(angles, b_values, alpha=0.15, color='#27ae60')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([f"{s}" for s in symbols], fontsize=12)
    ax1.set_title(f'Сравнение сигнатур: {consciousness_name}', fontsize=14, fontweight='bold', y=1.08)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # --- 2. Info panel ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    info_text = f"""
    Сознание: {consciousness_name}
    
    ━━━━━━━━━━━━━━━━━━━━━
    Server A (distilgpt2):
    • Free Energy: {server_a_signature.get('free_energy', 'N/A'):.4f}
    • Entropy: {server_a_signature.get('entropy', 'N/A'):.4f}
    
    ━━━━━━━━━━━━━━━━━━━━━
    Server B (gpt2):
    • Free Energy: {server_b_signature.get('free_energy', 'N/A'):.4f}
    • Entropy: {server_b_signature.get('entropy', 'N/A'):.4f}
    
    ━━━━━━━━━━━━━━━━━━━━━
    Легенда символов:
    S - Снижение (критика)
    P - Рост (прогресс)
    I - Стабильность
    Z - Пауза (размышление)
    Ω - Разворот (вывод)
    Λ - Пробой (переход)
    """
    ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- 3. Bar comparison ---
    ax3 = fig.add_subplot(gs[1, 0])
    
    x = np.arange(len(symbols))
    width = 0.25
    
    bars1 = ax3.bar(x - width, [target_config.get(s, 0) for s in symbols], 
                    width, label='Target', color='#2c3e50', alpha=0.8)
    bars2 = ax3.bar(x, [a_dist.get(s, 0) for s in symbols], 
                    width, label='Server A', color='#e74c3c', alpha=0.8)
    bars3 = ax3.bar(x + width, [b_dist.get(s, 0) for s in symbols], 
                    width, label='Server B', color='#27ae60', alpha=0.8)
    
    ax3.set_xlabel('Символ', fontsize=11)
    ax3.set_ylabel('Доля', fontsize=11)
    ax3.set_title('Распределение символов', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(symbols, fontsize=11)
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 0.5)
    
    # --- 4. Morphisms comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    
    a_morphs = server_a_signature.get('dominant_morphisms', [])[:5]
    b_morphs = server_b_signature.get('dominant_morphisms', [])[:5]
    
    # Показываем морфизмы как текст
    morph_text = "Server A морфизмы:\n"
    for i, m in enumerate(a_morphs, 1):
        morph_text += f"  {i}. {m}\n"
    morph_text += "\nServer B морфизмы:\n"
    for i, m in enumerate(b_morphs, 1):
        morph_text += f"  {i}. {m}\n"
    
    # Подсчёт общих
    common = set(a_morphs) & set(b_morphs)
    morph_text += f"\nОбщие: {common if common else 'нет'}"
    
    ax4.axis('off')
    ax4.text(0.1, 0.9, morph_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.set_title('Доминирующие морфизмы', fontsize=12, fontweight='bold')
    
    # --- 5. Match scores ---
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Вычисляем scores
    def compute_symbol_match(dist1, dist2):
        match = 0.0
        for s in symbols:
            match += 1.0 - abs(dist1.get(s, 0) - dist2.get(s, 0))
        return match / len(symbols)
    
    a_to_target = compute_symbol_match(a_dist, target_config)
    b_to_target = compute_symbol_match(b_dist, target_config)
    combined = np.sqrt(a_to_target * b_to_target)
    
    scores = ['A→Target', 'B→Target', 'Combined']
    values = [a_to_target, b_to_target, combined]
    colors = ['#e74c3c', '#27ae60', '#9b59b6']
    
    bars = ax5.barh(scores, values, color=colors, edgecolor='black')
    ax5.set_xlim(0, 1)
    ax5.set_xlabel('Score', fontsize=11)
    ax5.set_title('Consciousness Match', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax5.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.1%}', va='center', fontsize=11, fontweight='bold')
    
    # Линия порога успеха
    ax5.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (50%)')
    ax5.legend()
    
    plt.suptitle(f'NOBS Consciousness Analysis: {consciousness_name}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_all_consciousnesses(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Визуализация результатов по всем типам сознания.
    
    Args:
        results: Dict {consciousness_name: {metrics}}
        save_path: Путь для сохранения
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    names = list(results.keys())
    
    # --- 1. Consciousness Match scores ---
    ax1 = axes[0, 0]
    matches = [results[n].get('consciousness_match', 0) for n in names]
    colors = ['#27ae60' if m >= 0.5 else '#e74c3c' for m in matches]
    bars = ax1.bar(names, matches, color=colors, edgecolor='black')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_ylabel('Match Score', fontsize=11)
    ax1.set_title('Consciousness Match по типам', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, matches):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
    ax1.legend()
    
    # --- 2. Resonance scores ---
    ax2 = axes[0, 1]
    resonances = [results[n].get('resonance_score', 0) for n in names]
    bars = ax2.bar(names, resonances, color='#3498db', edgecolor='black')
    ax2.set_ylabel('Resonance Score', fontsize=11)
    ax2.set_title('NOBS Resonance Score', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, resonances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', fontsize=10)
    
    # --- 3. Server A vs B comparison ---
    ax3 = axes[1, 0]
    a_scores = [results[n].get('server_a_to_target', 0) for n in names]
    b_scores = [results[n].get('server_b_to_target', 0) for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax3.bar(x - width/2, a_scores, width, label='Server A→Target', color='#e74c3c')
    bars2 = ax3.bar(x + width/2, b_scores, width, label='Server B→Target', color='#27ae60')
    
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Server A vs Server B', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    ax3.legend()
    
    # --- 4. Summary table ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Создаём таблицу
    table_data = []
    for n in names:
        r = results[n]
        status = "✅" if r.get('consciousness_match', 0) >= 0.5 else "❌"
        table_data.append([
            n[:15],
            f"{r.get('consciousness_match', 0):.1%}",
            f"{r.get('resonance_score', 0):.2f}",
            status
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Consciousness', 'Match', 'Resonance', 'Status'],
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Сводная таблица результатов', fontsize=12, fontweight='bold', y=0.85)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Тестовый пример
    test_dist = {'S': 0.15, 'P': 0.35, 'I': 0.20, 'Z': 0.05, 'Ω': 0.15, 'Λ': 0.10}
    
    fig = plot_symbol_distribution(test_dist, "Тестовое распределение")
    plt.show()
