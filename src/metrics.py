"""
Метрики для оценки состояний агента в S

Включает:
- E_B: энергия понимания
- ρ_B: самоинвариантность
- J: интегральный reward для шамана
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnergyComponents:
    """Компоненты энергии понимания E_B."""
    length_penalty: float
    velocity_penalty: float
    entropy_penalty: float
    total: float


def compute_energy_E(
    latent_trajectory: np.ndarray,
    action_probs: Optional[np.ndarray] = None,
    response_lengths: Optional[List[int]] = None,
    config: Optional[Dict] = None
) -> EnergyComponents:
    """
    Вычисляет энергию понимания E_B для траектории.
    
    E_B = комбинация:
    - длины эпизодов (долгие = дороже)
    - скорости движения в S (∑||h_{t+1} - h_t||²)
    - энтропии политики (неуверенность)
    
    Args:
        latent_trajectory: [T, D] - траектория в латентном пространстве
        action_probs: [T, A] - вероятности действий (опционально)
        response_lengths: длины ответов (опционально)
        config: параметры весов
        
    Returns:
        EnergyComponents с разбивкой по компонентам
    """
    # Дефолтные веса
    if config is None:
        config = {
            "length_penalty": 0.01,
            "velocity_penalty": 1.0,
            "entropy_penalty": 0.1
        }
    
    T = len(latent_trajectory)
    
    # 1. Length penalty
    if response_lengths is not None:
        avg_length = np.mean(response_lengths)
    else:
        avg_length = T * 100  # proxy
    E_length = config["length_penalty"] * avg_length
    
    # 2. Velocity penalty (плавность траектории)
    if T > 1:
        velocities = latent_trajectory[1:] - latent_trajectory[:-1]
        E_velocity = config["velocity_penalty"] * np.sum(velocities ** 2)
    else:
        E_velocity = 0.0
    
    # 3. Entropy penalty
    if action_probs is not None:
        # H = -∑ p * log(p)
        eps = 1e-10
        entropy = -np.sum(action_probs * np.log(action_probs + eps), axis=-1)
        E_entropy = config["entropy_penalty"] * np.sum(entropy)
    else:
        E_entropy = 0.0
    
    E_total = E_length + E_velocity + E_entropy
    
    return EnergyComponents(
        length_penalty=float(E_length),
        velocity_penalty=float(E_velocity),
        entropy_penalty=float(E_entropy),
        total=float(E_total)
    )


def compute_rho(
    anchor_embedding: np.ndarray,
    sample_embeddings: np.ndarray,
    method: str = "inverse_variance"
) -> float:
    """
    Вычисляет самоинвариантность ρ_B.
    
    ρ высока, когда агент стабильно воспроизводит похожий смысл
    при вариациях условий/задач.
    
    Args:
        anchor_embedding: [D] - эталонный эмбеддинг (центр s_я)
        sample_embeddings: [N, D] - эмбеддинги от вариаций задач
        method: метод вычисления
        
    Returns:
        ρ ∈ [0, 1], где 1 = максимальная устойчивость
    """
    if len(sample_embeddings) == 0:
        return 1.0
    
    # Вычисляем расстояния от anchor
    diffs = sample_embeddings - anchor_embedding
    distances = np.linalg.norm(diffs, axis=-1)
    
    if method == "inverse_variance":
        # ρ = 1 / (1 + var(distances))
        variance = np.var(distances)
        return float(1.0 / (1.0 + variance))
    
    elif method == "inverse_mean":
        # ρ = 1 / (1 + mean(distances))
        mean_dist = np.mean(distances)
        return float(1.0 / (1.0 + mean_dist))
    
    elif method == "exponential":
        # ρ = exp(-mean(distances))
        mean_dist = np.mean(distances)
        return float(np.exp(-mean_dist))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_cluster_coherence(
    embeddings: np.ndarray,
    method: str = "silhouette"
) -> float:
    """
    Вычисляет когерентность кластера точек.
    
    Высокая когерентность = точки образуют плотный, чётко очерченный кластер.
    
    Args:
        embeddings: [N, D] - точки кластера
        method: метод оценки
        
    Returns:
        coherence ∈ [-1, 1] для silhouette, [0, 1] для остальных
    """
    if len(embeddings) < 3:
        return 0.0
    
    if method == "silhouette":
        # Упрощённый silhouette score для одного кластера
        # Сравниваем со случайным разбросом
        from sklearn.metrics import pairwise_distances
        
        dists = pairwise_distances(embeddings)
        
        # Средняя внутрикластерная дистанция
        mean_intra = np.mean(dists)
        
        # Нормируем на диапазон данных
        max_dist = np.max(dists)
        if max_dist < 1e-8:
            return 1.0
        
        # Чем меньше mean_intra относительно max_dist, тем выше coherence
        coherence = 1.0 - (mean_intra / max_dist)
        return float(coherence)
    
    elif method == "compactness":
        # Компактность = 1 / (1 + средний радиус от центроида)
        centroid = np.mean(embeddings, axis=0)
        radii = np.linalg.norm(embeddings - centroid, axis=-1)
        mean_radius = np.mean(radii)
        return float(1.0 / (1.0 + mean_radius))
    
    else:
        raise ValueError(f"Unknown method: {method}")


@dataclass
class ShamanReward:
    """Компоненты reward'а шамана."""
    energy_term: float
    rho_term: float
    cycle_term: float
    rarity_term: float
    struct_term: float
    total: float


def compute_shaman_reward(
    E_B: float,
    rho_B: float,
    cycle_persistence: float,
    rarity: float = 0.0,
    struct_score: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> ShamanReward:
    """
    Вычисляет интегральный reward J для шамана.
    
    J = w_E * E_B + w_rho * rho_B + w_cycle * cycle + w_rarity * rarity + w_struct * struct
    
    Args:
        E_B: энергия понимания (минимизируем)
        rho_B: самоинвариантность (максимизируем)
        cycle_persistence: цикличность γ (максимизируем)
        rarity: редкость симструктуры (максимизируем)
        struct_score: структурный показатель σ (максимизируем)
        weights: веса компонент
        
    Returns:
        ShamanReward с разбивкой по компонентам
    """
    if weights is None:
        weights = {
            "w_energy": -1.0,      # отрицательный - минимизируем E
            "w_rho": 2.0,
            "w_cycle": 1.0,
            "w_rarity": 0.5,
            "w_struct": 0.1
        }
    
    energy_term = weights["w_energy"] * E_B
    rho_term = weights["w_rho"] * rho_B
    cycle_term = weights["w_cycle"] * cycle_persistence
    rarity_term = weights["w_rarity"] * rarity
    struct_term = weights["w_struct"] * struct_score
    
    total = energy_term + rho_term + cycle_term + rarity_term + struct_term
    
    return ShamanReward(
        energy_term=energy_term,
        rho_term=rho_term,
        cycle_term=cycle_term,
        rarity_term=rarity_term,
        struct_term=struct_term,
        total=total
    )


def compute_d_P(
    cluster_A: np.ndarray,
    cluster_B: np.ndarray,
    method: str = "centroid_cosine"
) -> float:
    """
    Вычисляет расстояние d_P между двумя смысловыми кластерами.
    
    Это ключевая метрика эксперимента:
    d_P(s_A, s_B) должно уменьшаться после работы шамана.
    
    Args:
        cluster_A: [N_A, D] - точки кластера A (эталон)
        cluster_B: [N_B, D] - точки кластера B (целевой)
        method: метод вычисления
        
    Returns:
        d_P ∈ [0, 2] для cosine, [0, ∞) для euclidean
    """
    if len(cluster_A) == 0 or len(cluster_B) == 0:
        return float('inf')
    
    # Центроиды
    centroid_A = np.mean(cluster_A, axis=0)
    centroid_B = np.mean(cluster_B, axis=0)
    
    if method == "centroid_cosine":
        # d_P = 1 - cos(c_A, c_B)
        cos_sim = np.dot(centroid_A, centroid_B) / (
            np.linalg.norm(centroid_A) * np.linalg.norm(centroid_B) + 1e-8
        )
        return float(1.0 - cos_sim)
    
    elif method == "centroid_euclidean":
        return float(np.linalg.norm(centroid_A - centroid_B))
    
    elif method == "hausdorff":
        from scipy.spatial.distance import directed_hausdorff
        d1 = directed_hausdorff(cluster_A, cluster_B)[0]
        d2 = directed_hausdorff(cluster_B, cluster_A)[0]
        return float(max(d1, d2))
    
    elif method == "wasserstein":
        # 1D Wasserstein по каждой координате, усреднённый
        from scipy.stats import wasserstein_distance
        D = cluster_A.shape[1]
        w_dists = []
        for d in range(D):
            w = wasserstein_distance(cluster_A[:, d], cluster_B[:, d])
            w_dists.append(w)
        return float(np.mean(w_dists))
    
    elif method == "mmd":
        # Maximum Mean Discrepancy
        def rbf_kernel(X, Y, gamma=1.0):
            from scipy.spatial.distance import cdist
            dists = cdist(X, Y, 'sqeuclidean')
            return np.exp(-gamma * dists)
        
        K_xx = rbf_kernel(cluster_A, cluster_A).mean()
        K_yy = rbf_kernel(cluster_B, cluster_B).mean()
        K_xy = rbf_kernel(cluster_A, cluster_B).mean()
        
        mmd_sq = K_xx + K_yy - 2 * K_xy
        return float(np.sqrt(max(0, mmd_sq)))
    
    else:
        raise ValueError(f"Unknown method: {method}")


class MetricsTracker:
    """
    Отслеживает метрики в течение эксперимента.
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "d_P": [],
            "E_B": [],
            "rho_B": [],
            "cycle_persistence": [],
            "shaman_reward": []
        }
        
    def log(self, metrics: Dict[str, float]):
        """Добавляет метрики в историю."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Возвращает summary статистику."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last": float(values[-1])
                }
        return summary
    
    def get_trend(self, key: str, window: int = 10) -> float:
        """
        Возвращает тренд метрики (положительный = растёт).
        """
        values = self.history.get(key, [])
        if len(values) < window + 1:
            return 0.0
        
        recent = values[-window:]
        earlier = values[-(2*window):-window] if len(values) >= 2*window else values[:window]
        
        return float(np.mean(recent) - np.mean(earlier))
