"""
Энкодеры для латентного пространства смыслов S

PhiEncoder - преобразует текст/эпизоды в векторные представления h ∈ ℝ^D
"""

import torch
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer


class PhiEncoder:
    """
    Φ-энкодер: преобразует тексты в латентное пространство S.
    
    Использует sentence-transformers для получения семантических эмбеддингов.
    Для MVP используем легкую модель all-MiniLM-L6-v2 (22M параметров).
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        normalize: bool = True
    ):
        """
        Args:
            model_name: название sentence-transformer модели
            device: устройство для вычислений
            normalize: нормализовать ли эмбеддинги (L2)
        """
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode_text(self, text: str) -> np.ndarray:
        """
        Кодирует один текст в вектор h ∈ ℝ^D.
        
        Args:
            text: входной текст
            
        Returns:
            numpy array размерности [D]
        """
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Кодирует батч текстов.
        
        Args:
            texts: список текстов
            
        Returns:
            numpy array размерности [N, D]
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embeddings
    
    def encode_episode(self, episode: dict) -> np.ndarray:
        """
        Кодирует эпизод (задача + ответ) в один вектор.
        
        Args:
            episode: словарь с ключами 'task', 'response', опционально 'cot'
            
        Returns:
            numpy array размерности [D]
        """
        # Объединяем все части эпизода
        parts = []
        if 'task' in episode:
            parts.append(f"Task: {episode['task']}")
        if 'cot' in episode and episode['cot']:
            parts.append(f"Reasoning: {episode['cot']}")
        if 'response' in episode:
            parts.append(f"Response: {episode['response']}")
            
        combined_text = "\n".join(parts)
        return self.encode_text(combined_text)
    
    def encode_episodes_batch(self, episodes: List[dict]) -> np.ndarray:
        """
        Кодирует батч эпизодов.
        
        Args:
            episodes: список эпизодов
            
        Returns:
            numpy array размерности [N, D]
        """
        texts = []
        for ep in episodes:
            parts = []
            if 'task' in ep:
                parts.append(f"Task: {ep['task']}")
            if 'cot' in ep and ep['cot']:
                parts.append(f"Reasoning: {ep['cot']}")
            if 'response' in ep:
                parts.append(f"Response: {ep['response']}")
            texts.append("\n".join(parts))
            
        return self.encode_batch(texts)
    
    def compute_distance(
        self, 
        h1: np.ndarray, 
        h2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Вычисляет расстояние d_P между двумя точками в S.
        
        Args:
            h1, h2: векторы в латентном пространстве
            metric: тип метрики ("cosine", "euclidean")
            
        Returns:
            расстояние d_P(h1, h2)
        """
        if metric == "cosine":
            # d_P = 1 - cosine_similarity
            cos_sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
            return 1.0 - cos_sim
        elif metric == "euclidean":
            return float(np.linalg.norm(h1 - h2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_cluster_distance(
        self,
        cluster1: np.ndarray,
        cluster2: np.ndarray,
        method: str = "centroid"
    ) -> float:
        """
        Вычисляет расстояние между кластерами точек в S.
        
        Args:
            cluster1: точки первого кластера [N1, D]
            cluster2: точки второго кластера [N2, D]
            method: метод ("centroid", "hausdorff", "mmd")
            
        Returns:
            расстояние между кластерами
        """
        if method == "centroid":
            c1 = np.mean(cluster1, axis=0)
            c2 = np.mean(cluster2, axis=0)
            return self.compute_distance(c1, c2)
        
        elif method == "hausdorff":
            # Hausdorff distance
            from scipy.spatial.distance import directed_hausdorff
            d1 = directed_hausdorff(cluster1, cluster2)[0]
            d2 = directed_hausdorff(cluster2, cluster1)[0]
            return max(d1, d2)
        
        elif method == "mmd":
            # Maximum Mean Discrepancy (упрощённая версия)
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
            def rbf_kernel(X, Y, gamma=1.0):
                from scipy.spatial.distance import cdist
                dists = cdist(X, Y, 'sqeuclidean')
                return np.exp(-gamma * dists)
            
            K_xx = rbf_kernel(cluster1, cluster1).mean()
            K_yy = rbf_kernel(cluster2, cluster2).mean()
            K_xy = rbf_kernel(cluster1, cluster2).mean()
            
            mmd_sq = K_xx + K_yy - 2 * K_xy
            return float(np.sqrt(max(0, mmd_sq)))
        
        else:
            raise ValueError(f"Unknown method: {method}")


class LatentTrajectory:
    """
    Хранит и анализирует траекторию точек в латентном пространстве S.
    
    Используется для отслеживания динамики агента и вычисления метрик
    типа "скорости" движения в S, устойчивости и т.д.
    """
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.points: List[np.ndarray] = []
        self.timestamps: List[int] = []
        
    def add_point(self, h: np.ndarray, t: Optional[int] = None):
        """Добавляет точку в траекторию."""
        self.points.append(h.copy())
        self.timestamps.append(t if t is not None else len(self.points))
        
    def get_points_array(self) -> np.ndarray:
        """Возвращает все точки как numpy array [T, D]."""
        if not self.points:
            return np.zeros((0, self.embedding_dim))
        return np.stack(self.points, axis=0)
    
    def compute_velocity(self) -> np.ndarray:
        """
        Вычисляет "скорость" движения в S: v_t = h_{t+1} - h_t.
        
        Returns:
            numpy array [T-1, D] скоростей
        """
        points = self.get_points_array()
        if len(points) < 2:
            return np.zeros((0, self.embedding_dim))
        return points[1:] - points[:-1]
    
    def compute_total_path_length(self) -> float:
        """Вычисляет общую длину пути в S."""
        vel = self.compute_velocity()
        if len(vel) == 0:
            return 0.0
        return float(np.sum(np.linalg.norm(vel, axis=1)))
    
    def compute_displacement(self) -> float:
        """Вычисляет смещение от начала до конца."""
        points = self.get_points_array()
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(points[-1] - points[0]))
    
    def compute_tortuosity(self) -> float:
        """
        Вычисляет извилистость траектории.
        tortuosity = path_length / displacement
        Высокая извилистость = много петель/циклов.
        """
        path_len = self.compute_total_path_length()
        displacement = self.compute_displacement()
        if displacement < 1e-8:
            return float('inf') if path_len > 1e-8 else 1.0
        return path_len / displacement
    
    def get_centroid(self) -> np.ndarray:
        """Возвращает центроид траектории."""
        points = self.get_points_array()
        if len(points) == 0:
            return np.zeros(self.embedding_dim)
        return np.mean(points, axis=0)
    
    def get_variance(self) -> float:
        """Вычисляет дисперсию точек вокруг центроида."""
        points = self.get_points_array()
        if len(points) == 0:
            return 0.0
        centroid = self.get_centroid()
        diffs = points - centroid
        return float(np.mean(np.sum(diffs ** 2, axis=1)))
    
    def clear(self):
        """Очищает траекторию."""
        self.points = []
        self.timestamps = []
