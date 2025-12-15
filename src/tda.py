"""
TDA модуль - Топологический анализ данных для выявления циклов γ в S

Использует персистентную гомологию для:
- Вычисления числа Бетти β₁ (число независимых циклов)
- Оценки устойчивости (persistence) циклов
- Детекции структурных инвариантов
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PersistenceDiagram:
    """Диаграмма персистентности."""
    dimension: int
    birth_death_pairs: List[Tuple[float, float]]
    
    def get_persistence_values(self) -> List[float]:
        """Возвращает значения персистентности (death - birth)."""
        return [d - b for b, d in self.birth_death_pairs if d != float('inf')]
    
    def get_total_persistence(self) -> float:
        """Суммарная персистентность."""
        return sum(self.get_persistence_values())
    
    def get_max_persistence(self) -> float:
        """Максимальная персистентность."""
        pers = self.get_persistence_values()
        return max(pers) if pers else 0.0


class TDAModule:
    """
    Модуль топологического анализа данных.
    
    Вычисляет персистентную гомологию облака точек в латентном пространстве
    для выявления топологических структур (связные компоненты, циклы).
    """
    
    def __init__(
        self,
        max_dimension: int = 1,
        max_edge_length: float = 2.0,
        backend: str = "ripser"
    ):
        """
        Args:
            max_dimension: максимальная размерность гомологии (1 = циклы)
            max_edge_length: максимальная длина ребра для комплекса Рипса
            backend: библиотека для вычислений ("ripser" или "gudhi")
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.backend = backend
        
        # Ленивая загрузка бэкенда
        self._ripser = None
        self._gudhi = None
        
    def _get_ripser(self):
        """Ленивая загрузка ripser."""
        if self._ripser is None:
            try:
                import ripser
                self._ripser = ripser
            except ImportError:
                raise ImportError("ripser not installed. Run: pip install ripser")
        return self._ripser
    
    def _get_gudhi(self):
        """Ленивая загрузка gudhi."""
        if self._gudhi is None:
            try:
                import gudhi
                self._gudhi = gudhi
            except ImportError:
                raise ImportError("gudhi not installed. Run: pip install gudhi")
        return self._gudhi
    
    def compute_persistence(
        self, 
        points: np.ndarray
    ) -> Dict[int, PersistenceDiagram]:
        """
        Вычисляет персистентную гомологию для облака точек.
        
        Args:
            points: numpy array [N, D] - точки в латентном пространстве
            
        Returns:
            словарь {dimension: PersistenceDiagram}
        """
        if len(points) < 2:
            # Недостаточно точек для анализа
            return {
                d: PersistenceDiagram(dimension=d, birth_death_pairs=[])
                for d in range(self.max_dimension + 1)
            }
        
        if self.backend == "ripser":
            return self._compute_with_ripser(points)
        elif self.backend == "gudhi":
            return self._compute_with_gudhi(points)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _compute_with_ripser(
        self, 
        points: np.ndarray
    ) -> Dict[int, PersistenceDiagram]:
        """Вычисление через ripser."""
        ripser = self._get_ripser()
        
        result = ripser.ripser(
            points,
            maxdim=self.max_dimension,
            thresh=self.max_edge_length
        )
        
        diagrams = {}
        for dim in range(self.max_dimension + 1):
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                pairs = [(float(b), float(d)) for b, d in dgm]
            else:
                pairs = []
            diagrams[dim] = PersistenceDiagram(dimension=dim, birth_death_pairs=pairs)
        
        return diagrams
    
    def _compute_with_gudhi(
        self, 
        points: np.ndarray
    ) -> Dict[int, PersistenceDiagram]:
        """Вычисление через gudhi."""
        gudhi = self._get_gudhi()
        
        # Создаём комплекс Рипса
        rips = gudhi.RipsComplex(
            points=points.tolist(),
            max_edge_length=self.max_edge_length
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        
        # Вычисляем персистентность
        simplex_tree.compute_persistence()
        
        # Собираем результаты по размерностям
        diagrams = {d: [] for d in range(self.max_dimension + 1)}
        
        for dim, (birth, death) in simplex_tree.persistence():
            if dim <= self.max_dimension:
                diagrams[dim].append((float(birth), float(death)))
        
        return {
            dim: PersistenceDiagram(dimension=dim, birth_death_pairs=pairs)
            for dim, pairs in diagrams.items()
        }
    
    def compute_betti_numbers(
        self, 
        diagrams: Dict[int, PersistenceDiagram],
        threshold: float = 0.0
    ) -> Dict[int, int]:
        """
        Вычисляет числа Бетти из диаграмм персистентности.
        
        β₀ = число связных компонент
        β₁ = число независимых циклов (петель)
        
        Args:
            diagrams: диаграммы персистентности
            threshold: минимальная персистентность для учёта
            
        Returns:
            словарь {dimension: betti_number}
        """
        betti = {}
        for dim, dgm in diagrams.items():
            # Считаем только пары с достаточной персистентностью
            count = sum(
                1 for b, d in dgm.birth_death_pairs
                if (d - b) > threshold or d == float('inf')
            )
            betti[dim] = count
        return betti
    
    def cycle_persistence_score(
        self,
        diagrams: Dict[int, PersistenceDiagram],
        dim: int = 1,
        top_k: Optional[int] = None
    ) -> float:
        """
        Вычисляет суммарную персистентность циклов размерности dim.
        
        Высокий score означает наличие устойчивых, значимых циклов
        в латентном пространстве (γ-структуры).
        
        Args:
            diagrams: диаграммы персистентности
            dim: размерность (1 = 1-циклы / петли)
            top_k: учитывать только top_k самых персистентных циклов
            
        Returns:
            суммарная персистентность
        """
        if dim not in diagrams:
            return 0.0
        
        pers_values = diagrams[dim].get_persistence_values()
        
        if top_k is not None and len(pers_values) > top_k:
            pers_values = sorted(pers_values, reverse=True)[:top_k]
        
        return sum(pers_values)
    
    def analyze_trajectory(
        self,
        points: np.ndarray,
        window_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Полный анализ траектории в S.
        
        Args:
            points: numpy array [T, D] - точки траектории
            window_size: если задан, анализируем только последние window_size точек
            
        Returns:
            словарь с метриками:
            - beta_0: число связных компонент
            - beta_1: число циклов
            - total_persistence_1: суммарная персистентность 1-циклов
            - max_persistence_1: максимальная персистентность цикла
        """
        if window_size is not None and len(points) > window_size:
            points = points[-window_size:]
        
        diagrams = self.compute_persistence(points)
        betti = self.compute_betti_numbers(diagrams)
        
        return {
            'beta_0': betti.get(0, 0),
            'beta_1': betti.get(1, 0),
            'total_persistence_1': diagrams[1].get_total_persistence() if 1 in diagrams else 0.0,
            'max_persistence_1': diagrams[1].get_max_persistence() if 1 in diagrams else 0.0,
            'num_points': len(points)
        }


def compute_cycle_richness(
    tda: TDAModule,
    points: np.ndarray,
    num_samples: int = 5,
    sample_size: int = 20
) -> float:
    """
    Оценивает "богатство циклов" через bootstrap-сэмплирование.
    
    Стабильно высокий β₁ при разных подвыборках = устойчивая циклическая структура.
    
    Args:
        tda: экземпляр TDAModule
        points: все точки траектории
        num_samples: число bootstrap-выборок
        sample_size: размер каждой выборки
        
    Returns:
        средний β₁ по выборкам
    """
    if len(points) < sample_size:
        sample_size = len(points)
    if sample_size < 3:
        return 0.0
    
    beta1_values = []
    for _ in range(num_samples):
        indices = np.random.choice(len(points), size=sample_size, replace=False)
        sample = points[indices]
        
        diagrams = tda.compute_persistence(sample)
        betti = tda.compute_betti_numbers(diagrams)
        beta1_values.append(betti.get(1, 0))
    
    return float(np.mean(beta1_values))
