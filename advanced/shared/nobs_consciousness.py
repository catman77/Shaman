# -*- coding: utf-8 -*-
"""
NOBS-Based Consciousness Space.

Сознание нейросети кодируется через NOBS (Natural Observation-Based Space):
- Активации сети → символическая ДНК (Σ = {S, P, I, Z, Ω, Λ})
- Стиль решения → морфизмы в категории C_n
- Характер сознания → свободная энергия F = E - T·S

Ключевая идея: Bitcoin данные - это универсальный "носитель" структуры.
Любой паттерн (включая сознание) можно спроецировать на эту структуру
и найти его резонансный аналог.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd


# Add paths for NOBS modules
_nobs_path = Path(__file__).parent.parent.parent / "src" / "nobs"
if str(_nobs_path) not in sys.path:
    sys.path.insert(0, str(_nobs_path.parent))

try:
    from nobs.symbolic_dna import SymbolicDNAEncoder, StructuralDNA, Symbol
    from nobs.category import CategoryCn, CategoryHierarchy
    from nobs.free_energy import FreeEnergyMinimizer
    from nobs.space import NOBSSpace, NOBSEmbedding
    NOBS_AVAILABLE = True
except ImportError:
    NOBS_AVAILABLE = False
    print("Warning: NOBS modules not found, using fallback")


class ConsciousnessStyle(Enum):
    """
    Стили сознания, закодированные как NOBS-паттерны.
    
    Каждый стиль определяется:
    - Доминирующие символы
    - Характерные морфизмы (переходы)
    - Энергетический профиль
    """
    # Аналитический профессор: структурированный, последовательный
    ANALYTICAL = "analytical"
    
    # Творческий решатель: скачки между идеями, высокая энтропия
    CREATIVE = "creative"
    
    # Интуитивный угадыватель: низкая структура, быстрые ответы
    INTUITIVE = "intuitive"
    
    # Педантичный инженер: детальный, проверяющий
    PEDANTIC = "pedantic"
    
    # Философский мыслитель: абстрактный, рефлексивный
    PHILOSOPHICAL = "philosophical"


@dataclass
class ConsciousnessSignature:
    """
    NOBS-сигнатура сознания.
    
    Это "отпечаток" способа мышления нейросети,
    извлечённый из её активаций.
    """
    # Имя стиля
    style_name: str
    
    # Символьное распределение (что доминирует в активациях)
    symbol_distribution: Dict[str, float]
    
    # Доминирующие переходы (морфизмы)
    dominant_morphisms: List[str]
    
    # Энергетический профиль
    free_energy: float
    entropy: float
    temperature: float
    
    # Полный вектор в NOBS-пространстве
    embedding_vector: np.ndarray
    
    # Характерные n-граммы (паттерны)
    characteristic_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'style_name': self.style_name,
            'symbol_distribution': self.symbol_distribution,
            'dominant_morphisms': self.dominant_morphisms,
            'free_energy': float(self.free_energy),
            'entropy': float(self.entropy),
            'temperature': float(self.temperature),
            'characteristic_patterns': self.characteristic_patterns,
            'embedding_norm': float(np.linalg.norm(self.embedding_vector))
        }
    
    def distance_to(self, other: 'ConsciousnessSignature') -> float:
        """Расстояние до другой сигнатуры в NOBS-пространстве."""
        return float(np.linalg.norm(self.embedding_vector - other.embedding_vector))
    
    def cosine_similarity(self, other: 'ConsciousnessSignature') -> float:
        """Косинусное сходство с другой сигнатурой."""
        norm_self = np.linalg.norm(self.embedding_vector)
        norm_other = np.linalg.norm(other.embedding_vector)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(np.dot(self.embedding_vector, other.embedding_vector) / (norm_self * norm_other))


@dataclass
class ConsciousnessConfig:
    """
    A priori конфигурация стиля сознания.
    
    Это РАЗДЕЛЯЕМОЕ знание между серверами A и B.
    Server A учится достигать этого профиля.
    Server B ищет его через резонанс.
    """
    name: str
    description: str
    
    # Целевое распределение символов
    target_symbols: Dict[str, float]
    
    # Целевые морфизмы (характерные переходы)
    target_morphisms: List[str]
    
    # Целевой энергетический диапазон
    target_energy_range: Tuple[float, float]
    
    # Целевая энтропия (структурированность vs хаотичность)
    target_entropy_range: Tuple[float, float]
    
    # Промпт-шаблон для формирования стиля
    prompt_template: str
    
    # Характерные паттерны ответов
    response_patterns: List[str]


# Априорные конфигурации стилей сознания
CONSCIOUSNESS_CONFIGS: Dict[str, ConsciousnessConfig] = {
    "analytical_professor": ConsciousnessConfig(
        name="analytical_professor",
        description="Методичный аналитический подход профессора математики",
        target_symbols={
            'S': 0.1,   # Немного снижений (критика)
            'P': 0.35,  # Много роста (прогресс к решению)
            'I': 0.25,  # Стабильные шаги
            'Z': 0.05,  # Мало пауз
            'Ω': 0.15,  # Ключевые точки (выводы)
            'Λ': 0.1    # Переходы между этапами
        },
        target_morphisms=['PPP', 'PPI', 'IPP', 'PIΩ', 'ΩPP'],
        target_energy_range=(-0.5, 0.5),
        target_entropy_range=(0.3, 0.6),
        prompt_template="""You are an analytical professor of mathematics.
        
When solving problems, you:
1. First, clearly state what you're looking for
2. Break the problem into logical steps
3. Show each calculation explicitly
4. Verify your answer
5. Conclude with a clear final answer

Your responses should be structured, methodical, and educational.""",
        response_patterns=[
            "Let's analyze",
            "Step 1:",
            "Step 2:",
            "Therefore,",
            "We can verify",
            "The answer is"
        ]
    ),
    
    "creative_solver": ConsciousnessConfig(
        name="creative_solver",
        description="Творческий решатель с нестандартным мышлением",
        target_symbols={
            'S': 0.15,  # Откаты (попробовал, не работает)
            'P': 0.25,  # Прорывы
            'I': 0.15,  # Стабильность
            'Z': 0.1,   # Паузы на размышление
            'Ω': 0.2,   # Много инсайтов!
            'Λ': 0.15   # Частые смены подхода
        },
        target_morphisms=['ΩΛP', 'PΩΛ', 'ΛPP', 'ZΩP', 'SΛP'],
        target_energy_range=(-0.8, 0.8),
        target_entropy_range=(0.5, 0.8),
        prompt_template="""You are a creative problem solver who thinks outside the box.

Your style:
- Start with an unusual perspective or analogy
- Make unexpected connections
- Try unconventional approaches
- Have "aha!" moments
- Your solutions are elegant and surprising

Be imaginative but still arrive at the correct answer.""",
        response_patterns=[
            "Interesting!",
            "What if we",
            "Think of it like",
            "Aha!",
            "Surprisingly,",
            "The elegant solution"
        ]
    ),
    
    "intuitive_guesser": ConsciousnessConfig(
        name="intuitive_guesser",
        description="Интуитивный угадыватель с быстрыми ответами",
        target_symbols={
            'S': 0.1,
            'P': 0.3,
            'I': 0.2,
            'Z': 0.15,  # Больше пауз (интуиция работает)
            'Ω': 0.15,  # Прямые ответы
            'Λ': 0.1
        },
        target_morphisms=['ZΩP', 'IΩI', 'PΩZ', 'ZZΩ', 'ΩII'],
        target_energy_range=(-0.3, 0.3),
        target_entropy_range=(0.4, 0.7),
        prompt_template="""You solve problems intuitively and quickly.

Your style:
- You sense the answer before working through details
- Your explanations are brief and to the point
- You trust your gut feeling
- You may give the answer first, then explain

Be concise but correct.""",
        response_patterns=[
            "I sense",
            "Quick answer:",
            "Intuitively,",
            "It feels like",
            "Simply put,"
        ]
    ),
    
    "pedantic_engineer": ConsciousnessConfig(
        name="pedantic_engineer",
        description="Педантичный инженер, проверяющий каждый шаг",
        target_symbols={
            'S': 0.2,   # Много проверок/откатов
            'P': 0.25,
            'I': 0.3,   # Стабильный прогресс
            'Z': 0.05,
            'Ω': 0.1,
            'Λ': 0.1
        },
        target_morphisms=['PIS', 'IPI', 'SIP', 'IIS', 'PPS'],
        target_energy_range=(-0.6, 0.4),
        target_entropy_range=(0.2, 0.5),
        prompt_template="""You are a pedantic engineer who checks everything twice.

Your approach:
1. State assumptions explicitly
2. Check units and dimensions
3. Verify each step before proceeding
4. Double-check the final answer
5. Consider edge cases

Be thorough and precise, catching any potential errors.""",
        response_patterns=[
            "Let me verify",
            "Checking:",
            "Assumption:",
            "Double-checking:",
            "Edge case:",
            "Confirmed:"
        ]
    ),
    
    "philosophical_thinker": ConsciousnessConfig(
        name="philosophical_thinker",
        description="Философский мыслитель, размышляющий о сути",
        target_symbols={
            'S': 0.1,
            'P': 0.2,
            'I': 0.2,
            'Z': 0.2,   # Много размышлений
            'Ω': 0.15,  # Глубокие выводы
            'Λ': 0.15   # Переходы между уровнями абстракции
        },
        target_morphisms=['ZZΩ', 'ΩΛZ', 'ZΛΩ', 'IZΛ', 'ΛZI'],
        target_energy_range=(-0.7, 0.7),
        target_entropy_range=(0.5, 0.75),
        prompt_template="""You are a philosophical thinker who contemplates the deeper meaning.

Your style:
- Pause to reflect on the nature of the problem
- Consider what the problem reveals about underlying principles
- Draw connections to broader concepts
- Your answer transcends the mere calculation

Be thoughtful and profound, while still providing the correct answer.""",
        response_patterns=[
            "Let us contemplate",
            "The essence of",
            "This reveals",
            "On a deeper level",
            "The fundamental nature",
            "Thus we see"
        ]
    )
}


def get_consciousness_config(name: str) -> ConsciousnessConfig:
    """Получить конфигурацию стиля сознания."""
    if name not in CONSCIOUSNESS_CONFIGS:
        raise ValueError(f"Unknown consciousness: {name}. Available: {list(CONSCIOUSNESS_CONFIGS.keys())}")
    return CONSCIOUSNESS_CONFIGS[name]


def list_consciousness_styles() -> List[str]:
    """Список доступных стилей."""
    return list(CONSCIOUSNESS_CONFIGS.keys())


class NOBSConsciousnessSpace:
    """
    Пространство сознаний на основе NOBS.
    
    Использует Bitcoin данные как базу для семантического пространства.
    Активации нейросети проецируются в это пространство.
    """
    
    DATA_PATH = Path(__file__).parent.parent.parent / "data" / "BTC_USDT_USDT-4h-futures.feather"
    
    def __init__(
        self,
        embedding_dim: int = 128,
        window_size: int = 100,
        temperature: float = 1.0
    ):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.temperature = temperature
        
        self._space: Optional[NOBSSpace] = None
        self._df: Optional[pd.DataFrame] = None
        self._fitted = False
        
    def fit(self, data_path: Optional[str] = None) -> 'NOBSConsciousnessSpace':
        """
        Инициализировать пространство на Bitcoin данных.
        """
        if not NOBS_AVAILABLE:
            raise RuntimeError("NOBS modules not available")
        
        path = Path(data_path) if data_path else self.DATA_PATH
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        print(f"Loading Bitcoin data from {path}...")
        self._df = pd.read_feather(str(path))
        print(f"Loaded {len(self._df)} candles")
        
        # Нормализуем названия колонок
        col_map = {}
        for col in self._df.columns:
            lower = col.lower()
            if 'open' in lower:
                col_map[col] = 'open'
            elif 'high' in lower:
                col_map[col] = 'high'
            elif 'low' in lower:
                col_map[col] = 'low'
            elif 'close' in lower:
                col_map[col] = 'close'
            elif 'volume' in lower:
                col_map[col] = 'volume'
        
        if col_map:
            self._df = self._df.rename(columns=col_map)
        
        # Создаём NOBS space
        self._space = NOBSSpace(
            max_depth=5,
            window_size=self.window_size,
            temperature=self.temperature,
            embedding_dim=self.embedding_dim
        )
        
        print("Fitting NOBS space...")
        self._space.fit(self._df)
        
        self._fitted = True
        print("NOBS Consciousness Space ready")
        
        return self
    
    def encode_activations(
        self,
        activations: np.ndarray,
        style_name: str
    ) -> ConsciousnessSignature:
        """
        Закодировать активации нейросети в сигнатуру сознания.
        
        Алгоритм:
        1. Нормализуем активации
        2. Преобразуем в "псевдо-OHLCV" (рассматриваем активации как временной ряд)
        3. Кодируем через NOBS
        4. Извлекаем характеристики
        
        Args:
            activations: Массив активаций нейросети [batch, seq, hidden]
            style_name: Название целевого стиля
            
        Returns:
            ConsciousnessSignature
        """
        if not self._fitted:
            raise ValueError("Space not fitted. Call fit() first.")
        
        # Flatten to 1D if needed
        if activations.ndim > 1:
            activations = activations.reshape(-1)
        
        # Create pseudo-OHLCV from activations
        # Treat activations as a time series
        window = min(100, len(activations) // 4)
        if window < 4:
            window = 4
        
        pseudo_ohlcv = self._activations_to_ohlcv(activations, window)
        
        # Encode through NOBS
        embedding = self._space.embed_from_ohlcv(pseudo_ohlcv)
        
        # Get config for target style
        config = get_consciousness_config(style_name)
        
        # Build signature
        return ConsciousnessSignature(
            style_name=style_name,
            symbol_distribution=embedding.symbol_distribution,
            dominant_morphisms=self._extract_dominant_morphisms(embedding.symbolic_string),
            free_energy=embedding.free_energy,
            entropy=embedding.entropy,
            temperature=embedding.temperature,
            embedding_vector=embedding.vector,
            characteristic_patterns=self._extract_patterns(embedding.symbolic_string, 3)
        )
    
    def _activations_to_ohlcv(self, activations: np.ndarray, window: int) -> pd.DataFrame:
        """
        Преобразовать активации в псевдо-OHLCV данные.
        
        Интерпретация:
        - open: значение в начале окна
        - high: максимум в окне
        - low: минимум в окне  
        - close: значение в конце окна
        - volume: вариация в окне
        """
        n = len(activations)
        num_candles = n // window
        
        if num_candles < 10:
            # Слишком мало данных, создаём минимальный набор
            num_candles = min(10, n)
            window = max(1, n // num_candles)
        
        data = []
        for i in range(num_candles):
            start = i * window
            end = min(start + window, n)
            chunk = activations[start:end]
            
            if len(chunk) == 0:
                continue
                
            data.append({
                'open': chunk[0],
                'high': chunk.max(),
                'low': chunk.min(),
                'close': chunk[-1],
                'volume': chunk.std() * 1000 + 1  # Вариация как объём
            })
        
        return pd.DataFrame(data)
    
    def _extract_dominant_morphisms(self, symbolic_string: str, top_k: int = 5) -> List[str]:
        """Извлечь доминирующие 3-граммы (морфизмы)."""
        from collections import Counter
        
        if len(symbolic_string) < 3:
            return []
        
        ngrams = [symbolic_string[i:i+3] for i in range(len(symbolic_string) - 2)]
        counts = Counter(ngrams)
        return [ng for ng, _ in counts.most_common(top_k)]
    
    def _extract_patterns(self, symbolic_string: str, n: int) -> List[str]:
        """Извлечь характерные паттерны."""
        from collections import Counter
        
        if len(symbolic_string) < n:
            return []
        
        ngrams = [symbolic_string[i:i+n] for i in range(len(symbolic_string) - n + 1)]
        counts = Counter(ngrams)
        
        # Return patterns that occur more than once
        return [ng for ng, count in counts.most_common(10) if count > 1]
    
    def find_resonance(
        self,
        target_config: ConsciousnessConfig,
        num_samples: int = 1000
    ) -> Tuple[ConsciousnessSignature, float]:
        """
        Найти резонанс с целевым сознанием через сэмплирование NOBS пространства.
        
        Шаман использует этот метод для поиска сознания БЕЗ данных от Server A.
        Он знает только конфигурацию (априорное знание) и ищет резонанс.
        
        Args:
            target_config: Целевая конфигурация сознания
            num_samples: Количество сэмплов для поиска
            
        Returns:
            (best_signature, resonance_score)
        """
        if not self._fitted:
            raise ValueError("Space not fitted")
        
        print(f"\nSearching for resonance with '{target_config.name}'...")
        print(f"Sampling {num_samples} windows from NOBS space...")
        
        # Target embedding from config
        target_vector = self._config_to_vector(target_config)
        
        best_signature = None
        best_score = -float('inf')
        
        # Sample random windows from Bitcoin data
        n = len(self._space._dna_sequence)
        
        for i in range(num_samples):
            # Random window
            start = np.random.randint(0, max(1, n - self.window_size))
            end = start + self.window_size
            
            # Embed window
            embedding = self._space.embed_window(start, min(end, n))
            
            # Compute resonance score
            score = self._compute_resonance(embedding, target_config, target_vector)
            
            if score > best_score:
                best_score = score
                best_signature = ConsciousnessSignature(
                    style_name=target_config.name,
                    symbol_distribution=embedding.symbol_distribution,
                    dominant_morphisms=self._extract_dominant_morphisms(embedding.symbolic_string),
                    free_energy=embedding.free_energy,
                    entropy=embedding.entropy,
                    temperature=embedding.temperature,
                    embedding_vector=embedding.vector,
                    characteristic_patterns=self._extract_patterns(embedding.symbolic_string, 3)
                )
                
                if (i + 1) % 100 == 0:
                    print(f"  Iteration {i+1}: best resonance = {best_score:.4f}")
        
        print(f"Best resonance found: {best_score:.4f}")
        
        return best_signature, best_score
    
    def _config_to_vector(self, config: ConsciousnessConfig) -> np.ndarray:
        """Преобразовать конфигурацию в целевой вектор."""
        vector = np.zeros(self.embedding_dim)
        
        # Symbol distribution part (first 6 dims)
        symbols = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
        for i, sym in enumerate(symbols):
            vector[i] = config.target_symbols.get(sym, 0.0)
        
        # Energy range (middle of range)
        target_energy = (config.target_energy_range[0] + config.target_energy_range[1]) / 2
        target_entropy = (config.target_entropy_range[0] + config.target_entropy_range[1]) / 2
        
        # Set in appropriate positions (see NOBSSpace._build_embedding_vector)
        vector[118] = target_energy
        vector[119] = target_entropy
        
        return vector
    
    def _compute_resonance(
        self,
        embedding: NOBSEmbedding,
        config: ConsciousnessConfig,
        target_vector: np.ndarray
    ) -> float:
        """
        Вычислить резонанс между эмбеддингом и целевой конфигурацией.
        
        Резонанс = взвешенная комбинация:
        - Сходство символьного распределения
        - Соответствие морфизмов
        - Энергетическое соответствие
        """
        score = 0.0
        
        # 1. Symbol distribution alignment (weight: 0.4)
        symbols = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
        sym_score = 0.0
        for sym in symbols:
            target = config.target_symbols.get(sym, 0.0)
            actual = embedding.symbol_distribution.get(sym, 0.0)
            sym_score += 1.0 - abs(target - actual)
        sym_score /= len(symbols)
        score += 0.4 * sym_score
        
        # 2. Morphism alignment (weight: 0.3)
        # Check if dominant morphisms match target patterns
        morph_score = 0.0
        dominant = self._extract_dominant_morphisms(embedding.symbolic_string, 5)
        for morph in config.target_morphisms:
            if morph in dominant:
                morph_score += 1.0
            elif any(m in morph or morph in m for m in dominant):
                morph_score += 0.5
        if config.target_morphisms:
            morph_score /= len(config.target_morphisms)
        score += 0.3 * morph_score
        
        # 3. Energy alignment (weight: 0.2)
        e_min, e_max = config.target_energy_range
        if e_min <= embedding.free_energy <= e_max:
            energy_score = 1.0
        else:
            dist = min(abs(embedding.free_energy - e_min), abs(embedding.free_energy - e_max))
            energy_score = np.exp(-dist)
        score += 0.2 * energy_score
        
        # 4. Entropy alignment (weight: 0.1)
        s_min, s_max = config.target_entropy_range
        if s_min <= embedding.entropy <= s_max:
            entropy_score = 1.0
        else:
            dist = min(abs(embedding.entropy - s_min), abs(embedding.entropy - s_max))
            entropy_score = np.exp(-dist)
        score += 0.1 * entropy_score
        
        return score


if __name__ == "__main__":
    # Test
    space = NOBSConsciousnessSpace()
    space.fit()
    
    print("\n" + "="*60)
    print("Testing consciousness styles")
    print("="*60)
    
    for style in list_consciousness_styles():
        config = get_consciousness_config(style)
        print(f"\n{style}:")
        print(f"  Target symbols: {config.target_symbols}")
        print(f"  Target morphisms: {config.target_morphisms[:3]}")
        
        # Find resonance
        signature, score = space.find_resonance(config, num_samples=500)
        print(f"  Resonance score: {score:.4f}")
        print(f"  Found symbols: {signature.symbol_distribution}")
        print(f"  Found morphisms: {signature.dominant_morphisms[:3]}")
