"""
NOBS Semantic Space for Text-based Neural Networks

Адаптация NOBS-пространства для работы с текстовыми данными нейросетей.
Символическая ДНК строится на основе паттернов в текстовых ответах модели.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re
from collections import Counter


class TextSymbol(Enum):
    """
    Символы для текстовой "ДНК сознания".
    
    Отражают структурные и семантические паттерны в текстовых ответах.
    """
    P = "P"   # Progress - продвижение, развитие мысли
    S = "S"   # Stability - стабильность, повторение, подтверждение
    I = "I"   # Information - новая информация, факты
    Z = "Z"   # Zigzag - колебания, неопределённость
    Ω = "Ω"   # Omega - переход, смена состояния
    Λ = "Λ"   # Lambda - прорыв, инсайт


@dataclass
class TextDNA:
    """Один элемент текстовой ДНК"""
    symbol: TextSymbol
    confidence: float
    context: str = ""
    
    def to_symbol(self) -> str:
        return self.symbol.value


class TextPatternAnalyzer:
    """
    Анализатор текстовых паттернов.
    
    Преобразует текстовые ответы модели в символическую ДНК.
    """
    
    # Паттерны для распознавания символов
    PROGRESS_PATTERNS = [
        r'\bthen\b', r'\bnext\b', r'\btherefore\b', r'\bconsequently\b',
        r'\bstep\s+\d+', r'\bfirst\b.*\bthen\b', r'\b→\b', r'\b->\b',
        r'\bleads to\b', r'\bresults in\b', r'\bso\b'
    ]
    
    STABILITY_PATTERNS = [
        r'\bsame\b', r'\bagain\b', r'\bstill\b', r'\bremains\b',
        r'\balso\b', r'\bsimilarly\b', r'\bas before\b', r'\bconsistent\b'
    ]
    
    INFORMATION_PATTERNS = [
        r'\bbecause\b', r'\bsince\b', r'\bfact\b', r'\bdata\b',
        r'\bevidence\b', r'\bshows?\b', r'\bindicates?\b', r'\bproves?\b',
        r'\d+\s*[+\-*/=]\s*\d+', r'\$?\d+\.?\d*'  # Math expressions
    ]
    
    ZIGZAG_PATTERNS = [
        r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\byet\b',
        r'\bon the other hand\b', r'\bcontrary\b', r'\buncertain\b',
        r'\bmaybe\b', r'\bperhaps\b', r'\bor\b.*\bor\b'
    ]
    
    TRANSITION_PATTERNS = [
        r'\bnow\b', r'\blet\'?s\b', r'\bconsider\b', r'\bmoving on\b',
        r'\bfinally\b', r'\bin conclusion\b', r'\bto summarize\b',
        r'\boverall\b', r'\bin summary\b'
    ]
    
    BREAKTHROUGH_PATTERNS = [
        r'\b!+\b', r'\beureka\b', r'\baha\b', r'\bkey insight\b',
        r'\bimportant\b', r'\bcrucial\b', r'\bfundamental\b',
        r'\bsolution\b', r'\banswer\b', r'\bresult\b'
    ]
    
    def __init__(self):
        self.compiled_patterns = {
            TextSymbol.P: [re.compile(p, re.IGNORECASE) for p in self.PROGRESS_PATTERNS],
            TextSymbol.S: [re.compile(p, re.IGNORECASE) for p in self.STABILITY_PATTERNS],
            TextSymbol.I: [re.compile(p, re.IGNORECASE) for p in self.INFORMATION_PATTERNS],
            TextSymbol.Z: [re.compile(p, re.IGNORECASE) for p in self.ZIGZAG_PATTERNS],
            TextSymbol.Ω: [re.compile(p, re.IGNORECASE) for p in self.TRANSITION_PATTERNS],
            TextSymbol.Λ: [re.compile(p, re.IGNORECASE) for p in self.BREAKTHROUGH_PATTERNS],
        }
    
    def analyze_sentence(self, sentence: str) -> TextDNA:
        """Проанализировать предложение и вернуть символ ДНК"""
        scores = {}
        
        for symbol, patterns in self.compiled_patterns.items():
            count = sum(1 for p in patterns if p.search(sentence))
            scores[symbol] = count
        
        # Если нет паттернов, используем эвристики на основе длины и структуры
        if max(scores.values()) == 0:
            if len(sentence) < 20:
                symbol = TextSymbol.S  # Короткие - стабильность
            elif '?' in sentence:
                symbol = TextSymbol.Z  # Вопросы - неопределённость
            elif sentence.strip().endswith('.'):
                symbol = TextSymbol.I  # Утверждения - информация
            else:
                symbol = TextSymbol.P  # По умолчанию - прогресс
            confidence = 0.3
        else:
            symbol = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[symbol] / total if total > 0 else 0.5
        
        return TextDNA(symbol=symbol, confidence=confidence, context=sentence[:50])
    
    def extract_dna_sequence(self, text: str) -> List[TextDNA]:
        """Извлечь последовательность ДНК из текста"""
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        dna_sequence = []
        for sentence in sentences:
            dna = self.analyze_sentence(sentence)
            dna_sequence.append(dna)
        
        return dna_sequence


@dataclass
class TextEmbedding:
    """Эмбеддинг текстового окна"""
    vector: np.ndarray
    dna_sequence: str
    symbol_distribution: Dict[str, float]
    morphism_distribution: Dict[str, float]
    free_energy: float


class TextNOBSSpace:
    """
    NOBS-пространство для текстовых данных.
    
    Строит семантическое пространство на основе:
    - Символической ДНК текстовых ответов
    - Категориальной иерархии (морфизмы)
    - Свободной энергии (сложность/структурированность)
    """
    
    def __init__(self, embedding_dim: int = 128, window_size: int = 5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size  # В предложениях
        self.analyzer = TextPatternAnalyzer()
        
        self._dna_sequences: Dict[str, List[TextDNA]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Символьные базисные векторы
        self._symbol_bases = self._init_symbol_bases()
        
    def _init_symbol_bases(self) -> Dict[TextSymbol, np.ndarray]:
        """Инициализировать базисные векторы для символов"""
        np.random.seed(42)  # Для воспроизводимости
        bases = {}
        
        # Каждый символ имеет свой характерный паттерн
        for i, symbol in enumerate(TextSymbol):
            base = np.zeros(self.embedding_dim)
            # Основная компонента
            base[i * 20:(i + 1) * 20] = np.random.randn(20)
            # Нормализуем
            base = base / (np.linalg.norm(base) + 1e-8)
            bases[symbol] = base
        
        return bases
    
    def process_text(self, text: str, text_id: str) -> List[TextDNA]:
        """Обработать текст и сохранить ДНК-последовательность"""
        dna_seq = self.analyzer.extract_dna_sequence(text)
        self._dna_sequences[text_id] = dna_seq
        return dna_seq
    
    def embed_sequence(self, dna_sequence: List[TextDNA]) -> TextEmbedding:
        """Встроить ДНК-последовательность в векторное пространство"""
        if not dna_sequence:
            return TextEmbedding(
                vector=np.zeros(self.embedding_dim),
                dna_sequence="",
                symbol_distribution={},
                morphism_distribution={},
                free_energy=0.0
            )
        
        # 1. Символьная строка
        dna_str = ''.join(dna.to_symbol() for dna in dna_sequence)
        
        # 2. Распределение символов
        symbol_counts = Counter(dna.symbol for dna in dna_sequence)
        total = len(dna_sequence)
        symbol_dist = {s.value: symbol_counts.get(s, 0) / total for s in TextSymbol}
        
        # 3. Морфизмы (биграммы)
        morphisms = [dna_str[i:i+2] for i in range(len(dna_str) - 1)]
        morphism_counts = Counter(morphisms)
        morphism_total = len(morphisms) if morphisms else 1
        morphism_dist = {m: c / morphism_total for m, c in morphism_counts.items()}
        
        # 4. Вектор эмбеддинга
        vector = np.zeros(self.embedding_dim)
        
        # Добавляем взвешенную сумму базисных векторов
        for dna in dna_sequence:
            vector += dna.confidence * self._symbol_bases[dna.symbol]
        
        # Добавляем морфизмные компоненты
        for morphism, count in morphism_counts.items():
            if len(morphism) == 2:
                s1, s2 = TextSymbol(morphism[0]), TextSymbol(morphism[1])
                # Комбинируем базисы
                morphism_vec = self._symbol_bases[s1] * self._symbol_bases[s2]
                vector += (count / morphism_total) * morphism_vec
        
        # Нормализуем
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm * np.sqrt(self.embedding_dim)  # Сохраняем масштаб
        
        # 5. Свободная энергия (оценка сложности/структурированности)
        # Высокая энтропия символов = высокая энергия
        symbol_probs = np.array([symbol_dist.get(s.value, 0) for s in TextSymbol])
        symbol_probs = symbol_probs[symbol_probs > 0]
        entropy = -np.sum(symbol_probs * np.log(symbol_probs + 1e-10))
        
        # Энергия = энтропия + вариация уверенности
        confidence_var = np.var([dna.confidence for dna in dna_sequence])
        free_energy = entropy + confidence_var * 5
        
        return TextEmbedding(
            vector=vector,
            dna_sequence=dna_str,
            symbol_distribution=symbol_dist,
            morphism_distribution=morphism_dist,
            free_energy=free_energy
        )
    
    def embed_text(self, text: str) -> TextEmbedding:
        """Встроить текст напрямую"""
        dna_seq = self.analyzer.extract_dna_sequence(text)
        return self.embed_sequence(dna_seq)
    
    def compute_distance(self, emb1: TextEmbedding, emb2: TextEmbedding) -> float:
        """Вычислить расстояние между эмбеддингами"""
        # Косинусное расстояние
        cos_sim = np.dot(emb1.vector, emb2.vector) / (
            np.linalg.norm(emb1.vector) * np.linalg.norm(emb2.vector) + 1e-10
        )
        cos_dist = 1 - cos_sim
        
        # Расстояние по распределению символов
        symbol_dist = 0
        for s in TextSymbol:
            p1 = emb1.symbol_distribution.get(s.value, 0)
            p2 = emb2.symbol_distribution.get(s.value, 0)
            symbol_dist += (p1 - p2) ** 2
        symbol_dist = np.sqrt(symbol_dist)
        
        # Комбинируем
        return 0.6 * cos_dist + 0.4 * symbol_dist
    
    def compute_resonance(self, emb1: TextEmbedding, emb2: TextEmbedding) -> float:
        """
        Вычислить степень резонанса между эмбеддингами.
        
        Высокий резонанс = похожая структура сознания.
        """
        distance = self.compute_distance(emb1, emb2)
        
        # Резонанс = 1 / (1 + distance)
        resonance = 1 / (1 + distance)
        
        # Бонус за совпадение доминантных морфизмов
        top_morphisms_1 = set(sorted(emb1.morphism_distribution.keys(), 
                                     key=emb1.morphism_distribution.get, 
                                     reverse=True)[:3])
        top_morphisms_2 = set(sorted(emb2.morphism_distribution.keys(),
                                     key=emb2.morphism_distribution.get,
                                     reverse=True)[:3])
        
        morphism_overlap = len(top_morphisms_1 & top_morphisms_2) / 3
        
        return 0.7 * resonance + 0.3 * morphism_overlap


@dataclass 
class ConsciousnessSignature:
    """
    Подпись сознания - компактное представление стиля решения задач.
    
    Это то, что может быть использовано для поиска резонанса.
    """
    # Усреднённый эмбеддинг
    mean_embedding: np.ndarray
    # Распределение символов
    symbol_distribution: Dict[str, float]
    # Доминантные морфизмы
    dominant_morphisms: List[str]
    # Средняя свободная энергия
    mean_energy: float
    # Характерная ДНК-последовательность
    characteristic_dna: str
    
    def to_dict(self) -> dict:
        return {
            "symbol_distribution": self.symbol_distribution,
            "dominant_morphisms": self.dominant_morphisms,
            "mean_energy": self.mean_energy,
            "characteristic_dna": self.characteristic_dna
        }


class ConsciousnessExtractor:
    """
    Экстрактор сознания из текстовых ответов модели.
    
    Анализирует множество ответов и формирует "подпись сознания".
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.space = TextNOBSSpace(embedding_dim=embedding_dim)
        self._embeddings: List[TextEmbedding] = []
        
    def add_response(self, response: str):
        """Добавить ответ модели для анализа"""
        emb = self.space.embed_text(response)
        self._embeddings.append(emb)
    
    def extract_signature(self) -> ConsciousnessSignature:
        """Извлечь подпись сознания из всех ответов"""
        if not self._embeddings:
            raise ValueError("No responses added. Call add_response first.")
        
        # 1. Средний эмбеддинг
        mean_vec = np.mean([e.vector for e in self._embeddings], axis=0)
        
        # 2. Агрегированное распределение символов
        symbol_dist = {}
        for s in TextSymbol:
            symbol_dist[s.value] = np.mean([
                e.symbol_distribution.get(s.value, 0) 
                for e in self._embeddings
            ])
        
        # 3. Доминантные морфизмы
        all_morphisms = Counter()
        for e in self._embeddings:
            all_morphisms.update(e.morphism_distribution)
        dominant = [m for m, _ in all_morphisms.most_common(5)]
        
        # 4. Средняя энергия
        mean_energy = np.mean([e.free_energy for e in self._embeddings])
        
        # 5. Характерная ДНК (самая частая подпоследовательность)
        all_dna = ''.join(e.dna_sequence for e in self._embeddings)
        # Находим самые частые 3-граммы
        trigrams = [all_dna[i:i+3] for i in range(len(all_dna) - 2)]
        common_trigram = Counter(trigrams).most_common(1)
        char_dna = common_trigram[0][0] if common_trigram else "PSI"
        
        return ConsciousnessSignature(
            mean_embedding=mean_vec,
            symbol_distribution=symbol_dist,
            dominant_morphisms=dominant,
            mean_energy=mean_energy,
            characteristic_dna=char_dna
        )
    
    def compute_style_score(self, signature: ConsciousnessSignature,
                            target_style) -> float:
        """
        Вычислить насколько подпись соответствует целевому стилю сознания.
        
        target_style - из shared/consciousness.py
        """
        score = 0.0
        
        # 1. Совпадение доминантных символов (40%)
        if target_style.dominant_symbols:
            target_dom = target_style.dominant_symbols[0]
            actual_dom = max(signature.symbol_distribution.keys(),
                           key=lambda k: signature.symbol_distribution[k])
            if actual_dom == target_dom:
                score += 0.4
            elif actual_dom in target_style.dominant_symbols:
                score += 0.2
        
        # 2. Совпадение морфизмов (40%)
        if target_style.characteristic_morphisms:
            overlap = len(set(signature.dominant_morphisms) & 
                         set(target_style.characteristic_morphisms))
            max_overlap = min(len(signature.dominant_morphisms),
                            len(target_style.characteristic_morphisms))
            if max_overlap > 0:
                score += 0.4 * (overlap / max_overlap)
        
        # 3. Совпадение энергетического профиля (20%)
        if target_style.energy_profile == "low" and signature.mean_energy < 1.0:
            score += 0.2
        elif target_style.energy_profile == "high" and signature.mean_energy > 1.5:
            score += 0.2
        elif target_style.energy_profile == "medium" and 1.0 <= signature.mean_energy <= 1.5:
            score += 0.2
        
        return score
