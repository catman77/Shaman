"""
Server A - Agent A (Learner)

Сервер А обучается распознавать определённый "смысл" в своих данных.
Он НЕ передаёт никаких данных серверу Б - ни сырых, ни метаданных.

Входные данные:
- meaning_config.json - конфиг со смыслом для обучения
- Локальные данные (OHLCV)

Выходные данные:
- Обученная модель (локально)
- Метрики обучения (для самоконтроля)

НИКАКОГО ЭКСПОРТА ДАННЫХ ДЛЯ СЕРВЕРА Б!
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
import pandas as pd

# Add parent for shared imports
sys.path.insert(0, str(Path(__file__).parent))

from nobs.space import NOBSSpace
from shared.meanings import MeaningConfig, get_meaning_by_name


@dataclass
class TrainingMetrics:
    """Метрики обучения (только для внутреннего использования Server A)"""
    meaning_name: str
    samples_found: int
    mean_distance_to_expected: float
    symbol_match_score: float
    morphism_match_score: float
    energy_match: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


class MeaningLearner:
    """
    Обучение распознаванию смысла.
    
    Модель учится находить участки данных, соответствующие 
    заданному семантическому паттерну.
    """
    
    def __init__(self, meaning: MeaningConfig, window_size: int = 100):
        self.meaning = meaning
        self.window_size = window_size
        self.space: Optional[NOBSSpace] = None
        self.dna_sequence: str = ""
        
        # Внутренние данные (НИКУДА НЕ ПЕРЕДАЮТСЯ)
        self._learned_windows: List[int] = []
        self._learned_embeddings: Optional[np.ndarray] = None
        
    def build_space(self, ohlcv: np.ndarray):
        """Построить NOBS пространство из данных"""
        # Convert to DataFrame for NOBSSpace
        df = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume'])
        
        self.space = NOBSSpace(
            max_depth=5,
            window_size=self.window_size,
            embedding_dim=128
        )
        self.space.fit(df)
        
        # Get DNA sequence as string
        self.dna_sequence = ''.join([dna.to_symbol() for dna in self.space._dna_sequence])
        print(f"Built NOBS space: {len(self.dna_sequence)} symbols")
    
    def get_window_embedding(self, start_idx: int, size: int) -> np.ndarray:
        """Get embedding for a window"""
        end_idx = min(start_idx + size, len(self.space._dna_sequence))
        emb = self.space.embed_window(start_idx, end_idx)
        return emb.vector
        
    def _compute_symbol_score(self, window_symbols: str) -> float:
        """Оценка соответствия символов ожидаемым"""
        if not self.meaning.expected_symbols:
            return 0.5
            
        # Подсчёт частот
        freq = {}
        for s in window_symbols:
            freq[s] = freq.get(s, 0) + 1
        total = len(window_symbols)
        
        score = 0.0
        weights = {"dominant": 0.5, "secondary": 0.3, "rare": 0.2}
        
        for role, symbol in self.meaning.expected_symbols.items():
            if role in weights:
                actual_freq = freq.get(symbol, 0) / total if total > 0 else 0
                if role == "dominant":
                    # Доминантный символ должен быть самым частым
                    max_freq = max(freq.values()) / total if freq else 0
                    score += weights[role] * (1.0 if actual_freq == max_freq else 0.5)
                elif role == "rare":
                    # Редкий символ должен быть нечастым
                    score += weights[role] * (1.0 if actual_freq < 0.15 else 0.3)
                else:
                    # Вторичный - средняя частота
                    score += weights[role] * min(1.0, actual_freq * 3)
                    
        return score
        
    def _compute_morphism_score(self, window_start: int) -> float:
        """Оценка соответствия морфизмов ожидаемым"""
        if not self.meaning.expected_morphisms or self.space is None:
            return 0.5
            
        # Получаем морфизмы из категории C2
        window_end = min(window_start + self.window_size, len(self.dna_sequence))
        window_dna = self.dna_sequence[window_start:window_end]
        
        # Считаем биграммы
        bigrams = [window_dna[i:i+2] for i in range(len(window_dna)-1)]
        bigram_set = set(bigrams)
        
        # Сколько ожидаемых морфизмов найдено
        found = sum(1 for m in self.meaning.expected_morphisms 
                    if len(m) == 2 and m in bigram_set)
        expected_2grams = [m for m in self.meaning.expected_morphisms if len(m) == 2]
        
        if not expected_2grams:
            return 0.5
            
        return found / len(expected_2grams)
        
    def _compute_energy_score(self, embedding: np.ndarray) -> float:
        """Оценка соответствия энергии ожидаемому диапазону"""
        # Энергия связана с нормой эмбеддинга
        energy = np.linalg.norm(embedding)
        
        if self.meaning.expected_energy_range == "low":
            return 1.0 if energy < 5.0 else max(0, 1.0 - (energy - 5.0) / 10.0)
        elif self.meaning.expected_energy_range == "high":
            return 1.0 if energy > 10.0 else energy / 10.0
        else:  # medium
            return 1.0 if 5.0 <= energy <= 10.0 else 0.5
            
    def find_matching_windows(self, min_score: float = 0.6) -> List[Tuple[int, float]]:
        """
        Найти окна данных, соответствующие смыслу.
        
        Returns:
            List of (window_index, score) tuples
        """
        if self.space is None:
            raise ValueError("Space not built. Call build_space first.")
            
        matches = []
        step = self.window_size // 5  # Шаг с перекрытием
        
        for i in range(0, len(self.dna_sequence) - self.window_size, step):
            # Получаем символы окна
            window_dna = self.dna_sequence[i:i + self.window_size]
            
            # Получаем эмбеддинг
            embedding = self.get_window_embedding(i, self.window_size)
            
            # Считаем общий score
            symbol_score = self._compute_symbol_score(window_dna)
            morphism_score = self._compute_morphism_score(i)
            energy_score = self._compute_energy_score(embedding)
            
            total_score = 0.4 * symbol_score + 0.4 * morphism_score + 0.2 * energy_score
            
            if total_score >= min_score:
                matches.append((i, total_score))
                
        # Сортируем по score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
        
    def learn(self, ohlcv: np.ndarray, top_k: int = 10) -> TrainingMetrics:
        """
        Обучиться распознавать смысл.
        
        Находит лучшие примеры смысла в данных и запоминает их
        (локально, без экспорта).
        """
        print(f"\n{'='*60}")
        print(f"Learning meaning: {self.meaning.meaning_name}")
        print(f"Description: {self.meaning.description}")
        print(f"{'='*60}\n")
        
        # Строим пространство
        self.build_space(ohlcv)
        
        # Ищем соответствия
        matches = self.find_matching_windows(min_score=0.5)
        print(f"Found {len(matches)} matching windows")
        
        if not matches:
            return TrainingMetrics(
                meaning_name=self.meaning.meaning_name,
                samples_found=0,
                mean_distance_to_expected=float('inf'),
                symbol_match_score=0.0,
                morphism_match_score=0.0,
                energy_match=False
            )
            
        # Берём top_k лучших
        top_matches = matches[:top_k]
        
        # Сохраняем (локально!)
        self._learned_windows = [m[0] for m in top_matches]
        self._learned_embeddings = np.array([
            self.get_window_embedding(idx, self.window_size)
            for idx in self._learned_windows
        ])
        
        # Считаем метрики
        scores = [m[1] for m in top_matches]
        mean_score = np.mean(scores)
        
        # Детальные метрики
        symbol_scores = []
        morphism_scores = []
        energy_matches = []
        
        for idx in self._learned_windows:
            window_dna = self.dna_sequence[idx:idx + self.window_size]
            embedding = self.get_window_embedding(idx, self.window_size)
            
            symbol_scores.append(self._compute_symbol_score(window_dna))
            morphism_scores.append(self._compute_morphism_score(idx))
            
            energy = np.linalg.norm(embedding)
            if self.meaning.expected_energy_range == "low":
                energy_matches.append(energy < 5.0)
            elif self.meaning.expected_energy_range == "high":
                energy_matches.append(energy > 10.0)
            else:
                energy_matches.append(5.0 <= energy <= 10.0)
                
        metrics = TrainingMetrics(
            meaning_name=self.meaning.meaning_name,
            samples_found=len(top_matches),
            mean_distance_to_expected=1.0 - mean_score,  # Инвертируем score в distance
            symbol_match_score=float(np.mean(symbol_scores)),
            morphism_match_score=float(np.mean(morphism_scores)),
            energy_match=bool(sum(energy_matches) > len(energy_matches) // 2)
        )
        
        print(f"\nTraining Results:")
        print(f"  Samples found: {metrics.samples_found}")
        print(f"  Symbol match: {metrics.symbol_match_score:.3f}")
        print(f"  Morphism match: {metrics.morphism_match_score:.3f}")
        print(f"  Energy match: {metrics.energy_match}")
        print(f"  Mean distance: {metrics.mean_distance_to_expected:.3f}")
        
        return metrics


class ServerA:
    """
    Server A - обучается распознавать смысл.
    
    КРИТИЧНО: Этот сервер НЕ экспортирует никаких данных для Server B.
    Он только:
    1. Читает конфиг смысла
    2. Обучается на своих данных
    3. Сохраняет модель локально
    """
    
    def __init__(self, meaning_config_path: str, data_portion: Tuple[float, float] = (0.0, 0.5)):
        self.meaning_config_path = meaning_config_path
        self.data_portion = data_portion
        self.meaning: Optional[MeaningConfig] = None
        self.learner: Optional[MeaningLearner] = None
        
    def load_meaning_config(self):
        """Загрузить конфигурацию смысла"""
        # Пробуем загрузить как файл
        config_path = Path(self.meaning_config_path)
        if config_path.exists():
            self.meaning = MeaningConfig.load(str(config_path))
        else:
            # Пробуем как имя смысла
            self.meaning = get_meaning_by_name(self.meaning_config_path)
            
        if self.meaning is None:
            raise ValueError(f"Cannot load meaning config: {self.meaning_config_path}")
            
        print(f"Loaded meaning: {self.meaning.meaning_name}")
        
    def load_data(self, data_path: str) -> np.ndarray:
        """Загрузить свою порцию данных"""
        df = pd.read_feather(data_path)
        
        # Берём свою порцию
        n = len(df)
        start_idx = int(n * self.data_portion[0])
        end_idx = int(n * self.data_portion[1])
        
        df_portion = df.iloc[start_idx:end_idx]
        
        ohlcv = df_portion[['open', 'high', 'low', 'close', 'volume']].values
        print(f"Loaded data portion {self.data_portion}: {len(ohlcv)} candles")
        
        return ohlcv
        
    def train(self, ohlcv: np.ndarray) -> TrainingMetrics:
        """Обучить модель распознавать смысл"""
        if self.meaning is None:
            raise ValueError("Meaning not loaded")
            
        self.learner = MeaningLearner(self.meaning)
        return self.learner.learn(ohlcv)
        
    def save_model(self, output_dir: str):
        """
        Сохранить модель ЛОКАЛЬНО.
        
        КРИТИЧНО: Эти файлы НЕ передаются Server B!
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем только для себя
        if self.learner and self.learner._learned_embeddings is not None:
            np.save(
                output_path / "learned_embeddings.npy",
                self.learner._learned_embeddings
            )
            
            with open(output_path / "learned_windows.json", 'w') as f:
                json.dump(self.learner._learned_windows, f)
                
        print(f"Model saved locally to {output_dir}")
        print("NOTE: These files are NOT shared with Server B!")


def main():
    parser = argparse.ArgumentParser(description="Server A - Meaning Learner")
    parser.add_argument("--meaning", required=True,
                        help="Meaning name or path to meaning config JSON")
    parser.add_argument("--data", required=True,
                        help="Path to OHLCV data file")
    parser.add_argument("--output", default="./model",
                        help="Output directory for model (local only)")
    parser.add_argument("--portion-start", type=float, default=0.0,
                        help="Start of data portion (0.0-1.0)")
    parser.add_argument("--portion-end", type=float, default=0.5,
                        help="End of data portion (0.0-1.0)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SERVER A - MEANING LEARNER")
    print("="*60)
    print("\nNOTE: This server does NOT export any data to Server B!")
    print("Server B must find the meaning independently.\n")
    
    server = ServerA(
        meaning_config_path=args.meaning,
        data_portion=(args.portion_start, args.portion_end)
    )
    
    # 1. Загрузить конфиг смысла
    server.load_meaning_config()
    
    # 2. Загрузить данные
    ohlcv = server.load_data(args.data)
    
    # 3. Обучиться
    metrics = server.train(ohlcv)
    
    # 4. Сохранить модель (ЛОКАЛЬНО!)
    server.save_model(args.output)
    
    # 5. Сохранить метрики (для самоконтроля)
    metrics_path = Path(args.output) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
        
    print("\n" + "="*60)
    print("SERVER A COMPLETED")
    print("="*60)
    print(f"Meaning learned: {metrics.meaning_name}")
    print(f"Samples found: {metrics.samples_found}")
    print("\nServer B must now find this meaning independently!")
    

if __name__ == "__main__":
    main()
