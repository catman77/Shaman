"""
Server B - Agent B (Shaman)

Шаман получает ТОЛЬКО название/описание смысла из конфига.
Он НЕ получает никаких данных от Server A.

Задача: найти в своём пространстве данных конфигурации,
соответствующие заданному смыслу, используя ТОЛЬКО:
- Название смысла
- Априорное описание (из общего конфига meanings.py)

Это "чистый" эксперимент: два агента с одинаковыми априорными
знаниями о смыслах, но РАЗНЫМИ данными.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time

import numpy as np
import pandas as pd

# Add parent for shared imports
sys.path.insert(0, str(Path(__file__).parent))

from nobs.space import NOBSSpace
from shared.meanings import MeaningConfig, get_meaning_by_name


@dataclass
class SearchResult:
    """Результат поиска смысла"""
    window_index: int
    score: float
    symbol_distribution: Dict[str, float]
    dominant_morphisms: List[str]
    free_energy: float


@dataclass 
class ShamanReport:
    """Отчёт Шамана о поиске смысла"""
    meaning_name: str
    meaning_description: str
    search_successful: bool
    best_score: float
    total_matches: int
    top_results: List[Dict]  # Simplified SearchResult dicts
    search_time_seconds: float
    data_portion: Tuple[float, float]
    total_windows_scanned: int
    
    def to_dict(self) -> dict:
        return asdict(self)


class MeaningSearcher:
    """
    Поиск смысла в пространстве данных.
    
    Использует ТОЛЬКО априорные знания о смысле из конфига,
    НЕ получая никаких данных от другого агента.
    """
    
    def __init__(self, meaning: MeaningConfig, window_size: int = 100):
        self.meaning = meaning
        self.window_size = window_size
        self.space: Optional[NOBSSpace] = None
        self.dna_sequence: str = ""
        
    def build_space(self, ohlcv: np.ndarray):
        """Построить NOBS пространство из своих данных"""
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
        
    def _compute_symbol_score(self, window_symbols: str) -> Tuple[float, Dict[str, float]]:
        """
        Оценка соответствия символов ожидаемым.
        Возвращает (score, distribution)
        """
        # Подсчёт частот
        freq = {}
        for s in window_symbols:
            freq[s] = freq.get(s, 0) + 1
        total = len(window_symbols)
        
        distribution = {s: count / total for s, count in freq.items()}
        
        if not self.meaning.expected_symbols:
            return 0.5, distribution
            
        score = 0.0
        weights = {"dominant": 0.5, "secondary": 0.3, "rare": 0.2}
        
        for role, symbol in self.meaning.expected_symbols.items():
            if role in weights:
                actual_freq = freq.get(symbol, 0) / total if total > 0 else 0
                if role == "dominant":
                    # Доминантный символ должен быть самым частым
                    if freq:
                        max_freq = max(freq.values()) / total
                        score += weights[role] * (1.0 if actual_freq == max_freq else 0.5)
                elif role == "rare":
                    # Редкий символ должен быть нечастым
                    score += weights[role] * (1.0 if actual_freq < 0.15 else 0.3)
                else:
                    # Вторичный - средняя частота
                    score += weights[role] * min(1.0, actual_freq * 3)
                    
        return score, distribution
        
    def _compute_morphism_score(self, window_start: int) -> Tuple[float, List[str]]:
        """
        Оценка соответствия морфизмов ожидаемым.
        Возвращает (score, dominant_morphisms)
        """
        if self.space is None:
            return 0.5, []
            
        window_end = min(window_start + self.window_size, len(self.dna_sequence))
        window_dna = self.dna_sequence[window_start:window_end]
        
        # Считаем биграммы
        bigram_freq = {}
        for i in range(len(window_dna) - 1):
            bg = window_dna[i:i+2]
            bigram_freq[bg] = bigram_freq.get(bg, 0) + 1
            
        # Топ-5 морфизмов
        sorted_morphisms = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
        dominant = [m[0] for m in sorted_morphisms[:5]]
        
        if not self.meaning.expected_morphisms:
            return 0.5, dominant
            
        # Сколько ожидаемых морфизмов найдено
        bigram_set = set(bigram_freq.keys())
        found = sum(1 for m in self.meaning.expected_morphisms 
                    if len(m) == 2 and m in bigram_set)
        expected_2grams = [m for m in self.meaning.expected_morphisms if len(m) == 2]
        
        if not expected_2grams:
            return 0.5, dominant
            
        score = found / len(expected_2grams)
        return score, dominant
        
    def _compute_energy_score(self, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Оценка соответствия энергии ожидаемому диапазону.
        Возвращает (score, energy_value)
        """
        energy = np.linalg.norm(embedding)
        
        if self.meaning.expected_energy_range == "low":
            score = 1.0 if energy < 5.0 else max(0, 1.0 - (energy - 5.0) / 10.0)
        elif self.meaning.expected_energy_range == "high":
            score = 1.0 if energy > 10.0 else energy / 10.0
        else:  # medium
            score = 1.0 if 5.0 <= energy <= 10.0 else 0.5
            
        return score, float(energy)
        
    def search(self, min_score: float = 0.5) -> List[SearchResult]:
        """
        Искать смысл в пространстве данных.
        
        Использует ТОЛЬКО априорные знания о смысле!
        """
        if self.space is None:
            raise ValueError("Space not built. Call build_space first.")
            
        results = []
        step = self.window_size // 5  # Шаг с перекрытием
        
        total_windows = (len(self.dna_sequence) - self.window_size) // step
        print(f"Searching {total_windows} windows...")
        
        for i in range(0, len(self.dna_sequence) - self.window_size, step):
            # Получаем символы окна
            window_dna = self.dna_sequence[i:i + self.window_size]
            
            # Получаем эмбеддинг
            embedding = self.get_window_embedding(i, self.window_size)
            
            # Считаем все метрики
            symbol_score, distribution = self._compute_symbol_score(window_dna)
            morphism_score, dominant_morphisms = self._compute_morphism_score(i)
            energy_score, energy_value = self._compute_energy_score(embedding)
            
            # Общий score
            total_score = 0.4 * symbol_score + 0.4 * morphism_score + 0.2 * energy_score
            
            if total_score >= min_score:
                results.append(SearchResult(
                    window_index=i,
                    score=total_score,
                    symbol_distribution=distribution,
                    dominant_morphisms=dominant_morphisms,
                    free_energy=energy_value
                ))
                
        # Сортируем по score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


class ServerB:
    """
    Server B - Shaman (Искатель смысла)
    
    КРИТИЧНО: Этот сервер НЕ получает никаких данных от Server A!
    Он знает только:
    1. Название смысла (из конфига)
    2. Априорное описание смысла (из shared/meanings.py)
    
    Оба сервера используют ОДИНАКОВЫЙ shared/meanings.py,
    но работают с РАЗНЫМИ данными.
    """
    
    def __init__(self, meaning_name: str, data_portion: Tuple[float, float] = (0.5, 1.0)):
        self.meaning_name = meaning_name
        self.data_portion = data_portion
        self.meaning: Optional[MeaningConfig] = None
        self.searcher: Optional[MeaningSearcher] = None
        
    def load_meaning_config(self):
        """
        Загрузить конфигурацию смысла из ОБЩЕГО репозитория знаний.
        
        НЕ от Server A! Из shared/meanings.py, который одинаков
        для обоих серверов.
        """
        # Пробуем загрузить как файл
        config_path = Path(self.meaning_name)
        if config_path.exists():
            self.meaning = MeaningConfig.load(str(config_path))
        else:
            # Пробуем как имя смысла из общего репозитория
            self.meaning = get_meaning_by_name(self.meaning_name)
            
        if self.meaning is None:
            raise ValueError(f"Unknown meaning: {self.meaning_name}")
            
        print(f"Loaded meaning from shared knowledge base:")
        print(f"  Name: {self.meaning.meaning_name}")
        print(f"  Description: {self.meaning.description}")
        print(f"  Expected symbols: {self.meaning.expected_symbols}")
        print(f"  Expected morphisms: {self.meaning.expected_morphisms}")
        print(f"  Expected energy: {self.meaning.expected_energy_range}")
        
    def load_data(self, data_path: str) -> np.ndarray:
        """Загрузить свою (ДРУГУЮ) порцию данных"""
        df = pd.read_feather(data_path)
        
        # Берём свою порцию (отличную от Server A!)
        n = len(df)
        start_idx = int(n * self.data_portion[0])
        end_idx = int(n * self.data_portion[1])
        
        df_portion = df.iloc[start_idx:end_idx]
        
        ohlcv = df_portion[['open', 'high', 'low', 'close', 'volume']].values
        print(f"Loaded data portion {self.data_portion}: {len(ohlcv)} candles")
        print("NOTE: This is DIFFERENT data from Server A!")
        
        return ohlcv
        
    def search_meaning(self, ohlcv: np.ndarray, min_score: float = 0.5) -> Tuple[List[SearchResult], int]:
        """
        Искать смысл в своих данных.
        
        Использует ТОЛЬКО априорные знания!
        """
        if self.meaning is None:
            raise ValueError("Meaning not loaded")
            
        self.searcher = MeaningSearcher(self.meaning)
        self.searcher.build_space(ohlcv)
        
        results = self.searcher.search(min_score=min_score)
        
        total_windows = (len(self.searcher.dna_sequence) - 
                         self.searcher.window_size) // (self.searcher.window_size // 5)
        
        return results, total_windows
        
    def generate_report(self, results: List[SearchResult], 
                        total_windows: int,
                        search_time: float) -> ShamanReport:
        """Сгенерировать отчёт о поиске"""
        
        top_results = []
        for r in results[:10]:  # Top 10
            top_results.append({
                "window_index": r.window_index,
                "score": r.score,
                "symbol_distribution": r.symbol_distribution,
                "dominant_morphisms": r.dominant_morphisms,
                "free_energy": r.free_energy
            })
            
        return ShamanReport(
            meaning_name=self.meaning.meaning_name,
            meaning_description=self.meaning.description,
            search_successful=len(results) > 0,
            best_score=results[0].score if results else 0.0,
            total_matches=len(results),
            top_results=top_results,
            search_time_seconds=search_time,
            data_portion=self.data_portion,
            total_windows_scanned=total_windows
        )


def main():
    parser = argparse.ArgumentParser(description="Server B - Shaman (Meaning Searcher)")
    parser.add_argument("--meaning", required=True,
                        help="Meaning name (from shared/meanings.py) or path to config")
    parser.add_argument("--data", required=True,
                        help="Path to OHLCV data file")
    parser.add_argument("--output", default="./report",
                        help="Output directory for report")
    parser.add_argument("--portion-start", type=float, default=0.5,
                        help="Start of data portion (0.0-1.0)")
    parser.add_argument("--portion-end", type=float, default=1.0,
                        help="End of data portion (0.0-1.0)")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Minimum score threshold for matches")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SERVER B - SHAMAN (MEANING SEARCHER)")
    print("="*60)
    print("\nThis server receives NO DATA from Server A!")
    print("It only knows the MEANING NAME from shared config.\n")
    
    server = ServerB(
        meaning_name=args.meaning,
        data_portion=(args.portion_start, args.portion_end)
    )
    
    # 1. Загрузить конфиг смысла (из ОБЩЕЙ базы знаний, не от Server A!)
    print("\n--- Loading meaning from SHARED knowledge base ---")
    server.load_meaning_config()
    
    # 2. Загрузить свои данные (ДРУГИЕ, чем у Server A!)
    print("\n--- Loading OWN data (different from Server A) ---")
    ohlcv = server.load_data(args.data)
    
    # 3. Искать смысл
    print("\n--- Searching for meaning using ONLY a priori knowledge ---")
    start_time = time.time()
    results, total_windows = server.search_meaning(ohlcv, min_score=args.min_score)
    search_time = time.time() - start_time
    
    # 4. Сгенерировать отчёт
    report = server.generate_report(results, total_windows, search_time)
    
    # 5. Сохранить отчёт
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / "shaman_report.json"
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
        
    # 6. Вывести результаты
    print("\n" + "="*60)
    print("SHAMAN SEARCH COMPLETED")
    print("="*60)
    print(f"\nMeaning searched: {report.meaning_name}")
    print(f"Description: {report.meaning_description}")
    print(f"\nResults:")
    print(f"  Search successful: {report.search_successful}")
    print(f"  Best score: {report.best_score:.4f}")
    print(f"  Total matches: {report.total_matches}")
    print(f"  Windows scanned: {report.total_windows_scanned}")
    print(f"  Search time: {report.search_time_seconds:.2f}s")
    
    if report.top_results:
        print(f"\nTop 3 matches:")
        for i, r in enumerate(report.top_results[:3], 1):
            print(f"  {i}. Window {r['window_index']}: score={r['score']:.4f}")
            print(f"     Dominant morphisms: {r['dominant_morphisms'][:3]}")
            
    print(f"\nReport saved to: {report_file}")
    print("\nNOTE: This search was performed WITHOUT any data from Server A!")
    print("Only shared a priori knowledge about the meaning was used.")


if __name__ == "__main__":
    main()
