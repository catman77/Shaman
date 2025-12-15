# -*- coding: utf-8 -*-
"""
Symbolic DNA Encoder for OHLCV data.

Transforms financial market data into symbolic sequences using
the NOBS alphabet: Σ = {S, P, I, Z, Ω, Λ}

S - Supply/Short (падение, медвежий сигнал)
P - Price/Positive (рост, бычий сигнал)  
I - Invariant (нейтральное движение)
Z - Zero/Pause (консолидация, низкая волатильность)
Ω - Omega/Extremum (локальный максимум/минимум)
Λ - Lambda/Phase transition (смена фазы/тренда)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from enum import Enum


class Symbol(Enum):
    """Symbolic DNA alphabet."""
    S = 'S'  # Supply/Short - decline
    P = 'P'  # Price/Positive - growth  
    I = 'I'  # Invariant - neutral
    Z = 'Z'  # Zero/Pause - consolidation
    OMEGA = 'Ω'  # Extremum - local max/min
    LAMBDA = 'Λ'  # Phase transition


@dataclass
class StructuralDNA:
    """
    Structural DNA of a single observation.
    
    Attributes:
        base_pattern: Primary symbol from alphabet
        formation_tree: Sequence of recent elementary transitions
        topological_signature: Tuple (genus, euler_characteristic)
        timestamp: Original timestamp
        raw_features: Dict of normalized features used for encoding
    """
    base_pattern: Symbol
    formation_tree: Tuple[Symbol, ...] = field(default_factory=tuple)
    topological_signature: Tuple[int, int] = (0, 1)  # (genus=0, euler=1)
    timestamp: Optional[pd.Timestamp] = None
    raw_features: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.base_pattern.value
    
    def __repr__(self) -> str:
        tree_str = ''.join(s.value for s in self.formation_tree[-3:])
        return f"StructuralDNA({self.base_pattern.value}, tree=[...{tree_str}])"
    
    def to_symbol(self) -> str:
        """Get single character symbol."""
        return self.base_pattern.value


class SymbolicDNAEncoder:
    """
    Translator T: OHLCV → Σ*
    
    Transforms OHLCV data into symbolic DNA sequences.
    The encoding is functorial: preserves structural relationships.
    
    Parameters:
        volatility_threshold: Threshold for Z (pause) detection (default: 0.3 of avg volatility)
        trend_threshold: Threshold for P/S detection (default: 0.5 std of returns)
        extremum_lookback: Window for extremum detection (default: 5)
        phase_transition_sensitivity: Sensitivity for Λ detection (default: 2.0 std)
    """
    
    def __init__(
        self,
        volatility_threshold: float = 0.3,
        trend_threshold: float = 0.5,
        extremum_lookback: int = 5,
        phase_transition_sensitivity: float = 2.0
    ):
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.extremum_lookback = extremum_lookback
        self.phase_transition_sensitivity = phase_transition_sensitivity
        
        # Statistics computed during fit
        self._fitted = False
        self._mean_return = 0.0
        self._std_return = 1.0
        self._mean_volatility = 1.0
        self._mean_volume = 1.0
        
    def fit(self, df: pd.DataFrame) -> 'SymbolicDNAEncoder':
        """
        Fit encoder to data to compute normalization statistics.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            self (fitted encoder)
        """
        # Compute returns
        returns = (df['close'] - df['open']) / df['open']
        self._mean_return = returns.mean()
        self._std_return = returns.std() + 1e-10
        
        # Compute volatility (high-low range)
        volatility = (df['high'] - df['low']) / df['open']
        self._mean_volatility = volatility.mean() + 1e-10
        
        # Volume
        self._mean_volume = df['volume'].mean() + 1e-10
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> List[StructuralDNA]:
        """
        Transform OHLCV data to symbolic DNA sequence.
        
        Args:
            df: DataFrame with columns [date, open, high, low, close, volume]
            
        Returns:
            List of StructuralDNA objects
        """
        if not self._fitted:
            self.fit(df)
        
        dna_sequence: List[StructuralDNA] = []
        formation_tree: List[Symbol] = []
        
        # Precompute features
        returns = ((df['close'] - df['open']) / df['open']).values
        volatility = ((df['high'] - df['low']) / df['open']).values
        volume_ratio = (df['volume'] / self._mean_volume).values
        
        # Normalize
        norm_returns = (returns - self._mean_return) / self._std_return
        norm_volatility = volatility / self._mean_volatility
        
        # Detect local extrema
        closes = df['close'].values
        is_local_max = np.zeros(len(df), dtype=bool)
        is_local_min = np.zeros(len(df), dtype=bool)
        
        for i in range(self.extremum_lookback, len(df) - self.extremum_lookback):
            window = closes[i - self.extremum_lookback:i + self.extremum_lookback + 1]
            if closes[i] == window.max():
                is_local_max[i] = True
            if closes[i] == window.min():
                is_local_min[i] = True
        
        # Detect phase transitions (trend changes)
        # Using rolling mean of returns
        window_size = min(20, len(df) // 10)
        if window_size > 1:
            rolling_mean = pd.Series(returns).rolling(window_size, min_periods=1).mean().values
            rolling_std = pd.Series(returns).rolling(window_size, min_periods=1).std().fillna(self._std_return).values
            
            # Phase transition when return deviates significantly from rolling mean
            phase_deviation = np.abs(returns - rolling_mean) / (rolling_std + 1e-10)
            is_phase_transition = phase_deviation > self.phase_transition_sensitivity
        else:
            is_phase_transition = np.zeros(len(df), dtype=bool)
        
        # Encode each candle
        for i in range(len(df)):
            raw_features = {
                'return': float(returns[i]),
                'norm_return': float(norm_returns[i]),
                'volatility': float(volatility[i]),
                'norm_volatility': float(norm_volatility[i]),
                'volume_ratio': float(volume_ratio[i])
            }
            
            # Determine base symbol using priority rules
            symbol = self._classify_candle(
                norm_return=norm_returns[i],
                norm_volatility=norm_volatility[i],
                is_max=is_local_max[i],
                is_min=is_local_min[i],
                is_phase_trans=is_phase_transition[i]
            )
            
            # Update formation tree
            formation_tree.append(symbol)
            
            # Get timestamp if available
            timestamp = None
            if 'date' in df.columns:
                timestamp = df.iloc[i]['date']
            
            # Create structural DNA
            dna = StructuralDNA(
                base_pattern=symbol,
                formation_tree=tuple(formation_tree[-10:]),  # Keep last 10
                topological_signature=(0, 1),  # Trivial for single point
                timestamp=timestamp,
                raw_features=raw_features
            )
            
            dna_sequence.append(dna)
        
        return dna_sequence
    
    def fit_transform(self, df: pd.DataFrame) -> List[StructuralDNA]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def _classify_candle(
        self,
        norm_return: float,
        norm_volatility: float,
        is_max: bool,
        is_min: bool,
        is_phase_trans: bool
    ) -> Symbol:
        """
        Classify a single candle to symbol.
        
        Priority order:
        1. Λ (phase transition) - highest priority
        2. Ω (extremum)
        3. Z (pause/consolidation)
        4. P/S/I (direction)
        """
        # Phase transition has highest priority
        if is_phase_trans:
            return Symbol.LAMBDA
        
        # Extremum
        if is_max or is_min:
            return Symbol.OMEGA
        
        # Low volatility = consolidation
        if norm_volatility < self.volatility_threshold:
            return Symbol.Z
        
        # Direction based on normalized return
        if norm_return > self.trend_threshold:
            return Symbol.P  # Positive/Growth
        elif norm_return < -self.trend_threshold:
            return Symbol.S  # Supply/Decline
        else:
            return Symbol.I  # Invariant/Neutral
    
    def to_string(self, dna_sequence: List[StructuralDNA]) -> str:
        """Convert DNA sequence to string representation."""
        return ''.join(dna.to_symbol() for dna in dna_sequence)
    
    def get_ngrams(self, dna_sequence: List[StructuralDNA], n: int) -> List[str]:
        """
        Extract all n-grams from DNA sequence.
        
        Args:
            dna_sequence: List of StructuralDNA
            n: n-gram length
            
        Returns:
            List of n-gram strings
        """
        string = self.to_string(dna_sequence)
        return [string[i:i+n] for i in range(len(string) - n + 1)]


def encode_ohlcv_to_symbols(
    df: pd.DataFrame,
    **kwargs
) -> Tuple[List[StructuralDNA], str]:
    """
    Convenience function to encode OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        **kwargs: Parameters for SymbolicDNAEncoder
        
    Returns:
        Tuple of (DNA sequence, string representation)
    """
    encoder = SymbolicDNAEncoder(**kwargs)
    dna_seq = encoder.fit_transform(df)
    string_rep = encoder.to_string(dna_seq)
    return dna_seq, string_rep


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Load Bitcoin 4h data
    df = pd.read_feather("../../data/BTC_USDT_USDT-4h-futures.feather")
    
    encoder = SymbolicDNAEncoder()
    dna_sequence = encoder.fit_transform(df)
    
    print(f"Total candles: {len(df)}")
    print(f"DNA sequence length: {len(dna_sequence)}")
    
    # Print first 100 symbols
    dna_string = encoder.to_string(dna_sequence[:100])
    print(f"\nFirst 100 symbols:")
    print(dna_string)
    
    # Symbol distribution
    from collections import Counter
    symbols = [dna.base_pattern for dna in dna_sequence]
    dist = Counter(symbols)
    print(f"\nSymbol distribution:")
    for symbol, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {symbol.value}: {count} ({100*count/len(symbols):.1f}%)")
    
    # N-grams
    ngrams_3 = encoder.get_ngrams(dna_sequence, 3)
    ngram_dist = Counter(ngrams_3)
    print(f"\nTop 10 3-grams:")
    for ngram, count in ngram_dist.most_common(10):
        print(f"  {ngram}: {count}")
