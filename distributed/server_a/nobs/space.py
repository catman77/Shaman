# -*- coding: utf-8 -*-
"""
NOBS Semantic Space.

Unified space S combining:
- Symbolic DNA encoding
- Category hierarchy C_n
- Free energy landscape
- Morphism-based embeddings

This is the "rich" semantic space for the Shaman experiment,
replacing the simple sentence-transformer embeddings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict

from .symbolic_dna import SymbolicDNAEncoder, StructuralDNA, Symbol
from .category import CategoryCn, CategoryHierarchy, Morphism
from .free_energy import FreeEnergyMinimizer, AdaptiveTemperature


@dataclass
class NOBSEmbedding:
    """
    Embedding of a state in NOBS space.
    
    Multi-scale representation combining:
    - Symbolic features (symbol distribution, n-gram patterns)
    - Categorical features (morphism statistics)  
    - Energetic features (free energy, entropy)
    - Topological features (β₁ from TDA if available)
    """
    # Symbolic representation
    symbolic_string: str
    symbol_distribution: Dict[str, float]
    
    # N-gram features at different scales
    ngram_features: Dict[int, np.ndarray]  # n -> embedding vector
    
    # Energetic features
    free_energy: float
    entropy: float
    temperature: float
    
    # Morphism features  
    transition_entropy: float
    dominant_morphism: Optional[str]
    
    # Full embedding vector
    vector: np.ndarray
    
    def __repr__(self):
        return f"NOBSEmbedding(symbols={self.symbolic_string[:20]}..., dim={len(self.vector)})"
    
    def distance_to(self, other: 'NOBSEmbedding') -> float:
        """Compute distance to another embedding."""
        return float(np.linalg.norm(self.vector - other.vector))
    
    def cosine_similarity(self, other: 'NOBSEmbedding') -> float:
        """Compute cosine similarity to another embedding."""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))


class NOBSSpace:
    """
    NOBS Semantic Space S.
    
    Provides embedding of time series data into a rich structural space
    based on the NOBS framework.
    
    Architecture:
    1. Symbolic DNA encoding: OHLCV → Σ*
    2. Category hierarchy: C_1, C_2, ..., C_n
    3. Free energy computation
    4. Multi-scale embedding generation
    
    Parameters:
        max_depth: Maximum n-gram depth (default: 5)
        window_size: Window size for local embedding (default: 50)
        temperature: Free energy temperature (default: 1.0)
        embedding_dim: Output embedding dimension (default: 128)
    """
    
    # Symbol alphabet for one-hot encoding
    SYMBOLS = ['S', 'P', 'I', 'Z', 'Ω', 'Λ']
    
    def __init__(
        self,
        max_depth: int = 5,
        window_size: int = 50,
        temperature: float = 1.0,
        embedding_dim: int = 128
    ):
        self.max_depth = max_depth
        self.window_size = window_size
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        
        # Components
        self.encoder = SymbolicDNAEncoder()
        self.hierarchy: Optional[CategoryHierarchy] = None
        self.minimizer: Optional[FreeEnergyMinimizer] = None
        
        # State
        self._fitted = False
        self._dna_sequence: List[StructuralDNA] = []
        self._symbol_to_idx = {s: i for i, s in enumerate(self.SYMBOLS)}
        
        # Learned embeddings for n-grams
        self._ngram_embeddings: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
        
    def fit(self, df: pd.DataFrame) -> 'NOBSSpace':
        """
        Fit the NOBS space to data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            self (fitted space)
        """
        # 1. Symbolic DNA encoding
        self._dna_sequence = self.encoder.fit_transform(df)
        
        # 2. Build category hierarchy
        self.hierarchy = CategoryHierarchy(max_n=self.max_depth)
        self.hierarchy.build_from_sequence(self._dna_sequence)
        
        # 3. Compute free energy for primary depth (usually 3)
        primary_depth = min(3, self.max_depth)
        primary_cat = self.hierarchy.get_category(primary_depth)
        
        self.minimizer = FreeEnergyMinimizer(temperature=self.temperature)
        self.minimizer.fit(primary_cat)
        
        # 4. Learn n-gram embeddings via spectral decomposition
        self._learn_ngram_embeddings()
        
        self._fitted = True
        return self
    
    def _learn_ngram_embeddings(self):
        """
        Learn embeddings for n-grams using transition matrix spectral decomposition.
        """
        for n, cat in self.hierarchy.categories.items():
            if cat.num_objects == 0:
                continue
            
            # Get transition matrix
            trans_matrix, idx_map = cat.get_transition_matrix()
            
            # Spectral decomposition
            try:
                # Use SVD for embedding
                U, S, Vt = np.linalg.svd(trans_matrix, full_matrices=False)
                
                # Embedding dimension for this level
                embed_dim = min(32, len(S))
                
                # Embeddings are scaled left singular vectors
                embeddings = U[:, :embed_dim] * S[:embed_dim]
                
                # Store embeddings
                idx_to_ngram = {v: k for k, v in idx_map.items()}
                for idx, ngram in idx_to_ngram.items():
                    self._ngram_embeddings[n][ngram] = embeddings[idx]
                    
            except np.linalg.LinAlgError:
                # Fallback to random embeddings if SVD fails
                for ngram in cat.objects:
                    self._ngram_embeddings[n][ngram] = np.random.randn(32)
    
    def embed_sequence(
        self,
        dna_sequence: List[StructuralDNA],
        aggregation: str = 'mean'
    ) -> NOBSEmbedding:
        """
        Embed a DNA sequence into NOBS space.
        
        Args:
            dna_sequence: List of StructuralDNA objects
            aggregation: How to aggregate over sequence ('mean', 'last', 'attention')
            
        Returns:
            NOBSEmbedding object
        """
        if len(dna_sequence) == 0:
            return self._empty_embedding()
        
        # 1. Symbol distribution
        symbols = [dna.to_symbol() for dna in dna_sequence]
        symbol_dist = self._compute_symbol_distribution(symbols)
        
        # 2. N-gram features at each scale
        ngram_features = {}
        for n in range(1, self.max_depth + 1):
            ngrams = self._extract_ngrams(symbols, n)
            if ngrams:
                features = self._aggregate_ngram_embeddings(ngrams, n)
                ngram_features[n] = features
            else:
                ngram_features[n] = np.zeros(32)
        
        # 3. Energetic features
        # Compute for last window
        if len(dna_sequence) >= self.max_depth:
            window = dna_sequence[-self.window_size:] if len(dna_sequence) > self.window_size else dna_sequence
            
            # Build temporary category for this window
            temp_cat = CategoryCn(n=min(3, self.max_depth))
            temp_cat.build_from_sequence(window)
            
            temp_min = FreeEnergyMinimizer(temperature=self.temperature)
            temp_min.fit(temp_cat)
            
            stats = temp_min.get_statistics()
            free_energy = stats.get('mean_free_energy', 0.0)
            entropy = stats.get('mean_entropy', 0.0)
            transition_entropy = temp_cat.morphism_entropy()
        else:
            free_energy = 0.0
            entropy = 0.0
            transition_entropy = 0.0
        
        # 4. Find dominant morphism (most frequent recent transition)
        if len(symbols) >= 4:
            last_ngram = ''.join(symbols[-3:])
            dominant_morphism = last_ngram
        else:
            dominant_morphism = None
        
        # 5. Build full embedding vector
        vector = self._build_embedding_vector(
            symbol_dist=symbol_dist,
            ngram_features=ngram_features,
            free_energy=free_energy,
            entropy=entropy,
            transition_entropy=transition_entropy
        )
        
        return NOBSEmbedding(
            symbolic_string=''.join(symbols),
            symbol_distribution=symbol_dist,
            ngram_features=ngram_features,
            free_energy=free_energy,
            entropy=entropy,
            temperature=self.temperature,
            transition_entropy=transition_entropy,
            dominant_morphism=dominant_morphism,
            vector=vector
        )
    
    def embed_window(self, start_idx: int, end_idx: int) -> NOBSEmbedding:
        """
        Embed a window of the fitted sequence.
        
        Args:
            start_idx: Start index in DNA sequence
            end_idx: End index (exclusive)
            
        Returns:
            NOBSEmbedding
        """
        if not self._fitted:
            raise ValueError("Space not fitted. Call fit() first.")
        
        window = self._dna_sequence[start_idx:end_idx]
        return self.embed_sequence(window)
    
    def embed_from_ohlcv(self, df: pd.DataFrame) -> NOBSEmbedding:
        """
        Encode and embed new OHLCV data.
        
        Uses encoder fitted to original data for consistent encoding.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            NOBSEmbedding
        """
        dna_seq = self.encoder.transform(df)
        return self.embed_sequence(dna_seq)
    
    def _compute_symbol_distribution(self, symbols: List[str]) -> Dict[str, float]:
        """Compute normalized symbol frequency distribution."""
        dist = {s: 0.0 for s in self.SYMBOLS}
        for s in symbols:
            if s in dist:
                dist[s] += 1
        
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        
        return dist
    
    def _extract_ngrams(self, symbols: List[str], n: int) -> List[str]:
        """Extract all n-grams from symbol list."""
        if len(symbols) < n:
            return []
        return [''.join(symbols[i:i+n]) for i in range(len(symbols) - n + 1)]
    
    def _aggregate_ngram_embeddings(
        self,
        ngrams: List[str],
        n: int
    ) -> np.ndarray:
        """Aggregate n-gram embeddings into single vector."""
        embeddings = []
        
        for ngram in ngrams:
            if ngram in self._ngram_embeddings[n]:
                embeddings.append(self._ngram_embeddings[n][ngram])
        
        if not embeddings:
            return np.zeros(32)
        
        # Mean aggregation
        return np.mean(embeddings, axis=0)
    
    def _build_embedding_vector(
        self,
        symbol_dist: Dict[str, float],
        ngram_features: Dict[int, np.ndarray],
        free_energy: float,
        entropy: float,
        transition_entropy: float
    ) -> np.ndarray:
        """
        Build final embedding vector from all features.
        
        Structure:
        - [0:6] Symbol distribution (6 dims)
        - [6:38] N-gram level 1 features (32 dims)
        - [38:70] N-gram level 2 features (32 dims)
        - [70:102] N-gram level 3 features (32 dims)
        - [102:118] N-gram levels 4-5 compressed (16 dims)
        - [118:128] Energetic features (10 dims)
        """
        parts = []
        
        # Symbol distribution (6 dims)
        symbol_vec = np.array([symbol_dist.get(s, 0.0) for s in self.SYMBOLS])
        parts.append(symbol_vec)
        
        # N-gram features
        for n in range(1, 4):
            feat = ngram_features.get(n, np.zeros(32))
            if len(feat) < 32:
                feat = np.pad(feat, (0, 32 - len(feat)))
            parts.append(feat[:32])
        
        # Compressed higher-level features (levels 4-5)
        higher_feats = []
        for n in range(4, self.max_depth + 1):
            feat = ngram_features.get(n, np.zeros(32))
            higher_feats.append(feat[:8])  # Take first 8 dims
        
        if higher_feats:
            compressed = np.concatenate(higher_feats)[:16]
        else:
            compressed = np.zeros(16)
        
        if len(compressed) < 16:
            compressed = np.pad(compressed, (0, 16 - len(compressed)))
        parts.append(compressed)
        
        # Energetic features (10 dims)
        energy_vec = np.array([
            free_energy,
            entropy,
            transition_entropy,
            np.tanh(free_energy),
            np.tanh(entropy),
            np.exp(-abs(free_energy)),
            np.exp(-abs(entropy)),
            np.sin(free_energy * np.pi),
            np.cos(entropy * np.pi),
            free_energy * entropy
        ])
        parts.append(energy_vec)
        
        # Concatenate and pad/truncate to embedding_dim
        vector = np.concatenate(parts)
        
        if len(vector) < self.embedding_dim:
            vector = np.pad(vector, (0, self.embedding_dim - len(vector)))
        else:
            vector = vector[:self.embedding_dim]
        
        return vector.astype(np.float32)
    
    def _empty_embedding(self) -> NOBSEmbedding:
        """Return empty embedding for edge cases."""
        return NOBSEmbedding(
            symbolic_string='',
            symbol_distribution={s: 0.0 for s in self.SYMBOLS},
            ngram_features={},
            free_energy=0.0,
            entropy=0.0,
            temperature=self.temperature,
            transition_entropy=0.0,
            dominant_morphism=None,
            vector=np.zeros(self.embedding_dim, dtype=np.float32)
        )
    
    def distance(self, emb1: NOBSEmbedding, emb2: NOBSEmbedding) -> float:
        """Compute distance between two embeddings."""
        return emb1.distance_to(emb2)
    
    def batch_embed(
        self,
        windows: List[Tuple[int, int]]
    ) -> List[NOBSEmbedding]:
        """
        Embed multiple windows in batch.
        
        Args:
            windows: List of (start_idx, end_idx) tuples
            
        Returns:
            List of NOBSEmbedding objects
        """
        return [self.embed_window(s, e) for s, e in windows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fitted space."""
        if not self._fitted:
            return {'fitted': False}
        
        stats = {
            'fitted': True,
            'sequence_length': len(self._dna_sequence),
            'max_depth': self.max_depth,
            'embedding_dim': self.embedding_dim,
            'temperature': self.temperature,
        }
        
        # Category hierarchy stats
        if self.hierarchy:
            stats['depth_analysis'] = self.hierarchy.depth_analysis()
            stats['computational_depth'] = self.hierarchy.estimate_computational_depth()
        
        # Free energy stats
        if self.minimizer:
            stats['free_energy'] = self.minimizer.get_statistics()
        
        # Symbol distribution of full sequence
        symbols = [dna.to_symbol() for dna in self._dna_sequence]
        stats['symbol_distribution'] = self._compute_symbol_distribution(symbols)
        
        return stats
    
    def get_trajectory_embedding(
        self,
        window_size: int = 50,
        step: int = 10
    ) -> np.ndarray:
        """
        Get embedding trajectory over the full sequence.
        
        Args:
            window_size: Size of sliding window
            step: Step between windows
            
        Returns:
            Array of shape (num_windows, embedding_dim)
        """
        if not self._fitted:
            raise ValueError("Space not fitted")
        
        embeddings = []
        for start in range(0, len(self._dna_sequence) - window_size, step):
            emb = self.embed_window(start, start + window_size)
            embeddings.append(emb.vector)
        
        return np.array(embeddings)


if __name__ == "__main__":
    # Test NOBS Space
    import pandas as pd
    
    # Load data
    df = pd.read_feather("../../data/BTC_USDT_USDT-4h-futures.feather")
    print(f"Loaded {len(df)} candles")
    
    # Create and fit space
    space = NOBSSpace(max_depth=5, embedding_dim=128)
    space.fit(df)
    
    # Get statistics
    print("\nSpace Statistics:")
    stats = space.get_statistics()
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {val}")
    
    # Test embedding
    print("\n\nTest embeddings:")
    
    # Embed first window
    emb1 = space.embed_window(0, 100)
    print(f"Window [0:100]: {emb1}")
    print(f"  Vector shape: {emb1.vector.shape}")
    print(f"  Free energy: {emb1.free_energy:.4f}")
    print(f"  Entropy: {emb1.entropy:.4f}")
    
    # Embed last window
    emb2 = space.embed_window(-100, len(space._dna_sequence))
    print(f"\nWindow [-100:]: {emb2}")
    
    # Distance
    print(f"\nDistance between windows: {emb1.distance_to(emb2):.4f}")
    print(f"Cosine similarity: {emb1.cosine_similarity(emb2):.4f}")
    
    # Trajectory
    print("\n\nComputing trajectory...")
    trajectory = space.get_trajectory_embedding(window_size=50, step=20)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Mean embedding norm: {np.linalg.norm(trajectory, axis=1).mean():.4f}")
