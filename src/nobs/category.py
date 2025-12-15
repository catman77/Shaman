# -*- coding: utf-8 -*-
"""
Category C_n construction from symbolic DNA sequences.

Implements the categorical framework from NOBS:
- Objects: n-grams appearing in symbolic sequence
- Morphisms: Observed transitions between n-grams
- Functors N (extension) and R (reduction)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Iterator
from collections import defaultdict
import numpy as np
from .symbolic_dna import StructuralDNA, Symbol


@dataclass
class Morphism:
    """
    Morphism in category C_n.
    
    Represents an observed transition from source to target n-gram.
    
    Attributes:
        source: Source n-gram string
        target: Target n-gram string
        frequency: Number of times this transition was observed
        quality: Average "cost" (e.g., return after transition)
        timestamps: List of timestamps where this transition occurred
    """
    source: str
    target: str
    frequency: int = 1
    quality: float = 0.0
    qualities: List[float] = field(default_factory=list)
    timestamps: List = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        if not isinstance(other, Morphism):
            return False
        return self.source == other.source and self.target == other.target
    
    def __repr__(self):
        return f"Morphism({self.source} → {self.target}, freq={self.frequency})"
    
    @property
    def entropy(self) -> float:
        """Compute entropy of quality distribution."""
        if len(self.qualities) < 2:
            return 0.0
        
        # Discretize qualities into bins
        hist, _ = np.histogram(self.qualities, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zeros
        if len(hist) == 0:
            return 0.0
        
        hist = hist / hist.sum()  # Normalize
        return -np.sum(hist * np.log(hist + 1e-10))


class CategoryCn:
    """
    Category C_n of symbolic n-grams.
    
    Objects: All unique n-grams from symbolic DNA sequence
    Morphisms: Observed transitions (shifts) between n-grams
    
    Supports:
    - Building from DNA sequence
    - Morphism statistics computation
    - Functor operations (N: extend, R: reduce)
    
    Parameters:
        n: n-gram length (default: 3)
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        
        # Objects = set of n-gram strings
        self.objects: Set[str] = set()
        
        # Morphisms = dict: (source, target) -> Morphism
        self.morphisms: Dict[Tuple[str, str], Morphism] = {}
        
        # Adjacency structure for efficient lookup
        self._outgoing: Dict[str, List[Morphism]] = defaultdict(list)
        self._incoming: Dict[str, List[Morphism]] = defaultdict(list)
        
        # Statistics
        self._total_observations = 0
        self._entropy_cache: Optional[float] = None
    
    def build_from_sequence(
        self,
        dna_sequence: List[StructuralDNA],
        quality_fn: Optional[callable] = None
    ) -> 'CategoryCn':
        """
        Build category from DNA sequence.
        
        Args:
            dna_sequence: List of StructuralDNA objects
            quality_fn: Optional function(dna_i, dna_j) -> float to compute
                       transition quality. Default uses next candle return.
                       
        Returns:
            self (built category)
        """
        if len(dna_sequence) < self.n + 1:
            return self
        
        # Convert to string for n-gram extraction
        symbols = [dna.to_symbol() for dna in dna_sequence]
        
        # Default quality: next candle return
        if quality_fn is None:
            def quality_fn(dna_i, dna_j):
                if 'return' in dna_j.raw_features:
                    return dna_j.raw_features['return']
                return 0.0
        
        # Extract n-grams and morphisms
        for i in range(len(symbols) - self.n):
            source = ''.join(symbols[i:i + self.n])
            target = ''.join(symbols[i + 1:i + self.n + 1])
            
            self.objects.add(source)
            self.objects.add(target)
            
            # Get or create morphism
            key = (source, target)
            if key not in self.morphisms:
                self.morphisms[key] = Morphism(source=source, target=target)
                self._outgoing[source].append(self.morphisms[key])
                self._incoming[target].append(self.morphisms[key])
            
            morph = self.morphisms[key]
            morph.frequency += 1
            
            # Compute quality
            if i + self.n < len(dna_sequence):
                q = quality_fn(dna_sequence[i + self.n - 1], dna_sequence[i + self.n])
                morph.qualities.append(q)
            
            # Store timestamp if available
            if dna_sequence[i].timestamp is not None:
                morph.timestamps.append(dna_sequence[i].timestamp)
            
            self._total_observations += 1
        
        # Finalize quality computation
        for morph in self.morphisms.values():
            if morph.qualities:
                morph.quality = np.mean(morph.qualities)
        
        self._entropy_cache = None  # Invalidate cache
        return self
    
    def get_morphism(self, source: str, target: str) -> Optional[Morphism]:
        """Get morphism between source and target, if exists."""
        return self.morphisms.get((source, target))
    
    def get_outgoing(self, obj: str) -> List[Morphism]:
        """Get all morphisms leaving from object."""
        return self._outgoing.get(obj, [])
    
    def get_incoming(self, obj: str) -> List[Morphism]:
        """Get all morphisms arriving at object."""
        return self._incoming.get(obj, [])
    
    def identity(self, obj: str) -> Morphism:
        """Get identity morphism for object (self-loop)."""
        key = (obj, obj)
        if key not in self.morphisms:
            self.morphisms[key] = Morphism(source=obj, target=obj, frequency=0)
        return self.morphisms[key]
    
    def compose(self, f: Morphism, g: Morphism) -> Optional[Morphism]:
        """
        Compose morphisms: g ∘ f (f then g).
        Returns morphism from f.source to g.target if exists.
        """
        if f.target != g.source:
            return None  # Not composable
        return self.get_morphism(f.source, g.target)
    
    @property
    def num_objects(self) -> int:
        return len(self.objects)
    
    @property
    def num_morphisms(self) -> int:
        return len(self.morphisms)
    
    def transition_probability(self, source: str, target: str) -> float:
        """
        Compute transition probability P(target | source).
        """
        morph = self.get_morphism(source, target)
        if morph is None:
            return 0.0
        
        total_from_source = sum(m.frequency for m in self.get_outgoing(source))
        if total_from_source == 0:
            return 0.0
        
        return morph.frequency / total_from_source
    
    def get_transition_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Build transition probability matrix.
        
        Returns:
            Tuple of (matrix, index_map) where index_map[obj] = index in matrix
        """
        objects = sorted(self.objects)
        idx = {obj: i for i, obj in enumerate(objects)}
        n = len(objects)
        
        matrix = np.zeros((n, n))
        for (src, tgt), morph in self.morphisms.items():
            if src in idx and tgt in idx:
                matrix[idx[src], idx[tgt]] = morph.frequency
        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
        
        return matrix, idx
    
    def entropy(self) -> float:
        """
        Compute structural entropy H(σ) of the category.
        
        H = -Σ p_i log p_i where p_i is frequency of n-gram i.
        """
        if self._entropy_cache is not None:
            return self._entropy_cache
        
        if self._total_observations == 0:
            return 0.0
        
        # Count object frequencies (sum of incoming morphism frequencies)
        obj_freq = {}
        for obj in self.objects:
            freq = sum(m.frequency for m in self.get_incoming(obj))
            if freq > 0:
                obj_freq[obj] = freq
        
        total = sum(obj_freq.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for freq in obj_freq.values():
            p = freq / total
            if p > 0:
                entropy -= p * np.log(p)
        
        self._entropy_cache = entropy
        return entropy
    
    def morphism_entropy(self) -> float:
        """
        Compute entropy over morphisms (transitions).
        """
        if not self.morphisms:
            return 0.0
        
        total = sum(m.frequency for m in self.morphisms.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for morph in self.morphisms.values():
            p = morph.frequency / total
            if p > 0:
                entropy -= p * np.log(p)
        
        return entropy
    
    def extend(self, dna_sequence: List[StructuralDNA]) -> 'CategoryCn':
        """
        Functor N: C_n → C_{n+1}
        
        Extends context by creating C_{n+1} from same sequence.
        """
        extended = CategoryCn(n=self.n + 1)
        extended.build_from_sequence(dna_sequence)
        return extended
    
    def reduce(self) -> 'CategoryCn':
        """
        Functor R: C_n → C_{n-1}
        
        Reduces context by truncating n-grams.
        """
        if self.n <= 1:
            return self  # Can't reduce below 1-gram
        
        reduced = CategoryCn(n=self.n - 1)
        
        # Map objects
        for obj in self.objects:
            reduced.objects.add(obj[:-1])  # Truncate last symbol
        
        # Map morphisms
        for (src, tgt), morph in self.morphisms.items():
            red_src = src[:-1]
            red_tgt = tgt[:-1]
            
            key = (red_src, red_tgt)
            if key not in reduced.morphisms:
                reduced.morphisms[key] = Morphism(source=red_src, target=red_tgt)
                reduced._outgoing[red_src].append(reduced.morphisms[key])
                reduced._incoming[red_tgt].append(reduced.morphisms[key])
            
            # Accumulate statistics
            red_morph = reduced.morphisms[key]
            red_morph.frequency += morph.frequency
            red_morph.qualities.extend(morph.qualities)
        
        # Recompute qualities
        for morph in reduced.morphisms.values():
            if morph.qualities:
                morph.quality = np.mean(morph.qualities)
        
        reduced._total_observations = self._total_observations
        return reduced
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics about the category."""
        return {
            'n': self.n,
            'num_objects': self.num_objects,
            'num_morphisms': self.num_morphisms,
            'total_observations': self._total_observations,
            'object_entropy': self.entropy(),
            'morphism_entropy': self.morphism_entropy(),
            'avg_out_degree': np.mean([len(self.get_outgoing(o)) for o in self.objects]) if self.objects else 0,
            'max_out_degree': max([len(self.get_outgoing(o)) for o in self.objects]) if self.objects else 0,
        }
    
    def __repr__(self) -> str:
        return f"CategoryCn(n={self.n}, |Obj|={self.num_objects}, |Mor|={self.num_morphisms})"


class CategoryHierarchy:
    """
    Hierarchy of categories C_1, C_2, ..., C_max_n with functors.
    
    Provides unified interface for multi-scale analysis.
    """
    
    def __init__(self, max_n: int = 5):
        self.max_n = max_n
        self.categories: Dict[int, CategoryCn] = {}
    
    def build_from_sequence(
        self,
        dna_sequence: List[StructuralDNA],
        quality_fn: Optional[callable] = None
    ) -> 'CategoryHierarchy':
        """Build all categories from sequence."""
        for n in range(1, self.max_n + 1):
            cat = CategoryCn(n=n)
            cat.build_from_sequence(dna_sequence, quality_fn)
            self.categories[n] = cat
        return self
    
    def get_category(self, n: int) -> Optional[CategoryCn]:
        """Get category C_n."""
        return self.categories.get(n)
    
    def depth_analysis(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze structure at each depth.
        
        Returns dict mapping n -> statistics.
        """
        return {n: cat.get_statistics() for n, cat in self.categories.items()}
    
    def estimate_computational_depth(self) -> float:
        """
        Estimate self-computational depth d from entropy growth.
        
        d ≈ n* where entropy growth saturates.
        """
        if len(self.categories) < 2:
            return 1.0
        
        entropies = []
        for n in sorted(self.categories.keys()):
            entropies.append(self.categories[n].entropy())
        
        # Find where entropy growth rate drops below threshold
        growth_rates = np.diff(entropies)
        
        if len(growth_rates) == 0:
            return 1.0
        
        # Depth where growth drops below 10% of max growth
        max_growth = max(abs(g) for g in growth_rates) if growth_rates.any() else 1
        threshold = 0.1 * max_growth
        
        for i, rate in enumerate(growth_rates):
            if abs(rate) < threshold:
                return float(i + 1)
        
        return float(len(growth_rates))
    
    def __repr__(self) -> str:
        cats_str = ', '.join(f"C_{n}" for n in sorted(self.categories.keys()))
        return f"CategoryHierarchy([{cats_str}])"


if __name__ == "__main__":
    # Test with symbolic sequence
    from symbolic_dna import SymbolicDNAEncoder
    import pandas as pd
    
    # Load data
    df = pd.read_feather("../../data/BTC_USDT_USDT-4h-futures.feather")
    
    # Encode
    encoder = SymbolicDNAEncoder()
    dna_seq = encoder.fit_transform(df)
    
    print(f"DNA sequence length: {len(dna_seq)}")
    
    # Build category hierarchy
    hierarchy = CategoryHierarchy(max_n=5)
    hierarchy.build_from_sequence(dna_seq)
    
    print(f"\n{hierarchy}")
    
    # Analyze each depth
    print("\nDepth analysis:")
    for n, stats in hierarchy.depth_analysis().items():
        print(f"\n  C_{n}:")
        for key, val in stats.items():
            print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")
    
    # Estimate computational depth
    d = hierarchy.estimate_computational_depth()
    print(f"\nEstimated computational depth: d ≈ {d:.2f}")
    
    # Show top morphisms in C_3
    C3 = hierarchy.get_category(3)
    print(f"\nTop 10 morphisms in C_3:")
    top_morphisms = sorted(C3.morphisms.values(), key=lambda m: -m.frequency)[:10]
    for m in top_morphisms:
        print(f"  {m.source} → {m.target}: freq={m.frequency}, quality={m.quality:.4f}")
