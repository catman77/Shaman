# -*- coding: utf-8 -*-
"""
Free Energy Minimization for NOBS.

Implements the free energy functional:
F[σ] = E[σ] - T·S[σ]

Where:
- E[σ] = expected cost (quality of transitions)
- S[σ] = structural entropy
- T = temperature parameter
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .category import CategoryCn, Morphism


@dataclass
class FreeEnergyState:
    """
    State in free energy landscape.
    
    Attributes:
        ngram: Current n-gram string
        energy: Expected cost E
        entropy: Structural entropy S
        free_energy: F = E - T*S
        temperature: Temperature parameter used
    """
    ngram: str
    energy: float
    entropy: float
    free_energy: float
    temperature: float
    
    def __repr__(self):
        return f"FreeEnergyState({self.ngram}, F={self.free_energy:.4f})"


class FreeEnergyMinimizer:
    """
    Free Energy minimization for structural learning.
    
    Computes F[σ] = E[σ] - T·S[σ] for n-grams and finds
    minimum free energy paths through the category.
    
    Parameters:
        temperature: Temperature parameter T (default: 1.0)
        energy_scale: Scale factor for energy term (default: 1.0)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        energy_scale: float = 1.0
    ):
        self.temperature = temperature
        self.energy_scale = energy_scale
        
        self._category: Optional[CategoryCn] = None
        self._states: Dict[str, FreeEnergyState] = {}
    
    def fit(self, category: CategoryCn) -> 'FreeEnergyMinimizer':
        """
        Compute free energy for all objects in category.
        
        Args:
            category: CategoryCn to analyze
            
        Returns:
            self (fitted minimizer)
        """
        self._category = category
        self._states = {}
        
        # Global entropy for normalization
        global_entropy = category.entropy()
        
        for obj in category.objects:
            # Energy = negative average quality of outgoing morphisms
            # (lower quality = higher energy)
            outgoing = category.get_outgoing(obj)
            
            if outgoing:
                # Energy from quality of transitions
                qualities = [m.quality for m in outgoing]
                weights = [m.frequency for m in outgoing]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    weighted_quality = sum(q * w for q, w in zip(qualities, weights)) / total_weight
                else:
                    weighted_quality = 0.0
                
                # Convert quality to energy (higher quality = lower energy)
                energy = -weighted_quality * self.energy_scale
                
                # Local entropy: diversity of outgoing transitions
                probs = np.array(weights) / total_weight
                local_entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                energy = 0.0
                local_entropy = 0.0
            
            # Free energy
            free_energy = energy - self.temperature * local_entropy
            
            self._states[obj] = FreeEnergyState(
                ngram=obj,
                energy=energy,
                entropy=local_entropy,
                free_energy=free_energy,
                temperature=self.temperature
            )
        
        return self
    
    def get_state(self, ngram: str) -> Optional[FreeEnergyState]:
        """Get free energy state for n-gram."""
        return self._states.get(ngram)
    
    def get_minimum_free_energy_states(self, top_k: int = 10) -> List[FreeEnergyState]:
        """Get states with lowest free energy."""
        return sorted(self._states.values(), key=lambda s: s.free_energy)[:top_k]
    
    def get_maximum_entropy_states(self, top_k: int = 10) -> List[FreeEnergyState]:
        """Get states with highest entropy (most exploration potential)."""
        return sorted(self._states.values(), key=lambda s: -s.entropy)[:top_k]
    
    def softmax_weights(self) -> Dict[str, float]:
        """
        Compute softmax weights over all states based on negative free energy.
        
        States with lower free energy get higher weight.
        """
        if not self._states:
            return {}
        
        # Negative free energy (higher = better)
        neg_F = np.array([-s.free_energy for s in self._states.values()])
        
        # Softmax
        exp_F = np.exp(neg_F - np.max(neg_F))  # Subtract max for stability
        softmax = exp_F / exp_F.sum()
        
        return {
            ngram: float(w)
            for ngram, w in zip(self._states.keys(), softmax)
        }
    
    def morphism_free_energy(self, morphism: Morphism) -> float:
        """
        Compute free energy of a morphism.
        
        F_morph = -quality - T * entropy
        """
        return -morphism.quality * self.energy_scale - self.temperature * morphism.entropy
    
    def optimal_transition(self, from_state: str) -> Optional[Morphism]:
        """
        Find optimal (minimum free energy) transition from state.
        """
        if self._category is None:
            return None
        
        outgoing = self._category.get_outgoing(from_state)
        if not outgoing:
            return None
        
        # Find morphism with minimum F_target + F_morph
        best_morph = None
        best_F = float('inf')
        
        for morph in outgoing:
            target_state = self._states.get(morph.target)
            if target_state is None:
                continue
            
            F_total = target_state.free_energy + self.morphism_free_energy(morph)
            if F_total < best_F:
                best_F = F_total
                best_morph = morph
        
        return best_morph
    
    def compute_path_free_energy(self, path: List[str]) -> float:
        """
        Compute total free energy of a path through the category.
        """
        if not path:
            return 0.0
        
        total_F = 0.0
        
        # State energies
        for ngram in path:
            state = self._states.get(ngram)
            if state:
                total_F += state.free_energy
        
        # Transition energies
        if self._category:
            for i in range(len(path) - 1):
                morph = self._category.get_morphism(path[i], path[i + 1])
                if morph:
                    total_F += self.morphism_free_energy(morph)
        
        return total_F
    
    def find_minimum_energy_path(
        self,
        start: str,
        length: int = 10,
        greedy: bool = True
    ) -> Tuple[List[str], float]:
        """
        Find path of given length starting from state with minimum total free energy.
        
        Args:
            start: Starting n-gram
            length: Path length
            greedy: If True, use greedy selection; else use probabilistic
            
        Returns:
            Tuple of (path, total_free_energy)
        """
        if self._category is None or start not in self._states:
            return [start], 0.0
        
        path = [start]
        current = start
        
        for _ in range(length - 1):
            outgoing = self._category.get_outgoing(current)
            if not outgoing:
                break
            
            if greedy:
                # Greedy: pick minimum free energy transition
                morph = self.optimal_transition(current)
                if morph is None:
                    break
                current = morph.target
            else:
                # Probabilistic: sample based on softmax of -F
                F_values = []
                morphs = []
                for m in outgoing:
                    target_state = self._states.get(m.target)
                    if target_state:
                        F_values.append(target_state.free_energy + self.morphism_free_energy(m))
                        morphs.append(m)
                
                if not morphs:
                    break
                
                # Softmax sampling
                F_arr = np.array(F_values)
                probs = np.exp(-F_arr - np.max(-F_arr))
                probs = probs / probs.sum()
                
                idx = np.random.choice(len(morphs), p=probs)
                current = morphs[idx].target
            
            path.append(current)
        
        total_F = self.compute_path_free_energy(path)
        return path, total_F
    
    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self._states:
            return {}
        
        energies = [s.energy for s in self._states.values()]
        entropies = [s.entropy for s in self._states.values()]
        free_energies = [s.free_energy for s in self._states.values()]
        
        return {
            'temperature': self.temperature,
            'num_states': len(self._states),
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'mean_free_energy': np.mean(free_energies),
            'std_free_energy': np.std(free_energies),
            'min_free_energy': min(free_energies),
            'max_free_energy': max(free_energies),
        }


class AdaptiveTemperature:
    """
    Adaptive temperature scheduler for free energy minimization.
    
    Analogous to simulated annealing, but based on structural complexity.
    """
    
    def __init__(
        self,
        initial_temp: float = 2.0,
        min_temp: float = 0.1,
        cooling_rate: float = 0.95
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.current_temp = initial_temp
        self._step = 0
    
    def step(self) -> float:
        """Advance temperature and return new value."""
        self._step += 1
        self.current_temp = max(
            self.min_temp,
            self.initial_temp * (self.cooling_rate ** self._step)
        )
        return self.current_temp
    
    def reset(self):
        """Reset temperature to initial."""
        self.current_temp = self.initial_temp
        self._step = 0
    
    def adapt_to_entropy(self, category_entropy: float, target_entropy: float = 2.0):
        """
        Adapt temperature based on category entropy.
        
        Higher entropy -> higher temperature (more exploration)
        Lower entropy -> lower temperature (more exploitation)
        """
        ratio = category_entropy / (target_entropy + 1e-10)
        self.current_temp = self.min_temp + (self.initial_temp - self.min_temp) * min(ratio, 2.0)


if __name__ == "__main__":
    # Test free energy computation
    from symbolic_dna import SymbolicDNAEncoder
    from category import CategoryCn
    import pandas as pd
    
    # Load data
    df = pd.read_feather("../../data/BTC_USDT_USDT-4h-futures.feather")
    
    # Encode
    encoder = SymbolicDNAEncoder()
    dna_seq = encoder.fit_transform(df)
    
    # Build category
    C3 = CategoryCn(n=3)
    C3.build_from_sequence(dna_seq)
    
    print(f"Category: {C3}")
    
    # Fit free energy minimizer
    minimizer = FreeEnergyMinimizer(temperature=1.0)
    minimizer.fit(C3)
    
    # Statistics
    print("\nFree Energy Statistics:")
    for key, val in minimizer.get_statistics().items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    # Top minimum free energy states
    print("\nTop 10 minimum free energy states:")
    for state in minimizer.get_minimum_free_energy_states(10):
        print(f"  {state.ngram}: F={state.free_energy:.4f}, E={state.energy:.4f}, S={state.entropy:.4f}")
    
    # Top high entropy states
    print("\nTop 10 high entropy states:")
    for state in minimizer.get_maximum_entropy_states(10):
        print(f"  {state.ngram}: S={state.entropy:.4f}, F={state.free_energy:.4f}")
    
    # Find optimal path
    start = list(C3.objects)[0]
    path, total_F = minimizer.find_minimum_energy_path(start, length=20)
    print(f"\nOptimal path from {start}:")
    print(f"  Path: {' → '.join(path[:10])}...")
    print(f"  Total F: {total_F:.4f}")
