# -*- coding: utf-8 -*-
"""
NOBS-based Shaman Experiment.

This experiment uses the NOBS (Natural Observation-Based Computing) framework
for semantic space S instead of simple sentence-transformer embeddings.

The key insight from NOBS: meaning is encoded in STRUCTURAL patterns,
not just vector embeddings. The Shaman B_ш operates in a rich categorical
space where transitions between symbolic states carry semantic information.

Experiment Protocol:
1. Load Bitcoin OHLCV data
2. Build NOBS space (Symbolic DNA → Category Hierarchy → Free Energy)
3. Create two "agents" as different views/windows of the data
4. Shaman learns to find resonance between their structural patterns
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nobs.symbolic_dna import SymbolicDNAEncoder, StructuralDNA, Symbol
from nobs.category import CategoryCn, CategoryHierarchy, Morphism
from nobs.free_energy import FreeEnergyMinimizer, AdaptiveTemperature
from nobs.space import NOBSSpace, NOBSEmbedding


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NOBSExperimentConfig:
    """Configuration for NOBS experiment."""
    # Data
    data_path: str = "data/BTC_USDT_USDT-4h-futures.feather"
    
    # NOBS Space
    max_depth: int = 5
    embedding_dim: int = 128
    temperature: float = 1.0
    
    # Agent windows
    agent_a_window: Tuple[float, float] = (0.0, 0.5)  # First half of data
    agent_b_window: Tuple[float, float] = (0.5, 1.0)  # Second half
    
    # Shaman parameters
    window_size: int = 50  # Window size for local embedding
    step_size: int = 10    # Step between windows
    shaman_iterations: int = 200
    learning_rate: float = 0.01
    
    # Experiment
    num_runs: int = 3
    seed: int = 42
    
    # Success criteria
    distance_threshold: float = 0.3  # d_P threshold for success
    correlation_threshold: float = 0.7  # Pattern correlation threshold


@dataclass 
class PatternSignature:
    """
    Signature of agent's structural patterns.
    
    This is the "semantic invariant" s_A that we want to transfer.
    """
    # Symbol distribution
    symbol_dist: Dict[str, float]
    
    # Dominant morphisms (most frequent transitions)
    dominant_morphisms: List[Tuple[str, int]]  # (morphism_string, frequency)
    
    # Free energy profile
    mean_free_energy: float
    std_free_energy: float
    
    # Entropy profile
    mean_entropy: float
    morphism_entropy: float
    
    # Embedding centroid
    centroid: np.ndarray
    
    # N-gram pattern frequencies
    ngram_patterns: Dict[int, Dict[str, float]]  # depth -> pattern -> freq
    
    def distance_to(self, other: 'PatternSignature') -> float:
        """Compute distance to another signature."""
        # Weighted combination of different distance measures
        
        # 1. Symbol distribution distance (Jensen-Shannon)
        dist_symbols = self._js_divergence(
            self.symbol_dist, other.symbol_dist
        )
        
        # 2. Morphism overlap (Jaccard-like)
        my_morphs = set(m for m, _ in self.dominant_morphisms[:20])
        other_morphs = set(m for m, _ in other.dominant_morphisms[:20])
        if my_morphs or other_morphs:
            morphism_sim = len(my_morphs & other_morphs) / len(my_morphs | other_morphs)
        else:
            morphism_sim = 1.0
        dist_morphisms = 1 - morphism_sim
        
        # 3. Energy distance
        dist_energy = abs(self.mean_free_energy - other.mean_free_energy)
        
        # 4. Entropy distance
        dist_entropy = abs(self.mean_entropy - other.mean_entropy)
        
        # 5. Centroid distance
        dist_centroid = np.linalg.norm(self.centroid - other.centroid)
        
        # Weighted sum
        return (
            0.2 * dist_symbols +
            0.3 * dist_morphisms +
            0.1 * dist_energy +
            0.1 * dist_entropy +
            0.3 * min(1.0, dist_centroid / 2.0)  # Normalize
        )
    
    def _js_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Jensen-Shannon divergence."""
        all_keys = set(p.keys()) | set(q.keys())
        p_arr = np.array([p.get(k, 1e-10) for k in all_keys])
        q_arr = np.array([q.get(k, 1e-10) for k in all_keys])
        
        # Normalize
        p_arr = p_arr / (p_arr.sum() + 1e-10)
        q_arr = q_arr / (q_arr.sum() + 1e-10)
        
        # M = (P + Q) / 2
        m = (p_arr + q_arr) / 2
        
        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_pm = np.sum(p_arr * np.log(p_arr / (m + 1e-10) + 1e-10))
        kl_qm = np.sum(q_arr * np.log(q_arr / (m + 1e-10) + 1e-10))
        
        return 0.5 * kl_pm + 0.5 * kl_qm


class NOBSAgent:
    """
    Agent in NOBS space.
    
    An agent is characterized by its pattern signature over a window of data.
    """
    
    def __init__(
        self,
        name: str,
        space: NOBSSpace,
        dna_sequence: List[StructuralDNA],
        start_idx: int,
        end_idx: int
    ):
        self.name = name
        self.space = space
        self.dna_sequence = dna_sequence
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Build agent's view
        self.agent_dna = dna_sequence[start_idx:end_idx]
        
        # Compute signature
        self.signature = self._compute_signature()
    
    def _compute_signature(self) -> PatternSignature:
        """Compute structural pattern signature."""
        symbols = [dna.to_symbol() for dna in self.agent_dna]
        
        # Symbol distribution
        from collections import Counter
        sym_counts = Counter(symbols)
        total = len(symbols)
        symbol_dist = {s: sym_counts.get(s, 0) / total for s in NOBSSpace.SYMBOLS}
        
        # Build local category hierarchy
        hierarchy = CategoryHierarchy(max_n=self.space.max_depth)
        hierarchy.build_from_sequence(self.agent_dna)
        
        # Get dominant morphisms from C_3
        C3 = hierarchy.get_category(3)
        dominant_morphisms = []
        if C3:
            sorted_morphisms = sorted(
                C3.morphisms.values(),
                key=lambda m: -m.frequency
            )
            dominant_morphisms = [
                (f"{m.source}→{m.target}", m.frequency)
                for m in sorted_morphisms[:50]
            ]
        
        # Free energy profile
        minimizer = FreeEnergyMinimizer(temperature=self.space.temperature)
        if C3:
            minimizer.fit(C3)
            fe_stats = minimizer.get_statistics()
            mean_fe = fe_stats.get('mean_free_energy', 0.0)
            std_fe = fe_stats.get('std_free_energy', 0.0)
        else:
            mean_fe, std_fe = 0.0, 0.0
        
        # Entropy
        if C3:
            mean_entropy = C3.entropy()
            morph_entropy = C3.morphism_entropy()
        else:
            mean_entropy, morph_entropy = 0.0, 0.0
        
        # Compute embeddings and centroid
        embeddings = []
        window_size = min(50, len(self.agent_dna) // 10)
        step = max(1, window_size // 5)
        
        for i in range(0, len(self.agent_dna) - window_size, step):
            window = self.agent_dna[i:i + window_size]
            emb = self.space.embed_sequence(window)
            embeddings.append(emb.vector)
        
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
        else:
            centroid = np.zeros(self.space.embedding_dim)
        
        # N-gram patterns
        ngram_patterns = {}
        for n in range(1, self.space.max_depth + 1):
            cat = hierarchy.get_category(n)
            if cat and cat.num_objects > 0:
                total_freq = sum(
                    sum(m.frequency for m in cat.get_incoming(obj))
                    for obj in cat.objects
                )
                patterns = {}
                for obj in cat.objects:
                    freq = sum(m.frequency for m in cat.get_incoming(obj))
                    if freq > 0:
                        patterns[obj] = freq / (total_freq + 1e-10)
                ngram_patterns[n] = patterns
            else:
                ngram_patterns[n] = {}
        
        return PatternSignature(
            symbol_dist=symbol_dist,
            dominant_morphisms=dominant_morphisms,
            mean_free_energy=mean_fe,
            std_free_energy=std_fe,
            mean_entropy=mean_entropy,
            morphism_entropy=morph_entropy,
            centroid=centroid,
            ngram_patterns=ngram_patterns
        )
    
    def get_embedding_at(self, idx: int, window_size: int = 50) -> NOBSEmbedding:
        """Get embedding at specific index within agent's data."""
        start = max(0, idx - window_size // 2)
        end = min(len(self.agent_dna), start + window_size)
        window = self.agent_dna[start:end]
        return self.space.embed_sequence(window)


class NOBSShaman:
    """
    Shaman in NOBS space.
    
    The Shaman's task: given only the pattern signature of Agent A (the "anchor"),
    guide Agent B to produce patterns that resonate with A's signature.
    
    Key constraint: Shaman has NO direct access to A's data or responses,
    only the structural signature (centroid, dominant morphisms, energy profile).
    """
    
    def __init__(
        self,
        space: NOBSSpace,
        anchor_signature: PatternSignature,
        temperature: float = 1.0,
        learning_rate: float = 0.01
    ):
        self.space = space
        self.anchor = anchor_signature
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        # Shaman's learned parameters (what modifications to suggest)
        # These control how to transform B's patterns
        self.params = {
            'symbol_weights': np.ones(6) / 6,  # Weights for each symbol
            'depth_focus': np.array([0.1, 0.2, 0.4, 0.2, 0.1]),  # Focus on each depth
            'temperature_mod': 0.0,  # Modification to temperature
            'entropy_target': anchor_signature.mean_entropy,  # Target entropy
        }
        
        # Memory
        self.history: List[Dict[str, float]] = []
        self.best_distance = float('inf')
    
    def suggest_modification(self, current_signature: PatternSignature) -> Dict[str, Any]:
        """
        Suggest how to modify B's behavior to move closer to anchor.
        
        Returns dict of suggested modifications.
        """
        # Compute current distance
        distance = current_signature.distance_to(self.anchor)
        
        # Analyze what's different
        suggestions = {}
        
        # 1. Symbol distribution adjustment
        sym_diff = {}
        for sym in NOBSSpace.SYMBOLS:
            diff = self.anchor.symbol_dist.get(sym, 0) - current_signature.symbol_dist.get(sym, 0)
            sym_diff[sym] = diff
        suggestions['symbol_adjustments'] = sym_diff
        
        # 2. Target morphisms (ones that A has but B is missing)
        anchor_morphs = set(m for m, _ in self.anchor.dominant_morphisms[:20])
        current_morphs = set(m for m, _ in current_signature.dominant_morphisms[:20])
        missing = anchor_morphs - current_morphs
        suggestions['target_morphisms'] = list(missing)[:5]
        
        # 3. Energy/entropy guidance
        suggestions['energy_direction'] = np.sign(
            self.anchor.mean_free_energy - current_signature.mean_free_energy
        )
        suggestions['entropy_direction'] = np.sign(
            self.anchor.mean_entropy - current_signature.mean_entropy
        )
        
        # 4. Temperature suggestion
        if distance > 0.5:
            # High distance: explore more (higher T)
            suggestions['temperature'] = self.temperature * 1.2
        else:
            # Close: exploit (lower T)
            suggestions['temperature'] = self.temperature * 0.8
        
        return suggestions
    
    def compute_reward(
        self,
        before_signature: PatternSignature,
        after_signature: PatternSignature
    ) -> float:
        """
        Compute reward for a modification step.
        
        Reward = improvement in distance to anchor.
        """
        dist_before = before_signature.distance_to(self.anchor)
        dist_after = after_signature.distance_to(self.anchor)
        
        # Reward is reduction in distance
        improvement = dist_before - dist_after
        
        # Bonus for crossing thresholds
        if dist_after < 0.3 and dist_before >= 0.3:
            improvement += 0.5
        if dist_after < 0.2 and dist_before >= 0.2:
            improvement += 1.0
        
        return improvement
    
    def update(self, reward: float, suggestions: Dict[str, Any]):
        """Update Shaman's parameters based on reward."""
        # Simple gradient-like update
        if reward > 0:
            # Reinforce current direction
            self.temperature *= (1 - 0.1 * self.learning_rate)
        else:
            # Explore more
            self.temperature *= (1 + 0.1 * self.learning_rate)
        
        # Clamp temperature
        self.temperature = np.clip(self.temperature, 0.1, 3.0)
    
    def log_step(
        self,
        step: int,
        distance: float,
        reward: float,
        suggestions: Dict[str, Any]
    ):
        """Log a step in history."""
        self.history.append({
            'step': step,
            'distance': distance,
            'reward': reward,
            'temperature': self.temperature,
            'target_morphisms': len(suggestions.get('target_morphisms', [])),
        })
        
        if distance < self.best_distance:
            self.best_distance = distance


class NOBSExperiment:
    """
    NOBS-based Shaman experiment.
    
    Tests whether a Shaman can guide Agent B to produce structural patterns
    that resonate with Agent A's patterns, using only the pattern signature.
    """
    
    def __init__(self, config: NOBSExperimentConfig):
        self.config = config
        np.random.seed(config.seed)
        
        self.space: Optional[NOBSSpace] = None
        self.agent_a: Optional[NOBSAgent] = None
        self.agent_b: Optional[NOBSAgent] = None
        self.shaman: Optional[NOBSShaman] = None
        
        self.results: Dict[str, Any] = {}
    
    def setup(self):
        """Initialize experiment components."""
        logger.info("Loading data...")
        df = pd.read_feather(self.config.data_path)
        logger.info(f"Loaded {len(df)} candles")
        
        # Build NOBS space
        logger.info("Building NOBS space...")
        self.space = NOBSSpace(
            max_depth=self.config.max_depth,
            embedding_dim=self.config.embedding_dim,
            temperature=self.config.temperature
        )
        self.space.fit(df)
        
        # Get DNA sequence
        dna_seq = self.space._dna_sequence
        n_total = len(dna_seq)
        
        # Create agents
        logger.info("Creating agents...")
        
        # Agent A: first portion of data
        a_start = int(n_total * self.config.agent_a_window[0])
        a_end = int(n_total * self.config.agent_a_window[1])
        self.agent_a = NOBSAgent("Agent_A", self.space, dna_seq, a_start, a_end)
        
        # Agent B: second portion of data
        b_start = int(n_total * self.config.agent_b_window[0])
        b_end = int(n_total * self.config.agent_b_window[1])
        self.agent_b = NOBSAgent("Agent_B", self.space, dna_seq, b_start, b_end)
        
        logger.info(f"Agent A: {len(self.agent_a.agent_dna)} samples")
        logger.info(f"Agent B: {len(self.agent_b.agent_dna)} samples")
        
        # Initial distance
        initial_distance = self.agent_a.signature.distance_to(self.agent_b.signature)
        logger.info(f"Initial pattern distance: {initial_distance:.4f}")
        
        # Create Shaman with A's signature as anchor
        logger.info("Creating Shaman...")
        self.shaman = NOBSShaman(
            space=self.space,
            anchor_signature=self.agent_a.signature,
            temperature=self.config.temperature,
            learning_rate=self.config.learning_rate
        )
        
        self.results['initial_distance'] = initial_distance
        self.results['agent_a_entropy'] = self.agent_a.signature.mean_entropy
        self.results['agent_b_entropy'] = self.agent_b.signature.mean_entropy
    
    def run_resonance_search(self) -> Dict[str, float]:
        """
        Run the resonance search process.
        
        Shaman iteratively suggests modifications to B's interpretation,
        measuring how close B's patterns get to A's patterns.
        """
        logger.info("\n" + "="*50)
        logger.info("RESONANCE SEARCH")
        logger.info("="*50)
        logger.info(f"Shaman has ONLY Agent A's pattern signature (anchor)")
        logger.info(f"Goal: Guide B to produce resonant patterns")
        
        # Track current "view" of B (which window we're looking at)
        b_length = len(self.agent_b.agent_dna)
        window_size = self.config.window_size
        
        distances = []
        rewards = []
        
        # Current B signature (starts as full agent B)
        current_signature = self.agent_b.signature
        
        for step in tqdm(range(self.config.shaman_iterations), desc="Shaman searching"):
            # Get Shaman's suggestion
            suggestions = self.shaman.suggest_modification(current_signature)
            
            # Apply suggestion by selecting different window of B
            # (In real scenario, this would modify B's prompting/generation)
            # Here we simulate by moving to windows that better match suggestions
            
            # Select window based on suggestions
            best_window = None
            best_alignment = -float('inf')
            
            # Sample random windows and pick one aligned with suggestions
            for _ in range(10):
                start = np.random.randint(0, max(1, b_length - window_size))
                window = self.agent_b.agent_dna[start:start + window_size]
                
                # Quick alignment check
                symbols = [dna.to_symbol() for dna in window]
                sym_count = {s: symbols.count(s) / len(symbols) for s in NOBSSpace.SYMBOLS}
                
                # Alignment = how well this window matches target direction
                alignment = 0
                for sym, adj in suggestions['symbol_adjustments'].items():
                    if adj > 0:  # We want more of this symbol
                        alignment += sym_count.get(sym, 0) * adj
                    else:  # We want less
                        alignment -= sym_count.get(sym, 0) * abs(adj)
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_window = window
            
            # Compute new signature from selected window
            if best_window:
                new_sig = self._compute_window_signature(best_window)
            else:
                new_sig = current_signature
            
            # Compute reward
            reward = self.shaman.compute_reward(current_signature, new_sig)
            
            # Update shaman
            self.shaman.update(reward, suggestions)
            
            # Log
            distance = new_sig.distance_to(self.shaman.anchor)
            self.shaman.log_step(step, distance, reward, suggestions)
            
            distances.append(distance)
            rewards.append(reward)
            
            # Update current signature (momentum)
            current_signature = new_sig
        
        # Final metrics
        final_distance = distances[-1] if distances else self.results['initial_distance']
        mean_reward = np.mean(rewards) if rewards else 0.0
        
        return {
            'final_distance': final_distance,
            'mean_reward': mean_reward,
            'best_distance': self.shaman.best_distance,
            'distance_reduction': self.results['initial_distance'] - final_distance,
            'reduction_pct': (self.results['initial_distance'] - final_distance) / self.results['initial_distance'] * 100,
            'distances': distances,
            'rewards': rewards,
        }
    
    def _compute_window_signature(self, window: List[StructuralDNA]) -> PatternSignature:
        """Compute signature for a window."""
        symbols = [dna.to_symbol() for dna in window]
        
        # Symbol distribution
        from collections import Counter
        sym_counts = Counter(symbols)
        total = len(symbols)
        symbol_dist = {s: sym_counts.get(s, 0) / total for s in NOBSSpace.SYMBOLS}
        
        # Quick category
        hierarchy = CategoryHierarchy(max_n=3)
        hierarchy.build_from_sequence(window)
        
        C3 = hierarchy.get_category(3)
        dominant_morphisms = []
        if C3:
            sorted_morphisms = sorted(C3.morphisms.values(), key=lambda m: -m.frequency)
            dominant_morphisms = [(f"{m.source}→{m.target}", m.frequency) for m in sorted_morphisms[:20]]
        
        # Energy
        minimizer = FreeEnergyMinimizer(temperature=self.space.temperature)
        if C3 and C3.num_objects > 0:
            minimizer.fit(C3)
            fe_stats = minimizer.get_statistics()
            mean_fe = fe_stats.get('mean_free_energy', 0.0)
            std_fe = fe_stats.get('std_free_energy', 0.0)
            mean_entropy = C3.entropy()
            morph_entropy = C3.morphism_entropy()
        else:
            mean_fe, std_fe = 0.0, 0.0
            mean_entropy, morph_entropy = 0.0, 0.0
        
        # Centroid
        emb = self.space.embed_sequence(window)
        centroid = emb.vector
        
        return PatternSignature(
            symbol_dist=symbol_dist,
            dominant_morphisms=dominant_morphisms,
            mean_free_energy=mean_fe,
            std_free_energy=std_fe,
            mean_entropy=mean_entropy,
            morphism_entropy=morph_entropy,
            centroid=centroid,
            ngram_patterns={}
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the full experiment."""
        logger.info("\n" + "#"*60)
        logger.info("# NOBS SHAMAN EXPERIMENT")
        logger.info("#"*60)
        
        self.setup()
        
        all_results = []
        
        for run_idx in range(self.config.num_runs):
            logger.info(f"\n{'='*50}")
            logger.info(f"RUN {run_idx + 1}/{self.config.num_runs}")
            logger.info(f"{'='*50}")
            
            # Reset shaman for each run
            self.shaman = NOBSShaman(
                space=self.space,
                anchor_signature=self.agent_a.signature,
                temperature=self.config.temperature,
                learning_rate=self.config.learning_rate
            )
            
            run_results = self.run_resonance_search()
            all_results.append(run_results)
            
            logger.info(f"\nRun {run_idx + 1} Results:")
            logger.info(f"  Initial distance: {self.results['initial_distance']:.4f}")
            logger.info(f"  Final distance: {run_results['final_distance']:.4f}")
            logger.info(f"  Best distance: {run_results['best_distance']:.4f}")
            logger.info(f"  Reduction: {run_results['distance_reduction']:.4f} ({run_results['reduction_pct']:.1f}%)")
        
        # Aggregate results
        final_results = {
            'config': asdict(self.config),
            'initial_distance': self.results['initial_distance'],
            'runs': all_results,
            'mean_final_distance': np.mean([r['final_distance'] for r in all_results]),
            'std_final_distance': np.std([r['final_distance'] for r in all_results]),
            'mean_best_distance': np.mean([r['best_distance'] for r in all_results]),
            'mean_reduction_pct': np.mean([r['reduction_pct'] for r in all_results]),
        }
        
        # Determine success
        success = final_results['mean_best_distance'] < self.config.distance_threshold
        final_results['success'] = success
        
        # Print summary
        logger.info("\n" + "#"*60)
        logger.info("# FINAL RESULTS")
        logger.info("#"*60)
        logger.info(f"Initial distance: {final_results['initial_distance']:.4f}")
        logger.info(f"Mean final distance: {final_results['mean_final_distance']:.4f} ± {final_results['std_final_distance']:.4f}")
        logger.info(f"Mean best distance: {final_results['mean_best_distance']:.4f}")
        logger.info(f"Mean reduction: {final_results['mean_reduction_pct']:.1f}%")
        
        if success:
            logger.info("\n✓ EXPERIMENT SUCCESS: Pattern resonance achieved!")
        else:
            logger.info("\n✗ EXPERIMENT INCONCLUSIVE: Threshold not reached")
        
        return final_results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Save results to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nobs_experiment_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy arrays and numpy scalars to lists/Python types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results_json = convert(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NOBS Shaman Experiment")
    parser.add_argument("--data", default="data/BTC_USDT_USDT-4h-futures.feather",
                       help="Path to OHLCV data")
    parser.add_argument("--iterations", type=int, default=200,
                       help="Shaman iterations")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of experiment runs")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = NOBSExperimentConfig(
        data_path=args.data,
        shaman_iterations=args.iterations if not args.quick else 50,
        num_runs=args.runs if not args.quick else 1,
        seed=args.seed
    )
    
    if args.quick:
        logger.info("[QUICK MODE] Reduced iterations for testing")
    
    experiment = NOBSExperiment(config)
    results = experiment.run()
    experiment.save_results(results)
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
