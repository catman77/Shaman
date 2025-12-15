# -*- coding: utf-8 -*-
"""
NOBS-based Shaman Experiment v2.

Improved version with:
1. Pre-computed window embeddings for fast search
2. Gradient-based optimization toward anchor
3. Proper resonance detection via centroid alignment
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nobs.symbolic_dna import SymbolicDNAEncoder, StructuralDNA
from nobs.category import CategoryCn, CategoryHierarchy
from nobs.free_energy import FreeEnergyMinimizer
from nobs.space import NOBSSpace, NOBSEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Experiment configuration."""
    data_path: str = "data/BTC_USDT_USDT-4h-futures.feather"
    max_depth: int = 5
    embedding_dim: int = 128
    window_size: int = 100
    step_size: int = 20
    shaman_epochs: int = 100
    num_runs: int = 3
    seed: int = 42
    distance_threshold: float = 0.15  # Success threshold


class WindowDatabase:
    """
    Pre-computed database of window embeddings.
    Enables fast similarity search.
    """
    
    def __init__(self, space: NOBSSpace, dna_sequence: List[StructuralDNA], 
                 window_size: int, step: int):
        self.space = space
        self.window_size = window_size
        self.step = step
        
        # Pre-compute all window embeddings
        self.embeddings: List[np.ndarray] = []
        self.windows: List[Tuple[int, int]] = []  # (start, end) indices
        self.signatures: List[Dict[str, float]] = []  # Quick signature features
        
        logger.info(f"Pre-computing window embeddings (window={window_size}, step={step})...")
        
        for start in tqdm(range(0, len(dna_sequence) - window_size, step), desc="Building database"):
            end = start + window_size
            window = dna_sequence[start:end]
            
            # Embedding
            emb = space.embed_sequence(window)
            self.embeddings.append(emb.vector)
            self.windows.append((start, end))
            
            # Quick signature
            symbols = [dna.to_symbol() for dna in window]
            sym_dist = Counter(symbols)
            total = len(symbols)
            sig = {
                'symbol_dist': {s: sym_dist.get(s, 0) / total for s in NOBSSpace.SYMBOLS},
                'free_energy': emb.free_energy,
                'entropy': emb.entropy,
            }
            self.signatures.append(sig)
        
        self.embeddings = np.array(self.embeddings)
        logger.info(f"Database: {len(self.embeddings)} windows, shape {self.embeddings.shape}")
    
    def find_nearest(self, target: np.ndarray, k: int = 10) -> List[int]:
        """Find k nearest windows to target embedding."""
        distances = np.linalg.norm(self.embeddings - target, axis=1)
        return np.argsort(distances)[:k].tolist()
    
    def find_by_direction(self, current: np.ndarray, direction: np.ndarray, k: int = 10) -> List[int]:
        """Find windows in the direction from current."""
        # Project embeddings onto direction
        projections = np.dot(self.embeddings - current, direction)
        # Select positive projections (moving in the right direction)
        positive_mask = projections > 0
        if not positive_mask.any():
            return self.find_nearest(current + direction, k)
        
        # Among positive, select closest to current + direction
        target = current + direction
        distances = np.linalg.norm(self.embeddings - target, axis=1)
        distances[~positive_mask] = float('inf')
        return np.argsort(distances)[:k].tolist()


class ResonanceShaman:
    """
    Shaman that searches for resonance using gradient descent in embedding space.
    """
    
    def __init__(self, anchor_embedding: np.ndarray, anchor_signature: Dict,
                 learning_rate: float = 0.5, momentum: float = 0.9):
        self.anchor = anchor_embedding
        self.anchor_sig = anchor_signature
        self.lr = learning_rate
        self.momentum = momentum
        
        # State
        self.current_embedding = None
        self.velocity = np.zeros_like(anchor_embedding)
        self.history = []
        self.best_distance = float('inf')
        self.best_embedding = None
    
    def initialize(self, start_embedding: np.ndarray):
        """Initialize search from starting point."""
        self.current_embedding = start_embedding.copy()
        self.velocity = np.zeros_like(start_embedding)
    
    def compute_gradient(self) -> np.ndarray:
        """Compute gradient toward anchor (direction of improvement)."""
        # Simple: direction from current to anchor
        direction = self.anchor - self.current_embedding
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        return direction
    
    def step(self, database: WindowDatabase) -> Tuple[int, float]:
        """
        Take one optimization step.
        
        Returns (best_window_idx, distance_to_anchor)
        """
        # Compute gradient (direction toward anchor)
        gradient = self.compute_gradient()
        
        # Update velocity with momentum
        self.velocity = self.momentum * self.velocity + self.lr * gradient
        
        # Find best window in the direction we want to move
        candidates = database.find_by_direction(
            self.current_embedding, 
            self.velocity,
            k=20
        )
        
        # Select candidate closest to anchor
        best_idx = None
        best_dist = float('inf')
        
        for idx in candidates:
            emb = database.embeddings[idx]
            dist = np.linalg.norm(emb - self.anchor)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        if best_idx is not None:
            self.current_embedding = database.embeddings[best_idx].copy()
        
        # Track best
        if best_dist < self.best_distance:
            self.best_distance = best_dist
            self.best_embedding = self.current_embedding.copy()
        
        self.history.append({
            'distance': best_dist,
            'best_distance': self.best_distance,
        })
        
        return best_idx, best_dist


class NOBSExperimentV2:
    """
    NOBS Shaman Experiment v2.
    
    Setup:
    - Agent A: first half of Bitcoin data
    - Agent B: second half of Bitcoin data
    - Task: Can Shaman find windows in B that resonate with A's centroid?
    """
    
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.seed)
        
        self.space: Optional[NOBSSpace] = None
        self.database_a: Optional[WindowDatabase] = None
        self.database_b: Optional[WindowDatabase] = None
        self.anchor_embedding: Optional[np.ndarray] = None
        self.anchor_signature: Optional[Dict] = None
    
    def setup(self):
        """Initialize experiment."""
        logger.info("Loading data...")
        df = pd.read_feather(self.config.data_path)
        logger.info(f"Loaded {len(df)} candles")
        
        # Build NOBS space
        logger.info("Building NOBS space...")
        self.space = NOBSSpace(
            max_depth=self.config.max_depth,
            embedding_dim=self.config.embedding_dim
        )
        self.space.fit(df)
        
        dna_seq = self.space._dna_sequence
        n = len(dna_seq)
        mid = n // 2
        
        # Split into A (first half) and B (second half)
        dna_a = dna_seq[:mid]
        dna_b = dna_seq[mid:]
        
        logger.info(f"Agent A: {len(dna_a)} samples, Agent B: {len(dna_b)} samples")
        
        # Build window databases
        self.database_a = WindowDatabase(
            self.space, dna_a, 
            self.config.window_size, self.config.step_size
        )
        self.database_b = WindowDatabase(
            self.space, dna_b,
            self.config.window_size, self.config.step_size
        )
        
        # Compute anchor = centroid of A's embeddings
        self.anchor_embedding = np.mean(self.database_a.embeddings, axis=0)
        
        # Anchor signature (symbol distribution, etc)
        all_symbols = []
        for sig in self.database_a.signatures:
            for s, freq in sig['symbol_dist'].items():
                all_symbols.extend([s] * int(freq * 100))
        
        anchor_sym_dist = Counter(all_symbols)
        total = sum(anchor_sym_dist.values())
        self.anchor_signature = {
            'symbol_dist': {s: anchor_sym_dist.get(s, 0) / total for s in NOBSSpace.SYMBOLS},
            'mean_free_energy': np.mean([s['free_energy'] for s in self.database_a.signatures]),
            'mean_entropy': np.mean([s['entropy'] for s in self.database_a.signatures]),
        }
        
        logger.info(f"Anchor embedding computed (centroid of {len(self.database_a.embeddings)} windows)")
        
        # Initial baseline: distance of B's centroid to anchor
        b_centroid = np.mean(self.database_b.embeddings, axis=0)
        initial_dist = np.linalg.norm(b_centroid - self.anchor_embedding)
        logger.info(f"Initial distance (B centroid to A anchor): {initial_dist:.4f}")
        
        # Best possible: closest B window to anchor
        distances = np.linalg.norm(self.database_b.embeddings - self.anchor_embedding, axis=1)
        best_possible = distances.min()
        best_idx = distances.argmin()
        logger.info(f"Best possible distance (closest B window): {best_possible:.4f}")
        
        return {
            'initial_distance': initial_dist,
            'best_possible': best_possible,
            'best_possible_idx': best_idx,
        }
    
    def run_resonance_search(self, start_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Run resonance search.
        
        Shaman starts from a random B window and tries to find
        windows that resonate with A's anchor.
        """
        # Initialize from random or specified window
        if start_idx is None:
            start_idx = np.random.randint(len(self.database_b.embeddings))
        
        start_embedding = self.database_b.embeddings[start_idx]
        initial_distance = np.linalg.norm(start_embedding - self.anchor_embedding)
        
        # Create shaman
        shaman = ResonanceShaman(
            anchor_embedding=self.anchor_embedding,
            anchor_signature=self.anchor_signature,
            learning_rate=0.5,
            momentum=0.9
        )
        shaman.initialize(start_embedding)
        
        logger.info(f"Starting from window {start_idx}, distance={initial_distance:.4f}")
        
        # Run optimization
        distances = [initial_distance]
        selected_windows = [start_idx]
        
        for epoch in range(self.config.shaman_epochs):
            window_idx, dist = shaman.step(self.database_b)
            distances.append(dist)
            selected_windows.append(window_idx)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: distance={dist:.4f}, best={shaman.best_distance:.4f}")
        
        return {
            'initial_distance': initial_distance,
            'final_distance': distances[-1],
            'best_distance': shaman.best_distance,
            'distances': distances,
            'selected_windows': selected_windows,
            'reduction_pct': (initial_distance - shaman.best_distance) / initial_distance * 100,
        }
    
    def run(self) -> Dict[str, Any]:
        """Run full experiment."""
        logger.info("\n" + "#"*60)
        logger.info("# NOBS SHAMAN EXPERIMENT v2")
        logger.info("#"*60)
        
        baseline = self.setup()
        
        all_results = []
        
        for run_idx in range(self.config.num_runs):
            logger.info(f"\n{'='*50}")
            logger.info(f"RUN {run_idx + 1}/{self.config.num_runs}")
            logger.info(f"{'='*50}")
            
            # Different random start each run
            np.random.seed(self.config.seed + run_idx)
            
            results = self.run_resonance_search()
            all_results.append(results)
            
            logger.info(f"\nRun {run_idx + 1} Results:")
            logger.info(f"  Start distance: {results['initial_distance']:.4f}")
            logger.info(f"  Final distance: {results['final_distance']:.4f}")
            logger.info(f"  Best distance: {results['best_distance']:.4f}")
            logger.info(f"  Reduction: {results['reduction_pct']:.1f}%")
        
        # Aggregate
        final_results = {
            'baseline': baseline,
            'runs': all_results,
            'mean_best_distance': np.mean([r['best_distance'] for r in all_results]),
            'std_best_distance': np.std([r['best_distance'] for r in all_results]),
            'mean_reduction_pct': np.mean([r['reduction_pct'] for r in all_results]),
        }
        
        # Success if we find windows closer than threshold
        success = final_results['mean_best_distance'] < self.config.distance_threshold
        final_results['success'] = success
        
        # How close to optimal?
        final_results['optimality_gap'] = (
            final_results['mean_best_distance'] - baseline['best_possible']
        ) / baseline['best_possible'] * 100
        
        logger.info("\n" + "#"*60)
        logger.info("# FINAL RESULTS")
        logger.info("#"*60)
        logger.info(f"Baseline (B centroid to A anchor): {baseline['initial_distance']:.4f}")
        logger.info(f"Best possible (closest B window): {baseline['best_possible']:.4f}")
        logger.info(f"Shaman found: {final_results['mean_best_distance']:.4f} ± {final_results['std_best_distance']:.4f}")
        logger.info(f"Optimality gap: {final_results['optimality_gap']:.1f}%")
        logger.info(f"Mean reduction: {final_results['mean_reduction_pct']:.1f}%")
        
        if success:
            logger.info("\n✓ EXPERIMENT SUCCESS: Resonance found!")
        else:
            logger.info("\n✗ EXPERIMENT INCOMPLETE: Can improve further")
        
        return final_results
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save results."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"nobs_v2_{timestamp}.json")
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/BTC_USDT_USDT-4h-futures.feather")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = Config(
        data_path=args.data,
        shaman_epochs=args.epochs if not args.quick else 30,
        num_runs=args.runs if not args.quick else 1,
        seed=args.seed
    )
    
    if args.quick:
        logger.info("[QUICK MODE]")
    
    experiment = NOBSExperimentV2(config)
    results = experiment.run()
    experiment.save_results(results)
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
