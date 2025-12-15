# -*- coding: utf-8 -*-
"""
Shared Protocol for Distributed Shaman Experiment.

This module defines the data structures exchanged between servers.

CRITICAL CONSTRAINT: Server B receives ONLY the anchor signature,
NOT the raw data or full embeddings of Server A.

The anchor signature is a "semantic fingerprint" that allows resonance
search WITHOUT direct data transfer.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import json
import hashlib
import numpy as np
from datetime import datetime


@dataclass
class AnchorSignature:
    """
    The ONLY data that Server A shares with Server B.
    
    This is the "semantic fingerprint" of Agent A's patterns.
    It contains NO raw data, only structural invariants.
    
    Server B uses this anchor to search for resonant patterns
    in its own data.
    """
    
    # Identifier
    anchor_id: str
    created_at: str
    server_a_id: str
    
    # Centroid embedding (128-dim vector)
    # This is the MEAN of all window embeddings - no individual data points
    centroid: List[float]
    
    # Symbol distribution (aggregate statistics only)
    symbol_distribution: Dict[str, float]
    
    # Energy profile (aggregate statistics)
    mean_free_energy: float
    std_free_energy: float
    mean_entropy: float
    
    # Dominant morphisms (patterns, not data)
    # Only top patterns that characterize A's "style"
    dominant_morphisms: List[str]
    
    # Metadata
    num_windows_aggregated: int
    window_size: int
    
    # Hash of raw data (for verification, NOT the data itself)
    data_hash: str
    
    def to_json(self) -> str:
        """Serialize to JSON for transmission."""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AnchorSignature':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)
    
    def to_file(self, filepath: str):
        """Save to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AnchorSignature':
        """Load from file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())
    
    def get_centroid_array(self) -> np.ndarray:
        """Get centroid as numpy array."""
        return np.array(self.centroid, dtype=np.float32)


@dataclass 
class ResonanceReport:
    """
    Report from Server B about resonance search results.
    
    This can be sent back to Server A to verify the experiment.
    """
    
    # Identifier
    report_id: str
    created_at: str
    server_b_id: str
    anchor_id: str  # Reference to which anchor was used
    
    # Results
    initial_distance: float
    best_distance: float
    final_distance: float
    reduction_percent: float
    
    # Statistics
    num_epochs: int
    num_windows_searched: int
    
    # Verification hash (so A can verify B actually ran the search)
    result_hash: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ResonanceReport':
        data = json.loads(json_str)
        return cls(**data)


def compute_data_hash(data: bytes) -> str:
    """Compute SHA256 hash of data for verification."""
    return hashlib.sha256(data).hexdigest()[:16]


def generate_id(prefix: str) -> str:
    """Generate unique ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = hashlib.sha256(str(np.random.random()).encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_part}"
