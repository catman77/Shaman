# -*- coding: utf-8 -*-
"""
Natural Observation-Based Computing (NOBS) module.

Based on the paper "Natural Observation-Based Computing: 
A Novel Paradigm for Solving Computational Problems Through Structural Learning"
by Sergey Kotikov.

This module provides:
- Symbolic DNA encoding of OHLCV data
- Category C_n construction from n-grams
- Morphism statistics and free energy computation
- Self-computing functor construction
"""

from .symbolic_dna import SymbolicDNAEncoder, StructuralDNA
from .category import CategoryCn, Morphism
from .free_energy import FreeEnergyMinimizer
from .space import NOBSSpace

__all__ = [
    'SymbolicDNAEncoder',
    'StructuralDNA', 
    'CategoryCn',
    'Morphism',
    'FreeEnergyMinimizer',
    'NOBSSpace',
]
