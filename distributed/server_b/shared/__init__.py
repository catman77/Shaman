# -*- coding: utf-8 -*-
"""
Shared components for distributed Shaman experiment.

IMPORTANT: This is the ONLY thing shared between servers!
- meanings.py contains a priori knowledge about semantic patterns
- NO data, embeddings, or metadata is exchanged between servers
"""

from .meanings import (
    MeaningConfig,
    SemanticPattern,
    get_meaning,
    get_meaning_by_name,
    list_meanings,
    MEANINGS
)

__all__ = [
    'MeaningConfig',
    'SemanticPattern', 
    'get_meaning',
    'get_meaning_by_name',
    'list_meanings',
    'MEANINGS'
]
