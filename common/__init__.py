"""
Common utilities and base classes for cognitive computing.

This module contains shared components used across different cognitive
computing paradigms including base classes, interfaces, and utilities.
"""

from cognitive_computing.common.base import (
    # Base classes
    CognitiveMemory,
    MemoryConfig,
    VectorEncoder,
    
    # Utilities
    BinaryVector,
    MemoryPerformanceMetrics,
    
    # Enums
    DistanceMetric
)

__all__ = [
    # Base classes
    'CognitiveMemory',
    'MemoryConfig',
    'VectorEncoder',
    
    # Utilities
    'BinaryVector',
    'MemoryPerformanceMetrics',
    
    # Enums
    'DistanceMetric'
]