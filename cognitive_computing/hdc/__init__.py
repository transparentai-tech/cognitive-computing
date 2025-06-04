"""
Hyperdimensional Computing (HDC) module.

This module implements brain-inspired computing using high-dimensional vectors
for robust and efficient information processing, particularly suited for
classification, sensor fusion, and edge computing applications.
"""

from cognitive_computing.hdc.core import (
    HDC,
    HDCConfig,
    HypervectorType,
    create_hdc,
)

from cognitive_computing.hdc.hypervectors import (
    BinaryHypervector,
    BipolarHypervector,
    TernaryHypervector,
    LevelHypervector,
    generate_orthogonal_hypervectors,
    fractional_binding,
    protect_hypervector,
    unprotect_hypervector,
)

from cognitive_computing.hdc.operations import (
    BundlingMethod,
    PermutationMethod,
    bind_hypervectors,
    bundle_hypervectors,
    permute_hypervector,
    similarity,
    noise_hypervector,
    thin_hypervector,
    segment_hypervector,
    concatenate_hypervectors,
    power_hypervector,
    normalize_hypervector,
    protect_sequence,
)

from cognitive_computing.hdc.item_memory import ItemMemory

# Version information
__version__ = "0.1.0"

# Public API
__all__ = [
    # Core classes
    "HDC",
    "HDCConfig",
    "HypervectorType",
    
    # Hypervector types
    "BinaryHypervector",
    "BipolarHypervector", 
    "TernaryHypervector",
    "LevelHypervector",
    
    # Operations
    "BundlingMethod",
    "PermutationMethod",
    "bind_hypervectors",
    "bundle_hypervectors",
    "permute_hypervector",
    "similarity",
    "noise_hypervector",
    "thin_hypervector",
    "segment_hypervector",
    "concatenate_hypervectors",
    "power_hypervector",
    "normalize_hypervector",
    "protect_sequence",
    
    # Memory
    "ItemMemory",
    
    # Factory and utility functions
    "create_hdc",
    "generate_orthogonal_hypervectors",
    "fractional_binding",
    "protect_hypervector",
    "unprotect_hypervector",
]