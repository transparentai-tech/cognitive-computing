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

# Version information
__version__ = "0.1.0"

# Public API
__all__ = [
    # Core classes
    "HDC",
    "HDCConfig",
    "HypervectorType",
    
    # Factory functions
    "create_hdc",
]