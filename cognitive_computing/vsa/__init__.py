"""
Vector Symbolic Architectures (VSA) module.

This module implements a general framework for Vector Symbolic Architectures,
providing multiple binding operations, vector types, and VSA variants including
Binary Spatter Codes (BSC), Multiply-Add-Permute (MAP), and others.

VSA extends beyond HRR by offering:
- Multiple binding operations (XOR, multiplication, MAP, etc.)
- Different vector types (binary, bipolar, ternary, complex)
- Flexible architectures for different use cases
- Hardware-friendly implementations
"""

import logging

# Set up module logging
logger = logging.getLogger(__name__)

# Version info
__version__ = "0.1.0"

# Import core classes and functions
from .core import (
    VSA,
    VSAConfig,
    create_vsa,
    VSAType,
    VectorType
)

# Import vector types
from .vectors import (
    VSAVector,
    BinaryVector,
    BipolarVector,
    TernaryVector,
    ComplexVector,
    IntegerVector,
    create_vector
)

# Import binding operations
from .binding import (
    BindingOperation,
    XORBinding,
    MultiplicationBinding,
    ConvolutionBinding,
    MAPBinding,
    PermutationBinding,
    create_binding
)

# Import operations
from .operations import (
    permute,
    inverse_permute,
    thin,
    thicken,
    bundle,
    normalize_vector,
    generate_permutation
)

# Import encoding strategies
from .encoding import (
    VSAEncoder,
    RandomIndexingEncoder,
    SpatialEncoder,
    TemporalEncoder,
    LevelEncoder,
    GraphEncoder
)

# Import architectures
from .architectures import (
    BSC,  # Binary Spatter Codes
    MAP,  # Multiply-Add-Permute
    FHRR,  # Fourier HRR
    SparseVSA,
    HRRCompatibility
)

# Import utilities
from .utils import (
    generate_random_vector,
    analyze_binding_capacity,
    convert_vector,
    analyze_vector_distribution,
    compare_binding_methods
)

# Define public API
__all__ = [
    # Core
    'VSA', 'VSAConfig', 'create_vsa', 'VSAType', 'VectorType',
    
    # Vectors
    'VSAVector', 'BinaryVector', 'BipolarVector', 'TernaryVector',
    'ComplexVector', 'IntegerVector', 'create_vector',
    
    # Binding
    'BindingOperation', 'XORBinding', 'MultiplicationBinding',
    'ConvolutionBinding', 'MAPBinding', 'PermutationBinding',
    'create_binding',
    
    # Operations
    'permute', 'inverse_permute', 'thin', 'thicken', 'bundle',
    'normalize_vector', 'generate_permutation',
    
    # Encoding
    'VSAEncoder', 'RandomIndexingEncoder', 'SpatialEncoder',
    'TemporalEncoder', 'LevelEncoder', 'GraphEncoder',
    
    # Architectures
    'BSC', 'MAP', 'FHRR', 'SparseVSA', 'HRRCompatibility',
    
    # Utils
    'generate_random_vector', 'analyze_binding_capacity', 'convert_vector',
    'analyze_vector_distribution', 'compare_binding_methods',
]

# Log module initialization
logger.info(f"VSA module v{__version__} initialized")