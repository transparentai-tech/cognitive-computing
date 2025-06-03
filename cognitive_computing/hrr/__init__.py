"""
Holographic Reduced Representations (HRR) module for cognitive computing.

This module implements Holographic Reduced Representations, a method for
encoding compositional structures in fixed-size distributed representations
using circular convolution.

Key Features:
- Circular convolution binding and unbinding
- Cleanup memory for item retrieval
- Role-filler structures
- Sequence encoding
- Hierarchical composition

Example:
    >>> from cognitive_computing.hrr import create_hrr
    >>> hrr = create_hrr(dimension=1024)
    >>> 
    >>> # Bind two vectors
    >>> role = hrr.generate_vector()
    >>> filler = hrr.generate_vector()
    >>> binding = hrr.bind(role, filler)
    >>> 
    >>> # Unbind to retrieve filler
    >>> retrieved = hrr.unbind(binding, role)
    >>> similarity = hrr.similarity(retrieved, filler)
    >>> print(f"Similarity: {similarity:.3f}")
"""

import logging
from typing import Optional

# Import core classes
from .core import HRR, HRRConfig

# Import operations
from .operations import CircularConvolution, VectorOperations

# Import cleanup memory
from .cleanup import CleanupMemory, CleanupMemoryConfig

# Import encoders
from .encoding import RoleFillerEncoder, SequenceEncoder, HierarchicalEncoder

# Import utilities
from .utils import (
    generate_random_vector,
    generate_unitary_vector,
    generate_orthogonal_set,
    analyze_binding_capacity,
    measure_crosstalk,
    measure_associative_capacity,
    to_complex,
    from_complex,
)

# Import visualization functions
from .visualizations import (
    plot_similarity_matrix,
    plot_binding_accuracy,
    visualize_cleanup_space,
    plot_convolution_spectrum,
    animate_unbinding_process,
)

# Version info
__version__ = "0.1.0"

# Configure logging
logger = logging.getLogger(__name__)

# Public API
__all__ = [
    # Core classes
    "HRR",
    "HRRConfig",
    
    # Operations
    "CircularConvolution",
    "VectorOperations",
    
    # Cleanup memory
    "CleanupMemory",
    "CleanupMemoryConfig",
    
    # Encoders
    "RoleFillerEncoder",
    "SequenceEncoder",
    "HierarchicalEncoder",
    
    # Utilities
    "generate_random_vector",
    "generate_unitary_vector",
    "generate_orthogonal_set",
    "analyze_binding_capacity",
    "measure_crosstalk",
    "measure_associative_capacity",
    "to_complex",
    "from_complex",
    
    # Visualizations
    "plot_similarity_matrix",
    "plot_binding_accuracy",
    "visualize_cleanup_space",
    "plot_convolution_spectrum",
    "animate_unbinding_process",
    
    # Factory functions
    "create_hrr",
]


def create_hrr(
    dimension: int = 1024,
    normalize: bool = True,
    cleanup_threshold: float = 0.3,
    storage_method: str = "real",
    seed: Optional[int] = None
) -> HRR:
    """
    Create an HRR system with the specified parameters.
    
    This is a convenience function that creates an HRR instance with
    commonly used parameters.
    
    Parameters
    ----------
    dimension : int, optional
        Dimensionality of vectors (default: 1024)
    normalize : bool, optional
        Whether to normalize vectors after operations (default: True)
    cleanup_threshold : float, optional
        Similarity threshold for cleanup memory (default: 0.3)
    storage_method : str, optional
        Method for storing vectors: "real" or "complex" (default: "real")
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    HRR
        Configured HRR instance
        
    Examples
    --------
    >>> # Create a basic HRR system
    >>> hrr = create_hrr(dimension=512)
    >>> 
    >>> # Create HRR with complex storage for better capacity
    >>> hrr_complex = create_hrr(dimension=1024, storage_method="complex")
    >>> 
    >>> # Create HRR without normalization for specific applications
    >>> hrr_raw = create_hrr(dimension=2048, normalize=False)
    """
    config = HRRConfig(
        dimension=dimension,
        normalize=normalize,
        cleanup_threshold=cleanup_threshold,
        storage_method=storage_method,
        seed=seed
    )
    
    logger.info(f"Creating HRR with dimension={dimension}, "
                f"storage_method={storage_method}")
    
    return HRR(config)