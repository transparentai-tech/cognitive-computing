"""
Sparse Distributed Memory (SDM) Module

This module implements Sparse Distributed Memory, a content-addressable memory
system inspired by the properties of human long-term memory. SDM was introduced
by Pentti Kanerva in 1988.

Key Features:
- Content-addressable memory with high capacity
- Robust to noise and partial information
- Distributed storage across multiple locations
- Automatic generalization and pattern completion
- Scalable to high-dimensional spaces

Main Components:
- SDM: The main Sparse Distributed Memory class
- SDMConfig: Configuration class for SDM parameters
- AddressDecoder: Various address decoding strategies
- SDMUtils: Utility functions for SDM operations

Example Usage:
    >>> from cognitive_computing.sdm import SDM, SDMConfig
    >>> 
    >>> # Create SDM with 1000-dimensional binary vectors
    >>> config = SDMConfig(
    ...     dimension=1000,
    ...     num_hard_locations=1000,
    ...     activation_radius=451
    ... )
    >>> memory = SDM(config)
    >>> 
    >>> # Store a pattern
    >>> key = np.random.randint(0, 2, 1000)
    >>> value = np.random.randint(0, 2, 1000)
    >>> memory.store(key, value)
    >>> 
    >>> # Recall with noisy key
    >>> noisy_key = add_noise(key, noise_level=0.1)
    >>> recalled = memory.recall(noisy_key)

References:
    Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
"""

# Import main classes and functions
from cognitive_computing.sdm.core import SDM, SDMConfig
from cognitive_computing.sdm.memory import (
    HardLocation,
    MemoryContents,
    MemoryStatistics,
)
from cognitive_computing.sdm.address_decoder import (
    AddressDecoder,
    HammingDecoder,
    RandomDecoder,
    JaccardDecoder,
)
from cognitive_computing.sdm.utils import (
    add_noise,
    generate_random_patterns,
    compute_memory_capacity,
    analyze_activation_patterns,
)

# Import visualization tools if available
try:
    from cognitive_computing.sdm.visualizations import (
        plot_memory_distribution,
        plot_activation_pattern,
        plot_recall_accuracy,
        visualize_memory_contents,
    )
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

# Define public API
__all__ = [
    # Core classes
    "SDM",
    "SDMConfig",
    # Memory components
    "HardLocation",
    "MemoryContents",
    "MemoryStatistics",
    # Address decoders
    "AddressDecoder",
    "HammingDecoder",
    "RandomDecoder",
    "JaccardDecoder",
    # Utility functions
    "add_noise",
    "generate_random_patterns",
    "compute_memory_capacity",
    "analyze_activation_patterns",
]

# Add visualization functions if available
if _HAS_VIZ:
    __all__.extend([
        "plot_memory_distribution",
        "plot_activation_pattern",
        "plot_recall_accuracy",
        "visualize_memory_contents",
    ])

# Module metadata
__version__ = "0.1.0"
__author__ = "Cognitive Computing Contributors"

# Quick access functions
def create_sdm(dimension: int, num_locations: int = None, 
               activation_radius: int = None) -> SDM:
    """
    Quick function to create an SDM instance with sensible defaults.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of the address/data space
    num_locations : int, optional
        Number of hard locations. If None, uses sqrt(2^dimension)
    activation_radius : int, optional
        Hamming radius for activation. If None, uses dimension * 0.451
        
    Returns
    -------
    SDM
        Configured SDM instance
    """
    if num_locations is None:
        # Use Kanerva's recommendation
        num_locations = int(np.sqrt(2 ** min(dimension, 20)))
    
    if activation_radius is None:
        # Use critical distance for good performance
        activation_radius = int(dimension * 0.451)
    
    config = SDMConfig(
        dimension=dimension,
        num_hard_locations=num_locations,
        activation_radius=activation_radius
    )
    
    return SDM(config)

# Log module import
import logging
logger = logging.getLogger(__name__)
logger.info(f"SDM module v{__version__} loaded")