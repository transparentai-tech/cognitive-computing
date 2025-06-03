"""
Core implementation of Vector Symbolic Architectures (VSA).

This module provides the main VSA class and configuration, implementing
a flexible framework for various VSA variants including Binary Spatter Codes,
Multiply-Add-Permute, and others.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from ..common.base import CognitiveMemory, MemoryConfig

logger = logging.getLogger(__name__)


class VectorType(Enum):
    """Enumeration of supported vector types in VSA."""
    BINARY = "binary"  # {0, 1}
    BIPOLAR = "bipolar"  # {-1, +1}
    TERNARY = "ternary"  # {-1, 0, +1}
    COMPLEX = "complex"  # Complex unit vectors
    REAL = "real"  # Real-valued vectors


class VSAType(Enum):
    """Enumeration of VSA architecture types."""
    BSC = "bsc"  # Binary Spatter Codes
    MAP = "map"  # Multiply-Add-Permute
    FHRR = "fhrr"  # Fourier HRR
    HRR = "hrr"  # Holographic Reduced Representations (compatibility)
    SPARSE = "sparse"  # Sparse VSA
    CUSTOM = "custom"  # User-defined architecture


@dataclass
class VSAConfig(MemoryConfig):
    """
    Configuration for Vector Symbolic Architectures.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of vectors
    vector_type : VectorType or str
        Type of vectors to use (binary, bipolar, ternary, complex, real)
    vsa_type : VSAType or str
        VSA architecture type (bsc, map, fhrr, hrr, sparse, custom)
    binding_operation : str, optional
        Default binding operation to use. If None, uses architecture default
    sparsity : float
        Sparsity level for sparse vectors (0-1, where 0 is dense)
    normalize : bool
        Whether to normalize vectors after operations
    cleanup_threshold : float
        Similarity threshold for cleanup memory (0-1)
    seed : int, optional
        Random seed for reproducibility
    """
    dimension: int = 10000
    vector_type: Union[VectorType, str] = VectorType.BIPOLAR
    vsa_type: Union[VSAType, str] = VSAType.MAP
    binding_operation: Optional[str] = None
    sparsity: float = 0.0
    normalize: bool = True
    cleanup_threshold: float = 0.3
    
    def __post_init__(self):
        """Validate VSA configuration parameters."""
        super().__post_init__()
        
        # Convert string to enum if needed
        if isinstance(self.vector_type, str):
            try:
                self.vector_type = VectorType(self.vector_type.lower())
            except ValueError:
                raise ValueError(f"Invalid vector_type: {self.vector_type}. "
                               f"Must be one of {[v.value for v in VectorType]}")
        
        if isinstance(self.vsa_type, str):
            try:
                self.vsa_type = VSAType(self.vsa_type.lower())
            except ValueError:
                raise ValueError(f"Invalid vsa_type: {self.vsa_type}. "
                               f"Must be one of {[v.value for v in VSAType]}")
        
        # Validate sparsity
        if not 0 <= self.sparsity < 1:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        
        # Validate cleanup threshold
        if not 0 <= self.cleanup_threshold <= 1:
            raise ValueError(f"cleanup_threshold must be in [0, 1], "
                           f"got {self.cleanup_threshold}")
        
        # Set default binding operation based on architecture
        if self.binding_operation is None:
            self.binding_operation = self._get_default_binding()
    
    def _get_default_binding(self) -> str:
        """Get default binding operation for the VSA type."""
        defaults = {
            VSAType.BSC: "xor",
            VSAType.MAP: "map",
            VSAType.FHRR: "convolution",
            VSAType.HRR: "convolution",
            VSAType.SPARSE: "multiplication",
            VSAType.CUSTOM: "multiplication"
        }
        return defaults.get(self.vsa_type, "multiplication")


class VSA(CognitiveMemory):
    """
    Vector Symbolic Architecture implementation.
    
    VSA provides a flexible framework for cognitive computing with multiple
    vector types and binding operations. It supports various architectures
    including Binary Spatter Codes, MAP, and Fourier HRR.
    
    Parameters
    ----------
    config : VSAConfig
        Configuration object for the VSA system
        
    Attributes
    ----------
    config : VSAConfig
        System configuration
    memory : Dict[str, np.ndarray]
        Stored item vectors for cleanup
    binding_op : BindingOperation
        Current binding operation
    vector_factory : VSAVector
        Factory for creating vectors of the configured type
    _rng : np.random.RandomState
        Random number generator
    """
    
    def __init__(self, config: VSAConfig):
        """Initialize the VSA system."""
        super().__init__(config)
        self.config = config
        self.memory: Dict[str, np.ndarray] = {}
        self._rng = np.random.RandomState(config.seed)
        
        # These will be set by _initialize
        self.binding_op = None
        self.vector_factory = None
        
        self._initialize()
        
    def _initialize(self):
        """Initialize VSA components based on configuration."""
        logger.debug(f"Initializing VSA with dimension={self.config.dimension}, "
                    f"vector_type={self.config.vector_type.value}, "
                    f"vsa_type={self.config.vsa_type.value}")
        
        # Import here to avoid circular imports
        from .vectors import create_vector
        from .binding import create_binding
        
        # Create vector factory
        self.vector_factory = create_vector(
            self.config.vector_type,
            self.config.dimension,
            sparsity=self.config.sparsity,
            seed=self.config.seed
        )
        
        # Create binding operation
        self.binding_op = create_binding(
            self.config.binding_operation,
            vector_type=self.config.vector_type,
            dimension=self.config.dimension
        )
    
    def generate_vector(self, sparse: Optional[bool] = None) -> np.ndarray:
        """
        Generate a random vector suitable for VSA operations.
        
        Parameters
        ----------
        sparse : bool, optional
            Whether to generate a sparse vector. If None, uses config.sparsity
            
        Returns
        -------
        np.ndarray
            Random vector of the configured type
        """
        if sparse is None:
            sparse = self.config.sparsity > 0
            
        return self.vector_factory.generate(sparse=sparse)
    
    def bind(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Bind two vectors together using the configured binding operation.
        
        Parameters
        ----------
        x : np.ndarray
            First vector
        y : np.ndarray
            Second vector
            
        Returns
        -------
        np.ndarray
            Bound vector
        """
        result = self.binding_op.bind(x, y)
        
        if self.config.normalize:
            result = self.vector_factory.normalize(result)
            
        return result
    
    def unbind(self, xy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Unbind a vector from a bound pair.
        
        Parameters
        ----------
        xy : np.ndarray
            Bound vector
        y : np.ndarray
            Known vector to unbind
            
        Returns
        -------
        np.ndarray
            Retrieved vector (approximation of x where xy = bind(x, y))
        """
        result = self.binding_op.unbind(xy, y)
        
        if self.config.normalize:
            result = self.vector_factory.normalize(result)
            
        return result
    
    def bundle(self, vectors: List[np.ndarray], 
               weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle multiple vectors together (superposition).
        
        Parameters
        ----------
        vectors : List[np.ndarray]
            Vectors to bundle
        weights : List[float], optional
            Weights for weighted bundling
            
        Returns
        -------
        np.ndarray
            Bundled vector
        """
        if len(vectors) == 0:
            raise ValueError("Cannot bundle empty list of vectors")
        
        if weights is not None:
            if len(weights) != len(vectors):
                raise ValueError(f"Number of weights ({len(weights)}) must match "
                               f"number of vectors ({len(vectors)})")
            # Weighted sum
            result = sum(w * v for w, v in zip(weights, vectors))
        else:
            # Simple sum
            result = sum(vectors)
        
        # Apply vector-type specific bundling
        result = self.vector_factory.bundle_vectors(result)
        
        if self.config.normalize:
            result = self.vector_factory.normalize(result)
            
        return result
    
    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate similarity between two vectors.
        
        Parameters
        ----------
        x : np.ndarray
            First vector
        y : np.ndarray
            Second vector
            
        Returns
        -------
        float
            Similarity score (typically in [-1, 1])
        """
        return self.vector_factory.similarity(x, y)
    
    def store(self, key: str, vector: np.ndarray):
        """
        Store a vector in cleanup memory.
        
        Parameters
        ----------
        key : str
            Identifier for the vector
        vector : np.ndarray
            Vector to store
        """
        if self.config.normalize:
            vector = self.vector_factory.normalize(vector)
        self.memory[key] = vector.copy()
    
    def recall(self, query: np.ndarray, 
               return_similarity: bool = False) -> Union[str, Tuple[str, float]]:
        """
        Recall the most similar stored vector.
        
        Parameters
        ----------
        query : np.ndarray
            Query vector
        return_similarity : bool
            Whether to return similarity score
            
        Returns
        -------
        str or Tuple[str, float]
            Key of most similar vector, optionally with similarity
        """
        if len(self.memory) == 0:
            raise ValueError("No vectors stored in memory")
        
        best_key = None
        best_similarity = -float('inf')
        
        for key, stored_vector in self.memory.items():
            sim = self.similarity(query, stored_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_key = key
        
        if best_similarity < self.config.cleanup_threshold:
            if return_similarity:
                return None, best_similarity
            return None
        
        if return_similarity:
            return best_key, best_similarity
        return best_key
    
    def clear_memory(self):
        """Clear all stored vectors from cleanup memory."""
        self.memory.clear()
    
    def __repr__(self) -> str:
        """String representation of VSA."""
        return (f"VSA(dimension={self.config.dimension}, "
                f"vector_type={self.config.vector_type.value}, "
                f"vsa_type={self.config.vsa_type.value}, "
                f"binding={self.config.binding_operation}, "
                f"memory_size={len(self.memory)})")


def create_vsa(dimension: int = 10000,
               vector_type: Union[VectorType, str] = "bipolar",
               vsa_type: Union[VSAType, str] = "map",
               **kwargs) -> VSA:
    """
    Create a VSA system with the specified parameters.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    vector_type : VectorType or str
        Type of vectors (binary, bipolar, ternary, complex, real)
    vsa_type : VSAType or str
        VSA architecture (bsc, map, fhrr, hrr, sparse, custom)
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    VSA
        Configured VSA system
        
    Examples
    --------
    >>> # Create Binary Spatter Codes VSA
    >>> vsa = create_vsa(dimension=8192, vector_type="binary", vsa_type="bsc")
    
    >>> # Create MAP architecture with bipolar vectors
    >>> vsa = create_vsa(dimension=10000, vector_type="bipolar", vsa_type="map")
    
    >>> # Create sparse VSA with ternary vectors
    >>> vsa = create_vsa(dimension=10000, vector_type="ternary", 
    ...                  vsa_type="sparse", sparsity=0.9)
    """
    config = VSAConfig(
        dimension=dimension,
        vector_type=vector_type,
        vsa_type=vsa_type,
        **kwargs
    )
    return VSA(config)