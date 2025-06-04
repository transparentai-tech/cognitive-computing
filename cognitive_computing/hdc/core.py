"""
Core HDC implementation with configuration and base classes.

This module provides the main HDC class and configuration system for
hyperdimensional computing operations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from cognitive_computing.common.base import CognitiveMemory, MemoryConfig

logger = logging.getLogger(__name__)


class HypervectorType(Enum):
    """Enumeration of supported hypervector types."""
    BINARY = "binary"  # {0, 1}
    BIPOLAR = "bipolar"  # {-1, +1}
    TERNARY = "ternary"  # {-1, 0, +1}
    LEVEL = "level"  # Multi-level discrete values


@dataclass
class HDCConfig(MemoryConfig):
    """
    Configuration for HDC systems.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of hypervectors (default: 10000)
    hypervector_type : str
        Type of hypervectors: "binary", "bipolar", "ternary", "level"
    seed_orthogonal : bool
        Use orthogonal seed vectors for categories (default: True)
    similarity_threshold : float
        Threshold for classification decisions (default: 0.0)
    item_memory_size : int, optional
        Maximum number of items in associative memory
    levels : int
        Number of levels for level hypervectors (default: 5)
    sparsity : float
        Sparsity level for ternary vectors (default: 0.33)
    """
    dimension: int = 10000
    hypervector_type: str = "bipolar"
    seed_orthogonal: bool = True
    similarity_threshold: float = 0.0
    item_memory_size: Optional[int] = None
    levels: int = 5
    sparsity: float = 0.33
    
    def __post_init__(self):
        """Validate HDC configuration."""
        super().__post_init__()
        
        # Validate hypervector type
        valid_types = [t.value for t in HypervectorType]
        if self.hypervector_type not in valid_types:
            raise ValueError(
                f"hypervector_type must be one of {valid_types}, "
                f"got {self.hypervector_type}"
            )
        
        # Validate dimension
        if self.dimension < 100:
            raise ValueError(
                f"Dimension should be at least 100 for HDC, got {self.dimension}"
            )
        
        # Validate levels
        if self.levels < 2:
            raise ValueError(f"Levels must be at least 2, got {self.levels}")
        
        # Validate sparsity
        if not 0 < self.sparsity < 1:
            raise ValueError(f"Sparsity must be in (0, 1), got {self.sparsity}")


class HDC(CognitiveMemory):
    """
    Main Hyperdimensional Computing implementation.
    
    HDC uses high-dimensional vectors to represent and process information
    in a brain-inspired manner. It is particularly effective for classification,
    sensor fusion, and robust computing tasks.
    
    Parameters
    ----------
    config : HDCConfig
        Configuration object for the HDC system
    
    Attributes
    ----------
    config : HDCConfig
        System configuration
    dimension : int
        Hypervector dimensionality
    hypervector_type : HypervectorType
        Type of hypervectors used
    item_memory : Dict[str, np.ndarray]
        Associative memory for storing labeled hypervectors
    class_hypervectors : Dict[str, np.ndarray]
        Stored class prototypes for classification
    """
    
    def __init__(self, config: HDCConfig):
        """Initialize HDC system."""
        self.dimension = config.dimension
        self.hypervector_type = HypervectorType(config.hypervector_type)
        self.item_memory: Dict[str, np.ndarray] = {}
        self.class_hypervectors: Dict[str, np.ndarray] = {}
        self._rng = np.random.RandomState(config.seed)
        super().__init__(config)
        
    def _initialize(self):
        """Initialize HDC internals."""
        logger.info(
            f"Initialized HDC with dimension={self.dimension}, "
            f"type={self.hypervector_type.value}"
        )
        
    def store(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Store a key-value pair using HDC binding.
        
        Parameters
        ----------
        key : np.ndarray
            Key hypervector
        value : np.ndarray
            Value hypervector to associate with key
        """
        self._validate_hypervector(key, "key")
        self._validate_hypervector(value, "value")
        
        # Bind key and value
        bound = self.bind(key, value)
        
        # Store in item memory with generated ID
        item_id = f"item_{len(self.item_memory)}"
        self.item_memory[item_id] = bound
        
    def recall(self, key: np.ndarray) -> Optional[np.ndarray]:
        """
        Recall a value from memory given a key.
        
        Parameters
        ----------
        key : np.ndarray
            Key hypervector to search for
            
        Returns
        -------
        Optional[np.ndarray]
            Recalled value hypervector or None if not found
        """
        self._validate_hypervector(key, "key")
        
        # Search through item memory
        best_similarity = -np.inf
        best_value = None
        
        for item_id, bound in self.item_memory.items():
            # Unbind to get potential value
            value = self.unbind(bound, key)
            
            # Check similarity
            similarity = self.similarity(value, bound)
            if similarity > best_similarity:
                best_similarity = similarity
                best_value = value
                
        if best_similarity > self.config.similarity_threshold:
            return best_value
        return None
        
    def clear(self) -> None:
        """Clear all stored memories."""
        self.item_memory.clear()
        self.class_hypervectors.clear()
        
    @property
    def size(self) -> int:
        """Return the current number of stored items."""
        return len(self.item_memory)
        
    def generate_hypervector(self, orthogonal_to: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Generate a random hypervector of the configured type.
        
        Parameters
        ----------
        orthogonal_to : List[np.ndarray], optional
            List of hypervectors to be orthogonal to (for bipolar type)
            
        Returns
        -------
        np.ndarray
            Generated hypervector
        """
        if self.hypervector_type == HypervectorType.BINARY:
            return self._rng.randint(0, 2, size=self.dimension, dtype=np.uint8)
            
        elif self.hypervector_type == HypervectorType.BIPOLAR:
            if orthogonal_to and self.config.seed_orthogonal:
                # Generate orthogonal bipolar vector
                hv = self._rng.randn(self.dimension)
                
                # Orthogonalize against existing vectors
                for existing in orthogonal_to:
                    hv -= np.dot(hv, existing) * existing / np.dot(existing, existing)
                    
                # Normalize and binarize
                hv = np.sign(hv)
                hv[hv == 0] = 1
                return hv.astype(np.int8)
            else:
                return 2 * self._rng.randint(0, 2, size=self.dimension) - 1
                
        elif self.hypervector_type == HypervectorType.TERNARY:
            # Generate sparse ternary vector
            hv = np.zeros(self.dimension, dtype=np.int8)
            
            # Number of non-zero elements
            n_nonzero = int(self.dimension * self.config.sparsity)
            indices = self._rng.choice(self.dimension, n_nonzero, replace=False)
            
            # Assign random +1 or -1
            hv[indices] = 2 * self._rng.randint(0, 2, size=n_nonzero) - 1
            return hv
            
        elif self.hypervector_type == HypervectorType.LEVEL:
            # Generate multi-level hypervector
            levels = np.arange(self.config.levels)
            return self._rng.choice(levels, size=self.dimension).astype(np.int8)
            
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors using type-appropriate operation.
        
        Parameters
        ----------
        a, b : np.ndarray
            Hypervectors to bind
            
        Returns
        -------
        np.ndarray
            Bound hypervector
        """
        self._validate_hypervector(a, "a")
        self._validate_hypervector(b, "b")
        
        if self.hypervector_type == HypervectorType.BINARY:
            # XOR for binary
            return np.bitwise_xor(a, b)
            
        elif self.hypervector_type in [HypervectorType.BIPOLAR, HypervectorType.TERNARY]:
            # Element-wise multiplication
            return a * b
            
        elif self.hypervector_type == HypervectorType.LEVEL:
            # Modular addition for levels
            return (a + b) % self.config.levels
            
    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind a hypervector given one of the original vectors.
        
        For most HDC operations, unbind(bind(a, b), a) ≈ b
        
        Parameters
        ----------
        bound : np.ndarray
            Bound hypervector
        key : np.ndarray
            One of the original binding operands
            
        Returns
        -------
        np.ndarray
            Unbound hypervector
        """
        # For self-inverse operations, unbind is the same as bind
        return self.bind(bound, key)
        
    def bundle(self, hypervectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors into a single hypervector.
        
        Parameters
        ----------
        hypervectors : List[np.ndarray]
            List of hypervectors to bundle
            
        Returns
        -------
        np.ndarray
            Bundled hypervector
        """
        if not hypervectors:
            raise ValueError("Cannot bundle empty list of hypervectors")
            
        for i, hv in enumerate(hypervectors):
            self._validate_hypervector(hv, f"hypervector[{i}]")
            
        if self.hypervector_type == HypervectorType.BINARY:
            # Majority voting for binary
            summed = np.sum(hypervectors, axis=0)
            threshold = len(hypervectors) / 2
            return (summed > threshold).astype(np.uint8)
            
        elif self.hypervector_type == HypervectorType.BIPOLAR:
            # Sum and sign for bipolar
            summed = np.sum(hypervectors, axis=0)
            bundled = np.sign(summed)
            bundled[bundled == 0] = 1  # Handle ties
            return bundled.astype(np.int8)
            
        elif self.hypervector_type == HypervectorType.TERNARY:
            # Threshold-based bundling for ternary
            summed = np.sum(hypervectors, axis=0)
            threshold = len(hypervectors) * self.config.sparsity
            
            bundled = np.zeros(self.dimension, dtype=np.int8)
            bundled[summed > threshold] = 1
            bundled[summed < -threshold] = -1
            return bundled
            
        elif self.hypervector_type == HypervectorType.LEVEL:
            # Average and round for levels
            averaged = np.mean(hypervectors, axis=0)
            return np.round(averaged).astype(np.int8) % self.config.levels
            
    def permute(self, hypervector: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute a hypervector by cyclic shift.
        
        Parameters
        ----------
        hypervector : np.ndarray
            Hypervector to permute
        shift : int
            Number of positions to shift (default: 1)
            
        Returns
        -------
        np.ndarray
            Permuted hypervector
        """
        self._validate_hypervector(hypervector, "hypervector")
        return np.roll(hypervector, shift)
        
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate similarity between two hypervectors.
        
        Parameters
        ----------
        a, b : np.ndarray
            Hypervectors to compare
            
        Returns
        -------
        float
            Similarity score (range depends on hypervector type)
        """
        self._validate_hypervector(a, "a")
        self._validate_hypervector(b, "b")
        
        if self.hypervector_type == HypervectorType.BINARY:
            # Normalized Hamming similarity
            hamming_dist = np.sum(a != b)
            return 1 - (2 * hamming_dist / self.dimension)
            
        elif self.hypervector_type in [HypervectorType.BIPOLAR, HypervectorType.TERNARY]:
            # Cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
            
        elif self.hypervector_type == HypervectorType.LEVEL:
            # Normalized agreement
            agreements = np.sum(a == b)
            return agreements / self.dimension
            
    def _validate_hypervector(self, hv: np.ndarray, name: str) -> None:
        """Validate hypervector dimensions and type."""
        if not isinstance(hv, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
            
        if hv.shape != (self.dimension,):
            raise ValueError(
                f"{name} must have shape ({self.dimension},), "
                f"got {hv.shape}"
            )
            
        # Type-specific validation
        if self.hypervector_type == HypervectorType.BINARY:
            if not np.all(np.isin(hv, [0, 1])):
                raise ValueError(f"{name} must contain only 0 and 1 for binary type")
                
        elif self.hypervector_type == HypervectorType.BIPOLAR:
            if not np.all(np.isin(hv, [-1, 1])):
                raise ValueError(f"{name} must contain only -1 and 1 for bipolar type")
                
        elif self.hypervector_type == HypervectorType.TERNARY:
            if not np.all(np.isin(hv, [-1, 0, 1])):
                raise ValueError(f"{name} must contain only -1, 0, 1 for ternary type")
                
        elif self.hypervector_type == HypervectorType.LEVEL:
            if not np.all((0 <= hv) & (hv < self.config.levels)):
                raise ValueError(
                    f"{name} must contain values in [0, {self.config.levels}) "
                    f"for level type"
                )


def create_hdc(
    dimension: int = 10000,
    hypervector_type: str = "bipolar",
    **kwargs
) -> HDC:
    """
    Factory function to create an HDC instance.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of hypervectors (default: 10000)
    hypervector_type : str
        Type of hypervectors: "binary", "bipolar", "ternary", "level"
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    HDC
        Configured HDC instance
        
    Examples
    --------
    >>> hdc = create_hdc(dimension=5000, hypervector_type="binary")
    >>> hdc = create_hdc(dimension=10000, sparsity=0.1)
    """
    config = HDCConfig(
        dimension=dimension,
        hypervector_type=hypervector_type,
        **kwargs
    )
    return HDC(config)