"""
Core implementation of Holographic Reduced Representations (HRR).

This module provides the main HRR class and configuration, implementing
the fundamental operations of binding, unbinding, and bundling using
circular convolution.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal

from ..common.base import CognitiveMemory, MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class HRRConfig(MemoryConfig):
    """
    Configuration for Holographic Reduced Representations.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of vectors (should be even for complex storage)
    normalize : bool
        Whether to normalize vectors after operations
    cleanup_threshold : float
        Similarity threshold for cleanup memory (0-1)
    storage_method : str
        Method for storing vectors: "real" or "complex"
    seed : int, optional
        Random seed for reproducibility
    """
    dimension: int = 1024
    normalize: bool = True
    cleanup_threshold: float = 0.3
    storage_method: str = "real"  # "real" or "complex"
    
    def __post_init__(self):
        """Validate HRR configuration parameters."""
        super().__post_init__()
        
        if self.storage_method not in ["real", "complex"]:
            raise ValueError(f"storage_method must be 'real' or 'complex', "
                           f"got {self.storage_method}")
        
        if self.storage_method == "complex" and self.dimension % 2 != 0:
            raise ValueError(f"Dimension must be even for complex storage, "
                           f"got {self.dimension}")
        
        if not 0 <= self.cleanup_threshold <= 1:
            raise ValueError(f"cleanup_threshold must be in [0, 1], "
                           f"got {self.cleanup_threshold}")


class HRR(CognitiveMemory):
    """
    Holographic Reduced Representations implementation.
    
    HRR uses circular convolution to bind vectors together and circular
    correlation to unbind them. This allows for the creation of
    compositional structures in fixed-size distributed representations.
    
    Parameters
    ----------
    config : HRRConfig
        Configuration object for the HRR system
        
    Attributes
    ----------
    config : HRRConfig
        System configuration
    memory : Dict[str, np.ndarray]
        Stored item vectors for cleanup
    _rng : np.random.RandomState
        Random number generator
    """
    
    def __init__(self, config: HRRConfig):
        """Initialize the HRR system."""
        super().__init__(config)
        self.config = config
        self.memory: Dict[str, np.ndarray] = {}
        self._rng = np.random.RandomState(config.seed)
        
    def _initialize(self):
        """Initialize HRR internals."""
        logger.debug(f"Initializing HRR with dimension={self.config.dimension}, "
                    f"storage_method={self.config.storage_method}")
        
    def generate_vector(self, unitary: bool = False) -> np.ndarray:
        """
        Generate a random vector suitable for HRR operations.
        
        Parameters
        ----------
        unitary : bool, optional
            If True, generate a unitary vector (self-inverse under correlation)
            
        Returns
        -------
        np.ndarray
            Random vector of appropriate type and dimension
        """
        if self.config.storage_method == "complex":
            # Generate complex vector with random phases
            phases = self._rng.uniform(0, 2 * np.pi, self.config.dimension // 2)
            real_part = np.cos(phases)
            imag_part = np.sin(phases)
            vector = real_part + 1j * imag_part
        else:
            # Generate real-valued Gaussian vector
            vector = self._rng.randn(self.config.dimension)
            
        if self.config.normalize:
            vector = self._normalize(vector)
            
        if unitary and self.config.storage_method == "real":
            # Make vector unitary by ensuring it's its own inverse
            # For real vectors, this means making the Fourier transform real
            fft = np.fft.fft(vector)
            fft = np.abs(fft) * np.exp(1j * np.angle(fft) * 0)
            vector = np.real(np.fft.ifft(fft))
            if self.config.normalize:
                vector = self._normalize(vector)
                
        return vector
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two vectors using circular convolution.
        
        The binding operation creates a new vector that represents the
        association between the two input vectors. This operation is
        commutative but not associative.
        
        Parameters
        ----------
        a, b : np.ndarray
            Vectors to bind together
            
        Returns
        -------
        np.ndarray
            Bound vector representing the association
            
        Examples
        --------
        >>> hrr = HRR(HRRConfig(dimension=1024))
        >>> role = hrr.generate_vector()
        >>> filler = hrr.generate_vector()
        >>> binding = hrr.bind(role, filler)
        """
        self._validate_vector(a)
        self._validate_vector(b)
        
        # Use FFT for efficient circular convolution
        result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))
        
        if self.config.storage_method == "real":
            result = np.real(result)
            
        if self.config.normalize:
            result = self._normalize(result)
            
        return result
    
    def unbind(self, c: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Unbind a vector from a composite using circular correlation.
        
        Given a composite vector c = bind(a, b), unbinding with a
        retrieves an approximation of b.
        
        Parameters
        ----------
        c : np.ndarray
            Composite vector to unbind from
        a : np.ndarray
            Vector to unbind with
            
        Returns
        -------
        np.ndarray
            Retrieved vector
            
        Examples
        --------
        >>> hrr = HRR(HRRConfig(dimension=1024))
        >>> role = hrr.generate_vector()
        >>> filler = hrr.generate_vector()
        >>> binding = hrr.bind(role, filler)
        >>> retrieved = hrr.unbind(binding, role)
        >>> similarity = hrr.similarity(retrieved, filler)
        """
        self._validate_vector(c)
        self._validate_vector(a)
        
        # Circular correlation: corr(c, a) = IFFT(FFT(c) * conj(FFT(a)))
        result = np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(a)))
        
        if self.config.storage_method == "real":
            result = np.real(result)
            
        if self.config.normalize:
            result = self._normalize(result)
            
        return result
    
    def bundle(self, vectors: List[np.ndarray], 
               weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle multiple vectors using weighted superposition.
        
        Bundling creates a vector that is similar to all input vectors.
        This operation is used to create disjunctions or sets.
        
        Parameters
        ----------
        vectors : List[np.ndarray]
            Vectors to bundle together
        weights : List[float], optional
            Weights for each vector (default: equal weights)
            
        Returns
        -------
        np.ndarray
            Bundled vector
            
        Examples
        --------
        >>> hrr = HRR(HRRConfig(dimension=1024))
        >>> items = [hrr.generate_vector() for _ in range(3)]
        >>> bundle = hrr.bundle(items)
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
            
        for v in vectors:
            self._validate_vector(v)
            
        if weights is None:
            weights = [1.0] * len(vectors)
        elif len(weights) != len(vectors):
            raise ValueError(f"Number of weights ({len(weights)}) must match "
                           f"number of vectors ({len(vectors)})")
            
        # Weighted sum
        result = np.zeros_like(vectors[0])
        for v, w in zip(vectors, weights):
            result += w * v
            
        if self.config.normalize:
            result = self._normalize(result)
            
        return result
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Parameters
        ----------
        a, b : np.ndarray
            Vectors to compare
            
        Returns
        -------
        float
            Cosine similarity in range [-1, 1]
        """
        self._validate_vector(a)
        self._validate_vector(b)
        
        # Handle complex vectors
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            # Use complex dot product
            dot_product = np.real(np.vdot(a, b))
            norm_a = np.sqrt(np.real(np.vdot(a, a)))
            norm_b = np.sqrt(np.real(np.vdot(b, b)))
        else:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def store(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Store a key-value association in memory.
        
        This creates a composite vector that binds the key and value,
        which is then added to the memory trace.
        
        Parameters
        ----------
        key : np.ndarray
            Key vector
        value : np.ndarray
            Value vector to associate with key
        """
        self._validate_vector(key)
        self._validate_vector(value)
        
        # Create composite binding
        composite = self.bind(key, value)
        
        # Add to memory trace (superposition)
        if "memory_trace" not in self.memory:
            self.memory["memory_trace"] = composite.copy()
        else:
            self.memory["memory_trace"] += composite
            if self.config.normalize:
                self.memory["memory_trace"] = self._normalize(
                    self.memory["memory_trace"]
                )
    
    def recall(self, key: np.ndarray) -> Optional[np.ndarray]:
        """
        Recall a value from memory given a key.
        
        Parameters
        ----------
        key : np.ndarray
            Key vector to search for
            
        Returns
        -------
        Optional[np.ndarray]
            Retrieved value vector or None if not found
        """
        self._validate_vector(key)
        
        if "memory_trace" not in self.memory:
            return None
            
        # Unbind from memory trace
        retrieved = self.unbind(self.memory["memory_trace"], key)
        
        # Check if retrieval is above threshold
        # (In a full implementation, this would use cleanup memory)
        # For now, return the retrieved vector
        return retrieved
    
    def clear(self) -> None:
        """Clear all stored memories."""
        self.memory.clear()
    
    @property
    def size(self) -> int:
        """Return the current number of stored items."""
        # Count named items (excluding memory_trace)
        return len([k for k in self.memory.keys() if k != "memory_trace"])
    
    def add_item(self, name: str, vector: np.ndarray) -> None:
        """
        Add a named item to memory for cleanup operations.
        
        Parameters
        ----------
        name : str
            Name/label for the item
        vector : np.ndarray
            Vector representation of the item
        """
        self._validate_vector(vector)
        self.memory[name] = vector.copy()
    
    def get_item(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve a named item from memory.
        
        Parameters
        ----------
        name : str
            Name of the item to retrieve
            
        Returns
        -------
        Optional[np.ndarray]
            Item vector or None if not found
        """
        return self.memory.get(name)
    
    def _validate_vector(self, vector: np.ndarray) -> None:
        """Validate vector shape and type."""
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(vector)}")
            
        expected_shape = (self.config.dimension,)
        if self.config.storage_method == "complex":
            expected_shape = (self.config.dimension // 2,)
            
        if vector.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, "
                           f"got {vector.shape}")
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        if np.iscomplexobj(vector):
            norm = np.sqrt(np.real(np.vdot(vector, vector)))
        else:
            norm = np.linalg.norm(vector)
            
        if norm > 0:
            return vector / norm
        return vector
    
    def make_unitary(self, vector: np.ndarray) -> np.ndarray:
        """
        Make a vector unitary (self-inverse under correlation).
        
        Parameters
        ----------
        vector : np.ndarray
            Input vector
            
        Returns
        -------
        np.ndarray
            Unitary vector
        """
        self._validate_vector(vector)
        
        if self.config.storage_method == "complex":
            # For complex vectors, set all magnitudes to 1
            return np.exp(1j * np.angle(vector))
        else:
            # For real vectors, make Fourier transform have unit magnitude
            # and conjugate symmetry
            fft = np.fft.fft(vector)
            
            # Set magnitudes to 1 while preserving conjugate symmetry
            n = len(vector)
            angles = np.angle(fft)
            
            # First half (including DC and Nyquist if present)
            for i in range((n + 1) // 2):
                if i == 0 or (n % 2 == 0 and i == n // 2):
                    # DC and Nyquist must be real for real output
                    fft[i] = 1.0 if np.real(fft[i]) >= 0 else -1.0
                else:
                    fft[i] = np.exp(1j * angles[i])
            
            # Second half must be conjugate of first half
            for i in range(1, n // 2):
                fft[n - i] = np.conj(fft[i])
            
            result = np.real(np.fft.ifft(fft))
            if self.config.normalize:
                result = self._normalize(result)
            return result