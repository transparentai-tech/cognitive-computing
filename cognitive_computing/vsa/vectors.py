"""
Vector type implementations for Vector Symbolic Architectures.

This module provides different vector types used in VSA, including binary,
bipolar, ternary, and complex vectors. Each type has specific properties
and operations optimized for different use cases.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, List
import numpy as np
from scipy import stats

from .core import VectorType

logger = logging.getLogger(__name__)


class VSAVector(ABC):
    """
    Abstract base class for VSA vector types.
    
    Each vector type implements generation, normalization, similarity,
    and bundling operations specific to its representation.
    """
    
    def __init__(self, dimension: int, sparsity: float = 0.0, 
                 seed: Optional[int] = None):
        """
        Initialize vector factory.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        sparsity : float
            Sparsity level (0 = dense, approaching 1 = very sparse)
        seed : int, optional
            Random seed
        """
        self.dimension = dimension
        self.sparsity = sparsity
        self._rng = np.random.RandomState(seed)
        
    @abstractmethod
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random vector of this type."""
        pass
    
    @abstractmethod
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector according to type-specific rules."""
        pass
    
    @abstractmethod
    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        pass
    
    @abstractmethod
    def bundle_vectors(self, summed: np.ndarray) -> np.ndarray:
        """Apply type-specific bundling to a summed vector."""
        pass
    
    @abstractmethod
    def to_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert vector to bipolar representation."""
        pass
    
    @abstractmethod
    def from_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert from bipolar representation to this type."""
        pass


class BinaryVector(VSAVector):
    """
    Binary vector implementation using {0, 1} values.
    
    Optimized for XOR binding and hardware implementations.
    Uses Hamming distance for similarity.
    """
    
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random binary vector."""
        if sparse and self.sparsity > 0:
            # Generate sparse binary vector
            num_ones = int(self.dimension * (1 - self.sparsity) / 2)
            vector = np.zeros(self.dimension, dtype=np.uint8)
            ones_idx = self._rng.choice(self.dimension, num_ones, replace=False)
            vector[ones_idx] = 1
            return vector
        else:
            # Generate dense binary vector
            return self._rng.randint(0, 2, self.dimension, dtype=np.uint8)
    
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize binary vector by thresholding.
        
        Values >= 0.5 become 1, others become 0.
        """
        return (vector >= 0.5).astype(np.uint8)
    
    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate similarity using normalized Hamming distance.
        
        Returns value in [0, 1] where 1 is identical.
        """
        hamming_dist = np.sum(x != y)
        return 1.0 - (hamming_dist / self.dimension)
    
    def bundle_vectors(self, summed: np.ndarray) -> np.ndarray:
        """Bundle by majority voting."""
        threshold = len(summed.shape) / 2 if len(summed.shape) > 1 else 0.5
        return (summed > threshold).astype(np.uint8)
    
    def to_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert {0, 1} to {-1, +1}."""
        return 2 * vector.astype(np.float32) - 1
    
    def from_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert {-1, +1} to {0, 1}."""
        return ((vector + 1) / 2).astype(np.uint8)


class BipolarVector(VSAVector):
    """
    Bipolar vector implementation using {-1, +1} values.
    
    Standard for many VSA operations. Uses dot product similarity.
    Supports both dense and sparse representations.
    """
    
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random bipolar vector."""
        if sparse and self.sparsity > 0:
            # Generate sparse bipolar vector
            vector = np.zeros(self.dimension, dtype=np.float32)
            num_nonzero = int(self.dimension * (1 - self.sparsity))
            nonzero_idx = self._rng.choice(self.dimension, num_nonzero, replace=False)
            vector[nonzero_idx] = self._rng.choice([-1, 1], num_nonzero)
            return vector
        else:
            # Generate dense bipolar vector
            return self._rng.choice([-1, 1], self.dimension).astype(np.float32)
    
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize bipolar vector to unit length.
        
        Also ensures values are in {-1, 0, +1}.
        """
        # First threshold to {-1, 0, +1}
        vector = np.sign(vector)
        # Then normalize if non-zero
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate cosine similarity.
        
        Returns value in [-1, 1].
        """
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
            
        return dot_product / (norm_x * norm_y)
    
    def bundle_vectors(self, summed: np.ndarray) -> np.ndarray:
        """Bundle by sign of sum."""
        return np.sign(summed).astype(np.float32)
    
    def to_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Already bipolar."""
        return vector
    
    def from_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Already bipolar."""
        return vector


class TernaryVector(VSAVector):
    """
    Ternary vector implementation using {-1, 0, +1} values.
    
    Naturally sparse representation. Good for efficient storage
    and computation.
    """
    
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random ternary vector."""
        if sparse or self.sparsity > 0:
            # Control sparsity explicitly
            sparsity_level = max(self.sparsity, 0.5 if sparse else 0.0)
            vector = np.zeros(self.dimension, dtype=np.float32)
            num_nonzero = int(self.dimension * (1 - sparsity_level))
            nonzero_idx = self._rng.choice(self.dimension, num_nonzero, replace=False)
            vector[nonzero_idx] = self._rng.choice([-1, 1], num_nonzero)
            return vector
        else:
            # Generate dense ternary (equal probability for -1, 0, 1)
            return self._rng.choice([-1, 0, 1], self.dimension).astype(np.float32)
    
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize ternary vector by thresholding.
        
        Values > 0.33 become 1, < -0.33 become -1, else 0.
        """
        normalized = np.zeros_like(vector)
        normalized[vector > 0.33] = 1
        normalized[vector < -0.33] = -1
        return normalized.astype(np.float32)
    
    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate similarity for ternary vectors.
        
        Uses dot product normalized by number of active positions.
        """
        dot_product = np.dot(x, y)
        active_x = np.count_nonzero(x)
        active_y = np.count_nonzero(y)
        
        if active_x == 0 or active_y == 0:
            return 0.0
            
        # Normalize by geometric mean of active positions
        normalization = np.sqrt(active_x * active_y)
        return dot_product / normalization
    
    def bundle_vectors(self, summed: np.ndarray) -> np.ndarray:
        """Bundle by thresholded sign."""
        bundled = np.zeros_like(summed)
        threshold = 0.1 * np.max(np.abs(summed))
        bundled[summed > threshold] = 1
        bundled[summed < -threshold] = -1
        return bundled.astype(np.float32)
    
    def to_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert to bipolar by replacing 0 with random {-1, +1}."""
        bipolar = vector.copy()
        zero_idx = (vector == 0)
        num_zeros = np.sum(zero_idx)
        if num_zeros > 0:
            bipolar[zero_idx] = self._rng.choice([-1, 1], num_zeros)
        return bipolar
    
    def from_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert to ternary by thresholding."""
        return self.normalize(vector)


class ComplexVector(VSAVector):
    """
    Complex vector implementation using unit complex numbers.
    
    Supports phase-based binding. Used in Fourier HRR and
    frequency domain operations.
    """
    
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random complex unit vector."""
        if sparse and self.sparsity > 0:
            # Generate sparse complex vector
            vector = np.zeros(self.dimension, dtype=np.complex64)
            num_nonzero = int(self.dimension * (1 - self.sparsity))
            nonzero_idx = self._rng.choice(self.dimension, num_nonzero, replace=False)
            # Random phases
            phases = self._rng.uniform(0, 2 * np.pi, num_nonzero)
            vector[nonzero_idx] = np.exp(1j * phases)
            return vector
        else:
            # Generate dense complex unit vector
            phases = self._rng.uniform(0, 2 * np.pi, self.dimension)
            return np.exp(1j * phases).astype(np.complex64)
    
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize to unit magnitude while preserving phase."""
        magnitudes = np.abs(vector)
        # Avoid division by zero
        magnitudes[magnitudes == 0] = 1.0
        return (vector / magnitudes).astype(np.complex64)
    
    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate complex similarity.
        
        Uses real part of normalized dot product.
        """
        dot_product = np.vdot(x, y)  # Conjugate dot product
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
            
        return np.real(dot_product / (norm_x * norm_y))
    
    def bundle_vectors(self, summed: np.ndarray) -> np.ndarray:
        """Bundle by normalizing sum."""
        return self.normalize(summed)
    
    def to_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert complex to bipolar using real part sign."""
        return np.sign(np.real(vector)).astype(np.float32)
    
    def from_bipolar(self, vector: np.ndarray) -> np.ndarray:
        """Convert bipolar to complex with zero phase."""
        return vector.astype(np.complex64)


def create_vector(vector_type: Union[VectorType, str],
                  dimension: int,
                  sparsity: float = 0.0,
                  seed: Optional[int] = None) -> VSAVector:
    """
    Create a vector factory of the specified type.
    
    Parameters
    ----------
    vector_type : VectorType or str
        Type of vector (binary, bipolar, ternary, complex)
    dimension : int
        Vector dimension
    sparsity : float
        Sparsity level (0 = dense)
    seed : int, optional
        Random seed
        
    Returns
    -------
    VSAVector
        Vector factory instance
        
    Examples
    --------
    >>> binary_factory = create_vector("binary", 1024)
    >>> binary_vec = binary_factory.generate()
    
    >>> sparse_ternary = create_vector("ternary", 10000, sparsity=0.9)
    >>> ternary_vec = sparse_ternary.generate(sparse=True)
    """
    if isinstance(vector_type, str):
        try:
            vector_type = VectorType(vector_type.lower())
        except ValueError:
            raise ValueError(f"Unknown vector type: {vector_type}")
    
    factories = {
        VectorType.BINARY: BinaryVector,
        VectorType.BIPOLAR: BipolarVector,
        VectorType.TERNARY: TernaryVector,
        VectorType.COMPLEX: ComplexVector,
    }
    
    if vector_type == VectorType.REAL:
        # Real vectors use bipolar implementation
        logger.debug("Real vectors use BipolarVector implementation")
        vector_type = VectorType.BIPOLAR
    
    factory_class = factories.get(vector_type)
    if factory_class is None:
        raise ValueError(f"No implementation for vector type: {vector_type}")
    
    return factory_class(dimension, sparsity, seed)