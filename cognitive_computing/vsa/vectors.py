"""
Vector type implementations for Vector Symbolic Architectures.

This module provides different vector types used in VSA, including binary,
bipolar, ternary, complex, and integer vectors. Each type has specific properties
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
    
    def __init__(self, data: np.ndarray):
        """
        Initialize vector with data.
        
        Parameters
        ----------
        data : np.ndarray
            Vector data
        """
        self.data = data
        self.dimension = len(data)
        
    @abstractmethod
    def similarity(self, other: 'VSAVector') -> float:
        """Compute similarity with another vector."""
        pass
    
    @abstractmethod
    def normalize(self) -> 'VSAVector':
        """Return normalized version of vector."""
        pass
        
    @abstractmethod
    def to_bipolar(self) -> np.ndarray:
        """Convert to bipolar representation."""
        pass
        
    @classmethod
    @abstractmethod
    def random(cls, dimension: int, **kwargs) -> 'VSAVector':
        """Generate random vector of this type."""
        pass


class BinaryVector(VSAVector):
    """
    Binary vector implementation using {0, 1} values.
    
    Optimized for XOR binding and hardware implementations.
    Uses Hamming distance for similarity.
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize binary vector."""
        if not np.all(np.isin(data, [0, 1])):
            raise ValueError("Binary vector must contain only 0 and 1")
        super().__init__(data.astype(np.uint8))
        
    def similarity(self, other: 'BinaryVector') -> float:
        """
        Calculate similarity using normalized Hamming distance.
        
        Returns value in [0, 1] where 1 is identical.
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        hamming_dist = np.sum(self.data != other.data)
        return 1.0 - (hamming_dist / self.dimension)
    
    def normalize(self) -> 'BinaryVector':
        """Return self (binary vectors are already normalized)."""
        return BinaryVector(self.data.copy())
    
    def to_bipolar(self) -> np.ndarray:
        """Convert {0, 1} to {-1, +1}."""
        return 2 * self.data.astype(np.float32) - 1
    
    @classmethod
    def random(cls, dimension: int, rng: Optional[np.random.RandomState] = None) -> 'BinaryVector':
        """Generate random binary vector."""
        if rng is None:
            rng = np.random
        data = rng.randint(0, 2, dimension, dtype=np.uint8)
        return cls(data)
    
    @staticmethod
    def from_bipolar(vector: np.ndarray) -> 'BinaryVector':
        """Convert {-1, +1} to {0, 1}."""
        return BinaryVector(((vector + 1) / 2).astype(np.uint8))
        
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random binary vector."""
        if sparse and hasattr(self, 'sparsity') and self.sparsity > 0:
            # Generate sparse binary vector
            num_ones = int(self.dimension * (1 - self.sparsity) / 2)
            vector = np.zeros(self.dimension, dtype=np.uint8)
            ones_idx = self._rng.choice(self.dimension, num_ones, replace=False)
            vector[ones_idx] = 1
            return vector
        else:
            # Generate dense binary vector
            return self._rng.randint(0, 2, self.dimension, dtype=np.uint8)


class BipolarVector(VSAVector):
    """
    Bipolar vector implementation using {-1, +1} values.
    
    Standard for many VSA operations. Uses dot product similarity.
    Supports both dense and sparse representations.
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize bipolar vector."""
        if not np.all(np.isin(data, [-1, 1])):
            raise ValueError("Bipolar vector must contain only -1 and 1")
        super().__init__(data.astype(np.float32))
        
    def similarity(self, other: 'BipolarVector') -> float:
        """
        Calculate similarity using normalized dot product.
        
        Returns value in [-1, 1] where 1 is identical.
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        return np.dot(self.data, other.data) / self.dimension
    
    def normalize(self) -> 'BipolarVector':
        """Return self (bipolar vectors are already normalized)."""
        return BipolarVector(self.data.copy())
    
    def to_bipolar(self) -> np.ndarray:
        """Return self."""
        return self.data.copy()
    
    @classmethod
    def random(cls, dimension: int, rng: Optional[np.random.RandomState] = None) -> 'BipolarVector':
        """Generate random bipolar vector."""
        if rng is None:
            rng = np.random
        data = rng.choice([-1, 1], dimension).astype(np.float32)
        return cls(data)
    
    @staticmethod
    def from_binary(vector: np.ndarray) -> 'BipolarVector':
        """Convert {0, 1} to {-1, +1}."""
        return BipolarVector(2 * vector.astype(np.float32) - 1)
        
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random bipolar vector."""
        if sparse and hasattr(self, 'sparsity') and self.sparsity > 0:
            # Generate sparse bipolar vector with mostly zeros
            vector = np.zeros(self.dimension, dtype=np.float32)
            num_nonzero = int(self.dimension * (1 - self.sparsity))
            nonzero_idx = self._rng.choice(self.dimension, num_nonzero, replace=False)
            vector[nonzero_idx] = self._rng.choice([-1, 1], num_nonzero).astype(np.float32)
            return vector
        else:
            # Generate dense bipolar vector
            return self._rng.choice([-1, 1], self.dimension).astype(np.float32)


class TernaryVector(VSAVector):
    """
    Ternary vector implementation using {-1, 0, +1} values.
    
    Provides natural sparsity with zero values. Good for
    selective attention and feature gating.
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize ternary vector."""
        if not np.all(np.isin(data, [-1, 0, 1])):
            raise ValueError("Ternary vector must contain only -1, 0, and 1")
        super().__init__(data.astype(np.float32))
        
    def similarity(self, other: 'TernaryVector') -> float:
        """
        Calculate similarity using normalized dot product.
        
        Only considers non-zero positions.
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        
        # Only consider positions where at least one vector is non-zero
        mask = (self.data != 0) | (other.data != 0)
        if np.sum(mask) == 0:
            return 0.0
            
        active_x = self.data[mask]
        active_y = other.data[mask]
        
        return np.dot(active_x, active_y) / np.sum(mask)
    
    def normalize(self) -> 'TernaryVector':
        """Normalize to unit length."""
        norm = np.linalg.norm(self.data)
        if norm == 0:
            return TernaryVector(self.data.copy())
        normalized_data = self.data / norm
        # Return a special normalized ternary vector
        result = TernaryVector(np.zeros_like(self.data))
        result.data = normalized_data
        return result
    
    def to_bipolar(self) -> np.ndarray:
        """Convert to bipolar, mapping 0 to -1."""
        return np.where(self.data == 0, -1, self.data).astype(np.float32)
    
    @classmethod
    def random(cls, dimension: int, sparsity: float = 0.3,
               rng: Optional[np.random.RandomState] = None) -> 'TernaryVector':
        """
        Generate random ternary vector.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        sparsity : float
            Fraction of zero values
        rng : np.random.RandomState, optional
            Random number generator
        """
        if rng is None:
            rng = np.random
            
        # Generate sparse ternary vector
        vector = np.zeros(dimension, dtype=np.float32)
        num_nonzero = int(dimension * (1 - sparsity))
        nonzero_idx = rng.choice(dimension, num_nonzero, replace=False)
        vector[nonzero_idx] = rng.choice([-1, 1], num_nonzero).astype(np.float32)
        
        return cls(vector)
    
    @property
    def sparsity(self) -> float:
        """Get actual sparsity of vector."""
        return np.mean(self.data == 0)
        
    def generate(self, sparse: bool = True) -> np.ndarray:
        """Generate a random ternary vector."""
        if sparse:
            # Control sparsity level
            sparsity = getattr(self, 'sparsity', 0.3)
            vector = np.zeros(self.dimension, dtype=np.float32)
            num_nonzero = int(self.dimension * (1 - sparsity))
            nonzero_idx = self._rng.choice(self.dimension, num_nonzero, replace=False)
            vector[nonzero_idx] = self._rng.choice([-1, 1], num_nonzero).astype(np.float32)
            return vector
        else:
            # Generate dense ternary (all positions active)
            return self._rng.choice([-1, 0, 1], self.dimension).astype(np.float32)


class ComplexVector(VSAVector):
    """
    Complex vector implementation using unit magnitude phasors.
    
    Each component has magnitude 1 and arbitrary phase. Supports
    continuous rotation operations and phase-based binding.
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize complex vector."""
        if not np.allclose(np.abs(data), 1.0, atol=1e-6):
            raise ValueError("Complex vector components must have unit magnitude")
        super().__init__(data.astype(np.complex64))
        
    def similarity(self, other: 'ComplexVector') -> float:
        """
        Calculate similarity as real part of normalized dot product.
        
        Returns value in [-1, 1].
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        
        # Complex dot product normalized by dimension
        dot_product = np.dot(self.data, np.conj(other.data))
        return np.real(dot_product) / self.dimension
    
    def normalize(self) -> 'ComplexVector':
        """Normalize to unit magnitude per component."""
        magnitudes = np.abs(self.data)
        magnitudes[magnitudes == 0] = 1  # Avoid division by zero
        normalized = self.data / magnitudes
        return ComplexVector(normalized)
    
    def to_bipolar(self) -> np.ndarray:
        """Convert to bipolar based on real part sign."""
        return np.sign(np.real(self.data)).astype(np.float32)
    
    @classmethod
    def random(cls, dimension: int, rng: Optional[np.random.RandomState] = None) -> 'ComplexVector':
        """Generate random complex vector with uniform phase distribution."""
        if rng is None:
            rng = np.random
        
        # Random phases uniformly distributed in [-pi, pi]
        phases = rng.uniform(-np.pi, np.pi, dimension)
        data = np.exp(1j * phases).astype(np.complex64)
        return cls(data)
    
    @staticmethod
    def from_phases(phases: np.ndarray) -> 'ComplexVector':
        """Create complex vector from phase angles."""
        return ComplexVector(np.exp(1j * phases).astype(np.complex64))
        
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random complex vector."""
        if sparse and hasattr(self, 'sparsity') and self.sparsity > 0:
            # Sparse complex vector with some zero components
            phases = self._rng.uniform(-np.pi, np.pi, self.dimension)
            vector = np.exp(1j * phases)
            # Zero out some components
            num_zeros = int(self.dimension * self.sparsity)
            zero_idx = self._rng.choice(self.dimension, num_zeros, replace=False)
            vector[zero_idx] = 0
            # Renormalize non-zero components
            nonzero_mask = vector != 0
            vector[nonzero_mask] = vector[nonzero_mask] / np.abs(vector[nonzero_mask])
            return vector.astype(np.complex64)
        else:
            # Generate dense complex vector
            phases = self._rng.uniform(-np.pi, np.pi, self.dimension)
            return np.exp(1j * phases).astype(np.complex64)


class IntegerVector(VSAVector):
    """
    Integer vector implementation for modular arithmetic VSA.
    
    Uses integer values in range [0, modulus-1] with operations
    performed modulo the specified modulus.
    """
    
    def __init__(self, data: np.ndarray, modulus: int = 256):
        """
        Initialize integer vector.
        
        Parameters
        ----------
        data : np.ndarray
            Integer values in range [0, modulus-1]
        modulus : int
            Modulus for arithmetic operations
        """
        if modulus < 2:
            raise ValueError("Modulus must be at least 2")
            
        if not np.all((data >= 0) & (data < modulus)):
            raise ValueError(f"Integer vector values must be in range [0, {modulus-1}]")
            
        super().__init__(data.astype(np.int32))
        self.modulus = modulus
        
    def similarity(self, other: 'IntegerVector') -> float:
        """
        Compute similarity with another integer vector.
        
        Uses cosine similarity after mapping to [-1, 1] range.
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
            
        if self.modulus != other.modulus:
            raise ValueError("Vectors must have same modulus")
            
        # Map to [-1, 1] range
        x = (2 * self.data / (self.modulus - 1)) - 1
        y = (2 * other.data / (other.modulus - 1)) - 1
        
        # Cosine similarity
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
            
        return np.dot(x, y) / (norm_x * norm_y)
        
    def normalize(self) -> 'IntegerVector':
        """
        Normalize integer vector to [-1, 1] range.
        
        Returns a new vector with float values.
        """
        normalized_data = (2 * self.data / (self.modulus - 1)) - 1
        # Return as a special normalized integer vector
        normalized = IntegerVector(np.zeros(self.dimension), self.modulus)
        normalized.data = normalized_data
        return normalized
        
    def to_bipolar(self) -> np.ndarray:
        """Convert to bipolar representation."""
        # Values < modulus/2 -> -1, others -> 1
        return np.where(self.data < self.modulus / 2, -1, 1).astype(np.float32)
        
    @classmethod
    def random(cls, dimension: int, modulus: int = 256,
               rng: Optional[np.random.RandomState] = None) -> 'IntegerVector':
        """
        Generate random integer vector.
        
        Parameters
        ----------
        dimension : int
            Vector dimension
        modulus : int
            Modulus for values
        rng : np.random.RandomState, optional
            Random number generator
            
        Returns
        -------
        IntegerVector
            Random integer vector
        """
        if rng is None:
            rng = np.random
        data = rng.randint(0, modulus, size=dimension)
        return cls(data, modulus)
        
    def generate(self, sparse: bool = False) -> np.ndarray:
        """Generate a random integer vector."""
        if sparse and hasattr(self, 'sparsity') and self.sparsity > 0:
            # For sparse integer vectors, use fewer unique values
            num_values = max(2, int(self.modulus * (1 - self.sparsity)))
            values = self._rng.choice(self.modulus, size=num_values, replace=False)
            data = self._rng.choice(values, size=self.dimension)
        else:
            data = self._rng.randint(0, self.modulus, size=self.dimension)
        return data
        
    @staticmethod
    def from_data(data: np.ndarray, modulus: int = 256) -> np.ndarray:
        """Create integer vector from data."""
        return data.astype(np.int32) % modulus


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
    
    # For now, return a factory-like object
    class VectorFactory:
        def __init__(self, vtype, dim, sp, s):
            self.vector_type = vtype
            self.dimension = dim
            self.sparsity = sp
            self._rng = np.random.RandomState(s)
            
        def generate(self, sparse=False):
            if self.vector_type == VectorType.BINARY:
                return BinaryVector.random(self.dimension, self._rng).data
            elif self.vector_type == VectorType.BIPOLAR:
                return BipolarVector.random(self.dimension, self._rng).data
            elif self.vector_type == VectorType.TERNARY:
                return TernaryVector.random(self.dimension, self.sparsity if sparse else 0.0, self._rng).data
            elif self.vector_type == VectorType.COMPLEX:
                return ComplexVector.random(self.dimension, self._rng).data
    
    return VectorFactory(vector_type, dimension, sparsity, seed)