"""
Hypervector types and operations for HDC.

This module provides various hypervector implementations and operations
specific to hyperdimensional computing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.special import erfinv


class Hypervector(ABC):
    """Abstract base class for hypervector types."""
    
    def __init__(self, dimension: int, seed: Optional[int] = None):
        """
        Initialize hypervector.
        
        Parameters
        ----------
        dimension : int
            Dimensionality of the hypervector
        seed : int, optional
            Random seed for reproducibility
        """
        self.dimension = dimension
        self._rng = np.random.RandomState(seed)
        
    @abstractmethod
    def random(self) -> np.ndarray:
        """Generate a random hypervector."""
        pass
        
    @abstractmethod
    def zero(self) -> np.ndarray:
        """Generate a zero/neutral hypervector."""
        pass
        
    @abstractmethod
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors."""
        pass
        
    @abstractmethod
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors."""
        pass
        
    @abstractmethod
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between hypervectors."""
        pass


class BinaryHypervector(Hypervector):
    """Binary hypervector implementation using {0, 1} values."""
    
    def random(self) -> np.ndarray:
        """Generate random binary hypervector."""
        return self._rng.randint(0, 2, size=self.dimension, dtype=np.uint8)
        
    def zero(self) -> np.ndarray:
        """Generate zero hypervector (all 0.5 probability)."""
        # For binary, we use equal probability as "zero"
        return self.random()
        
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind using XOR operation."""
        return np.bitwise_xor(a, b)
        
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle using majority voting."""
        if not vectors:
            raise ValueError("Cannot bundle empty list")
            
        summed = np.sum(vectors, axis=0)
        threshold = len(vectors) / 2
        bundled = (summed > threshold).astype(np.uint8)
        
        # Handle ties randomly
        ties = summed == threshold
        if np.any(ties):
            bundled[ties] = self._rng.randint(0, 2, size=np.sum(ties))
            
        return bundled
        
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Normalized Hamming similarity."""
        hamming_dist = np.sum(a != b)
        return 1 - (2 * hamming_dist / self.dimension)


class BipolarHypervector(Hypervector):
    """Bipolar hypervector implementation using {-1, +1} values."""
    
    def random(self) -> np.ndarray:
        """Generate random bipolar hypervector."""
        return 2 * self._rng.randint(0, 2, size=self.dimension) - 1
        
    def zero(self) -> np.ndarray:
        """Generate zero hypervector (random)."""
        return self.random()
        
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind using element-wise multiplication."""
        return a * b
        
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle using sum and sign."""
        if not vectors:
            raise ValueError("Cannot bundle empty list")
            
        summed = np.sum(vectors, axis=0)
        bundled = np.sign(summed)
        
        # Handle zeros (ties) randomly
        zeros = bundled == 0
        if np.any(zeros):
            bundled[zeros] = 2 * self._rng.randint(0, 2, size=np.sum(zeros)) - 1
            
        return bundled.astype(np.int8)
        
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
        return np.dot(a, b) / self.dimension


class TernaryHypervector(Hypervector):
    """Ternary hypervector implementation using {-1, 0, +1} values."""
    
    def __init__(self, dimension: int, sparsity: float = 0.33, seed: Optional[int] = None):
        """
        Initialize ternary hypervector.
        
        Parameters
        ----------
        dimension : int
            Dimensionality of the hypervector
        sparsity : float
            Proportion of non-zero elements (default: 0.33)
        seed : int, optional
            Random seed
        """
        super().__init__(dimension, seed)
        self.sparsity = sparsity
        
    def random(self) -> np.ndarray:
        """Generate random sparse ternary hypervector."""
        hv = np.zeros(self.dimension, dtype=np.int8)
        
        # Number of non-zero elements
        n_nonzero = int(self.dimension * self.sparsity)
        indices = self._rng.choice(self.dimension, n_nonzero, replace=False)
        
        # Assign random +1 or -1
        hv[indices] = 2 * self._rng.randint(0, 2, size=n_nonzero) - 1
        return hv
        
    def zero(self) -> np.ndarray:
        """Generate zero hypervector (all zeros)."""
        return np.zeros(self.dimension, dtype=np.int8)
        
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind using element-wise multiplication."""
        return a * b
        
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle using threshold-based approach."""
        if not vectors:
            raise ValueError("Cannot bundle empty list")
            
        summed = np.sum(vectors, axis=0)
        threshold = len(vectors) * self.sparsity / 2
        
        bundled = np.zeros(self.dimension, dtype=np.int8)
        bundled[summed > threshold] = 1
        bundled[summed < -threshold] = -1
        
        return bundled
        
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for sparse vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)


class LevelHypervector(Hypervector):
    """Multi-level hypervector with discrete levels."""
    
    def __init__(self, dimension: int, levels: int = 5, seed: Optional[int] = None):
        """
        Initialize level hypervector.
        
        Parameters
        ----------
        dimension : int
            Dimensionality of the hypervector
        levels : int
            Number of discrete levels (default: 5)
        seed : int, optional
            Random seed
        """
        super().__init__(dimension, seed)
        self.levels = levels
        
    def random(self) -> np.ndarray:
        """Generate random level hypervector."""
        return self._rng.randint(0, self.levels, size=self.dimension, dtype=np.int8)
        
    def zero(self) -> np.ndarray:
        """Generate zero hypervector (middle level)."""
        return np.full(self.dimension, self.levels // 2, dtype=np.int8)
        
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind using modular addition."""
        return (a + b) % self.levels
        
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle using averaging and rounding."""
        if not vectors:
            raise ValueError("Cannot bundle empty list")
            
        averaged = np.mean(vectors, axis=0)
        return np.round(averaged).astype(np.int8) % self.levels
        
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Normalized agreement similarity."""
        agreements = np.sum(a == b)
        return agreements / self.dimension


def generate_orthogonal_hypervectors(
    dimension: int,
    n_vectors: int,
    hypervector_type: str = "bipolar",
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate a set of quasi-orthogonal hypervectors.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of hypervectors
    n_vectors : int
        Number of hypervectors to generate
    hypervector_type : str
        Type of hypervectors: "binary" or "bipolar"
    seed : int, optional
        Random seed
        
    Returns
    -------
    List[np.ndarray]
        List of quasi-orthogonal hypervectors
        
    Notes
    -----
    For high dimensions, random vectors are nearly orthogonal with high
    probability. This function generates random vectors and optionally
    applies Gram-Schmidt orthogonalization for small sets.
    """
    rng = np.random.RandomState(seed)
    
    if n_vectors > dimension:
        raise ValueError(
            f"Cannot generate {n_vectors} orthogonal vectors "
            f"in {dimension} dimensions"
        )
        
    vectors = []
    
    if hypervector_type == "binary":
        # Generate random binary vectors
        for _ in range(n_vectors):
            vec = rng.randint(0, 2, size=dimension, dtype=np.uint8)
            vectors.append(vec)
            
    elif hypervector_type == "bipolar":
        if n_vectors <= dimension // 10:
            # For small sets, use Gram-Schmidt
            # Start with random Gaussian vectors
            gaussian_vecs = rng.randn(n_vectors, dimension)
            
            # Gram-Schmidt orthogonalization
            for i in range(n_vectors):
                vec = gaussian_vecs[i]
                
                # Orthogonalize against previous vectors
                for j in range(i):
                    vec -= np.dot(vec, gaussian_vecs[j]) * gaussian_vecs[j]
                    
                # Normalize
                vec = vec / np.linalg.norm(vec)
                gaussian_vecs[i] = vec
                
            # Convert to bipolar
            for vec in gaussian_vecs:
                bipolar = np.sign(vec)
                bipolar[bipolar == 0] = 1
                vectors.append(bipolar.astype(np.int8))
        else:
            # For large sets, rely on high-dimensional randomness
            for _ in range(n_vectors):
                vec = 2 * rng.randint(0, 2, size=dimension) - 1
                vectors.append(vec)
                
    else:
        raise ValueError(f"Unsupported hypervector type: {hypervector_type}")
        
    return vectors


def fractional_binding(
    a: np.ndarray,
    b: np.ndarray,
    weight: float,
    hypervector_type: str = "bipolar"
) -> np.ndarray:
    """
    Perform fractional binding between two hypervectors.
    
    Parameters
    ----------
    a, b : np.ndarray
        Hypervectors to bind
    weight : float
        Binding weight in [0, 1], where 0 gives a and 1 gives bind(a, b)
    hypervector_type : str
        Type of hypervectors
        
    Returns
    -------
    np.ndarray
        Fractionally bound hypervector
        
    Notes
    -----
    This implements the FLiPR (Fractional Power) binding from the literature,
    allowing continuous interpolation between unbound and fully bound states.
    """
    if not 0 <= weight <= 1:
        raise ValueError(f"Weight must be in [0, 1], got {weight}")
        
    if weight == 0:
        return a.copy()
    elif weight == 1:
        if hypervector_type == "binary":
            return np.bitwise_xor(a, b)
        elif hypervector_type == "bipolar":
            return a * b
        else:
            raise ValueError(f"Unsupported type for fractional binding: {hypervector_type}")
            
    # For intermediate weights, use probabilistic binding
    if hypervector_type == "bipolar":
        # Interpolate between a and a*b
        bound = a * b
        # Use inverse error function for proper probability mapping
        threshold = erfinv(2 * weight - 1) * np.sqrt(2)
        noise = np.random.randn(len(a))
        
        result = np.where(noise < threshold, bound, a)
        return result.astype(np.int8)
        
    else:
        raise NotImplementedError(
            f"Fractional binding not implemented for {hypervector_type}"
        )


def protect_hypervector(
    hypervector: np.ndarray,
    n_protections: int = 1,
    permutation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Protect a hypervector using permutation-based encoding.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Hypervector to protect
    n_protections : int
        Number of protection layers (default: 1)
    permutation : np.ndarray, optional
        Permutation to use (generated if not provided)
        
    Returns
    -------
    protected : np.ndarray
        Protected hypervector
    permutation : np.ndarray
        Permutation used for protection
        
    Notes
    -----
    Protection makes hypervectors more robust to noise and interference
    by applying reversible permutations.
    """
    dimension = len(hypervector)
    
    if permutation is None:
        # Generate random permutation
        permutation = np.random.permutation(dimension)
        
    protected = hypervector.copy()
    
    for _ in range(n_protections):
        protected = protected[permutation]
        
    return protected, permutation


def unprotect_hypervector(
    protected: np.ndarray,
    permutation: np.ndarray,
    n_protections: int = 1
) -> np.ndarray:
    """
    Unprotect a hypervector by reversing permutation-based encoding.
    
    Parameters
    ----------
    protected : np.ndarray
        Protected hypervector
    permutation : np.ndarray
        Permutation used for protection
    n_protections : int
        Number of protection layers to reverse
        
    Returns
    -------
    np.ndarray
        Unprotected hypervector
    """
    # Create inverse permutation
    inverse_perm = np.argsort(permutation)
    
    unprotected = protected.copy()
    
    for _ in range(n_protections):
        unprotected = unprotected[inverse_perm]
        
    return unprotected