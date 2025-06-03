"""
VSA-specific operations including permutation, thinning, and bundling.

This module provides operations that are unique to VSA or have special
implementations for different vector types.
"""

import logging
from typing import List, Optional, Union, Tuple
import numpy as np
from scipy import sparse

from .core import VectorType

logger = logging.getLogger(__name__)


def permute(vector: np.ndarray, 
            permutation: Optional[np.ndarray] = None,
            shift: Optional[int] = None) -> np.ndarray:
    """
    Permute a vector using either explicit permutation or cyclic shift.
    
    Parameters
    ----------
    vector : np.ndarray
        Vector to permute
    permutation : np.ndarray, optional
        Explicit permutation array. If None and shift is None, 
        generates random permutation
    shift : int, optional
        Cyclic shift amount. Takes precedence over permutation
        
    Returns
    -------
    np.ndarray
        Permuted vector
        
    Examples
    --------
    >>> # Cyclic shift
    >>> shifted = permute(vector, shift=1)
    
    >>> # Random permutation
    >>> permuted = permute(vector)
    
    >>> # Explicit permutation
    >>> perm = np.array([2, 0, 1, 3, 4])
    >>> permuted = permute(vector[:5], permutation=perm)
    """
    if shift is not None:
        # Cyclic shift
        return np.roll(vector, shift)
    elif permutation is not None:
        # Explicit permutation
        return vector[permutation]
    else:
        # Random permutation
        perm = np.random.permutation(len(vector))
        return vector[perm]


def inverse_permute(vector: np.ndarray,
                    permutation: Optional[np.ndarray] = None,
                    shift: Optional[int] = None) -> np.ndarray:
    """
    Apply inverse permutation to a vector.
    
    Parameters
    ----------
    vector : np.ndarray
        Permuted vector
    permutation : np.ndarray, optional
        Original permutation array
    shift : int, optional
        Original cyclic shift amount
        
    Returns
    -------
    np.ndarray
        Original vector before permutation
    """
    if shift is not None:
        # Inverse cyclic shift
        return np.roll(vector, -shift)
    elif permutation is not None:
        # Inverse permutation
        inverse_perm = np.argsort(permutation)
        return vector[inverse_perm]
    else:
        raise ValueError("Either permutation or shift must be provided")


def generate_permutation(dimension: int, 
                        permutation_type: str = "random",
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a permutation for the given dimension.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    permutation_type : str
        Type of permutation: 'random', 'cyclic', 'block', 'hierarchical'
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Permutation array
    """
    rng = np.random.RandomState(seed)
    
    if permutation_type == "random":
        return rng.permutation(dimension)
    
    elif permutation_type == "cyclic":
        # Cyclic shift by 1
        return np.roll(np.arange(dimension), 1)
    
    elif permutation_type == "block":
        # Block permutation (swap halves)
        mid = dimension // 2
        return np.concatenate([np.arange(mid, dimension), np.arange(mid)])
    
    elif permutation_type == "hierarchical":
        # Hierarchical bit-reversal permutation
        if dimension & (dimension - 1) != 0:
            # Not a power of 2, use random
            return rng.permutation(dimension)
        
        # Bit reversal for powers of 2
        n_bits = int(np.log2(dimension))
        indices = np.arange(dimension)
        reversed_indices = np.zeros(dimension, dtype=int)
        
        for i in range(dimension):
            reversed_indices[i] = int(
                bin(i)[2:].zfill(n_bits)[::-1], 2
            )
        
        return reversed_indices
    
    else:
        raise ValueError(f"Unknown permutation type: {permutation_type}")


def thin(vector: np.ndarray, 
         sparsity: float,
         method: str = "threshold",
         preserve_sign: bool = True) -> np.ndarray:
    """
    Make a vector sparser by setting values to zero.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector
    sparsity : float
        Target sparsity (0 = no change, 1 = all zeros)
    method : str
        Thinning method: 'threshold', 'random', 'magnitude'
    preserve_sign : bool
        Whether to preserve signs when thinning
        
    Returns
    -------
    np.ndarray
        Thinned vector
    """
    if sparsity <= 0:
        return vector.copy()
    
    if sparsity >= 1:
        return np.zeros_like(vector)
    
    num_zeros = int(len(vector) * sparsity)
    
    if method == "threshold":
        # Zero out values below threshold
        threshold = np.percentile(np.abs(vector), sparsity * 100)
        result = vector.copy()
        result[np.abs(result) < threshold] = 0
        return result
    
    elif method == "random":
        # Randomly zero out elements
        result = vector.copy()
        zero_idx = np.random.choice(len(vector), num_zeros, replace=False)
        result[zero_idx] = 0
        return result
    
    elif method == "magnitude":
        # Keep only top magnitude values
        result = np.zeros_like(vector)
        num_keep = len(vector) - num_zeros
        top_idx = np.argpartition(np.abs(vector), -num_keep)[-num_keep:]
        result[top_idx] = vector[top_idx]
        return result
    
    else:
        raise ValueError(f"Unknown thinning method: {method}")


def thicken(vector: np.ndarray,
            density: float,
            method: str = "random",
            value_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """
    Make a sparse vector denser by adding non-zero values.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector (possibly sparse)
    density : float
        Target density (1 = fully dense)
    method : str
        Thickening method: 'random', 'interpolate', 'duplicate'
    value_range : Tuple[float, float]
        Range for new values
        
    Returns
    -------
    np.ndarray
        Thickened vector
    """
    current_density = np.count_nonzero(vector) / len(vector)
    
    if current_density >= density:
        return vector.copy()
    
    result = vector.copy()
    zero_idx = np.where(vector == 0)[0]
    
    if len(zero_idx) == 0:
        return result
    
    # Number of zeros to fill
    num_fill = int((density - current_density) * len(vector))
    num_fill = min(num_fill, len(zero_idx))
    
    if method == "random":
        # Fill with random values
        fill_idx = np.random.choice(zero_idx, num_fill, replace=False)
        if value_range == (-1, 1):
            # Bipolar values
            result[fill_idx] = np.random.choice([-1, 1], num_fill)
        else:
            # Random values in range
            result[fill_idx] = np.random.uniform(
                value_range[0], value_range[1], num_fill
            )
    
    elif method == "interpolate":
        # Interpolate between non-zero values
        nonzero_idx = np.where(vector != 0)[0]
        if len(nonzero_idx) < 2:
            # Fall back to random
            return thicken(vector, density, method="random", value_range=value_range)
        
        # Simple nearest neighbor interpolation
        fill_idx = np.random.choice(zero_idx, num_fill, replace=False)
        for idx in fill_idx:
            # Find nearest non-zero neighbors
            distances = np.abs(nonzero_idx - idx)
            nearest = nonzero_idx[np.argmin(distances)]
            result[idx] = vector[nearest] * 0.5  # Dampened copy
    
    elif method == "duplicate":
        # Duplicate existing non-zero values
        nonzero_idx = np.where(vector != 0)[0]
        if len(nonzero_idx) == 0:
            # Fall back to random
            return thicken(vector, density, method="random", value_range=value_range)
        
        fill_idx = np.random.choice(zero_idx, num_fill, replace=False)
        source_idx = np.random.choice(nonzero_idx, num_fill, replace=True)
        result[fill_idx] = vector[source_idx]
    
    else:
        raise ValueError(f"Unknown thickening method: {method}")
    
    return result


def bundle(vectors: List[np.ndarray],
           weights: Optional[List[float]] = None,
           method: str = "sum",
           normalize: bool = True,
           vector_type: Optional[VectorType] = None) -> np.ndarray:
    """
    Bundle multiple vectors into a single vector.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        Vectors to bundle
    weights : List[float], optional
        Weights for weighted bundling
    method : str
        Bundling method: 'sum', 'average', 'majority', 'sample'
    normalize : bool
        Whether to normalize the result
    vector_type : VectorType, optional
        Vector type for type-specific bundling
        
    Returns
    -------
    np.ndarray
        Bundled vector
    """
    if len(vectors) == 0:
        raise ValueError("Cannot bundle empty list of vectors")
    
    if weights is not None:
        if len(weights) != len(vectors):
            raise ValueError("Number of weights must match number of vectors")
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
    
    if method == "sum":
        if weights is not None:
            result = sum(w * v for w, v in zip(weights, vectors))
        else:
            result = sum(vectors)
    
    elif method == "average":
        if weights is not None:
            result = sum(w * v for w, v in zip(weights, vectors))
        else:
            result = np.mean(vectors, axis=0)
    
    elif method == "majority":
        # Majority voting (good for binary/ternary)
        if weights is not None:
            # Weighted voting
            weighted_sum = sum(w * v for w, v in zip(weights, vectors))
            result = np.sign(weighted_sum)
        else:
            # Simple majority
            result = np.sign(np.sum(vectors, axis=0))
    
    elif method == "sample":
        # Randomly sample from vectors (with optional weights)
        if weights is not None:
            # Weighted sampling
            indices = np.random.choice(len(vectors), len(vectors[0]), p=weights)
            result = np.array([vectors[i][j] for j, i in enumerate(indices)])
        else:
            # Uniform sampling
            indices = np.random.randint(0, len(vectors), len(vectors[0]))
            result = np.array([vectors[i][j] for j, i in enumerate(indices)])
    
    else:
        raise ValueError(f"Unknown bundling method: {method}")
    
    # Type-specific post-processing
    if vector_type == VectorType.BINARY:
        result = (result > 0).astype(np.uint8)
    elif vector_type in [VectorType.BIPOLAR, VectorType.TERNARY]:
        if normalize:
            result = np.sign(result)
    elif vector_type == VectorType.COMPLEX:
        if normalize:
            # Normalize complex vectors to unit magnitude
            magnitudes = np.abs(result)
            magnitudes[magnitudes == 0] = 1.0
            result = result / magnitudes
    
    return result


def normalize_vector(vector: np.ndarray,
                    vector_type: Optional[VectorType] = None,
                    method: str = "default") -> np.ndarray:
    """
    Normalize a vector according to its type.
    
    Parameters
    ----------
    vector : np.ndarray
        Vector to normalize
    vector_type : VectorType, optional
        Vector type for type-specific normalization
    method : str
        Normalization method: 'default', 'L2', 'L1', 'max', 'sign'
        
    Returns
    -------
    np.ndarray
        Normalized vector
    """
    if method == "default":
        # Type-specific default normalization
        if vector_type == VectorType.BINARY:
            return (vector > 0.5).astype(np.uint8)
        elif vector_type == VectorType.BIPOLAR:
            return np.sign(vector).astype(np.float32)
        elif vector_type == VectorType.TERNARY:
            normalized = np.zeros_like(vector)
            normalized[vector > 0.33] = 1
            normalized[vector < -0.33] = -1
            return normalized.astype(np.float32)
        elif vector_type == VectorType.COMPLEX:
            magnitudes = np.abs(vector)
            magnitudes[magnitudes == 0] = 1.0
            return vector / magnitudes
        else:
            # Default to L2
            method = "L2"
    
    if method == "L2":
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    elif method == "L1":
        norm = np.sum(np.abs(vector))
        if norm > 0:
            return vector / norm
        return vector
    
    elif method == "max":
        max_val = np.max(np.abs(vector))
        if max_val > 0:
            return vector / max_val
        return vector
    
    elif method == "sign":
        return np.sign(vector)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def measure_sparsity(vector: np.ndarray) -> float:
    """
    Measure the sparsity of a vector.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector
        
    Returns
    -------
    float
        Sparsity level (0 = dense, 1 = all zeros)
    """
    num_zeros = np.count_nonzero(vector == 0)
    return num_zeros / len(vector)


def create_sparse_vector(dimension: int,
                        num_active: int,
                        vector_type: VectorType = VectorType.BIPOLAR,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Create a sparse vector with specified number of active elements.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    num_active : int
        Number of non-zero elements
    vector_type : VectorType
        Type of vector to create
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Sparse vector
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with zeros
    if vector_type == VectorType.BINARY:
        vector = np.zeros(dimension, dtype=np.uint8)
    elif vector_type == VectorType.COMPLEX:
        vector = np.zeros(dimension, dtype=np.complex64)
    else:
        vector = np.zeros(dimension, dtype=np.float32)
    
    # Select active positions
    active_idx = rng.choice(dimension, num_active, replace=False)
    
    # Fill active positions
    if vector_type == VectorType.BINARY:
        vector[active_idx] = 1
    elif vector_type == VectorType.BIPOLAR:
        vector[active_idx] = rng.choice([-1, 1], num_active)
    elif vector_type == VectorType.TERNARY:
        vector[active_idx] = rng.choice([-1, 1], num_active)
    elif vector_type == VectorType.COMPLEX:
        phases = rng.uniform(0, 2 * np.pi, num_active)
        vector[active_idx] = np.exp(1j * phases)
    else:
        # Real values
        vector[active_idx] = rng.randn(num_active)
    
    return vector