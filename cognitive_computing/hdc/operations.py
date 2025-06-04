"""
Core HDC operations for hypervector manipulation.

This module provides fundamental operations for hyperdimensional computing
including binding, bundling, permutation, and similarity measures.
"""

from typing import List, Optional, Tuple, Union, Callable
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BundlingMethod(Enum):
    """Enumeration of bundling methods."""
    MAJORITY = "majority"  # Majority voting
    AVERAGE = "average"    # Averaging with threshold
    SAMPLE = "sample"      # Random sampling
    WEIGHTED = "weighted"  # Weighted bundling


class PermutationMethod(Enum):
    """Enumeration of permutation methods."""
    CYCLIC = "cyclic"      # Cyclic shift
    RANDOM = "random"      # Random permutation
    BLOCK = "block"        # Block permutation
    INVERSE = "inverse"    # Inverse permutation


def bind_hypervectors(
    a: np.ndarray,
    b: np.ndarray,
    hypervector_type: str = "bipolar"
) -> np.ndarray:
    """
    Bind two hypervectors using type-appropriate operation.
    
    Parameters
    ----------
    a, b : np.ndarray
        Hypervectors to bind
    hypervector_type : str
        Type of hypervectors: "binary", "bipolar", "ternary", "level"
        
    Returns
    -------
    np.ndarray
        Bound hypervector
        
    Notes
    -----
    Binding is typically self-inverse, meaning bind(bind(a,b), b) = a
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
    if hypervector_type == "binary":
        return np.bitwise_xor(a, b)
    elif hypervector_type in ["bipolar", "ternary"]:
        return a * b
    elif hypervector_type == "level":
        # Assume levels are known from context
        raise ValueError("Level binding requires levels parameter")
    else:
        raise ValueError(f"Unknown hypervector type: {hypervector_type}")


def bundle_hypervectors(
    hypervectors: List[np.ndarray],
    method: Union[str, BundlingMethod] = BundlingMethod.MAJORITY,
    hypervector_type: str = "bipolar",
    weights: Optional[np.ndarray] = None,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Bundle multiple hypervectors into a single hypervector.
    
    Parameters
    ----------
    hypervectors : List[np.ndarray]
        List of hypervectors to bundle
    method : str or BundlingMethod
        Bundling method to use
    hypervector_type : str
        Type of hypervectors
    weights : np.ndarray, optional
        Weights for weighted bundling
    threshold : float
        Threshold for thresholding methods
        
    Returns
    -------
    np.ndarray
        Bundled hypervector
    """
    if not hypervectors:
        raise ValueError("Cannot bundle empty list")
        
    if isinstance(method, str):
        method = BundlingMethod(method)
        
    dimension = hypervectors[0].shape[0]
    
    # Validate all hypervectors have same shape
    for i, hv in enumerate(hypervectors):
        if hv.shape[0] != dimension:
            raise ValueError(f"Dimension mismatch at index {i}")
            
    if method == BundlingMethod.MAJORITY:
        if hypervector_type == "binary":
            # Majority voting for binary
            summed = np.sum(hypervectors, axis=0)
            threshold_val = len(hypervectors) / 2
            bundled = (summed > threshold_val).astype(np.uint8)
            
            # Handle ties randomly
            ties = summed == threshold_val
            if np.any(ties):
                bundled[ties] = np.random.randint(0, 2, size=np.sum(ties))
                
            return bundled
            
        elif hypervector_type == "bipolar":
            # Sum and sign for bipolar
            summed = np.sum(hypervectors, axis=0)
            bundled = np.sign(summed)
            
            # Handle zeros randomly
            zeros = bundled == 0
            if np.any(zeros):
                bundled[zeros] = 2 * np.random.randint(0, 2, size=np.sum(zeros)) - 1
                
            return bundled.astype(np.int8)
            
    elif method == BundlingMethod.AVERAGE:
        # Average and threshold
        averaged = np.mean(hypervectors, axis=0)
        
        if hypervector_type == "binary":
            return (averaged > 0.5).astype(np.uint8)
        elif hypervector_type == "bipolar":
            return np.sign(averaged).astype(np.int8)
            
    elif method == BundlingMethod.SAMPLE:
        # Random sampling from inputs
        n_vectors = len(hypervectors)
        indices = np.random.randint(0, n_vectors, size=dimension)
        bundled = np.zeros(dimension, dtype=hypervectors[0].dtype)
        
        for i in range(dimension):
            bundled[i] = hypervectors[indices[i]][i]
            
        return bundled
        
    elif method == BundlingMethod.WEIGHTED:
        if weights is None:
            raise ValueError("Weights required for weighted bundling")
            
        if len(weights) != len(hypervectors):
            raise ValueError("Number of weights must match number of hypervectors")
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted sum
        weighted_sum = np.zeros(dimension)
        for w, hv in zip(weights, hypervectors):
            weighted_sum += w * hv
            
        if hypervector_type == "binary":
            return (weighted_sum > 0.5).astype(np.uint8)
        elif hypervector_type == "bipolar":
            return np.sign(weighted_sum).astype(np.int8)
            
    else:
        raise ValueError(f"Unknown bundling method: {method}")


def permute_hypervector(
    hypervector: np.ndarray,
    method: Union[str, PermutationMethod] = PermutationMethod.CYCLIC,
    shift: int = 1,
    permutation: Optional[np.ndarray] = None,
    block_size: Optional[int] = None
) -> np.ndarray:
    """
    Permute a hypervector using various methods.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Hypervector to permute
    method : str or PermutationMethod
        Permutation method
    shift : int
        Shift amount for cyclic permutation
    permutation : np.ndarray, optional
        Custom permutation array
    block_size : int, optional
        Block size for block permutation
        
    Returns
    -------
    np.ndarray
        Permuted hypervector
    """
    if isinstance(method, str):
        method = PermutationMethod(method)
        
    if method == PermutationMethod.CYCLIC:
        return np.roll(hypervector, shift)
        
    elif method == PermutationMethod.RANDOM:
        if permutation is None:
            permutation = np.random.permutation(len(hypervector))
        return hypervector[permutation]
        
    elif method == PermutationMethod.BLOCK:
        if block_size is None:
            raise ValueError("Block size required for block permutation")
            
        dimension = len(hypervector)
        if dimension % block_size != 0:
            raise ValueError("Dimension must be divisible by block size")
            
        # Reshape into blocks, permute blocks, flatten
        n_blocks = dimension // block_size
        reshaped = hypervector.reshape(n_blocks, block_size)
        
        # Cyclic shift blocks
        permuted_blocks = np.roll(reshaped, shift, axis=0)
        return permuted_blocks.flatten()
        
    elif method == PermutationMethod.INVERSE:
        if permutation is None:
            raise ValueError("Permutation array required for inverse")
            
        # Create inverse permutation
        inverse_perm = np.argsort(permutation)
        return hypervector[inverse_perm]
        
    else:
        raise ValueError(f"Unknown permutation method: {method}")


def similarity(
    a: np.ndarray,
    b: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Calculate similarity between two hypervectors.
    
    Parameters
    ----------
    a, b : np.ndarray
        Hypervectors to compare
    metric : str
        Similarity metric: "cosine", "hamming", "euclidean", "jaccard"
        
    Returns
    -------
    float
        Similarity score
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
    if metric == "cosine":
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
        
    elif metric == "hamming":
        # Normalized Hamming similarity
        hamming_dist = np.sum(a != b)
        return 1 - (2 * hamming_dist / len(a))
        
    elif metric == "euclidean":
        # Normalized Euclidean similarity
        dist = np.linalg.norm(a - b)
        max_dist = np.sqrt(len(a) * 4)  # Max distance for bipolar
        return 1 - (dist / max_dist)
        
    elif metric == "jaccard":
        # Jaccard similarity for binary vectors
        intersection = np.sum((a == 1) & (b == 1))
        union = np.sum((a == 1) | (b == 1))
        
        if union == 0:
            return 0.0
            
        return intersection / union
        
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def noise_hypervector(
    hypervector: np.ndarray,
    noise_level: float,
    hypervector_type: str = "bipolar"
) -> np.ndarray:
    """
    Add noise to a hypervector.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Original hypervector
    noise_level : float
        Proportion of components to flip (0 to 1)
    hypervector_type : str
        Type of hypervector
        
    Returns
    -------
    np.ndarray
        Noisy hypervector
    """
    if not 0 <= noise_level <= 1:
        raise ValueError(f"Noise level must be in [0, 1], got {noise_level}")
        
    dimension = len(hypervector)
    n_flips = int(dimension * noise_level)
    
    if n_flips == 0:
        return hypervector.copy()
        
    # Select random positions to flip
    flip_positions = np.random.choice(dimension, n_flips, replace=False)
    noisy = hypervector.copy()
    
    if hypervector_type == "binary":
        noisy[flip_positions] = 1 - noisy[flip_positions]
        
    elif hypervector_type == "bipolar":
        noisy[flip_positions] = -noisy[flip_positions]
        
    elif hypervector_type == "ternary":
        # For ternary, randomly change to one of the other values
        for pos in flip_positions:
            current = noisy[pos]
            if current == 0:
                noisy[pos] = np.random.choice([-1, 1])
            elif current == 1:
                noisy[pos] = np.random.choice([-1, 0])
            else:  # current == -1
                noisy[pos] = np.random.choice([0, 1])
                
    else:
        raise ValueError(f"Noise not implemented for {hypervector_type}")
        
    return noisy


def thin_hypervector(
    hypervector: np.ndarray,
    sparsity: float
) -> np.ndarray:
    """
    Make a hypervector sparse by setting elements to zero.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Original hypervector
    sparsity : float
        Target sparsity (proportion of zeros)
        
    Returns
    -------
    np.ndarray
        Sparse hypervector
    """
    if not 0 <= sparsity <= 1:
        raise ValueError(f"Sparsity must be in [0, 1], got {sparsity}")
        
    dimension = len(hypervector)
    n_zeros = int(dimension * sparsity)
    
    # Select positions to zero out
    zero_positions = np.random.choice(dimension, n_zeros, replace=False)
    
    sparse = hypervector.copy()
    sparse[zero_positions] = 0
    
    return sparse


def segment_hypervector(
    hypervector: np.ndarray,
    n_segments: int
) -> List[np.ndarray]:
    """
    Segment a hypervector into multiple parts.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Hypervector to segment
    n_segments : int
        Number of segments
        
    Returns
    -------
    List[np.ndarray]
        List of segments
    """
    dimension = len(hypervector)
    
    if dimension % n_segments != 0:
        raise ValueError(
            f"Dimension {dimension} not divisible by {n_segments} segments"
        )
        
    segment_size = dimension // n_segments
    segments = []
    
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        segments.append(hypervector[start:end])
        
    return segments


def concatenate_hypervectors(
    hypervectors: List[np.ndarray]
) -> np.ndarray:
    """
    Concatenate multiple hypervectors.
    
    Parameters
    ----------
    hypervectors : List[np.ndarray]
        Hypervectors to concatenate
        
    Returns
    -------
    np.ndarray
        Concatenated hypervector
    """
    if not hypervectors:
        raise ValueError("Cannot concatenate empty list")
        
    return np.concatenate(hypervectors)


def power_hypervector(
    hypervector: np.ndarray,
    exponent: int,
    bind_operation: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Compute the power of a hypervector (self-binding).
    
    Parameters
    ----------
    hypervector : np.ndarray
        Base hypervector
    exponent : int
        Power to raise to
    bind_operation : Callable
        Binding function to use
        
    Returns
    -------
    np.ndarray
        Hypervector raised to the given power
        
    Notes
    -----
    For self-inverse binding operations and even exponents,
    this returns the identity element.
    """
    if exponent == 0:
        # Return identity (depends on binding operation)
        # For XOR/multiplication, identity is all ones
        return np.ones_like(hypervector)
        
    if exponent == 1:
        return hypervector.copy()
        
    result = hypervector.copy()
    for _ in range(exponent - 1):
        result = bind_operation(result, hypervector)
        
    return result


def normalize_hypervector(
    hypervector: np.ndarray,
    hypervector_type: str = "bipolar"
) -> np.ndarray:
    """
    Normalize a hypervector to its standard form.
    
    Parameters
    ----------
    hypervector : np.ndarray
        Hypervector to normalize
    hypervector_type : str
        Type of hypervector
        
    Returns
    -------
    np.ndarray
        Normalized hypervector
    """
    if hypervector_type == "binary":
        # Ensure binary values
        return np.clip(hypervector, 0, 1).astype(np.uint8)
        
    elif hypervector_type == "bipolar":
        # Ensure bipolar values
        normalized = np.sign(hypervector)
        normalized[normalized == 0] = 1
        return normalized.astype(np.int8)
        
    elif hypervector_type == "ternary":
        # Threshold to {-1, 0, 1}
        normalized = np.zeros_like(hypervector, dtype=np.int8)
        normalized[hypervector > 0.5] = 1
        normalized[hypervector < -0.5] = -1
        return normalized
        
    else:
        return hypervector


def protect_sequence(
    sequence: List[np.ndarray],
    position_vectors: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Protect a sequence of hypervectors with position encoding.
    
    Parameters
    ----------
    sequence : List[np.ndarray]
        Sequence of hypervectors
    position_vectors : List[np.ndarray], optional
        Position encoding vectors
        
    Returns
    -------
    np.ndarray
        Protected sequence representation
    """
    if not sequence:
        raise ValueError("Cannot protect empty sequence")
        
    dimension = sequence[0].shape[0]
    
    if position_vectors is None:
        # Generate random position vectors
        position_vectors = []
        for i in range(len(sequence)):
            pos_vec = 2 * np.random.randint(0, 2, size=dimension) - 1
            position_vectors.append(pos_vec)
            
    elif len(position_vectors) != len(sequence):
        raise ValueError("Number of position vectors must match sequence length")
        
    # Bind each item with its position and bundle
    protected_items = []
    for item, pos in zip(sequence, position_vectors):
        protected = item * pos  # Binding
        protected_items.append(protected)
        
    # Bundle all protected items
    return bundle_hypervectors(protected_items, hypervector_type="bipolar")