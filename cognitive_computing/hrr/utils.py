"""
Utility functions for HRR.

This module provides helper functions for vector generation, analysis,
and HRR system evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import time

from .core import HRR, HRRConfig
from .operations import CircularConvolution, VectorOperations
from .cleanup import CleanupMemory, CleanupMemoryConfig

logger = logging.getLogger(__name__)


# Vector Generation Functions

def generate_random_vector(dimension: int, 
                         method: str = "gaussian",
                         seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random vector suitable for HRR.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    method : str
        Generation method: "gaussian", "binary", "ternary", "sparse"
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Generated vector
        
    Examples
    --------
    >>> v = generate_random_vector(1024, method="gaussian")
    >>> v_binary = generate_random_vector(1024, method="binary")
    """
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")
    
    rng = np.random.RandomState(seed)
    
    if method == "gaussian":
        # Standard Gaussian
        vector = rng.randn(dimension)
        
    elif method == "binary":
        # Binary {-1, +1}
        vector = 2 * rng.randint(0, 2, dimension) - 1
        
    elif method == "ternary":
        # Ternary {-1, 0, +1}
        vector = rng.randint(-1, 2, dimension)
        
    elif method == "sparse":
        # Sparse vector with ~10% non-zero
        vector = np.zeros(dimension)
        n_nonzero = max(1, dimension // 10)
        indices = rng.choice(dimension, n_nonzero, replace=False)
        vector[indices] = rng.randn(n_nonzero)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
        
    return vector


def generate_unitary_vector(dimension: int, 
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a unitary vector (self-inverse under correlation).
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Unitary vector
    """
    rng = np.random.RandomState(seed)
    
    # Start with random phases
    phases = rng.uniform(0, 2 * np.pi, dimension)
    
    # Create vector with unit magnitude in frequency domain
    fft = np.exp(1j * phases)
    
    # Ensure conjugate symmetry for real result
    for i in range(1, dimension // 2):
        fft[dimension - i] = np.conj(fft[i])
    
    # DC and Nyquist must be real
    fft[0] = np.abs(fft[0])
    if dimension % 2 == 0:
        fft[dimension // 2] = np.abs(fft[dimension // 2])
    
    # Transform to time domain
    vector = np.real(np.fft.ifft(fft))
    
    # Normalize
    return vector / np.linalg.norm(vector)


def generate_orthogonal_set(dimension: int, n_vectors: int,
                          method: str = "gram_schmidt",
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a set of orthogonal vectors.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    n_vectors : int
        Number of vectors to generate
    method : str
        Method: "gram_schmidt" or "hadamard"
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Array of shape (n_vectors, dimension) with orthogonal vectors
    """
    if n_vectors > dimension:
        raise ValueError(f"Cannot generate {n_vectors} orthogonal vectors "
                        f"in {dimension} dimensions")
    
    rng = np.random.RandomState(seed)
    
    if method == "gram_schmidt":
        # Start with random vectors
        vectors = rng.randn(n_vectors, dimension)
        
        # Gram-Schmidt orthogonalization
        for i in range(n_vectors):
            # Orthogonalize against previous vectors
            for j in range(i):
                vectors[i] -= (np.dot(vectors[i], vectors[j]) * vectors[j])
            
            # Normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
                
    elif method == "hadamard":
        # Use Hadamard matrix if dimension is power of 2
        if dimension & (dimension - 1) != 0:
            raise ValueError(f"Hadamard method requires dimension to be "
                           f"power of 2, got {dimension}")
        
        # Generate Hadamard matrix
        H = np.array([[1]])
        while H.shape[0] < dimension:
            H = np.block([[H, H], [H, -H]])
        
        # Select random rows
        indices = rng.permutation(dimension)[:n_vectors]
        vectors = H[indices].astype(float) / np.sqrt(dimension)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return vectors


# Analysis Functions

@dataclass
class HRRAnalysisResult:
    """Results from HRR analysis."""
    binding_accuracy: float
    unbinding_accuracy: float
    bundling_capacity: int
    crosstalk_level: float
    orthogonality: float
    stats: Dict[str, Any]


def analyze_binding_capacity(hrr: HRR, n_pairs: int,
                           n_trials: int = 10,
                           noise_level: float = 0.0) -> Dict[str, float]:
    """
    Analyze the binding capacity of an HRR system.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to analyze
    n_pairs : int
        Number of role-filler pairs to test
    n_trials : int
        Number of trials to average
    noise_level : float
        Noise level to add to queries (0-1)
        
    Returns
    -------
    Dict[str, float]
        Analysis results including accuracy and capacity metrics
    """
    if n_pairs == 0:
        return {
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "mean_similarity": 0.0,
            "min_similarity": 0.0,
            "capacity_estimate": 0,
        }
    
    results = []
    
    for trial in range(n_trials):
        # Generate roles and fillers
        roles = [hrr.generate_vector() for _ in range(n_pairs)]
        fillers = [hrr.generate_vector() for _ in range(n_pairs)]
        
        # Create bundled structure
        pairs = []
        for role, filler in zip(roles, fillers):
            pairs.append(hrr.bind(role, filler))
        
        structure = hrr.bundle(pairs)
        
        # Test retrieval
        correct = 0
        similarities = []
        
        for i, (role, filler) in enumerate(zip(roles, fillers)):
            # Add noise to role if specified
            if noise_level > 0:
                noise = np.random.randn(hrr.config.dimension) * noise_level
                noisy_role = role + noise
                noisy_role = noisy_role / np.linalg.norm(noisy_role)
            else:
                noisy_role = role
            
            # Retrieve filler
            retrieved = hrr.unbind(structure, noisy_role)
            
            # Check similarity with correct filler
            sim = hrr.similarity(retrieved, filler)
            similarities.append(sim)
            
            # Check if correct filler has highest similarity
            max_sim = sim
            correct_retrieval = True
            
            for j, other_filler in enumerate(fillers):
                if i != j:
                    other_sim = hrr.similarity(retrieved, other_filler)
                    if other_sim > max_sim:
                        correct_retrieval = False
                        break
            
            if correct_retrieval:
                correct += 1
        
        accuracy = correct / n_pairs
        results.append({
            "accuracy": accuracy,
            "mean_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
        })
    
    # Aggregate results
    return {
        "mean_accuracy": np.mean([r["accuracy"] for r in results]),
        "std_accuracy": np.std([r["accuracy"] for r in results]),
        "mean_similarity": np.mean([r["mean_similarity"] for r in results]),
        "min_similarity": np.mean([r["min_similarity"] for r in results]),
        "capacity_estimate": n_pairs if np.mean([r["accuracy"] for r in results]) > 0.9 else -1,
    }


def measure_crosstalk(hrr: HRR, vectors: List[np.ndarray]) -> float:
    """
    Measure crosstalk between vectors in superposition.
    
    Parameters
    ----------
    hrr : HRR
        HRR system
    vectors : List[np.ndarray]
        Vectors to test
        
    Returns
    -------
    float
        Average crosstalk level (0 = no crosstalk, 1 = complete interference)
    """
    if len(vectors) < 2:
        return 0.0
    
    # Bundle all vectors
    bundle = hrr.bundle(vectors)
    
    # Measure how much each vector interferes with others
    crosstalks = []
    
    for i, vec in enumerate(vectors):
        # Get similarity with bundle
        sim_bundle = hrr.similarity(vec, bundle)
        
        # Get similarity with individual vector
        sim_self = hrr.similarity(vec, vec)  # Should be 1
        
        # Estimate contribution from other vectors
        expected_self_contribution = 1.0 / len(vectors)
        crosstalk = max(0, sim_bundle - expected_self_contribution)
        crosstalks.append(crosstalk)
    
    return np.mean(crosstalks)


def measure_associative_capacity(hrr: HRR, n_items: int,
                            item_dimension: Optional[int] = None) -> Dict[str, Any]:
    """
    Test the associative memory capacity of HRR.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to test
    n_items : int
        Number of associations to store
    item_dimension : int, optional
        Dimension of item vectors (if different from HRR dimension)
        
    Returns
    -------
    Dict[str, Any]
        Test results including capacity and performance metrics
    """
    if item_dimension is None:
        item_dimension = hrr.config.dimension
    
    # Generate random associations using HRR's own method to handle storage type
    keys = [hrr.generate_vector() for _ in range(n_items)]
    values = [hrr.generate_vector() for _ in range(n_items)]
    
    # Store associations
    start_time = time.time()
    for key, value in zip(keys, values):
        hrr.store(key, value)
    store_time = time.time() - start_time
    
    # Test recall
    correct_recalls = 0
    similarities = []
    recall_times = []
    
    for key, value in zip(keys, values):
        start_time = time.time()
        recalled = hrr.recall(key)
        recall_time = time.time() - start_time
        recall_times.append(recall_time)
        
        if recalled is not None:
            sim = hrr.similarity(recalled, value)
            similarities.append(sim)
            
            # Check if this is the best match
            best_match = True
            for other_value in values:
                if other_value is not value:
                    if hrr.similarity(recalled, other_value) > sim:
                        best_match = False
                        break
            
            if best_match:
                correct_recalls += 1
    
    return {
        "n_items": n_items,
        "accuracy": correct_recalls / n_items if n_items > 0 else 0,
        "mean_similarity": np.mean(similarities) if similarities else 0,
        "std_similarity": np.std(similarities) if similarities else 0,
        "total_store_time": store_time,
        "mean_recall_time": np.mean(recall_times) if recall_times else 0,
        "items_per_second_store": n_items / store_time if store_time > 0 else 0,
        "items_per_second_recall": 1 / np.mean(recall_times) if recall_times else 0,
    }


# Conversion Utilities

def to_complex(vector: np.ndarray) -> np.ndarray:
    """
    Convert real vector to complex representation.
    
    Parameters
    ----------
    vector : np.ndarray
        Real-valued vector
        
    Returns
    -------
    np.ndarray
        Complex vector (half the dimension)
    """
    if np.iscomplexobj(vector):
        return vector
    
    # Use FFT to get complex representation
    fft = np.fft.fft(vector)
    
    # Take first half (positive frequencies)
    # This preserves all information due to conjugate symmetry
    n = len(vector)
    return fft[:n // 2 + 1]


def from_complex(vector: np.ndarray, dimension: int) -> np.ndarray:
    """
    Convert complex vector back to real representation.
    
    Parameters
    ----------
    vector : np.ndarray
        Complex vector
    dimension : int
        Target dimension for real vector
        
    Returns
    -------
    np.ndarray
        Real-valued vector
    """
    if not np.iscomplexobj(vector):
        return vector
    
    # Reconstruct full FFT with conjugate symmetry
    n = dimension
    fft = np.zeros(n, dtype=complex)
    
    # Copy positive frequencies
    fft[:len(vector)] = vector
    
    # Create negative frequencies (conjugate symmetry)
    for i in range(1, n // 2):
        if i < len(vector):
            fft[n - i] = np.conj(vector[i])
    
    # Transform back to real
    return np.real(np.fft.ifft(fft))


# Performance Testing

@dataclass 
class HRRPerformanceResult:
    """Results from HRR performance testing."""
    dimension: int
    n_operations: int
    bind_time_mean: float
    bind_time_std: float
    unbind_time_mean: float
    unbind_time_std: float
    bundle_time_mean: float
    bundle_time_std: float
    operations_per_second: float


def benchmark_hrr_performance(dimension: int = 1024,
                             n_operations: int = 1000,
                             storage_method: str = "real") -> HRRPerformanceResult:
    """
    Benchmark HRR operations performance.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    n_operations : int
        Number of operations to test
    storage_method : str
        Storage method: "real" or "complex"
        
    Returns
    -------
    HRRPerformanceResult
        Performance metrics
    """
    # Create HRR system
    config = HRRConfig(
        dimension=dimension,
        storage_method=storage_method
    )
    hrr = HRR(config)
    
    # Generate test vectors
    vectors_a = [hrr.generate_vector() for _ in range(n_operations)]
    vectors_b = [hrr.generate_vector() for _ in range(n_operations)]
    
    # Test binding
    bind_times = []
    for a, b in zip(vectors_a, vectors_b):
        start = time.time()
        _ = hrr.bind(a, b)
        bind_times.append(time.time() - start)
    
    # Test unbinding
    unbind_times = []
    bindings = [hrr.bind(a, b) for a, b in zip(vectors_a, vectors_b)]
    
    for binding, a in zip(bindings, vectors_a):
        start = time.time()
        _ = hrr.unbind(binding, a)
        unbind_times.append(time.time() - start)
    
    # Test bundling
    bundle_times = []
    chunk_size = 10
    
    for i in range(0, n_operations - chunk_size, chunk_size):
        chunk = vectors_a[i:i + chunk_size]
        start = time.time()
        _ = hrr.bundle(chunk)
        bundle_times.append(time.time() - start)
    
    # Calculate statistics
    total_time = (sum(bind_times) + sum(unbind_times) + sum(bundle_times))
    total_ops = len(bind_times) + len(unbind_times) + len(bundle_times)
    
    return HRRPerformanceResult(
        dimension=dimension,
        n_operations=n_operations,
        bind_time_mean=np.mean(bind_times),
        bind_time_std=np.std(bind_times),
        unbind_time_mean=np.mean(unbind_times),
        unbind_time_std=np.std(unbind_times),
        bundle_time_mean=np.mean(bundle_times),
        bundle_time_std=np.std(bundle_times),
        operations_per_second=total_ops / total_time if total_time > 0 else 0
    )


# Utility Functions

def create_cleanup_memory(hrr: HRR, items: Dict[str, np.ndarray],
                        threshold: float = 0.3) -> CleanupMemory:
    """
    Create a cleanup memory from a dictionary of items.
    
    Parameters
    ----------
    hrr : HRR
        HRR system
    items : Dict[str, np.ndarray]
        Dictionary mapping names to vectors
    threshold : float
        Similarity threshold for cleanup
        
    Returns
    -------
    CleanupMemory
        Initialized cleanup memory
    """
    config = CleanupMemoryConfig(threshold=threshold)
    cleanup = CleanupMemory(config, hrr.config.dimension)
    
    for name, vector in items.items():
        cleanup.add_item(name, vector)
    
    return cleanup


def compare_storage_methods(dimension: int = 1024,
                          n_items: int = 100) -> Dict[str, Any]:
    """
    Compare real vs complex storage methods.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    n_items : int
        Number of items to test
        
    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    results = {}
    
    for method in ["real", "complex"]:
        # Adjust dimension for complex
        dim = dimension if method == "real" else dimension + (dimension % 2)
        
        # Create HRR
        hrr = HRR(HRRConfig(dimension=dim, storage_method=method))
        
        # Test associative capacity
        capacity_results = measure_associative_capacity(hrr, n_items)
        
        # Test performance
        perf_results = benchmark_hrr_performance(dim, 100, method)
        
        results[method] = {
            "accuracy": capacity_results["accuracy"],
            "mean_similarity": capacity_results["mean_similarity"],
            "bind_time": perf_results.bind_time_mean,
            "unbind_time": perf_results.unbind_time_mean,
            "ops_per_second": perf_results.operations_per_second,
        }
    
    return results