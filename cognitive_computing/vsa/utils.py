"""
Utility functions and analysis tools for Vector Symbolic Architectures.

This module provides helper functions for generating vectors, analyzing VSA
capacity, performance benchmarking, and cross-architecture conversions.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Type
from dataclasses import dataclass
import time
from collections import defaultdict
import warnings

from .vectors import (
    VSAVector, BinaryVector, BipolarVector, TernaryVector, 
    ComplexVector, IntegerVector
)
from .binding import (
    BindingOperation, XORBinding, MultiplicationBinding,
    ConvolutionBinding, MAPBinding, PermutationBinding
)
from .core import VSA, VSAConfig


@dataclass
class VSACapacityMetrics:
    """Metrics for VSA capacity analysis."""
    dimension: int
    vector_type: str
    binding_method: str
    max_reliable_bindings: int
    noise_tolerance: float
    similarity_threshold: float
    theoretical_capacity: float
    empirical_capacity: float
    
    
@dataclass 
class VSAPerformanceMetrics:
    """Performance metrics for VSA operations."""
    operation: str
    vector_type: str
    dimension: int
    num_operations: int
    total_time: float
    mean_time: float
    std_time: float
    operations_per_second: float


def generate_random_vector(
    dimension: int,
    vector_type: Type[VSAVector],
    rng: Optional[np.random.RandomState] = None,
    **kwargs
) -> VSAVector:
    """
    Generate a random VSA vector of specified type.
    
    Parameters
    ----------
    dimension : int
        Dimension of the vector
    vector_type : Type[VSAVector]
        Class of vector to generate
    rng : RandomState, optional
        Random number generator
    **kwargs
        Additional arguments for specific vector types:
        - sparsity: float (for TernaryVector)
        - modulus: int (for IntegerVector)
        
    Returns
    -------
    VSAVector
        Random vector of specified type
        
    Examples
    --------
    >>> binary_vec = generate_random_vector(1000, BinaryVector)
    >>> ternary_vec = generate_random_vector(1000, TernaryVector, sparsity=0.1)
    """
    if rng is None:
        rng = np.random.RandomState()
        
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")
        
    if vector_type == BinaryVector:
        data = rng.randint(0, 2, dimension)
        return BinaryVector(data)
        
    elif vector_type == BipolarVector:
        data = 2 * rng.randint(0, 2, dimension) - 1
        return BipolarVector(data)
        
    elif vector_type == TernaryVector:
        sparsity = kwargs.get('sparsity', 0.1)
        data = np.zeros(dimension)
        num_nonzero = int(dimension * sparsity)
        indices = rng.choice(dimension, num_nonzero, replace=False)
        values = 2 * rng.randint(0, 2, num_nonzero) - 1
        data[indices] = values
        return TernaryVector(data)
        
    elif vector_type == ComplexVector:
        phases = rng.uniform(0, 2 * np.pi, dimension)
        data = np.exp(1j * phases)
        return ComplexVector(data)
        
    elif vector_type == IntegerVector:
        modulus = kwargs.get('modulus', 256)
        data = rng.randint(0, modulus, dimension)
        return IntegerVector(data, modulus)
        
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")


def generate_orthogonal_vectors(
    num_vectors: int,
    dimension: int, 
    vector_type: Type[VSAVector],
    rng: Optional[np.random.RandomState] = None,
    max_similarity: float = 0.1
) -> List[VSAVector]:
    """
    Generate approximately orthogonal VSA vectors.
    
    Parameters
    ----------
    num_vectors : int
        Number of vectors to generate
    dimension : int
        Dimension of each vector
    vector_type : Type[VSAVector]
        Class of vectors to generate
    rng : RandomState, optional
        Random number generator
    max_similarity : float
        Maximum allowed similarity between vectors
        
    Returns
    -------
    List[VSAVector]
        List of approximately orthogonal vectors
        
    Notes
    -----
    For high dimensions, random vectors are approximately orthogonal.
    This function generates random vectors and checks similarity.
    """
    if num_vectors * num_vectors > dimension:
        warnings.warn(
            f"Generating {num_vectors} orthogonal vectors in {dimension}D "
            "may not achieve desired orthogonality"
        )
        
    vectors = []
    attempts = 0
    max_attempts = num_vectors * 100
    
    while len(vectors) < num_vectors and attempts < max_attempts:
        attempts += 1
        candidate = generate_random_vector(dimension, vector_type, rng)
        
        # Check similarity with existing vectors
        is_orthogonal = True
        for existing in vectors:
            sim = candidate.similarity(existing)
            if abs(sim) > max_similarity:
                is_orthogonal = False
                break
                
        if is_orthogonal:
            vectors.append(candidate)
            
    if len(vectors) < num_vectors:
        raise RuntimeError(
            f"Could only generate {len(vectors)} orthogonal vectors "
            f"out of {num_vectors} requested"
        )
        
    return vectors


def analyze_binding_capacity(
    vsa: VSA,
    num_items: int = 100,
    num_trials: int = 10,
    similarity_threshold: float = 0.7,
    rng: Optional[np.random.RandomState] = None
) -> VSACapacityMetrics:
    """
    Analyze the binding capacity of a VSA system.
    
    Parameters
    ----------
    vsa : VSA
        VSA instance to analyze
    num_items : int
        Number of items to test binding
    num_trials : int
        Number of trials to average
    similarity_threshold : float
        Threshold for successful retrieval
    rng : RandomState, optional
        Random number generator
        
    Returns
    -------
    VSACapacityMetrics
        Capacity analysis results
    """
    if rng is None:
        rng = np.random.RandomState()
        
    dimension = vsa.config.dimension
    vector_type = vsa._vector_class
    
    max_reliable = 0
    noise_tolerances = []
    
    for trial in range(num_trials):
        # Generate random item vectors
        items = [
            generate_random_vector(dimension, vector_type, rng)
            for _ in range(num_items)
        ]
        
        # Test increasing numbers of bindings
        for n in range(1, num_items):
            # Create superposition of n bindings
            bound_pairs = []
            superposition = None
            
            for i in range(n):
                key = items[i]
                value = items[num_items - i - 1]
                bound = vsa.bind(key.data, value.data)
                bound_pairs.append((key, value, bound))
                
                if superposition is None:
                    superposition = bound
                else:
                    superposition = vsa.bundle([superposition, bound])
                    
            # Test retrieval
            successful_retrievals = 0
            for key, value, _ in bound_pairs:
                retrieved = vsa.unbind(superposition, key.data)
                similarity = np.corrcoef(retrieved.flatten(), value.data.flatten())[0, 1]
                if similarity >= similarity_threshold:
                    successful_retrievals += 1
                    
            # Check if all retrievals successful
            if successful_retrievals == n:
                max_reliable = max(max_reliable, n)
            else:
                break
                
        # Test noise tolerance
        if max_reliable > 0:
            # Add noise to superposition
            noise_levels = np.linspace(0, 0.5, 10)
            for noise_level in noise_levels:
                noisy = superposition + noise_level * rng.randn(*superposition.shape)
                
                # Test retrieval with noise
                key, value, _ = bound_pairs[0]
                retrieved = vsa.unbind(noisy, key.data)
                similarity = np.corrcoef(retrieved.flatten(), value.data.flatten())[0, 1]
                
                if similarity >= similarity_threshold:
                    noise_tolerances.append(noise_level)
                    
    # Calculate metrics
    avg_noise_tolerance = np.mean(noise_tolerances) if noise_tolerances else 0.0
    
    # Theoretical capacity estimates
    if vsa.config.binding_method == "xor":
        theoretical_capacity = dimension / (2 * np.log(2))
    elif vsa.config.binding_method == "multiplication":
        theoretical_capacity = dimension / 4
    elif vsa.config.binding_method == "convolution":
        theoretical_capacity = np.sqrt(dimension) / 2
    else:
        theoretical_capacity = dimension / 8  # Conservative estimate
        
    empirical_capacity = max_reliable / num_trials
    
    return VSACapacityMetrics(
        dimension=dimension,
        vector_type=vsa.config.vector_type,
        binding_method=vsa.config.binding_method,
        max_reliable_bindings=max_reliable,
        noise_tolerance=avg_noise_tolerance,
        similarity_threshold=similarity_threshold,
        theoretical_capacity=theoretical_capacity,
        empirical_capacity=empirical_capacity
    )


def benchmark_vsa_operations(
    vsa: VSA,
    num_operations: int = 1000,
    operations: Optional[List[str]] = None,
    rng: Optional[np.random.RandomState] = None
) -> Dict[str, VSAPerformanceMetrics]:
    """
    Benchmark VSA operations performance.
    
    Parameters
    ----------
    vsa : VSA
        VSA instance to benchmark
    num_operations : int
        Number of operations to perform
    operations : List[str], optional
        Operations to benchmark (default: all)
    rng : RandomState, optional
        Random number generator
        
    Returns
    -------
    Dict[str, VSAPerformanceMetrics]
        Performance metrics for each operation
    """
    if rng is None:
        rng = np.random.RandomState()
        
    if operations is None:
        operations = ["bind", "unbind", "bundle", "permute"]
        
    dimension = vsa.config.dimension
    vector_type = vsa._vector_class
    
    # Pre-generate test vectors
    vectors = [
        generate_random_vector(dimension, vector_type, rng)
        for _ in range(num_operations * 2)
    ]
    
    results = {}
    
    for op in operations:
        times = []
        
        if op == "bind":
            for i in range(num_operations):
                start = time.perf_counter()
                _ = vsa.bind(vectors[i].data, vectors[i + num_operations].data)
                times.append(time.perf_counter() - start)
                
        elif op == "unbind":
            # Pre-bind vectors
            bound = [
                vsa.bind(vectors[i].data, vectors[i + num_operations].data)
                for i in range(num_operations)
            ]
            
            for i in range(num_operations):
                start = time.perf_counter()
                _ = vsa.unbind(bound[i], vectors[i].data)
                times.append(time.perf_counter() - start)
                
        elif op == "bundle":
            for i in range(0, num_operations, 2):
                start = time.perf_counter()
                _ = vsa.bundle([vectors[i].data, vectors[i + 1].data])
                times.append(time.perf_counter() - start)
                
        elif op == "permute":
            for i in range(num_operations):
                start = time.perf_counter()
                _ = vsa.permute(vectors[i].data, shift=1)
                times.append(time.perf_counter() - start)
                
        times = np.array(times)
        total_time = np.sum(times)
        
        results[op] = VSAPerformanceMetrics(
            operation=op,
            vector_type=vsa.config.vector_type,
            dimension=dimension,
            num_operations=len(times),
            total_time=total_time,
            mean_time=np.mean(times),
            std_time=np.std(times),
            operations_per_second=len(times) / total_time
        )
        
    return results


def compare_binding_methods(
    dimension: int,
    vector_type: str = "bipolar",
    num_items: int = 10,
    rng: Optional[np.random.RandomState] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different binding methods for a given vector type.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    vector_type : str
        Type of vectors to use
    num_items : int
        Number of items to bind
    rng : RandomState, optional
        Random number generator
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Comparison metrics for each binding method
    """
    if rng is None:
        rng = np.random.RandomState()
        
    # Determine compatible binding methods
    if vector_type == "binary":
        methods = ["xor", "map"]
    elif vector_type == "bipolar":
        methods = ["multiplication", "convolution", "map"]
    elif vector_type == "complex":
        methods = ["multiplication", "convolution"]
    else:
        methods = ["convolution"]
        
    results = {}
    
    for method in methods:
        # Create VSA with this binding method
        config = VSAConfig(
            dimension=dimension,
            vector_type=vector_type,
            binding_method=method
        )
        vsa = VSA(config)
        
        # Generate test vectors
        vector_class = vsa._vector_class
        items = [
            generate_random_vector(dimension, vector_class, rng)
            for _ in range(num_items * 2)
        ]
        
        # Test binding/unbinding accuracy
        similarities = []
        for i in range(num_items):
            key = items[i]
            value = items[i + num_items]
            
            bound = vsa.bind(key.data, value.data)
            retrieved = vsa.unbind(bound, key.data)
            
            similarity = np.corrcoef(retrieved.flatten(), value.data.flatten())[0, 1]
            similarities.append(similarity)
            
        # Test associativity
        a, b, c = items[0], items[1], items[2]
        ab_c = vsa.bind(vsa.bind(a.data, b.data), c.data)
        a_bc = vsa.bind(a.data, vsa.bind(b.data, c.data))
        associativity = np.corrcoef(ab_c.flatten(), a_bc.flatten())[0, 1]
        
        # Test commutativity
        ab = vsa.bind(a.data, b.data)
        ba = vsa.bind(b.data, a.data)
        commutativity = np.corrcoef(ab.flatten(), ba.flatten())[0, 1]
        
        # Benchmark performance
        perf = benchmark_vsa_operations(vsa, num_operations=100, operations=["bind", "unbind"], rng=rng)
        
        results[method] = {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "associativity": associativity,
            "commutativity": commutativity,
            "bind_ops_per_second": perf["bind"].operations_per_second,
            "unbind_ops_per_second": perf["unbind"].operations_per_second
        }
        
    return results


def convert_vector(
    vector: VSAVector,
    target_type: Type[VSAVector],
    **kwargs
) -> VSAVector:
    """
    Convert a VSA vector from one type to another.
    
    Parameters
    ----------
    vector : VSAVector
        Vector to convert
    target_type : Type[VSAVector]
        Target vector type
    **kwargs
        Additional arguments for conversion:
        - threshold: float (for binary conversion)
        - sparsity: float (for ternary conversion)
        - modulus: int (for integer conversion)
        
    Returns
    -------
    VSAVector
        Converted vector
        
    Examples
    --------
    >>> binary = BinaryVector(np.array([0, 1, 0, 1]))
    >>> bipolar = convert_vector(binary, BipolarVector)
    >>> ternary = convert_vector(bipolar, TernaryVector, sparsity=0.5)
    """
    if type(vector) == target_type:
        return vector
        
    # Get raw data
    data = vector.data
    
    # Convert to target type
    if target_type == BinaryVector:
        threshold = kwargs.get('threshold', 0.5)
        if isinstance(vector, BipolarVector):
            new_data = (data > 0).astype(int)
        elif isinstance(vector, ComplexVector):
            new_data = (np.real(data) > 0).astype(int)
        else:
            # Normalize to [0, 1] range
            min_val, max_val = np.min(data), np.max(data)
            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
                new_data = (normalized > threshold).astype(int)
            else:
                new_data = np.zeros_like(data, dtype=int)
        return BinaryVector(new_data)
        
    elif target_type == BipolarVector:
        if isinstance(vector, BinaryVector):
            new_data = 2 * data - 1
        elif isinstance(vector, ComplexVector):
            new_data = np.sign(np.real(data))
            new_data[new_data == 0] = 1
        else:
            new_data = np.sign(data)
            new_data[new_data == 0] = 1
        return BipolarVector(new_data.astype(int))
        
    elif target_type == TernaryVector:
        sparsity = kwargs.get('sparsity', 0.1)
        if isinstance(vector, (BinaryVector, BipolarVector)):
            # Convert to bipolar first
            if isinstance(vector, BinaryVector):
                bipolar_data = 2 * data - 1
            else:
                bipolar_data = data
                
            # Sparsify by keeping only top values
            num_keep = int(len(data) * sparsity)
            indices = np.argsort(np.abs(bipolar_data))[-num_keep:]
            new_data = np.zeros_like(bipolar_data)
            new_data[indices] = bipolar_data[indices]
        else:
            # General sparsification
            num_keep = int(len(data) * sparsity)
            indices = np.argsort(np.abs(data))[-num_keep:]
            new_data = np.zeros_like(data, dtype=float)
            new_data[indices] = np.sign(data[indices])
        return TernaryVector(new_data)
        
    elif target_type == ComplexVector:
        if isinstance(vector, BinaryVector):
            phases = data * np.pi  # 0 -> 0, 1 -> pi
        elif isinstance(vector, BipolarVector):
            phases = (1 - data) * np.pi / 2  # 1 -> 0, -1 -> pi
        else:
            # Map to phases
            normalized = data / (np.abs(data).max() + 1e-10)
            phases = np.arccos(normalized)
        new_data = np.exp(1j * phases)
        return ComplexVector(new_data)
        
    elif target_type == IntegerVector:
        modulus = kwargs.get('modulus', 256)
        if isinstance(vector, BinaryVector):
            new_data = data * (modulus - 1)
        else:
            # Normalize to [0, modulus) range
            min_val, max_val = np.min(data), np.max(data)
            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
                new_data = (normalized * (modulus - 1)).astype(int)
            else:
                new_data = np.zeros_like(data, dtype=int)
        return IntegerVector(new_data, modulus)
        
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def analyze_vector_distribution(
    vectors: List[VSAVector],
    num_bins: int = 50
) -> Dict[str, np.ndarray]:
    """
    Analyze the distribution of vector components.
    
    Parameters
    ----------
    vectors : List[VSAVector]
        Vectors to analyze
    num_bins : int
        Number of histogram bins
        
    Returns
    -------
    Dict[str, np.ndarray]
        Distribution statistics
    """
    if not vectors:
        raise ValueError("No vectors provided")
        
    # Stack all vectors
    data = np.stack([v.data for v in vectors])
    
    # Calculate statistics
    stats = {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "histogram": np.histogram(data.flatten(), bins=num_bins)[0],
        "bin_edges": np.histogram(data.flatten(), bins=num_bins)[1]
    }
    
    # Type-specific statistics
    if isinstance(vectors[0], ComplexVector):
        stats["mean_magnitude"] = np.mean(np.abs(data))
        stats["mean_phase"] = np.mean(np.angle(data))
        stats["phase_histogram"] = np.histogram(np.angle(data).flatten(), bins=num_bins)[0]
        
    elif isinstance(vectors[0], TernaryVector):
        stats["sparsity"] = np.mean(data == 0)
        stats["positive_fraction"] = np.mean(data > 0)
        stats["negative_fraction"] = np.mean(data < 0)
        
    return stats


def estimate_memory_requirements(
    num_vectors: int,
    dimension: int,
    vector_type: str,
    include_operations: bool = True
) -> Dict[str, float]:
    """
    Estimate memory requirements for VSA operations.
    
    Parameters
    ----------
    num_vectors : int
        Number of vectors to store
    dimension : int
        Dimension of each vector
    vector_type : str
        Type of vectors
    include_operations : bool
        Include temporary storage for operations
        
    Returns
    -------
    Dict[str, float]
        Memory estimates in MB
    """
    # Bytes per element
    bytes_per_element = {
        "binary": 1,  # uint8
        "bipolar": 1,  # int8
        "ternary": 1,  # int8 with special encoding
        "complex": 16,  # complex128
        "integer": 2   # int16
    }
    
    if vector_type not in bytes_per_element:
        raise ValueError(f"Unknown vector type: {vector_type}")
        
    # Basic storage
    basic_storage = num_vectors * dimension * bytes_per_element[vector_type]
    
    # Operation overhead (temporary vectors)
    if include_operations:
        # Assume 3x overhead for operations
        operation_overhead = 3 * dimension * bytes_per_element[vector_type]
    else:
        operation_overhead = 0
        
    # Convert to MB
    total_mb = (basic_storage + operation_overhead) / (1024 * 1024)
    
    return {
        "basic_storage_mb": basic_storage / (1024 * 1024),
        "operation_overhead_mb": operation_overhead / (1024 * 1024),
        "total_mb": total_mb,
        "bytes_per_vector": dimension * bytes_per_element[vector_type]
    }


def find_optimal_dimension(
    num_items: int,
    desired_capacity: float = 0.9,
    vector_type: str = "bipolar",
    binding_method: str = "multiplication"
) -> int:
    """
    Find optimal dimension for desired capacity.
    
    Parameters
    ----------
    num_items : int
        Number of items to store
    desired_capacity : float
        Desired retrieval accuracy
    vector_type : str
        Type of vectors
    binding_method : str
        Binding method to use
        
    Returns
    -------
    int
        Recommended dimension
    """
    # Empirical scaling factors
    scaling_factors = {
        ("binary", "xor"): 2.5,
        ("bipolar", "multiplication"): 2.0,
        ("bipolar", "convolution"): 3.0,
        ("complex", "multiplication"): 1.8,
        ("complex", "convolution"): 2.5,
    }
    
    key = (vector_type, binding_method)
    if key in scaling_factors:
        factor = scaling_factors[key]
    else:
        factor = 3.0  # Conservative default
        
    # Add safety margin for desired capacity
    capacity_factor = 1.0 / desired_capacity
    
    # Calculate dimension
    dimension = int(factor * num_items * capacity_factor * np.log(num_items))
    
    # Round to nearest power of 2 for efficiency
    dimension = 2 ** int(np.ceil(np.log2(dimension)))
    
    return dimension