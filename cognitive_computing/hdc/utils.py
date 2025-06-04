"""
Utility functions for hyperdimensional computing.

This module provides various utility functions for analyzing, benchmarking,
and working with hyperdimensional computing systems.
"""

import time
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import logging
from dataclasses import dataclass, field
import json

from cognitive_computing.hdc.core import HDC, HDCConfig
from cognitive_computing.hdc.hypervectors import generate_orthogonal_hypervectors
from cognitive_computing.hdc.operations import (
    bind_hypervectors,
    bundle_hypervectors,
    similarity,
    noise_hypervector,
)
from cognitive_computing.hdc.item_memory import ItemMemory
from cognitive_computing.hdc.classifiers import HDClassifier

logger = logging.getLogger(__name__)


@dataclass
class HDCPerformanceMetrics:
    """Performance metrics for HDC operations."""
    dimension: int
    hypervector_type: str
    capacity_results: Dict[str, float] = field(default_factory=dict)
    noise_tolerance: Dict[float, float] = field(default_factory=dict)
    operation_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, int] = field(default_factory=dict)
    similarity_distribution: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "dimension": self.dimension,
            "hypervector_type": self.hypervector_type,
            "capacity_results": self.capacity_results,
            "noise_tolerance": self.noise_tolerance,
            "operation_times": self.operation_times,
            "memory_usage": self.memory_usage,
            "similarity_distribution": self.similarity_distribution,
        }
        
    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'HDCPerformanceMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to floats for noise_tolerance
        if "noise_tolerance" in data:
            data["noise_tolerance"] = {
                float(k): v for k, v in data["noise_tolerance"].items()
            }
            
        return cls(**data)


def measure_capacity(
    hdc: HDC,
    num_items: int = 1000,
    noise_levels: List[float] = None,
    similarity_threshold: float = 0.1
) -> HDCPerformanceMetrics:
    """
    Measure the capacity and noise tolerance of an HDC system.
    
    Parameters
    ----------
    hdc : HDC
        HDC instance to test
    num_items : int
        Number of items to test with
    noise_levels : List[float], optional
        Noise levels to test (fraction of bits to flip)
    similarity_threshold : float
        Threshold for considering vectors as interfering
        
    Returns
    -------
    HDCPerformanceMetrics
        Performance metrics
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
    metrics = HDCPerformanceMetrics(
        dimension=hdc.dimension,
        hypervector_type=hdc.hypervector_type.value
    )
    
    # Generate random hypervectors
    logger.info(f"Generating {num_items} random hypervectors...")
    vectors = []
    for i in range(num_items):
        hv = hdc.generate_hypervector()
        vectors.append(hv)
        
    # Measure pairwise similarities
    logger.info("Measuring pairwise similarities...")
    similarities = []
    for i in range(min(100, num_items)):
        for j in range(i + 1, min(100, num_items)):
            sim = similarity(vectors[i], vectors[j])
            similarities.append(sim)
            
    metrics.similarity_distribution["random_pairs"] = similarities
    
    # Calculate interference statistics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    max_sim = np.max(similarities)
    
    metrics.capacity_results["mean_similarity"] = mean_sim
    metrics.capacity_results["std_similarity"] = std_sim
    metrics.capacity_results["max_similarity"] = max_sim
    metrics.capacity_results["interference_rate"] = np.mean(np.array(similarities) > similarity_threshold)
    
    # Estimate capacity based on similarity distribution
    # Capacity estimation: number of vectors that can be stored with < threshold interference
    expected_interference = mean_sim + 3 * std_sim  # 3-sigma bound
    if expected_interference < similarity_threshold:
        estimated_capacity = int(2 ** (hdc.dimension * (1 - expected_interference)))
    else:
        estimated_capacity = int(hdc.dimension / (10 * expected_interference))
        
    metrics.capacity_results["estimated_capacity"] = estimated_capacity
    
    # Test noise tolerance
    logger.info("Testing noise tolerance...")
    item_memory = ItemMemory(
        dimension=hdc.dimension,
        similarity_metric="cosine",
        max_items=100
    )
    
    # Store subset of vectors
    test_vectors = vectors[:100]
    for i, vec in enumerate(test_vectors):
        item_memory.add(f"item_{i}", vec)
        
    # Test recovery with noise
    for noise_level in noise_levels:
        correct_recoveries = 0
        
        for i in range(min(50, len(test_vectors))):
            # Add noise
            noisy_vec = noise_hypervector(
                test_vectors[i],
                noise_level,
                hdc.hypervector_type.value
            )
            
            # Try to recover
            recovered, label = item_memory.cleanup(noisy_vec)
            if label == f"item_{i}":
                correct_recoveries += 1
                
        recovery_rate = correct_recoveries / min(50, len(test_vectors))
        metrics.noise_tolerance[noise_level] = recovery_rate
        
    return metrics


def benchmark_operations(
    hdc: HDC,
    num_trials: int = 100
) -> Dict[str, float]:
    """
    Benchmark HDC operations.
    
    Parameters
    ----------
    hdc : HDC
        HDC instance to benchmark
    num_trials : int
        Number of trials for each operation
        
    Returns
    -------
    Dict[str, float]
        Operation times in milliseconds
    """
    times = {}
    
    # Generate test vectors
    vectors = [hdc.generate_hypervector() for _ in range(10)]
    
    # Benchmark hypervector generation
    start = time.perf_counter()
    for _ in range(num_trials):
        hdc.generate_hypervector()
    end = time.perf_counter()
    times["generate_hypervector"] = (end - start) / num_trials * 1000
    
    # Benchmark binding
    start = time.perf_counter()
    for _ in range(num_trials):
        bind_hypervectors(vectors[0], vectors[1], hdc.hypervector_type.value)
    end = time.perf_counter()
    times["bind"] = (end - start) / num_trials * 1000
    
    # Benchmark bundling
    start = time.perf_counter()
    for _ in range(num_trials):
        bundle_hypervectors(vectors[:5], hypervector_type=hdc.hypervector_type.value)
    end = time.perf_counter()
    times["bundle_5"] = (end - start) / num_trials * 1000
    
    # Benchmark similarity
    start = time.perf_counter()
    for _ in range(num_trials):
        similarity(vectors[0], vectors[1])
    end = time.perf_counter()
    times["similarity"] = (end - start) / num_trials * 1000
    
    # Benchmark permutation
    start = time.perf_counter()
    for _ in range(num_trials):
        hdc.permute(vectors[0], shift=1)
    end = time.perf_counter()
    times["permute"] = (end - start) / num_trials * 1000
    
    return times


def analyze_binding_properties(
    dimension: int,
    hypervector_type: str = "bipolar",
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Analyze properties of binding operations.
    
    Parameters
    ----------
    dimension : int
        Hypervector dimension
    hypervector_type : str
        Type of hypervectors
    num_samples : int
        Number of samples to test
        
    Returns
    -------
    Dict[str, float]
        Analysis results
    """
    results = {}
    
    # Test self-inverse property
    self_inverse_errors = []
    for _ in range(num_samples):
        # Generate random vectors
        if hypervector_type == "binary":
            a = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
            b = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
        else:
            a = 2 * np.random.randint(0, 2, size=dimension) - 1
            a = a.astype(np.int8)
            b = 2 * np.random.randint(0, 2, size=dimension) - 1
            b = b.astype(np.int8)
            
        # Bind
        bound = bind_hypervectors(a, b, hypervector_type)
        
        # Unbind (bind with b again for self-inverse)
        unbound = bind_hypervectors(bound, b, hypervector_type)
        
        # Check similarity
        sim = similarity(a, unbound)
        self_inverse_errors.append(1.0 - sim)
        
    results["mean_self_inverse_error"] = np.mean(self_inverse_errors)
    results["max_self_inverse_error"] = np.max(self_inverse_errors)
    
    # Test preservation of distance
    distance_ratios = []
    for _ in range(num_samples // 2):
        # Generate vectors
        if hypervector_type == "binary":
            a = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
            b = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
            k = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
        else:
            a = 2 * np.random.randint(0, 2, size=dimension) - 1
            a = a.astype(np.int8)
            b = 2 * np.random.randint(0, 2, size=dimension) - 1
            b = b.astype(np.int8)
            k = 2 * np.random.randint(0, 2, size=dimension) - 1
            k = k.astype(np.int8)
            
        # Original distance
        orig_dist = 1.0 - similarity(a, b)
        
        # Bind both with same key
        bound_a = bind_hypervectors(a, k, hypervector_type)
        bound_b = bind_hypervectors(b, k, hypervector_type)
        
        # Distance after binding
        bound_dist = 1.0 - similarity(bound_a, bound_b)
        
        if orig_dist > 0:
            distance_ratios.append(bound_dist / orig_dist)
            
    results["mean_distance_preservation"] = np.mean(distance_ratios)
    results["std_distance_preservation"] = np.std(distance_ratios)
    
    return results


def compare_hypervector_types(
    dimension: int = 10000,
    num_items: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Compare different hypervector types.
    
    Parameters
    ----------
    dimension : int
        Hypervector dimension
    num_items : int
        Number of items to test
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Comparison results for each type
    """
    types = ["binary", "bipolar", "ternary"]
    results = {}
    
    for hv_type in types:
        logger.info(f"Testing {hv_type} hypervectors...")
        
        # Create HDC instance
        config = HDCConfig(
            dimension=dimension,
            hypervector_type=hv_type
        )
        hdc = HDC(config)
        
        # Measure capacity
        metrics = measure_capacity(hdc, num_items=num_items)
        
        # Benchmark operations
        op_times = benchmark_operations(hdc)
        
        results[hv_type] = {
            "mean_similarity": metrics.capacity_results["mean_similarity"],
            "estimated_capacity": metrics.capacity_results["estimated_capacity"],
            "noise_tolerance_0.1": metrics.noise_tolerance.get(0.1, 0.0),
            "noise_tolerance_0.2": metrics.noise_tolerance.get(0.2, 0.0),
            "generate_time_ms": op_times["generate_hypervector"],
            "bind_time_ms": op_times["bind"],
            "bundle_time_ms": op_times["bundle_5"],
            "similarity_time_ms": op_times["similarity"],
        }
        
    return results


def generate_similarity_matrix(
    vectors: List[np.ndarray],
    labels: Optional[List[str]] = None,
    metric: str = "cosine"
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate similarity matrix for a set of hypervectors.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of hypervectors
    labels : List[str], optional
        Labels for vectors
    metric : str
        Similarity metric to use
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Similarity matrix and labels
    """
    n = len(vectors)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = similarity(vectors[i], vectors[j], metric=metric)
            
    if labels is None:
        labels = [f"V{i}" for i in range(n)]
        
    return sim_matrix, labels


def measure_associativity(
    dimension: int,
    hypervector_type: str = "bipolar",
    num_trials: int = 100
) -> Dict[str, float]:
    """
    Measure associativity of bundling operation.
    
    Parameters
    ----------
    dimension : int
        Hypervector dimension
    hypervector_type : str
        Type of hypervectors
    num_trials : int
        Number of trials
        
    Returns
    -------
    Dict[str, float]
        Test results
    """
    associativity_errors = []
    
    for _ in range(num_trials):
        # Generate three random vectors
        if hypervector_type == "binary":
            a = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
            b = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
            c = np.random.randint(0, 2, size=dimension, dtype=np.uint8)
        else:
            a = 2 * np.random.randint(0, 2, size=dimension) - 1
            a = a.astype(np.int8)
            b = 2 * np.random.randint(0, 2, size=dimension) - 1
            b = b.astype(np.int8)
            c = 2 * np.random.randint(0, 2, size=dimension) - 1
            c = c.astype(np.int8)
            
        # (a + b) + c
        ab = bundle_hypervectors([a, b], hypervector_type=hypervector_type)
        abc1 = bundle_hypervectors([ab, c], hypervector_type=hypervector_type)
        
        # a + (b + c)
        bc = bundle_hypervectors([b, c], hypervector_type=hypervector_type)
        abc2 = bundle_hypervectors([a, bc], hypervector_type=hypervector_type)
        
        # Compare
        sim = similarity(abc1, abc2)
        associativity_errors.append(1.0 - sim)
        
    return {
        "mean_associativity_error": np.mean(associativity_errors),
        "max_associativity_error": np.max(associativity_errors),
        "perfect_associations": np.sum(np.array(associativity_errors) < 1e-10) / num_trials
    }


def estimate_required_dimension(
    num_items: int,
    similarity_threshold: float = 0.1,
    confidence: float = 0.99
) -> int:
    """
    Estimate required dimension for storing items.
    
    Parameters
    ----------
    num_items : int
        Number of items to store
    similarity_threshold : float
        Maximum acceptable similarity
    confidence : float
        Confidence level for estimate
        
    Returns
    -------
    int
        Required dimension
    """
    # Based on Johnson-Lindenstrauss lemma and HDC theory
    # For binary vectors, expected similarity is 0.5
    # For bipolar vectors, expected similarity is 0.0
    
    # Conservative estimate
    epsilon = similarity_threshold
    n = num_items
    
    # J-L bound: d >= (4 * log(n) / epsilon^2)
    d_jl = int(np.ceil(4 * np.log(n) / (epsilon ** 2)))
    
    # HDC heuristic: d >= 10 * log2(n) for good separation
    d_hdc = int(np.ceil(10 * np.log2(n)))
    
    # Use the larger estimate
    required_dim = max(d_jl, d_hdc, 1000)  # Minimum 1000
    
    # Round up to nearest multiple of 100
    required_dim = int(np.ceil(required_dim / 100) * 100)
    
    return required_dim


def create_codebook(
    num_symbols: int,
    dimension: int,
    hypervector_type: str = "bipolar"
) -> Dict[str, np.ndarray]:
    """
    Create a codebook of quasi-orthogonal hypervectors.
    
    Parameters
    ----------
    num_symbols : int
        Number of symbols
    dimension : int
        Hypervector dimension  
    hypervector_type : str
        Type of hypervectors
        
    Returns
    -------
    Dict[str, np.ndarray]
        Symbol to hypervector mapping
    """
    # Generate orthogonal hypervectors
    vectors = generate_orthogonal_hypervectors(
        dimension,
        num_symbols,
        hypervector_type
    )
    
    # Create codebook
    codebook = {}
    for i in range(num_symbols):
        symbol = f"symbol_{i}"
        codebook[symbol] = vectors[i]
        
    return codebook


def measure_classifier_performance(
    classifier: HDClassifier,
    X_test: List[any],
    y_test: List[str],
    X_train: Optional[List[any]] = None,
    y_train: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Measure classifier performance metrics.
    
    Parameters
    ----------
    classifier : HDClassifier
        Trained classifier
    X_test : List[any]
        Test data
    y_test : List[str]
        Test labels
    X_train : List[any], optional
        Training data for overfitting check
    y_train : List[str], optional
        Training labels
        
    Returns
    -------
    Dict[str, float]
        Performance metrics
    """
    metrics = {}
    
    # Test accuracy
    test_score = classifier.score(X_test, y_test)
    metrics["test_accuracy"] = test_score
    
    # Train accuracy (if provided)
    if X_train is not None and y_train is not None:
        train_score = classifier.score(X_train, y_train)
        metrics["train_accuracy"] = train_score
        metrics["overfitting_gap"] = train_score - test_score
        
    # Per-class accuracy
    predictions = classifier.predict(X_test)
    class_correct = {}
    class_total = {}
    
    for pred, true in zip(predictions, y_test):
        if true not in class_total:
            class_total[true] = 0
            class_correct[true] = 0
            
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
            
    for cls in class_total:
        metrics[f"accuracy_{cls}"] = class_correct[cls] / class_total[cls]
        
    # Confusion statistics
    unique_classes = list(set(y_test))
    confusion_pairs = {}
    
    for true_cls in unique_classes:
        for pred_cls in unique_classes:
            if true_cls != pred_cls:
                count = sum(1 for p, t in zip(predictions, y_test) 
                           if t == true_cls and p == pred_cls)
                if count > 0:
                    confusion_pairs[f"{true_cls}_as_{pred_cls}"] = count
                    
    metrics["confusion_pairs"] = confusion_pairs
    
    return metrics