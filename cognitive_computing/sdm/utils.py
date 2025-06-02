"""
Utility functions for Sparse Distributed Memory operations.

This module provides helper functions for working with SDM, including:
- Pattern generation and manipulation
- Noise addition and analysis
- Capacity calculations
- Performance testing utilities
- Data encoding/decoding helpers
- Activation pattern analysis
- Similarity metrics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from scipy import stats
from scipy.spatial.distance import hamming, jaccard
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict
import json
import hashlib
import pickle
import warnings

logger = logging.getLogger(__name__)


def add_noise(pattern: np.ndarray, noise_level: float, 
              noise_type: str = 'flip', seed: Optional[int] = None) -> np.ndarray:
    """
    Add noise to a binary pattern.
    
    Parameters
    ----------
    pattern : np.ndarray
        Binary pattern to add noise to
    noise_level : float
        Amount of noise (0 to 1)
        - For 'flip': probability of bit flip
        - For 'swap': proportion of bits to swap
        - For 'burst': length of burst as proportion
    noise_type : str, optional
        Type of noise: 'flip', 'swap', 'burst', 'salt_pepper'
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Noisy pattern
        
    Examples
    --------
    >>> pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    >>> noisy = add_noise(pattern, 0.25, 'flip')
    >>> # Approximately 25% of bits will be flipped
    """
    if not 0 <= noise_level <= 1:
        raise ValueError(f"Noise level must be in [0, 1], got {noise_level}")
    
    rng = np.random.RandomState(seed)
    noisy_pattern = pattern.copy()
    
    if noise_type == 'flip':
        # Random bit flips
        flip_mask = rng.random(len(pattern)) < noise_level
        noisy_pattern[flip_mask] = 1 - noisy_pattern[flip_mask]
        
    elif noise_type == 'swap':
        # Swap pairs of bits
        n_swaps = int(len(pattern) * noise_level / 2)
        for _ in range(n_swaps):
            i, j = rng.choice(len(pattern), 2, replace=False)
            noisy_pattern[i], noisy_pattern[j] = noisy_pattern[j], noisy_pattern[i]
            
    elif noise_type == 'burst':
        # Burst errors (contiguous flips)
        burst_length = int(len(pattern) * noise_level)
        if burst_length > 0:
            start_pos = rng.randint(0, max(1, len(pattern) - burst_length))
            noisy_pattern[start_pos:start_pos + burst_length] = \
                1 - noisy_pattern[start_pos:start_pos + burst_length]
            
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise (force to 0 or 1)
        noise_mask = rng.random(len(pattern)) < noise_level
        noise_values = rng.randint(0, 2, np.sum(noise_mask))
        noisy_pattern[noise_mask] = noise_values
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy_pattern


def generate_random_patterns(num_patterns: int, dimension: int,
                           sparsity: float = 0.5, 
                           correlation: float = 0.0,
                           seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate random binary patterns for testing.
    
    Parameters
    ----------
    num_patterns : int
        Number of patterns to generate
    dimension : int
        Dimension of each pattern
    sparsity : float, optional
        Proportion of 1s in patterns (default: 0.5)
    correlation : float, optional
        Correlation between addresses and data (0 to 1)
    seed : int, optional
        Random seed
        
    Returns
    -------
    addresses : list
        List of address patterns
    data : list
        List of data patterns
        
    Examples
    --------
    >>> addrs, data = generate_random_patterns(100, 1000, sparsity=0.3)
    >>> # Generates 100 patterns with 30% ones
    """
    rng = np.random.RandomState(seed)
    
    addresses = []
    data = []
    
    for i in range(num_patterns):
        # Generate address
        addr = (rng.random(dimension) < sparsity).astype(np.uint8)
        addresses.append(addr)
        
        if correlation > 0:
            # Generate correlated data
            base_data = (rng.random(dimension) < sparsity).astype(np.uint8)
            # Copy some bits from address based on correlation
            copy_mask = rng.random(dimension) < correlation
            data_pattern = base_data.copy()
            data_pattern[copy_mask] = addr[copy_mask]
            data.append(data_pattern)
        else:
            # Independent data
            data_pattern = (rng.random(dimension) < sparsity).astype(np.uint8)
            data.append(data_pattern)
    
    return addresses, data


def compute_memory_capacity(dimension: int, num_locations: int,
                          activation_radius: int, 
                          error_tolerance: float = 0.01) -> Dict[str, float]:
    """
    Compute theoretical capacity of SDM configuration.
    
    Based on Kanerva's analysis and information theory bounds.
    
    Parameters
    ----------
    dimension : int
        Address/data space dimension
    num_locations : int
        Number of hard locations
    activation_radius : int
        Hamming radius for activation
    error_tolerance : float, optional
        Acceptable bit error rate
        
    Returns
    -------
    dict
        Capacity estimates using different methods
        
    References
    ----------
    Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
    """
    # Critical distance
    critical_distance = 0.451 * dimension
    
    # Probability of activation
    p_activation = sum(stats.binom.pmf(k, dimension, 0.5) 
                      for k in range(activation_radius + 1))
    
    # Expected number of activated locations
    expected_activated = num_locations * p_activation
    
    # Method 1: Kanerva's original estimate
    if activation_radius <= critical_distance:
        kanerva_capacity = 0.15 * num_locations
    else:
        ratio = activation_radius / critical_distance
        kanerva_capacity = 0.15 * num_locations / (ratio ** 2)
    
    # Method 2: Information theoretic bound
    # Based on signal-to-noise ratio
    if expected_activated > 0:
        snr = 1 / (error_tolerance * expected_activated)
        info_capacity = num_locations * np.log2(1 + snr) / dimension
    else:
        info_capacity = 0
    
    # Method 3: Sphere packing bound
    # Number of non-overlapping spheres of radius r
    sphere_volume = sum(stats.binom.pmf(k, dimension, 0.5) 
                       for k in range(2 * activation_radius + 1))
    packing_capacity = 1 / sphere_volume if sphere_volume > 0 else 0
    
    # Method 4: Coverage-based estimate
    # Based on how well activated locations cover the space
    coverage = 1 - (1 - p_activation) ** num_locations
    coverage_capacity = num_locations * coverage * (1 - error_tolerance)
    
    return {
        'kanerva_estimate': kanerva_capacity,
        'information_theoretic': info_capacity,
        'sphere_packing': packing_capacity,
        'coverage_based': coverage_capacity,
        'expected_activated': expected_activated,
        'activation_probability': p_activation,
        'recommended_capacity': int(min(kanerva_capacity, coverage_capacity))
    }


def analyze_activation_patterns(sdm, sample_size: int = 1000,
                              visualize: bool = False) -> Dict[str, any]:
    """
    Analyze activation patterns in an SDM instance.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to analyze
    sample_size : int, optional
        Number of random addresses to sample
    visualize : bool, optional
        Whether to create visualization plots
        
    Returns
    -------
    dict
        Analysis results including overlap statistics and distributions
    """
    # Sample random addresses
    addresses = []
    activation_sets = []
    activation_counts = []
    
    for _ in range(sample_size):
        addr = np.random.randint(0, 2, sdm.config.dimension)
        addresses.append(addr)
        
        activated = sdm._get_activated_locations(addr)
        activation_sets.append(set(activated))
        activation_counts.append(len(activated))
    
    # Compute pairwise overlaps
    overlaps = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            overlap = len(activation_sets[i] & activation_sets[j])
            overlaps.append(overlap)
            
    # Compute address similarities vs activation overlaps
    correlations = []
    for i in range(min(100, sample_size)):  # Subsample for efficiency
        for j in range(i + 1, min(100, sample_size)):
            addr_similarity = 1 - hamming(addresses[i], addresses[j])
            # Handle case where no locations are activated
            max_size = max(len(activation_sets[i]), len(activation_sets[j]))
            if max_size > 0:
                activation_overlap = len(activation_sets[i] & activation_sets[j]) / max_size
            else:
                # Both sets are empty, so overlap is undefined (we'll use 1.0)
                activation_overlap = 1.0
            correlations.append((addr_similarity, activation_overlap))
    
    # Location usage distribution
    location_activations = defaultdict(int)
    for act_set in activation_sets:
        for loc in act_set:
            location_activations[loc] += 1
    
    usage_counts = list(location_activations.values())
    unused_locations = sdm.config.num_hard_locations - len(location_activations)
    
    results = {
        'mean_activation_count': np.mean(activation_counts),
        'std_activation_count': np.std(activation_counts),
        'min_activations': np.min(activation_counts),
        'max_activations': np.max(activation_counts),
        'mean_overlap': np.mean(overlaps),
        'std_overlap': np.std(overlaps),
        'location_usage_mean': np.mean(usage_counts) if usage_counts else 0,
        'location_usage_std': np.std(usage_counts) if usage_counts else 0,
        'unused_locations': unused_locations,
        'usage_uniformity': 1 - (np.std(usage_counts) / np.mean(usage_counts)) if usage_counts else 0
    }
    
    # Compute correlation between address similarity and activation overlap
    if correlations:
        addr_sims, act_overlaps = zip(*correlations)
        results['similarity_correlation'] = np.corrcoef(addr_sims, act_overlaps)[0, 1]
    
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Activation count distribution
        axes[0, 0].hist(activation_counts, bins=30, edgecolor='black')
        axes[0, 0].axvline(results['mean_activation_count'], color='red', 
                           linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('Number of Activated Locations')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Activation Count Distribution')
        axes[0, 0].legend()
        
        # Overlap distribution
        axes[0, 1].hist(overlaps, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Overlap Size')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Pairwise Overlap Distribution')
        
        # Location usage
        if usage_counts:
            axes[1, 0].hist(usage_counts, bins=30, edgecolor='black')
            axes[1, 0].set_xlabel('Activation Count per Location')
            axes[1, 0].set_ylabel('Number of Locations')
            axes[1, 0].set_title('Location Usage Distribution')
        
        # Similarity correlation
        if correlations:
            addr_sims, act_overlaps = zip(*correlations)
            axes[1, 1].scatter(addr_sims, act_overlaps, alpha=0.5)
            axes[1, 1].set_xlabel('Address Similarity')
            axes[1, 1].set_ylabel('Activation Overlap')
            axes[1, 1].set_title(f'Correlation: {results["similarity_correlation"]:.3f}')
        
        plt.tight_layout()
        results['figure'] = fig
    
    return results


@dataclass
class PerformanceTestResult:
    """Results from performance testing."""
    pattern_count: int
    dimension: int
    write_time_mean: float
    write_time_std: float
    read_time_mean: float
    read_time_std: float
    recall_accuracy_mean: float
    recall_accuracy_std: float
    noise_tolerance: Dict[float, float]
    capacity_utilization: float


def test_sdm_performance(sdm, test_patterns: int = 100,
                        noise_levels: List[float] = None,
                        progress: bool = True) -> PerformanceTestResult:
    """
    Comprehensive performance test for SDM.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to test
    test_patterns : int, optional
        Number of test patterns
    noise_levels : list, optional
        Noise levels to test
    progress : bool, optional
        Show progress bar
        
    Returns
    -------
    PerformanceTestResult
        Test results
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    # Generate test patterns
    addresses, data = generate_random_patterns(
        test_patterns, sdm.config.dimension, sparsity=0.5
    )
    
    # Test write performance
    write_times = []
    iterator = tqdm(zip(addresses, data), total=test_patterns, 
                   desc="Write test") if progress else zip(addresses, data)
    
    for addr, dat in iterator:
        start_time = time.time()
        sdm.store(addr, dat)
        write_times.append(time.time() - start_time)
    
    # Test read performance and accuracy
    read_times = []
    accuracies = []
    
    iterator = tqdm(zip(addresses, data), total=test_patterns,
                   desc="Read test") if progress else zip(addresses, data)
    
    for addr, original_data in iterator:
        start_time = time.time()
        recalled = sdm.recall(addr)
        read_times.append(time.time() - start_time)
        
        if recalled is not None:
            accuracy = np.mean(recalled == original_data)
            accuracies.append(accuracy)
    
    # Test noise tolerance
    noise_tolerance = {}
    
    for noise in noise_levels:
        noise_accuracies = []
        
        # Sample subset for noise testing
        test_indices = np.random.choice(len(addresses), 
                                      min(50, len(addresses)), 
                                      replace=False)
        
        for idx in test_indices:
            noisy_addr = add_noise(addresses[idx], noise, 'flip')
            recalled = sdm.recall(noisy_addr)
            
            if recalled is not None:
                accuracy = np.mean(recalled == data[idx])
                noise_accuracies.append(accuracy)
        
        noise_tolerance[noise] = np.mean(noise_accuracies) if noise_accuracies else 0.0
    
    # Calculate capacity utilization
    memory_stats = sdm.get_memory_stats()
    capacity_util = memory_stats.get('locations_used', 0) / sdm.config.num_hard_locations
    
    return PerformanceTestResult(
        pattern_count=test_patterns,
        dimension=sdm.config.dimension,
        write_time_mean=np.mean(write_times),
        write_time_std=np.std(write_times),
        read_time_mean=np.mean(read_times),
        read_time_std=np.std(read_times),
        recall_accuracy_mean=np.mean(accuracies) if accuracies else 0.0,
        recall_accuracy_std=np.std(accuracies) if accuracies else 0.0,
        noise_tolerance=noise_tolerance,
        capacity_utilization=capacity_util
    )


def calculate_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray,
                               metric: str = 'hamming') -> float:
    """
    Calculate similarity between two binary patterns.
    
    Parameters
    ----------
    pattern1, pattern2 : np.ndarray
        Binary patterns to compare
    metric : str, optional
        Similarity metric: 'hamming', 'jaccard', 'cosine', 'mutual_info'
        
    Returns
    -------
    float
        Similarity score (interpretation depends on metric)
    """
    if pattern1.shape != pattern2.shape:
        raise ValueError("Patterns must have the same shape")
    
    if metric == 'hamming':
        # Normalized Hamming similarity (1 - distance)
        return 1 - hamming(pattern1, pattern2)
    
    elif metric == 'jaccard':
        # Jaccard similarity
        # Ensure patterns are binary integers for bitwise operations
        p1 = pattern1.astype(np.uint8)
        p2 = pattern2.astype(np.uint8)
        intersection = np.sum(p1 & p2)
        union = np.sum(p1 | p2)
        return intersection / union if union > 0 else 0.0
    
    elif metric == 'cosine':
        # Cosine similarity
        dot_product = np.dot(pattern1, pattern2)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    elif metric == 'mutual_info':
        # Normalized mutual information
        return mutual_info_score(pattern1, pattern2)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_orthogonal_patterns(num_patterns: int, dimension: int,
                              min_distance: Optional[int] = None) -> List[np.ndarray]:
    """
    Create approximately orthogonal binary patterns.
    
    Parameters
    ----------
    num_patterns : int
        Number of patterns to create
    dimension : int
        Dimension of patterns
    min_distance : int, optional
        Minimum Hamming distance between patterns
        
    Returns
    -------
    list
        List of binary patterns
    """
    if min_distance is None:
        min_distance = dimension // 3
    
    patterns = []
    max_attempts = 1000
    
    while len(patterns) < num_patterns:
        attempts = 0
        found = False
        
        while not found and attempts < max_attempts:
            # Generate candidate pattern
            candidate = np.random.randint(0, 2, dimension)
            
            # Check distance to existing patterns
            if all(np.sum(candidate != p) >= min_distance for p in patterns):
                patterns.append(candidate)
                found = True
            
            attempts += 1
        
        if not found:
            warnings.warn(f"Could only create {len(patterns)} patterns with "
                         f"min_distance={min_distance}")
            break
    
    return patterns


class PatternEncoder:
    """
    Encode various data types into binary patterns for SDM.
    
    This class provides methods to encode different data types
    (integers, floats, strings, etc.) into binary vectors suitable
    for storage in SDM.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize pattern encoder.
        
        Parameters
        ----------
        dimension : int
            Target dimension for encoded patterns
        """
        self.dimension = dimension
    
    def encode_integer(self, value: int, bits: Optional[int] = None) -> np.ndarray:
        """
        Encode integer to binary pattern.
        
        Parameters
        ----------
        value : int
            Integer value to encode
        bits : int, optional
            Number of bits to use (default: as needed)
            
        Returns
        -------
        np.ndarray
            Binary pattern
        """
        if bits is None:
            bits = max(1, int(np.log2(abs(value) + 1)) + 2)  # +1 for sign bit
        
        # Convert to binary string
        if value >= 0:
            binary = format(value, f'0{bits}b')
        else:
            # Two's complement for negative numbers
            binary = format((1 << bits) + value, f'0{bits}b')
        
        # Convert to numpy array
        pattern = np.array([int(b) for b in binary], dtype=np.uint8)
        
        # Pad or truncate to dimension
        if len(pattern) < self.dimension:
            pattern = np.pad(pattern, (0, self.dimension - len(pattern)))
        elif len(pattern) > self.dimension:
            pattern = pattern[:self.dimension]
        
        return pattern
    
    def encode_float(self, value: float, precision: int = 16) -> np.ndarray:
        """
        Encode float to binary pattern.
        
        Parameters
        ----------
        value : float
            Float value to encode
        precision : int, optional
            Number of bits for fractional part
            
        Returns
        -------
        np.ndarray
            Binary pattern
        """
        # Scale and convert to integer
        scaled = int(value * (2 ** precision))
        return self.encode_integer(scaled, self.dimension)
    
    def encode_string(self, text: str, method: str = 'hash') -> np.ndarray:
        """
        Encode string to binary pattern.
        
        Parameters
        ----------
        text : str
            String to encode
        method : str, optional
            Encoding method: 'hash', 'char', 'semantic'
            
        Returns
        -------
        np.ndarray
            Binary pattern
        """
        if method == 'hash':
            # Use cryptographic hash
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert to binary
            pattern = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
            
        elif method == 'char':
            # Character-based encoding
            char_bits = []
            for char in text:
                char_bits.extend([int(b) for b in format(ord(char), '08b')])
            pattern = np.array(char_bits, dtype=np.uint8)
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        # Adjust to dimension
        if len(pattern) < self.dimension:
            # Cycle pattern to fill dimension
            pattern = np.tile(pattern, (self.dimension // len(pattern) + 1))[:self.dimension]
        else:
            pattern = pattern[:self.dimension]
        
        return pattern
    
    def encode_vector(self, vector: np.ndarray, method: str = 'threshold') -> np.ndarray:
        """
        Encode continuous vector to binary pattern.
        
        Parameters
        ----------
        vector : np.ndarray
            Continuous vector
        method : str, optional
            Encoding method: 'threshold', 'rank', 'random_projection'
            
        Returns
        -------
        np.ndarray
            Binary pattern
        """
        if method == 'threshold':
            # Threshold at median
            threshold = np.median(vector)
            pattern = (vector > threshold).astype(np.uint8)
            
        elif method == 'rank':
            # Rank-based encoding
            ranks = stats.rankdata(vector)
            threshold = len(vector) / 2
            pattern = (ranks > threshold).astype(np.uint8)
            
        elif method == 'random_projection':
            # Random projection to binary
            rng = np.random.RandomState(42)  # Fixed seed for consistency
            projection = rng.randn(len(vector), self.dimension)
            projected = vector @ projection
            pattern = (projected > 0).astype(np.uint8)
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        # Ensure correct dimension
        if len(pattern) != self.dimension:
            # Resample to match dimension
            indices = np.linspace(0, len(pattern)-1, self.dimension).astype(int)
            pattern = pattern[indices]
        
        return pattern


def save_sdm_state(sdm, filepath: str, include_patterns: bool = True):
    """
    Save SDM state to file.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to save
    filepath : str
        Path to save file
    include_patterns : bool, optional
        Whether to include stored patterns
    """
    # Save only the constructor parameters of SDMConfig
    config_params = {
        'dimension': sdm.config.dimension,
        'num_hard_locations': sdm.config.num_hard_locations,
        'activation_radius': sdm.config.activation_radius,
        'storage_method': sdm.config.storage_method,
        'saturation_value': sdm.config.saturation_value,
        'seed': sdm.config.seed
    }
    
    state = {
        'config': config_params,
        'hard_locations': sdm.hard_locations,
        'location_usage': sdm.location_usage,
        'metrics': sdm.metrics.__dict__
    }
    
    # Storage state
    if sdm.config.storage_method == 'counters':
        state['counters'] = sdm.counters
    else:
        state['binary_storage'] = sdm.binary_storage
    
    # Stored patterns
    if include_patterns and len(sdm._stored_addresses) > 0:
        state['stored_addresses'] = sdm._stored_addresses
        state['stored_data'] = sdm._stored_data
    
    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    
    logger.info(f"SDM state saved to {filepath}")


def load_sdm_state(filepath: str, sdm_class=None):
    """
    Load SDM state from file.
    
    Parameters
    ----------
    filepath : str
        Path to saved file
    sdm_class : class, optional
        SDM class to use (default: SDM)
        
    Returns
    -------
    SDM
        Loaded SDM instance
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    # Import SDM if not provided
    if sdm_class is None:
        from cognitive_computing.sdm.core import SDM, SDMConfig
    else:
        from cognitive_computing.sdm.core import SDMConfig
        SDM = sdm_class
    
    # Reconstruct config
    config = SDMConfig(**state['config'])
    
    # Create SDM instance
    sdm = SDM(config)
    
    # Restore state
    sdm.hard_locations = state['hard_locations']
    sdm.location_usage = state['location_usage']
    
    if 'counters' in state:
        sdm.counters = state['counters']
    elif 'binary_storage' in state:
        sdm.binary_storage = state['binary_storage']
    
    if 'stored_addresses' in state:
        sdm._stored_addresses = state['stored_addresses']
        sdm._stored_data = state['stored_data']
    
    logger.info(f"SDM state loaded from {filepath}")
    
    return sdm


# Time is imported at the module level
import time