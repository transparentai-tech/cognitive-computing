"""
Memory storage and analysis components for Sparse Distributed Memory.

This module provides classes and utilities for managing the internal memory
structures of SDM, including hard locations, memory contents analysis, and
statistical tracking of memory usage patterns.

Key Components:
- HardLocation: Represents a single hard location in SDM
- MemoryContents: Analyzer for SDM memory contents
- MemoryStatistics: Advanced statistical analysis of SDM behavior
- MemoryOptimizer: Utilities for optimizing SDM parameters

The module supports both counter-based and binary storage methods, providing
tools for visualization, analysis, and optimization of memory usage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)


@dataclass
class HardLocation:
    """
    Represents a single hard location in Sparse Distributed Memory.
    
    A hard location consists of an address in the binary space and
    associated storage (counters or binary values). This class tracks
    usage statistics and provides analysis methods for individual locations.
    
    Parameters
    ----------
    index : int
        Index of this location in the SDM
    address : np.ndarray
        Binary address vector of shape (dimension,)
    dimension : int
        Dimensionality of the address/data space
    storage_type : str
        Type of storage: 'counters' or 'binary'
    
    Attributes
    ----------
    counters : np.ndarray or None
        Counter values if storage_type is 'counters'
    binary_data : np.ndarray or None
        Binary values if storage_type is 'binary'
    access_count : int
        Number of times this location has been activated
    write_count : int
        Number of times data has been written to this location
    last_access_time : int
        Timestamp of last access (in terms of operation count)
    activation_history : List[int]
        History of activation timestamps
    """
    
    index: int
    address: np.ndarray
    dimension: int
    storage_type: str = "counters"
    counters: Optional[np.ndarray] = None
    binary_data: Optional[np.ndarray] = None
    access_count: int = 0
    write_count: int = 0
    last_access_time: int = 0
    activation_history: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize storage arrays based on storage type."""
        if self.storage_type == "counters":
            if self.counters is None:
                self.counters = np.zeros(self.dimension, dtype=np.int16)
        elif self.storage_type == "binary":
            if self.binary_data is None:
                self.binary_data = np.zeros(self.dimension, dtype=np.uint8)
        else:
            raise ValueError(f"Invalid storage_type: {self.storage_type}")
    
    def write(self, data: np.ndarray, timestamp: int = 0):
        """
        Write data to this location.
        
        Parameters
        ----------
        data : np.ndarray
            Data to write (binary for 'binary' storage, bipolar for 'counters')
        timestamp : int, optional
            Current timestamp for tracking access patterns
        """
        self.write_count += 1
        self.access_count += 1
        self.last_access_time = timestamp
        self.activation_history.append(timestamp)
        
        if self.storage_type == "counters":
            self.counters += data
        else:
            self.binary_data = np.logical_or(self.binary_data, data).astype(np.uint8)
    
    def read(self, timestamp: int = 0) -> np.ndarray:
        """
        Read data from this location.
        
        Parameters
        ----------
        timestamp : int, optional
            Current timestamp for tracking access patterns
            
        Returns
        -------
        np.ndarray
            Stored data (counters or binary values)
        """
        self.access_count += 1
        self.last_access_time = timestamp
        
        if self.storage_type == "counters":
            return self.counters
        else:
            return self.binary_data
    
    def get_saturation_level(self, max_value: int = 127) -> float:
        """
        Calculate saturation level for counter-based storage.
        
        Parameters
        ----------
        max_value : int, optional
            Maximum counter value (default: 127)
            
        Returns
        -------
        float
            Proportion of counters at maximum absolute value
        """
        if self.storage_type != "counters":
            return 0.0
        
        saturated = np.sum(np.abs(self.counters) >= max_value)
        return saturated / self.dimension
    
    def get_bit_density(self) -> float:
        """
        Calculate bit density for binary storage.
        
        Returns
        -------
        float
            Proportion of bits set to 1
        """
        if self.storage_type != "binary":
            return 0.0
        
        return np.mean(self.binary_data)
    
    def get_entropy(self) -> float:
        """
        Calculate entropy of stored data.
        
        Returns
        -------
        float
            Shannon entropy of the stored pattern
        """
        if self.storage_type == "counters":
            # Normalize counters to probabilities
            if np.all(self.counters == 0):
                return 0.0
            probs = np.abs(self.counters) / np.sum(np.abs(self.counters))
            probs = probs[probs > 0]  # Remove zeros for log calculation
            return -np.sum(probs * np.log2(probs))
        else:
            # Binary entropy
            p = np.mean(self.binary_data)
            if p == 0 or p == 1:
                return 0.0
            return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def reset(self):
        """Reset this location to initial state."""
        if self.storage_type == "counters":
            self.counters.fill(0)
        else:
            self.binary_data.fill(0)
        
        self.access_count = 0
        self.write_count = 0
        self.last_access_time = 0
        self.activation_history.clear()


class MemoryContents:
    """
    Analyzer for SDM memory contents.
    
    This class provides tools for analyzing the contents of an SDM,
    including pattern distribution, clustering, and quality metrics.
    
    Parameters
    ----------
    sdm : SDM
        The SDM instance to analyze
    """
    
    def __init__(self, sdm):
        """Initialize memory contents analyzer."""
        self.sdm = sdm
        self.hard_locations = self._create_hard_location_objects()
    
    def _create_hard_location_objects(self) -> List[HardLocation]:
        """Create HardLocation objects for all locations in SDM."""
        locations = []
        for i in range(self.sdm.config.num_hard_locations):
            loc = HardLocation(
                index=i,
                address=self.sdm.hard_locations[i],
                dimension=self.sdm.config.dimension,
                storage_type=self.sdm.config.storage_method
            )
            
            # Copy current storage state
            if self.sdm.config.storage_method == "counters":
                loc.counters = self.sdm.counters[i].copy()
            else:
                loc.binary_data = self.sdm.binary_storage[i].copy()
            
            # Copy usage statistics
            loc.access_count = self.sdm.location_usage[i]
            loc.write_count = self.sdm.location_usage[i]  # Approximation
            
            locations.append(loc)
        
        return locations
    
    def get_memory_map(self) -> Dict[str, np.ndarray]:
        """
        Generate a memory map showing usage patterns.
        
        Returns
        -------
        dict
            Dictionary containing various memory maps:
            - 'usage_map': Location usage frequencies
            - 'saturation_map': Saturation levels (for counters)
            - 'density_map': Bit densities (for binary)
            - 'entropy_map': Information entropy per location
        """
        num_locs = self.sdm.config.num_hard_locations
        
        maps = {
            'usage_map': np.array([loc.access_count for loc in self.hard_locations]),
            'entropy_map': np.array([loc.get_entropy() for loc in self.hard_locations])
        }
        
        if self.sdm.config.storage_method == "counters":
            maps['saturation_map'] = np.array([
                loc.get_saturation_level(self.sdm.config.saturation_value)
                for loc in self.hard_locations
            ])
            maps['avg_counter_magnitude'] = np.array([
                np.mean(np.abs(loc.counters)) for loc in self.hard_locations
            ])
        else:
            maps['density_map'] = np.array([
                loc.get_bit_density() for loc in self.hard_locations
            ])
        
        return maps
    
    def analyze_pattern_distribution(self, sample_size: int = 1000) -> Dict[str, float]:
        """
        Analyze how patterns are distributed across memory.
        
        Parameters
        ----------
        sample_size : int, optional
            Number of random addresses to sample
            
        Returns
        -------
        dict
            Distribution statistics
        """
        # Sample random addresses and check activation patterns
        activation_counts = []
        overlap_matrix = np.zeros((sample_size, sample_size))
        
        sampled_addresses = []
        activated_sets = []
        
        for i in range(sample_size):
            # Generate random address
            addr = np.random.randint(0, 2, self.sdm.config.dimension)
            sampled_addresses.append(addr)
            
            # Get activated locations
            activated = self.sdm._get_activated_locations(addr)
            activated_sets.append(set(activated))
            activation_counts.append(len(activated))
        
        # Compute pairwise overlaps
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                overlap = len(activated_sets[i] & activated_sets[j])
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
        
        # Compute statistics
        stats = {
            'mean_activation_count': np.mean(activation_counts),
            'std_activation_count': np.std(activation_counts),
            'min_activation_count': np.min(activation_counts),
            'max_activation_count': np.max(activation_counts),
            'mean_overlap': np.mean(overlap_matrix[np.triu_indices(sample_size, k=1)]),
            'std_overlap': np.std(overlap_matrix[np.triu_indices(sample_size, k=1)]),
            'activation_uniformity': 1.0 - (np.std(activation_counts) / np.mean(activation_counts))
        }
        
        # Check if activation follows expected distribution
        expected_activation = self._compute_expected_activation()
        stats['activation_deviation'] = abs(stats['mean_activation_count'] - expected_activation) / expected_activation
        
        return stats
    
    def _compute_expected_activation(self) -> float:
        """Compute expected number of activated locations."""
        # Probability that a random location is within activation radius
        # Using binomial approximation
        n = self.sdm.config.dimension
        r = self.sdm.config.activation_radius
        p = sum(stats.binom.pmf(k, n, 0.5) for k in range(r + 1))
        
        return self.sdm.config.num_hard_locations * p
    
    def find_similar_locations(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Find pairs of hard locations with similar contents.
        
        Parameters
        ----------
        threshold : float, optional
            Similarity threshold (0 to 1)
            
        Returns
        -------
        list
            List of tuples (loc1_idx, loc2_idx, similarity)
        """
        similar_pairs = []
        
        for i in range(len(self.hard_locations)):
            for j in range(i+1, len(self.hard_locations)):
                loc1, loc2 = self.hard_locations[i], self.hard_locations[j]
                
                # Compute similarity based on storage type
                if self.sdm.config.storage_method == "counters":
                    # Cosine similarity of counter vectors
                    norm1 = np.linalg.norm(loc1.counters)
                    norm2 = np.linalg.norm(loc2.counters)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(loc1.counters, loc2.counters) / (norm1 * norm2)
                    else:
                        similarity = 0.0
                else:
                    # Jaccard similarity for binary data
                    intersection = np.sum(loc1.binary_data & loc2.binary_data)
                    union = np.sum(loc1.binary_data | loc2.binary_data)
                    similarity = intersection / union if union > 0 else 0.0
                
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))
        
        return sorted(similar_pairs, key=lambda x: x[2], reverse=True)
    
    def get_capacity_estimate(self) -> Dict[str, float]:
        """
        Estimate current and maximum capacity of the SDM.
        
        Returns
        -------
        dict
            Capacity estimates and related metrics
        """
        # Get memory maps
        maps = self.get_memory_map()
        
        # Current utilization
        locations_used = np.sum(maps['usage_map'] > 0)
        utilization = locations_used / self.sdm.config.num_hard_locations
        
        # Estimate remaining capacity based on saturation/density
        if self.sdm.config.storage_method == "counters":
            avg_saturation = np.mean(maps['saturation_map'])
            capacity_used = avg_saturation  # Rough estimate
        else:
            avg_density = np.mean(maps['density_map'])
            capacity_used = avg_density
        
        # Signal-to-noise ratio estimate
        if len(self.sdm._stored_addresses) > 0:
            # Test recall on stored patterns
            snr_samples = min(100, len(self.sdm._stored_addresses))
            errors = []
            
            for i in np.random.choice(len(self.sdm._stored_addresses), snr_samples, replace=False):
                addr = self.sdm._stored_addresses[i]
                data = self.sdm._stored_data[i]
                recalled = self.sdm.recall(addr)
                
                if recalled is not None:
                    error = np.mean(recalled != data)
                    errors.append(error)
            
            avg_error = np.mean(errors) if errors else 0.0
            snr = -10 * np.log10(avg_error + 1e-10)  # In dB
        else:
            avg_error = 0.0
            snr = float('inf')
        
        return {
            'theoretical_capacity': self.sdm.config.capacity,
            'patterns_stored': len(self.sdm._stored_addresses),
            'location_utilization': utilization,
            'capacity_used_estimate': capacity_used,
            'remaining_capacity_estimate': max(0, 1.0 - capacity_used),
            'average_recall_error': avg_error,
            'signal_to_noise_ratio_db': snr,
            'recommended_max_patterns': int(self.sdm.config.capacity * (1.0 - avg_error))
        }


class MemoryStatistics:
    """
    Advanced statistical analysis for SDM behavior.
    
    This class provides comprehensive statistical analysis of SDM performance,
    including temporal patterns, correlation analysis, and predictive metrics.
    
    Parameters
    ----------
    sdm : SDM
        The SDM instance to analyze
    """
    
    def __init__(self, sdm):
        """Initialize statistics analyzer."""
        self.sdm = sdm
        self.contents = MemoryContents(sdm)
        
        # Storage for temporal analysis
        self.operation_history = []
        self.performance_history = defaultdict(list)
    
    def record_operation(self, operation: str, success: bool, 
                        details: Optional[Dict] = None):
        """
        Record an operation for temporal analysis.
        
        Parameters
        ----------
        operation : str
            Type of operation ('store' or 'recall')
        success : bool
            Whether the operation was successful
        details : dict, optional
            Additional details about the operation
        """
        timestamp = len(self.operation_history)
        record = {
            'timestamp': timestamp,
            'operation': operation,
            'success': success,
            'details': details or {}
        }
        self.operation_history.append(record)
    
    def analyze_temporal_patterns(self, window_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Analyze temporal patterns in memory usage.
        
        Parameters
        ----------
        window_size : int, optional
            Size of sliding window for analysis
            
        Returns
        -------
        dict
            Temporal pattern analysis results
        """
        if len(self.operation_history) < window_size:
            return {"error": "Insufficient operation history"}
        
        # Extract time series data
        operations = [op['operation'] for op in self.operation_history]
        successes = [op['success'] for op in self.operation_history]
        
        # Compute sliding window statistics
        store_rates = []
        recall_rates = []
        success_rates = []
        
        for i in range(len(operations) - window_size + 1):
            window_ops = operations[i:i+window_size]
            window_success = successes[i:i+window_size]
            
            store_rate = sum(1 for op in window_ops if op == 'store') / window_size
            recall_rate = sum(1 for op in window_ops if op == 'recall') / window_size
            success_rate = sum(window_success) / window_size
            
            store_rates.append(store_rate)
            recall_rates.append(recall_rate)
            success_rates.append(success_rate)
        
        return {
            'store_rates': np.array(store_rates),
            'recall_rates': np.array(recall_rates),
            'success_rates': np.array(success_rates),
            'operation_balance': np.array(store_rates) - np.array(recall_rates)
        }
    
    def compute_correlation_matrix(self, sample_size: int = 100) -> np.ndarray:
        """
        Compute correlation matrix between memory locations.
        
        Parameters
        ----------
        sample_size : int, optional
            Number of locations to sample for analysis
            
        Returns
        -------
        np.ndarray
            Correlation matrix of shape (sample_size, sample_size)
        """
        # Sample locations
        num_locs = min(sample_size, self.sdm.config.num_hard_locations)
        sampled_indices = np.random.choice(
            self.sdm.config.num_hard_locations, 
            num_locs, 
            replace=False
        )
        
        # Extract data for correlation
        if self.sdm.config.storage_method == "counters":
            data = self.sdm.counters[sampled_indices]
        else:
            data = self.sdm.binary_storage[sampled_indices].astype(float)
        
        # Compute correlation matrix
        return np.corrcoef(data)
    
    def analyze_recall_quality(self, test_size: int = 100, 
                              noise_levels: List[float] = None) -> Dict[str, List[float]]:
        """
        Analyze recall quality under different noise conditions.
        
        Parameters
        ----------
        test_size : int, optional
            Number of test patterns to use
        noise_levels : list, optional
            List of noise levels to test (default: [0, 0.05, 0.1, 0.2, 0.3])
            
        Returns
        -------
        dict
            Recall quality metrics for each noise level
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
        
        if len(self.sdm._stored_addresses) == 0:
            return {"error": "No patterns stored in memory"}
        
        results = {
            'noise_levels': noise_levels,
            'recall_accuracies': [],
            'recall_success_rates': [],
            'bit_error_rates': []
        }
        
        # Test each noise level
        for noise in noise_levels:
            accuracies = []
            successes = []
            bit_errors = []
            
            # Sample test patterns
            test_indices = np.random.choice(
                len(self.sdm._stored_addresses),
                min(test_size, len(self.sdm._stored_addresses)),
                replace=True
            )
            
            for idx in test_indices:
                original_addr = self.sdm._stored_addresses[idx]
                original_data = self.sdm._stored_data[idx]
                
                # Add noise to address
                if noise > 0:
                    noise_mask = np.random.random(self.sdm.config.dimension) < noise
                    noisy_addr = original_addr.copy()
                    noisy_addr[noise_mask] = 1 - noisy_addr[noise_mask]
                else:
                    noisy_addr = original_addr
                
                # Attempt recall
                recalled = self.sdm.recall(noisy_addr)
                
                if recalled is not None:
                    successes.append(1)
                    accuracy = np.mean(recalled == original_data)
                    accuracies.append(accuracy)
                    bit_errors.append(1 - accuracy)
                else:
                    successes.append(0)
                    bit_errors.append(1.0)
            
            results['recall_accuracies'].append(np.mean(accuracies) if accuracies else 0.0)
            results['recall_success_rates'].append(np.mean(successes))
            results['bit_error_rates'].append(np.mean(bit_errors))
        
        return results
    
    def generate_report(self) -> Dict[str, any]:
        """
        Generate comprehensive statistical report.
        
        Returns
        -------
        dict
            Complete statistical analysis report
        """
        report = {
            'configuration': {
                'dimension': self.sdm.config.dimension,
                'num_hard_locations': self.sdm.config.num_hard_locations,
                'activation_radius': self.sdm.config.activation_radius,
                'storage_method': self.sdm.config.storage_method
            },
            'basic_stats': self.sdm.get_memory_stats(),
            'capacity_analysis': self.contents.get_capacity_estimate(),
            'distribution_analysis': self.contents.analyze_pattern_distribution(sample_size=500),
            'memory_maps': self.contents.get_memory_map()
        }
        
        # Add temporal analysis if available
        if len(self.operation_history) >= 100:
            report['temporal_patterns'] = self.analyze_temporal_patterns()
        
        # Add noise tolerance analysis
        if len(self.sdm._stored_addresses) > 0:
            report['noise_tolerance'] = self.analyze_recall_quality(test_size=50)
        
        # Add correlation analysis for smaller memories
        if self.sdm.config.num_hard_locations <= 1000:
            report['location_correlations'] = {
                'mean': np.mean(self.compute_correlation_matrix()),
                'std': np.std(self.compute_correlation_matrix())
            }
        
        return report
    
    def plot_analysis(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Generate visualization plots for memory analysis.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size for the plots
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('SDM Memory Analysis', fontsize=16)
        
        # Plot 1: Location usage distribution
        ax = axes[0, 0]
        usage_map = self.contents.get_memory_map()['usage_map']
        ax.hist(usage_map[usage_map > 0], bins=30, edgecolor='black')
        ax.set_xlabel('Usage Count')
        ax.set_ylabel('Number of Locations')
        ax.set_title('Location Usage Distribution')
        
        # Plot 2: Activation pattern
        ax = axes[0, 1]
        if len(self.sdm._stored_addresses) > 0:
            sample_addr = self.sdm._stored_addresses[0]
            activated = self.sdm._get_activated_locations(sample_addr)
            activation_pattern = np.zeros(self.sdm.config.num_hard_locations)
            activation_pattern[activated] = 1
            
            ax.imshow(activation_pattern.reshape(-1, 1), aspect='auto', cmap='binary')
            ax.set_xlabel('Activated')
            ax.set_ylabel('Location Index')
            ax.set_title('Sample Activation Pattern')
        
        # Plot 3: Noise tolerance
        ax = axes[0, 2]
        noise_analysis = self.analyze_recall_quality(test_size=50)
        if 'error' not in noise_analysis:
            ax.plot(noise_analysis['noise_levels'], 
                   noise_analysis['recall_accuracies'], 
                   'o-', label='Accuracy')
            ax.plot(noise_analysis['noise_levels'], 
                   noise_analysis['recall_success_rates'], 
                   's-', label='Success Rate')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Performance')
            ax.set_title('Noise Tolerance')
            ax.legend()
            ax.grid(True)
        
        # Plot 4: Memory saturation/density
        ax = axes[1, 0]
        if self.sdm.config.storage_method == "counters":
            saturation = usage_map = self.contents.get_memory_map()['saturation_map']
            ax.hist(saturation, bins=30, edgecolor='black')
            ax.set_xlabel('Saturation Level')
            ax.set_title('Counter Saturation Distribution')
        else:
            density = self.contents.get_memory_map()['density_map']
            ax.hist(density, bins=30, edgecolor='black')
            ax.set_xlabel('Bit Density')
            ax.set_title('Bit Density Distribution')
        ax.set_ylabel('Number of Locations')
        
        # Plot 5: Temporal patterns (if available)
        ax = axes[1, 1]
        if len(self.operation_history) >= 100:
            temporal = self.analyze_temporal_patterns()
            ax.plot(temporal['success_rates'], label='Success Rate')
            ax.set_xlabel('Time Window')
            ax.set_ylabel('Rate')
            ax.set_title('Temporal Success Rate')
            ax.grid(True)
        
        # Plot 6: Capacity utilization
        ax = axes[1, 2]
        capacity = self.contents.get_capacity_estimate()
        labels = ['Used', 'Remaining']
        sizes = [capacity['capacity_used_estimate'], 
                capacity['remaining_capacity_estimate']]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Estimated Capacity Utilization')
        
        plt.tight_layout()
        return fig


class MemoryOptimizer:
    """
    Utilities for optimizing SDM parameters.
    
    This class provides methods for finding optimal SDM configurations
    based on specific use cases and requirements.
    """
    
    @staticmethod
    def find_optimal_radius(dimension: int, num_locations: int, 
                           target_activation: int = None) -> int:
        """
        Find optimal activation radius for given parameters.
        
        Parameters
        ----------
        dimension : int
            Address space dimension
        num_locations : int
            Number of hard locations
        target_activation : int, optional
            Target number of activated locations (default: sqrt(num_locations))
            
        Returns
        -------
        int
            Optimal activation radius
        """
        if target_activation is None:
            target_activation = int(np.sqrt(num_locations))
        
        # Binary search for optimal radius
        low, high = 0, dimension
        best_radius = int(0.451 * dimension)  # Default to critical distance
        
        while low <= high:
            mid = (low + high) // 2
            
            # Calculate expected activations
            p = sum(stats.binom.pmf(k, dimension, 0.5) for k in range(mid + 1))
            expected_activations = num_locations * p
            
            if abs(expected_activations - target_activation) < 1:
                return mid
            elif expected_activations < target_activation:
                low = mid + 1
            else:
                high = mid - 1
                best_radius = mid
        
        return best_radius
    
    @staticmethod
    def estimate_required_locations(dimension: int, capacity: int, 
                                  activation_radius: int = None) -> int:
        """
        Estimate number of hard locations needed for desired capacity.
        
        Parameters
        ----------
        dimension : int
            Address space dimension
        capacity : int
            Desired storage capacity
        activation_radius : int, optional
            Activation radius (default: critical distance)
            
        Returns
        -------
        int
            Estimated number of hard locations needed
        """
        if activation_radius is None:
            activation_radius = int(0.451 * dimension)
        
        # Use Kanerva's capacity formula
        # capacity ≈ 0.15 * num_locations (at critical distance)
        critical_distance = int(0.451 * dimension)
        
        if activation_radius <= critical_distance:
            required_locations = int(capacity / 0.15)
        else:
            # Adjust for larger radius
            ratio = activation_radius / critical_distance
            required_locations = int(capacity / 0.15 * (ratio ** 2))
        
        return required_locations
    
    @staticmethod
    def analyze_parameter_space(dimension_range: Tuple[int, int],
                               location_range: Tuple[int, int],
                               samples: int = 10) -> List[Dict]:
        """
        Analyze SDM performance across parameter space.
        
        Parameters
        ----------
        dimension_range : tuple
            Range of dimensions to test (min, max)
        location_range : tuple
            Range of location counts to test (min, max)
        samples : int, optional
            Number of samples per dimension
            
        Returns
        -------
        list
            Analysis results for each parameter combination
        """
        results = []
        
        dimensions = np.linspace(dimension_range[0], dimension_range[1], samples, dtype=int)
        
        for dim in dimensions:
            num_locs = np.linspace(location_range[0], location_range[1], samples, dtype=int)
            
            for locs in num_locs:
                # Find optimal radius
                radius = MemoryOptimizer.find_optimal_radius(dim, locs)
                
                # Estimate capacity
                critical_dist = int(0.451 * dim)
                if radius <= critical_dist:
                    capacity = int(0.15 * locs)
                else:
                    ratio = radius / critical_dist
                    capacity = int(0.15 * locs / (ratio ** 2))
                
                # Calculate efficiency metrics
                bits_per_location = dim * 8  # Assuming 8-bit counters
                total_bits = locs * bits_per_location
                bits_per_pattern = total_bits / capacity if capacity > 0 else float('inf')
                
                results.append({
                    'dimension': dim,
                    'num_locations': locs,
                    'optimal_radius': radius,
                    'estimated_capacity': capacity,
                    'bits_per_pattern': bits_per_pattern,
                    'efficiency': capacity / locs if locs > 0 else 0
                })
        
        return results