"""
Core implementation of Sparse Distributed Memory (SDM).

This module contains the main SDM class and configuration, implementing
Kanerva's Sparse Distributed Memory algorithm. SDM is a content-addressable
memory system that exhibits properties similar to human long-term memory.

Key Concepts:
- Hard Locations: Fixed random addresses in the memory space
- Activation Radius: Hamming distance threshold for activating locations
- Distributed Storage: Data is stored across multiple activated locations
- Superposition: Multiple patterns can be stored in overlapping locations
- Auto-association: Ability to recall complete patterns from partial/noisy inputs

Mathematical Foundation:
- Address space: {0,1}^n (n-dimensional binary vectors)
- Critical distance: ~0.451n for optimal performance
- Capacity: ~0.15 * num_hard_locations patterns

References:
    Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
    Kanerva, P. (1993). Sparse Distributed Memory and Related Models.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from copy import deepcopy
import time

from cognitive_computing.common.base import (
    CognitiveMemory, 
    MemoryConfig, 
    BinaryVector,
    MemoryPerformanceMetrics,
    DistanceMetric
)

logger = logging.getLogger(__name__)


@dataclass
class SDMConfig(MemoryConfig):
    """
    Configuration for Sparse Distributed Memory.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of address and data vectors
    num_hard_locations : int
        Number of hard (physical) memory locations
    activation_radius : int
        Hamming radius for location activation
    threshold : float, optional
        Threshold for reading from counters (default: 0.0)
    storage_method : str, optional
        Method for storing data: 'counters' or 'binary' (default: 'counters')
    parallel : bool, optional
        Whether to use parallel processing (default: False)
    num_workers : int, optional
        Number of worker threads for parallel processing (default: 4)
    counter_bits : int, optional
        Number of bits per counter (default: 8)
    saturation_value : int, optional
        Maximum absolute value for counters (default: 127)
    
    Attributes
    ----------
    capacity : int
        Theoretical capacity estimate based on parameters
    critical_distance : int
        Critical distance for the given dimension
    """
    
    num_hard_locations: int = 1000
    activation_radius: int = 451
    threshold: float = 0.0
    storage_method: str = "counters"  # "counters" or "binary"
    parallel: bool = False
    num_workers: int = 4
    counter_bits: int = 8
    saturation_value: int = 127
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        super().__post_init__()
        
        # Validate parameters
        if self.num_hard_locations <= 0:
            raise ValueError(f"num_hard_locations must be positive, got {self.num_hard_locations}")
        
        if not 0 <= self.activation_radius <= self.dimension:
            raise ValueError(f"activation_radius must be in [0, {self.dimension}], got {self.activation_radius}")
        
        if self.storage_method not in ["counters", "binary"]:
            raise ValueError(f"storage_method must be 'counters' or 'binary', got {self.storage_method}")
        
        if self.counter_bits < 1 or self.counter_bits > 32:
            raise ValueError(f"counter_bits must be in [1, 32], got {self.counter_bits}")
        
        # Compute derived parameters
        self.critical_distance = int(0.451 * self.dimension)
        
        # Estimate capacity (Kanerva's formula)
        # Capacity ≈ 0.15 * num_hard_locations when activation_radius ≈ critical_distance
        if self.activation_radius <= self.critical_distance:
            self.capacity = int(0.15 * self.num_hard_locations)
        else:
            # Reduced capacity for larger activation radius
            ratio = self.activation_radius / self.critical_distance
            self.capacity = int(0.15 * self.num_hard_locations / (ratio ** 2))
        
        logger.info(f"SDM configured: dimension={self.dimension}, "
                   f"locations={self.num_hard_locations}, "
                   f"radius={self.activation_radius}, "
                   f"estimated_capacity={self.capacity}")


class SDM(CognitiveMemory):
    """
    Sparse Distributed Memory implementation.
    
    This class implements Kanerva's SDM algorithm with support for both
    counter-based and binary storage methods. The memory uses randomly
    distributed hard locations and activates locations within a Hamming
    radius of the input address.
    
    Parameters
    ----------
    config : SDMConfig
        Configuration object containing SDM parameters
        
    Attributes
    ----------
    hard_locations : np.ndarray
        Array of hard location addresses, shape (num_hard_locations, dimension)
    counters : np.ndarray or None
        Counter array for storage method 'counters', shape (num_hard_locations, dimension)
    binary_storage : np.ndarray or None
        Binary storage for method 'binary', shape (num_hard_locations, dimension)
    location_usage : np.ndarray
        Count of how many times each location has been activated
    metrics : MemoryPerformanceMetrics
        Performance tracking metrics
        
    Examples
    --------
    >>> # Create SDM with 1000-bit addresses
    >>> config = SDMConfig(dimension=1000, num_hard_locations=1000, activation_radius=451)
    >>> sdm = SDM(config)
    >>> 
    >>> # Store a pattern
    >>> address = np.random.randint(0, 2, 1000)
    >>> data = np.random.randint(0, 2, 1000)
    >>> sdm.store(address, data)
    >>> 
    >>> # Recall the pattern
    >>> recalled = sdm.recall(address)
    >>> accuracy = np.mean(recalled == data)
    """
    
    def __init__(self, config: SDMConfig):
        """Initialize SDM with given configuration."""
        self.config = deepcopy(config)
        self.metrics = MemoryPerformanceMetrics()
        super().__init__(config)
        
    def _initialize(self):
        """Initialize SDM internal structures."""
        # Generate random hard locations
        self.hard_locations = BinaryVector.random(
            dimension=self.config.dimension * self.config.num_hard_locations,
            density=0.5,
            seed=self.config.seed
        ).reshape(self.config.num_hard_locations, self.config.dimension)
        
        # Initialize storage based on method
        if self.config.storage_method == "counters":
            # Initialize counters to zero
            self.counters = np.zeros(
                (self.config.num_hard_locations, self.config.dimension),
                dtype=np.int16
            )
            self.binary_storage = None
        else:  # binary
            # Initialize binary storage
            self.binary_storage = np.zeros(
                (self.config.num_hard_locations, self.config.dimension),
                dtype=np.uint8
            )
            self.counters = None
        
        # Track location usage statistics
        self.location_usage = np.zeros(self.config.num_hard_locations, dtype=np.int32)
        
        # Track stored patterns for analysis
        self._stored_addresses = []
        self._stored_data = []
        
        # Initialize parallel executor if needed
        if self.config.parallel:
            self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.executor = None
            
        logger.debug(f"SDM initialized with {self.config.num_hard_locations} hard locations")
    
    def _get_activated_locations(self, address: np.ndarray) -> np.ndarray:
        """
        Get indices of activated hard locations for given address.
        
        Parameters
        ----------
        address : np.ndarray
            Query address vector
            
        Returns
        -------
        np.ndarray
            Indices of activated locations
        """
        if self.config.parallel and self.executor is not None:
            return self._get_activated_locations_parallel(address)
        else:
            return self._get_activated_locations_sequential(address)
    
    def _get_activated_locations_sequential(self, address: np.ndarray) -> np.ndarray:
        """Sequential implementation of location activation."""
        # Compute Hamming distances to all hard locations
        distances = np.sum(self.hard_locations != address, axis=1)
        
        # Find locations within activation radius
        activated = np.where(distances <= self.config.activation_radius)[0]
        
        return activated
    
    def _get_activated_locations_parallel(self, address: np.ndarray) -> np.ndarray:
        """Parallel implementation of location activation."""
        def compute_chunk(start_idx, end_idx):
            distances = np.sum(self.hard_locations[start_idx:end_idx] != address, axis=1)
            local_activated = np.where(distances <= self.config.activation_radius)[0]
            return local_activated + start_idx
        
        # Split work into chunks
        chunk_size = max(1, self.config.num_hard_locations // self.config.num_workers)
        futures = []
        
        for i in range(0, self.config.num_hard_locations, chunk_size):
            end_idx = min(i + chunk_size, self.config.num_hard_locations)
            future = self.executor.submit(compute_chunk, i, end_idx)
            futures.append(future)
        
        # Collect results
        activated_locations = []
        for future in as_completed(futures):
            activated_locations.extend(future.result())
        
        return np.array(activated_locations)
    
    def store(self, address: np.ndarray, data: np.ndarray) -> None:
        """
        Store a data pattern at the given address.
        
        The data is distributed across all activated hard locations.
        For counter-based storage, the counters are incremented/decremented
        based on the data bits. For binary storage, the data is OR-ed.
        
        Parameters
        ----------
        address : np.ndarray
            Address vector of shape (dimension,)
        data : np.ndarray
            Data vector of shape (dimension,)
        """
        start_time = time.time()
        
        # Validate inputs
        if address.shape != (self.config.dimension,):
            raise ValueError(f"Address shape must be ({self.config.dimension},), got {address.shape}")
        if data.shape != (self.config.dimension,):
            raise ValueError(f"Data shape must be ({self.config.dimension},), got {data.shape}")
        
        # Get activated locations
        activated_locations = self._get_activated_locations(address)
        
        if len(activated_locations) == 0:
            warnings.warn("No locations activated for the given address")
            return
        
        # Store data in activated locations
        if self.config.storage_method == "counters":
            # Convert binary data to bipolar (-1, +1) for counter updates
            bipolar_data = BinaryVector.to_bipolar(data)
            
            # Update counters at activated locations
            for loc_idx in activated_locations:
                self.counters[loc_idx] += bipolar_data
                
                # Apply saturation to prevent overflow
                self.counters[loc_idx] = np.clip(
                    self.counters[loc_idx],
                    -self.config.saturation_value,
                    self.config.saturation_value
                )
        else:  # binary storage
            # OR the data into activated locations
            for loc_idx in activated_locations:
                self.binary_storage[loc_idx] = np.logical_or(
                    self.binary_storage[loc_idx], data
                ).astype(np.uint8)
        
        # Update usage statistics
        self.location_usage[activated_locations] += 1
        
        # Store pattern for analysis (limited to prevent memory issues)
        if len(self._stored_addresses) < 1000:
            self._stored_addresses.append(address.copy())
            self._stored_data.append(data.copy())
        
        # Record metrics
        elapsed_time = time.time() - start_time
        self.metrics.record_store(elapsed_time)
        
        logger.debug(f"Stored pattern in {len(activated_locations)} locations "
                    f"(took {elapsed_time:.4f}s)")
    
    def recall(self, address: np.ndarray) -> Optional[np.ndarray]:
        """
        Recall data from the given address.
        
        The recall process activates locations near the address and
        reads out the superimposed data, using majority voting or
        thresholding to reconstruct the original pattern.
        
        Parameters
        ----------
        address : np.ndarray
            Address vector of shape (dimension,)
            
        Returns
        -------
        np.ndarray or None
            Recalled data vector of shape (dimension,), or None if no data found
        """
        start_time = time.time()
        
        # Validate input
        if address.shape != (self.config.dimension,):
            raise ValueError(f"Address shape must be ({self.config.dimension},), got {address.shape}")
        
        # Get activated locations
        activated_locations = self._get_activated_locations(address)
        
        if len(activated_locations) == 0:
            logger.debug("No locations activated for recall")
            self.metrics.record_recall(time.time() - start_time, success=False)
            return None
        
        # Read and sum data from activated locations
        if self.config.storage_method == "counters":
            # Sum counters from activated locations
            sum_counters = np.sum(self.counters[activated_locations], axis=0)
            
            # Apply threshold to get binary output
            recalled_data = (sum_counters > self.config.threshold).astype(np.uint8)
        else:  # binary storage
            # Majority voting on binary values
            sum_values = np.sum(self.binary_storage[activated_locations], axis=0)
            recalled_data = (sum_values > len(activated_locations) // 2).astype(np.uint8)
        
        # Record metrics
        elapsed_time = time.time() - start_time
        self.metrics.record_recall(elapsed_time, success=True)
        
        logger.debug(f"Recalled pattern from {len(activated_locations)} locations "
                    f"(took {elapsed_time:.4f}s)")
        
        return recalled_data
    
    def recall_with_confidence(self, address: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Recall data with confidence scores for each bit.
        
        Parameters
        ----------
        address : np.ndarray
            Address vector of shape (dimension,)
            
        Returns
        -------
        data : np.ndarray or None
            Recalled data vector, or None if no data found
        confidence : np.ndarray
            Confidence scores for each bit (0 to 1)
        """
        # Get activated locations
        activated_locations = self._get_activated_locations(address)
        
        if len(activated_locations) == 0:
            return None, np.zeros(self.config.dimension)
        
        if self.config.storage_method == "counters":
            # Sum counters from activated locations
            sum_counters = np.sum(self.counters[activated_locations], axis=0)
            
            # For confidence, calculate based on the agreement among activated locations
            # When a single pattern is stored, counters will be -1 or 1
            # Perfect recall should give high confidence
            
            # Simple approach: confidence based on absolute sum normalized by voters
            # If all locations vote the same way, abs(sum) = num_locations
            # If votes are split, abs(sum) < num_locations
            num_activated = len(activated_locations)
            confidence = np.abs(sum_counters) / num_activated
            
            # Since counters can accumulate over multiple stores, cap at 1
            confidence = np.clip(confidence, 0, 1)
            
            # Apply threshold to get binary output
            recalled_data = (sum_counters > self.config.threshold).astype(np.uint8)
        else:  # binary storage
            # Count votes for each bit
            sum_values = np.sum(self.binary_storage[activated_locations], axis=0)
            total_votes = len(activated_locations)
            
            # Confidence is how far from 50/50 the vote is
            confidence = np.abs(sum_values - total_votes / 2) / (total_votes / 2)
            confidence = np.clip(confidence, 0, 1)
            
            # Majority voting
            recalled_data = (sum_values > total_votes // 2).astype(np.uint8)
        
        return recalled_data, confidence
    
    def clear(self) -> None:
        """Clear all stored data from memory."""
        if self.config.storage_method == "counters":
            self.counters.fill(0)
        else:
            self.binary_storage.fill(0)
        
        self.location_usage.fill(0)
        self._stored_addresses.clear()
        self._stored_data.clear()
        self.metrics.reset()
        
        logger.info("SDM memory cleared")
    
    @property
    def size(self) -> int:
        """Return the number of stored patterns."""
        return len(self._stored_addresses)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get statistics about memory usage and distribution.
        
        Returns
        -------
        dict
            Dictionary containing various memory statistics
        """
        stats = {
            "num_patterns_stored": self.size,
            "num_hard_locations": self.config.num_hard_locations,
            "activation_radius": self.config.activation_radius,
            "dimension": self.config.dimension,
        }
        
        # Location usage statistics
        if np.any(self.location_usage > 0):
            stats["locations_used"] = np.sum(self.location_usage > 0)
            stats["avg_location_usage"] = np.mean(self.location_usage[self.location_usage > 0])
            stats["max_location_usage"] = np.max(self.location_usage)
            stats["location_usage_std"] = np.std(self.location_usage[self.location_usage > 0])
        else:
            stats["locations_used"] = 0
            stats["avg_location_usage"] = 0
            stats["max_location_usage"] = 0
            stats["location_usage_std"] = 0
        
        # Storage statistics
        if self.config.storage_method == "counters":
            stats["avg_counter_magnitude"] = np.mean(np.abs(self.counters))
            stats["max_counter_value"] = np.max(np.abs(self.counters))
            stats["counter_saturation_rate"] = np.mean(
                np.abs(self.counters) == self.config.saturation_value
            )
        else:
            stats["avg_bit_density"] = np.mean(self.binary_storage)
            stats["fully_saturated_bits"] = np.sum(
                np.all(self.binary_storage == 1, axis=0)
            )
        
        # Performance metrics
        stats.update(self.metrics.summary())
        
        return stats
    
    def analyze_crosstalk(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Analyze crosstalk between stored patterns.
        
        Parameters
        ----------
        num_samples : int, optional
            Number of pattern pairs to sample for analysis
            
        Returns
        -------
        dict
            Crosstalk analysis results
        """
        if self.size < 2:
            return {"error": "Need at least 2 stored patterns for crosstalk analysis"}
        
        # Sample pattern pairs
        num_pairs = min(num_samples, self.size * (self.size - 1) // 2)
        sampled_pairs = []
        
        for _ in range(num_pairs):
            i, j = np.random.choice(self.size, 2, replace=False)
            addr_i, data_i = self._stored_addresses[i], self._stored_data[i]
            addr_j, data_j = self._stored_addresses[j], self._stored_data[j]
            
            # Check if patterns share activated locations
            locs_i = set(self._get_activated_locations(addr_i))
            locs_j = set(self._get_activated_locations(addr_j))
            
            overlap = len(locs_i & locs_j)
            if overlap > 0:
                # Measure interference
                recalled_i = self.recall(addr_i)
                recalled_j = self.recall(addr_j)
                
                if recalled_i is not None and recalled_j is not None:
                    error_i = np.mean(recalled_i != data_i)
                    error_j = np.mean(recalled_j != data_j)
                    
                    sampled_pairs.append({
                        "overlap": overlap,
                        "error_i": error_i,
                        "error_j": error_j,
                        "avg_error": (error_i + error_j) / 2
                    })
        
        if not sampled_pairs:
            return {"error": "No overlapping pattern pairs found"}
        
        # Compute statistics
        avg_overlap = np.mean([p["overlap"] for p in sampled_pairs])
        avg_error = np.mean([p["avg_error"] for p in sampled_pairs])
        max_error = np.max([p["avg_error"] for p in sampled_pairs])
        
        # Compute correlation safely
        overlaps = [p["overlap"] for p in sampled_pairs]
        errors = [p["avg_error"] for p in sampled_pairs]
        
        # Check if correlation can be computed (need variance in both arrays)
        try:
            # Need at least 2 different values in each array for correlation
            if len(set(overlaps)) > 1 and len(set(errors)) > 1:
                correlation = np.corrcoef(overlaps, errors)[0, 1]
            else:
                # Cannot compute correlation if one or both arrays have no variance
                correlation = np.nan
        except (FloatingPointError, RuntimeWarning, ValueError):
            # Handle any other numerical issues
            correlation = np.nan
        
        return {
            "num_pairs_analyzed": len(sampled_pairs),
            "avg_location_overlap": avg_overlap,
            "avg_recall_error": avg_error,
            "max_recall_error": max_error,
            "correlation": correlation
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=False)