"""
Address decoding mechanisms for Sparse Distributed Memory.

This module implements various address decoding strategies that determine
which hard locations are activated for a given input address. Different
decoders offer trade-offs between performance, capacity, and noise tolerance.

Address Decoder Types:
- HammingDecoder: Classic Hamming distance-based activation
- JaccardDecoder: Jaccard similarity for sparse binary vectors
- RandomDecoder: Random fixed mapping for fast access
- AdaptiveDecoder: Dynamically adjusts activation patterns
- HierarchicalDecoder: Multi-level hierarchical activation
- LSHDecoder: Locality-Sensitive Hashing based decoder

The choice of decoder affects:
- Memory capacity and interference patterns
- Computational complexity of read/write operations
- Noise tolerance and generalization ability
- Suitability for different data distributions
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Set, Callable
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.stats import binom
import warnings
from sklearn.random_projection import GaussianRandomProjection
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DecoderConfig:
    """
    Configuration for address decoders.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of address space
    num_hard_locations : int
        Number of hard locations in memory
    activation_radius : int
        Base activation radius (interpretation varies by decoder)
    seed : int, optional
        Random seed for reproducibility
    """
    dimension: int
    num_hard_locations: int
    activation_radius: int
    seed: Optional[int] = None


class AddressDecoder(ABC):
    """
    Abstract base class for address decoders.
    
    Address decoders determine which hard locations should be activated
    for a given input address. Different decoding strategies offer various
    trade-offs in terms of performance, capacity, and properties.
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration
    hard_locations : np.ndarray
        Array of hard location addresses, shape (num_hard_locations, dimension)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray):
        """Initialize address decoder."""
        self.config = config
        self.hard_locations = hard_locations
        self._initialize()
        
    @abstractmethod
    def _initialize(self):
        """Initialize decoder-specific structures."""
        pass
    
    @abstractmethod
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode an address to activated location indices.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector of shape (dimension,)
            
        Returns
        -------
        np.ndarray
            Indices of activated hard locations
        """
        pass
    
    def decode_batch(self, addresses: np.ndarray) -> List[np.ndarray]:
        """
        Decode multiple addresses in batch.
        
        Parameters
        ----------
        addresses : np.ndarray
            Array of addresses, shape (n_addresses, dimension)
            
        Returns
        -------
        list
            List of activated location indices for each address
        """
        return [self.decode(addr) for addr in addresses]
    
    def get_activation_stats(self, address: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about activation pattern for an address.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        dict
            Activation statistics
        """
        activated = self.decode(address)
        return {
            'num_activated': len(activated),
            'activation_ratio': len(activated) / self.config.num_hard_locations,
            'activated_indices': activated
        }
    
    @abstractmethod
    def expected_activations(self) -> float:
        """Return expected number of activations per address."""
        pass


class HammingDecoder(AddressDecoder):
    """
    Classic Hamming distance-based address decoder.
    
    This decoder activates all hard locations within a fixed Hamming
    distance (activation radius) from the input address. It's the
    original decoder proposed by Kanerva.
    
    Properties:
    - Uniform activation probability across space
    - Good for uniformly distributed data
    - Predictable capacity and interference
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration
    hard_locations : np.ndarray
        Hard location addresses
    use_fast_hamming : bool, optional
        Use optimized Hamming distance computation (default: True)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray,
                 use_fast_hamming: bool = True):
        """Initialize Hamming decoder."""
        self.use_fast_hamming = use_fast_hamming
        super().__init__(config, hard_locations)
    
    def _initialize(self):
        """Initialize Hamming-specific structures."""
        # Precompute hard location norms for fast Hamming distance
        if self.use_fast_hamming:
            self.hard_location_sums = np.sum(self.hard_locations, axis=1)
        
        # Log decoder info
        logger.info(f"HammingDecoder initialized with radius {self.config.activation_radius}")
    
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode address using Hamming distance.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        np.ndarray
            Indices of locations within activation radius
        """
        if self.use_fast_hamming:
            # Fast Hamming distance using precomputed sums
            addr_sum = np.sum(address)
            # Hamming distance = addr_sum + loc_sum - 2 * dot(addr, loc)
            dot_products = self.hard_locations @ address
            distances = addr_sum + self.hard_location_sums - 2 * dot_products
        else:
            # Standard Hamming distance computation
            distances = np.sum(self.hard_locations != address, axis=1)
        
        # Find locations within activation radius
        activated = np.where(distances <= self.config.activation_radius)[0]
        
        return activated
    
    def expected_activations(self) -> float:
        """
        Calculate expected number of activations.
        
        For Hamming decoder, this is based on the binomial distribution
        of Hamming distances in binary space.
        
        Returns
        -------
        float
            Expected number of activated locations
        """
        n = self.config.dimension
        r = self.config.activation_radius
        
        # Probability that a random location is within radius r
        p = sum(binom.pmf(k, n, 0.5) for k in range(r + 1))
        
        return self.config.num_hard_locations * p
    
    def get_activation_distribution(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Analyze activation distribution through sampling.
        
        Parameters
        ----------
        num_samples : int, optional
            Number of random addresses to sample
            
        Returns
        -------
        dict
            Distribution statistics
        """
        activation_counts = []
        
        for _ in range(num_samples):
            # Generate random address
            addr = np.random.randint(0, 2, self.config.dimension)
            activated = self.decode(addr)
            activation_counts.append(len(activated))
        
        return {
            'counts': np.array(activation_counts),
            'mean': np.mean(activation_counts),
            'std': np.std(activation_counts),
            'min': np.min(activation_counts),
            'max': np.max(activation_counts)
        }


class JaccardDecoder(AddressDecoder):
    """
    Jaccard similarity-based address decoder.
    
    This decoder uses Jaccard similarity (intersection over union) for
    activation, which is particularly suitable for sparse binary vectors
    where the number of 1s matters more than the total dimension.
    
    Properties:
    - Better for sparse data
    - Activation depends on bit density
    - More flexible than fixed Hamming radius
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration (radius interpreted as similarity threshold * 1000)
    hard_locations : np.ndarray
        Hard location addresses
    min_similarity : float, optional
        Minimum Jaccard similarity for activation (default: from radius)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray,
                 min_similarity: Optional[float] = None):
        """Initialize Jaccard decoder."""
        if min_similarity is None:
            # Convert radius to similarity threshold (0 to 1)
            self.min_similarity = config.activation_radius / 1000.0
        else:
            self.min_similarity = min_similarity
        
        super().__init__(config, hard_locations)
    
    def _initialize(self):
        """Initialize Jaccard-specific structures."""
        # Precompute number of 1s in each hard location
        self.hard_location_ones = np.sum(self.hard_locations, axis=1)
        
        # Filter out all-zero locations (would cause division by zero)
        self.valid_locations = self.hard_location_ones > 0
        
        if not np.all(self.valid_locations):
            num_invalid = np.sum(~self.valid_locations)
            warnings.warn(f"Found {num_invalid} all-zero hard locations")
        
        logger.info(f"JaccardDecoder initialized with similarity threshold {self.min_similarity}")
    
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode address using Jaccard similarity.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        np.ndarray
            Indices of locations with sufficient similarity
        """
        addr_ones = np.sum(address)
        
        if addr_ones == 0:
            # Special case: all-zero address activates all-zero locations
            return np.where(~self.valid_locations)[0]
        
        # Compute intersections
        intersections = self.hard_locations @ address
        
        # Compute unions
        unions = addr_ones + self.hard_location_ones - intersections
        
        # Compute Jaccard similarities
        similarities = np.zeros(self.config.num_hard_locations)
        valid_mask = unions > 0
        similarities[valid_mask] = intersections[valid_mask] / unions[valid_mask]
        
        # Find locations above threshold
        activated = np.where(similarities >= self.min_similarity)[0]
        
        return activated
    
    def expected_activations(self) -> float:
        """
        Estimate expected number of activations.
        
        This is more complex for Jaccard similarity and depends on
        the bit density distribution.
        
        Returns
        -------
        float
            Estimated expected activations
        """
        # Simplified estimate assuming uniform bit density of 0.5
        # In practice, this would need to consider actual data distribution
        estimated_prob = 0.1 * self.min_similarity  # Rough approximation
        return self.config.num_hard_locations * estimated_prob


class RandomDecoder(AddressDecoder):
    """
    Random hash-based address decoder.
    
    This decoder uses random hash functions to map addresses to a fixed
    set of locations. It provides fast, consistent activation with no
    distance calculations required.
    
    Properties:
    - Very fast (O(1) decoding)
    - Fixed number of activations
    - No distance-based generalization
    - Good for load balancing
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration (radius interpreted as number of activations)
    hard_locations : np.ndarray
        Hard location addresses (not used, but kept for interface compatibility)
    num_hashes : int, optional
        Number of hash functions (default: from radius)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray,
                 num_hashes: Optional[int] = None):
        """Initialize random decoder."""
        self.num_hashes = num_hashes or config.activation_radius
        super().__init__(config, hard_locations)
    
    def _initialize(self):
        """Initialize random hash functions."""
        # Create random projection matrices for hashing
        rng = np.random.RandomState(self.config.seed)
        
        # Each hash function is a random projection
        self.hash_projections = []
        for i in range(self.num_hashes):
            # Random binary projection matrix
            projection = rng.choice([-1, 1], 
                                  size=(self.config.dimension, 32))
            self.hash_projections.append(projection)
        
        logger.info(f"RandomDecoder initialized with {self.num_hashes} hash functions")
    
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode address using random hashing.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        np.ndarray
            Fixed set of activated location indices
        """
        activated_set = set()
        
        for projection in self.hash_projections:
            # Project address and compute hash
            projected = address @ projection
            hash_bits = (projected > 0).astype(np.uint8)
            
            # Convert to integer hash
            hash_value = 0
            for bit in hash_bits:
                hash_value = (hash_value << 1) | bit
            
            # Map to location index
            location_idx = hash_value % self.config.num_hard_locations
            activated_set.add(location_idx)
        
        return np.array(list(activated_set))
    
    def expected_activations(self) -> float:
        """
        Return expected number of activations.
        
        For random decoder, this is deterministic.
        
        Returns
        -------
        float
            Number of hash functions (minus possible collisions)
        """
        # Account for possible hash collisions
        collision_prob = 1 - (1 - 1/self.config.num_hard_locations) ** self.num_hashes
        expected_unique = self.num_hashes * (1 - collision_prob)
        return expected_unique


class AdaptiveDecoder(AddressDecoder):
    """
    Adaptive address decoder that adjusts activation based on memory state.
    
    This decoder dynamically adjusts its activation pattern based on
    the current memory utilization and interference patterns. It can
    expand or contract activation radius to maintain optimal performance.
    
    Properties:
    - Self-adjusting activation
    - Maintains target activation count
    - Adapts to memory load
    - Better capacity utilization
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration
    hard_locations : np.ndarray
        Hard location addresses
    target_activations : int, optional
        Target number of activations to maintain
    adaptation_rate : float, optional
        Rate of adaptation (0 to 1, default: 0.1)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray,
                 target_activations: Optional[int] = None,
                 adaptation_rate: float = 0.1):
        """Initialize adaptive decoder."""
        self.target_activations = target_activations or int(np.sqrt(config.num_hard_locations))
        self.adaptation_rate = adaptation_rate
        
        # Adaptive radius for each location
        self.location_radii = np.full(config.num_hard_locations, 
                                     config.activation_radius, 
                                     dtype=np.float32)
        
        # Track activation history
        self.activation_history = defaultdict(list)
        self.operation_count = 0
        
        super().__init__(config, hard_locations)
    
    def _initialize(self):
        """Initialize adaptive structures."""
        # Base Hamming decoder for distance calculations
        self.base_decoder = HammingDecoder(self.config, self.hard_locations)
        
        logger.info(f"AdaptiveDecoder initialized with target {self.target_activations} activations")
    
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode address with adaptive activation.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        np.ndarray
            Adaptively selected location indices
        """
        # Compute distances to all locations
        distances = np.sum(self.hard_locations != address, axis=1)
        
        # Apply individual location radii
        activated_mask = distances <= self.location_radii
        activated_initial = np.where(activated_mask)[0]
        
        # Adjust if too many or too few activations
        num_activated = len(activated_initial)
        
        if num_activated < self.target_activations * 0.8:
            # Too few activations - expand radius
            # Find next closest locations
            sorted_indices = np.argsort(distances)
            activated = sorted_indices[:self.target_activations]
            
            # Update radii for learning
            for idx in activated:
                if idx not in activated_initial:
                    self.location_radii[idx] = min(
                        self.location_radii[idx] + self.adaptation_rate,
                        self.config.dimension
                    )
        
        elif num_activated > self.target_activations * 1.2:
            # Too many activations - select closest
            distances_activated = distances[activated_initial]
            sorted_local = np.argsort(distances_activated)
            keep_indices = sorted_local[:self.target_activations]
            activated = activated_initial[keep_indices]
            
            # Update radii for learning
            for idx in activated_initial:
                if idx not in activated:
                    self.location_radii[idx] = max(
                        self.location_radii[idx] - self.adaptation_rate,
                        0
                    )
        else:
            activated = activated_initial
        
        # Track activation history
        self.operation_count += 1
        for idx in activated:
            self.activation_history[idx].append(self.operation_count)
        
        return activated
    
    def adapt_radii(self):
        """
        Globally adapt radii based on activation history.
        
        This method should be called periodically to balance
        activation load across locations.
        """
        if self.operation_count == 0:
            return
        
        # Calculate activation frequencies
        frequencies = np.zeros(self.config.num_hard_locations)
        for idx, history in self.activation_history.items():
            frequencies[idx] = len(history) / self.operation_count
        
        # Target frequency
        target_freq = self.target_activations / self.config.num_hard_locations
        
        # Adjust radii based on frequency deviation
        for idx in range(self.config.num_hard_locations):
            if frequencies[idx] > target_freq * 1.5:
                # Over-activated - reduce radius
                self.location_radii[idx] *= (1 - self.adaptation_rate)
            elif frequencies[idx] < target_freq * 0.5:
                # Under-activated - increase radius
                self.location_radii[idx] *= (1 + self.adaptation_rate)
        
        # Clamp radii to valid range
        self.location_radii = np.clip(self.location_radii, 0, self.config.dimension)
        
        logger.debug(f"Adapted radii: mean={np.mean(self.location_radii):.2f}, "
                    f"std={np.std(self.location_radii):.2f}")
    
    def expected_activations(self) -> float:
        """Return target number of activations."""
        return self.target_activations
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """
        Get statistics about adaptation behavior.
        
        Returns
        -------
        dict
            Adaptation statistics
        """
        frequencies = np.zeros(self.config.num_hard_locations)
        for idx, history in self.activation_history.items():
            if self.operation_count > 0:
                frequencies[idx] = len(history) / self.operation_count
        
        return {
            'mean_radius': np.mean(self.location_radii),
            'std_radius': np.std(self.location_radii),
            'min_radius': np.min(self.location_radii),
            'max_radius': np.max(self.location_radii),
            'mean_frequency': np.mean(frequencies[frequencies > 0]),
            'frequency_uniformity': 1.0 - np.std(frequencies) / (np.mean(frequencies) + 1e-10),
            'operation_count': self.operation_count
        }


class HierarchicalDecoder(AddressDecoder):
    """
    Hierarchical address decoder with multi-level activation.
    
    This decoder organizes hard locations in a hierarchy and activates
    locations at multiple levels, allowing for both coarse and fine-grained
    pattern matching.
    
    Properties:
    - Multi-resolution activation
    - Efficient for structured data
    - Natural clustering behavior
    - Scalable to large memories
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration
    hard_locations : np.ndarray
        Hard location addresses
    num_levels : int, optional
        Number of hierarchy levels (default: 3)
    branching_factor : int, optional
        Branching factor at each level (default: 4)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray,
                 num_levels: int = 3, branching_factor: int = 4):
        """Initialize hierarchical decoder."""
        self.num_levels = num_levels
        self.branching_factor = branching_factor
        super().__init__(config, hard_locations)
    
    def _initialize(self):
        """Build hierarchical structure."""
        # Build hierarchy using k-means clustering at each level
        from sklearn.cluster import KMeans
        
        self.hierarchy = []
        self.cluster_assignments = []
        
        # Start with all locations
        current_data = self.hard_locations.copy()
        current_indices = np.arange(self.config.num_hard_locations)
        
        for level in range(self.num_levels - 1):
            n_clusters = min(
                self.branching_factor ** (level + 1),
                len(current_indices) // 2
            )
            
            if n_clusters <= 1:
                break
            
            # Cluster current level
            kmeans = KMeans(n_clusters=n_clusters, 
                           random_state=self.config.seed,
                           n_init=10)
            labels = kmeans.fit_predict(current_data)
            
            # Store cluster information
            level_info = {
                'centers': kmeans.cluster_centers_,
                'labels': labels,
                'indices': current_indices.copy()
            }
            self.hierarchy.append(level_info)
            
            # Prepare next level (cluster centers)
            current_data = kmeans.cluster_centers_
            current_indices = np.arange(n_clusters)
        
        logger.info(f"HierarchicalDecoder built {len(self.hierarchy)} levels")
    
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode address using hierarchical activation.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        np.ndarray
            Hierarchically selected location indices
        """
        activated = set()
        
        # Start from top level and work down
        for level_idx in range(len(self.hierarchy) - 1, -1, -1):
            level = self.hierarchy[level_idx]
            
            # Find closest clusters at this level
            distances = cdist([address], level['centers'], metric='hamming')[0]
            distances *= self.config.dimension  # Convert to actual Hamming distance
            
            # Activate clusters within radius (scaled by level)
            level_radius = self.config.activation_radius * (level_idx + 1) / self.num_levels
            close_clusters = np.where(distances <= level_radius)[0]
            
            if len(close_clusters) == 0:
                # Take closest cluster
                close_clusters = [np.argmin(distances)]
            
            # Find locations in selected clusters
            for cluster_idx in close_clusters:
                if level_idx == 0:
                    # Bottom level - activate actual locations
                    cluster_locs = np.where(level['labels'] == cluster_idx)[0]
                    activated.update(level['indices'][cluster_locs])
                else:
                    # Higher level - continue to next level
                    # This is handled by the loop
                    pass
        
        # Final activation at location level
        if len(activated) == 0:
            # Fallback to closest locations
            distances = cdist([address], self.hard_locations, metric='hamming')[0]
            distances *= self.config.dimension
            activated = set(np.argsort(distances)[:self.config.activation_radius])
        
        return np.array(list(activated))
    
    def expected_activations(self) -> float:
        """
        Estimate expected activations for hierarchical decoder.
        
        Returns
        -------
        float
            Estimated expected activations
        """
        # Estimate based on the hierarchical structure
        # At each level, we expect to activate approximately 1/branching_factor of clusters
        # The final activation count depends on cluster sizes at the bottom level
        
        # For a balanced hierarchy, we expect roughly:
        # num_locations / (branching_factor ^ (num_levels - 1))
        expected_fraction = 1.0 / (self.branching_factor ** (self.num_levels - 1))
        
        # Adjust by activation radius effect (normalized)
        radius_factor = min(1.0, self.config.activation_radius / self.config.dimension)
        
        # Expected activations should be a fraction of total locations
        expected = self.config.num_hard_locations * expected_fraction * (1 + radius_factor)
        
        # Ensure it's at least 1 and at most num_hard_locations
        return max(1.0, min(expected, self.config.num_hard_locations))
    
    def visualize_hierarchy(self) -> Dict[str, np.ndarray]:
        """
        Get hierarchy structure for visualization.
        
        Returns
        -------
        dict
            Hierarchy visualization data
        """
        viz_data = {
            'num_levels': len(self.hierarchy),
            'level_sizes': [],
            'level_centers': []
        }
        
        for level in self.hierarchy:
            viz_data['level_sizes'].append(len(level['centers']))
            viz_data['level_centers'].append(level['centers'])
        
        return viz_data


class LSHDecoder(AddressDecoder):
    """
    Locality-Sensitive Hashing (LSH) based decoder.
    
    This decoder uses LSH to efficiently find similar addresses without
    computing distances to all hard locations. It's particularly efficient
    for very large address spaces.
    
    Properties:
    - Sub-linear query time
    - Probabilistic activation
    - Tunable accuracy/speed trade-off
    - Excellent for large-scale systems
    
    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration
    hard_locations : np.ndarray
        Hard location addresses
    num_tables : int, optional
        Number of LSH hash tables (default: 10)
    hash_size : int, optional
        Size of each hash (default: 8 bits)
    """
    
    def __init__(self, config: DecoderConfig, hard_locations: np.ndarray,
                 num_tables: int = 10, hash_size: int = 8):
        """Initialize LSH decoder."""
        self.num_tables = num_tables
        self.hash_size = hash_size
        super().__init__(config, hard_locations)
    
    def _initialize(self):
        """Initialize LSH structures."""
        rng = np.random.RandomState(self.config.seed)
        
        # Create hash tables
        self.hash_tables = []
        self.hash_functions = []
        
        for table_idx in range(self.num_tables):
            # Random hyperplanes for this table
            hyperplanes = rng.randn(self.hash_size, self.config.dimension)
            hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1, keepdims=True)
            self.hash_functions.append(hyperplanes)
            
            # Hash all hard locations
            hash_table = defaultdict(list)
            for loc_idx, location in enumerate(self.hard_locations):
                # Convert binary to bipolar for LSH
                bipolar_loc = 2 * location - 1
                hash_value = self._compute_hash(bipolar_loc, hyperplanes)
                hash_table[hash_value].append(loc_idx)
            
            self.hash_tables.append(dict(hash_table))
        
        logger.info(f"LSHDecoder initialized with {self.num_tables} tables, "
                   f"{self.hash_size}-bit hashes")
    
    def _compute_hash(self, vector: np.ndarray, hyperplanes: np.ndarray) -> int:
        """
        Compute LSH hash for a vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Input vector (bipolar format)
        hyperplanes : np.ndarray
            Hash function hyperplanes
            
        Returns
        -------
        int
            Hash value
        """
        # Project onto hyperplanes
        projections = hyperplanes @ vector
        
        # Convert to bits
        hash_bits = (projections > 0).astype(np.uint8)
        
        # Convert to integer
        hash_value = 0
        for bit in hash_bits:
            hash_value = (hash_value << 1) | bit
        
        return int(hash_value)
    
    def decode(self, address: np.ndarray) -> np.ndarray:
        """
        Decode address using LSH.
        
        Parameters
        ----------
        address : np.ndarray
            Input address vector
            
        Returns
        -------
        np.ndarray
            LSH-selected location indices
        """
        # Convert to bipolar
        bipolar_addr = 2 * address - 1
        
        # Collect candidates from all hash tables
        candidates = set()
        
        for table_idx in range(self.num_tables):
            hyperplanes = self.hash_functions[table_idx]
            hash_value = self._compute_hash(bipolar_addr, hyperplanes)
            
            # Look up in hash table
            if hash_value in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_value])
            
            # Also check nearby hash values (1-bit differences)
            for bit_flip in range(self.hash_size):
                neighbor_hash = hash_value ^ (1 << bit_flip)
                if neighbor_hash in self.hash_tables[table_idx]:
                    candidates.update(self.hash_tables[table_idx][neighbor_hash])
        
        if not candidates:
            # No candidates found - use random selection
            return np.random.choice(self.config.num_hard_locations, 
                                  size=min(10, self.config.num_hard_locations),
                                  replace=False)
        
        # Verify candidates with actual distance
        candidates = np.array(list(candidates))
        distances = np.sum(self.hard_locations[candidates] != address, axis=1)
        
        # Return those within activation radius
        within_radius = candidates[distances <= self.config.activation_radius]
        
        if len(within_radius) == 0:
            # Return closest candidates
            sorted_candidates = candidates[np.argsort(distances)]
            return sorted_candidates[:min(10, len(sorted_candidates))]
        
        return within_radius
    
    def expected_activations(self) -> float:
        """
        Estimate expected activations for LSH decoder.
        
        Returns
        -------
        float
            Estimated expected activations
        """
        # Depends on hash collision probability
        # This is a rough estimate
        collision_prob = 2 ** (-self.hash_size)
        candidates_per_table = self.config.num_hard_locations * collision_prob
        total_candidates = self.num_tables * candidates_per_table
        
        # Assume some fraction pass distance check
        return min(total_candidates * 0.5, self.config.activation_radius)
    
    def get_hash_statistics(self) -> Dict[str, float]:
        """
        Get statistics about hash table distribution.
        
        Returns
        -------
        dict
            Hash table statistics
        """
        stats = {
            'num_tables': self.num_tables,
            'hash_size': self.hash_size,
            'table_stats': []
        }
        
        for table in self.hash_tables:
            bucket_sizes = [len(locations) for locations in table.values()]
            stats['table_stats'].append({
                'num_buckets': len(table),
                'avg_bucket_size': np.mean(bucket_sizes),
                'max_bucket_size': np.max(bucket_sizes),
                'empty_bucket_ratio': 1 - len(table) / (2 ** self.hash_size)
            })
        
        return stats


def create_decoder(decoder_type: str, config: DecoderConfig, 
                  hard_locations: np.ndarray, **kwargs) -> AddressDecoder:
    """
    Factory function to create address decoders.
    
    Parameters
    ----------
    decoder_type : str
        Type of decoder: 'hamming', 'jaccard', 'random', 'adaptive', 
        'hierarchical', or 'lsh'
    config : DecoderConfig
        Decoder configuration
    hard_locations : np.ndarray
        Hard location addresses
    **kwargs
        Additional decoder-specific parameters
        
    Returns
    -------
    AddressDecoder
        Configured address decoder instance
    """
    decoders = {
        'hamming': HammingDecoder,
        'jaccard': JaccardDecoder,
        'random': RandomDecoder,
        'adaptive': AdaptiveDecoder,
        'hierarchical': HierarchicalDecoder,
        'lsh': LSHDecoder
    }
    
    if decoder_type not in decoders:
        raise ValueError(f"Unknown decoder type: {decoder_type}. "
                        f"Available: {list(decoders.keys())}")
    
    decoder_class = decoders[decoder_type]
    return decoder_class(config, hard_locations, **kwargs)