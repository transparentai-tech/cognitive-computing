"""
Tests for SDM address decoder implementations.

This module contains comprehensive tests for all address decoder strategies:
- HammingDecoder
- JaccardDecoder
- RandomDecoder
- AdaptiveDecoder
- HierarchicalDecoder
- LSHDecoder
"""

import pytest
import numpy as np
from typing import List, Set
import warnings
from unittest.mock import Mock, patch

from cognitive_computing.sdm.address_decoder import (
    DecoderConfig,
    AddressDecoder,
    HammingDecoder,
    JaccardDecoder,
    RandomDecoder,
    AdaptiveDecoder,
    HierarchicalDecoder,
    LSHDecoder,
    create_decoder
)
from cognitive_computing.sdm.core import SDM, SDMConfig
from cognitive_computing.sdm.utils import add_noise, generate_random_patterns


class TestDecoderConfig:
    """Test DecoderConfig dataclass."""
    
    def test_valid_config(self):
        """Test creating valid decoder configuration."""
        config = DecoderConfig(
            dimension=1000,
            num_hard_locations=1000,
            activation_radius=451,
            seed=42
        )
        
        assert config.dimension == 1000
        assert config.num_hard_locations == 1000
        assert config.activation_radius == 451
        assert config.seed == 42
    
    def test_default_seed(self):
        """Test that seed defaults to None."""
        config = DecoderConfig(
            dimension=500,
            num_hard_locations=100,
            activation_radius=225
        )
        assert config.seed is None


class TestHammingDecoder:
    """Test Hamming distance-based decoder."""
    
    @pytest.fixture
    def hamming_decoder(self):
        """Create a Hamming decoder for testing."""
        config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115,
            seed=42
        )
        rng = np.random.RandomState(config.seed)
        hard_locations = rng.randint(0, 2, (config.num_hard_locations, config.dimension))
        return HammingDecoder(config, hard_locations)
    
    def test_initialization(self, hamming_decoder):
        """Test Hamming decoder initialization."""
        assert hamming_decoder.config.dimension == 256
        assert hamming_decoder.config.activation_radius == 115
        assert hamming_decoder.hard_locations.shape == (100, 256)
        
        # Check fast Hamming optimization
        assert hasattr(hamming_decoder, 'hard_location_sums')
        assert len(hamming_decoder.hard_location_sums) == 100
    
    def test_decode_basic(self, hamming_decoder):
        """Test basic decode operation."""
        address = np.random.randint(0, 2, 256)
        activated = hamming_decoder.decode(address)
        
        assert isinstance(activated, np.ndarray)
        assert len(activated) >= 0
        assert len(activated) <= hamming_decoder.config.num_hard_locations
        
        # All indices should be valid
        assert np.all(activated >= 0)
        assert np.all(activated < hamming_decoder.config.num_hard_locations)
    
    def test_decode_distance_threshold(self, hamming_decoder):
        """Test that only locations within radius are activated."""
        address = np.random.randint(0, 2, 256)
        activated = hamming_decoder.decode(address)
        
        # Manually check distances
        for idx in activated:
            distance = np.sum(hamming_decoder.hard_locations[idx] != address)
            assert distance <= hamming_decoder.config.activation_radius
    
    def test_decode_batch(self, hamming_decoder):
        """Test batch decoding."""
        addresses = np.random.randint(0, 2, (5, 256))
        results = hamming_decoder.decode_batch(addresses)
        
        assert len(results) == 5
        assert all(isinstance(r, np.ndarray) for r in results)
    
    def test_expected_activations(self, hamming_decoder):
        """Test expected activation calculation."""
        expected = hamming_decoder.expected_activations()
        
        assert expected > 0
        assert expected < hamming_decoder.config.num_hard_locations
        
        # For 256D with radius 115, should activate reasonable number
        assert 1 < expected < 50
    
    def test_activation_distribution(self, hamming_decoder):
        """Test activation distribution analysis."""
        dist = hamming_decoder.get_activation_distribution(num_samples=100)
        
        assert 'counts' in dist
        assert 'mean' in dist
        assert 'std' in dist
        assert len(dist['counts']) == 100
        
        # Mean should be close to expected
        expected = hamming_decoder.expected_activations()
        assert abs(dist['mean'] - expected) < expected * 0.5
    
    def test_fast_vs_standard_hamming(self):
        """Test that fast and standard Hamming give same results."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=57)
        locations = np.random.randint(0, 2, (50, 128))
        
        # Create decoders with different methods
        decoder_fast = HammingDecoder(config, locations, use_fast_hamming=True)
        decoder_standard = HammingDecoder(config, locations, use_fast_hamming=False)
        
        # Test with same address
        address = np.random.randint(0, 2, 128)
        activated_fast = decoder_fast.decode(address)
        activated_standard = decoder_standard.decode(address)
        
        assert np.array_equal(activated_fast, activated_standard)
    
    def test_get_activation_stats(self, hamming_decoder):
        """Test activation statistics."""
        address = np.random.randint(0, 2, 256)
        stats = hamming_decoder.get_activation_stats(address)
        
        assert 'num_activated' in stats
        assert 'activation_ratio' in stats
        assert 'activated_indices' in stats
        
        assert stats['num_activated'] == len(stats['activated_indices'])
        assert 0 <= stats['activation_ratio'] <= 1


class TestJaccardDecoder:
    """Test Jaccard similarity-based decoder."""
    
    @pytest.fixture
    def jaccard_decoder(self):
        """Create a Jaccard decoder for testing."""
        config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=200,  # Interpreted as similarity * 1000 (0.2 threshold)
            seed=42
        )
        rng = np.random.RandomState(config.seed)
        # Create sparse hard locations for Jaccard
        hard_locations = (rng.random((config.num_hard_locations, config.dimension)) < 0.3).astype(np.uint8)
        return JaccardDecoder(config, hard_locations)
    
    def test_initialization(self, jaccard_decoder):
        """Test Jaccard decoder initialization."""
        assert jaccard_decoder.min_similarity == 0.2  # 200/1000
        assert hasattr(jaccard_decoder, 'hard_location_ones')
        assert hasattr(jaccard_decoder, 'valid_locations')
    
    def test_decode_sparse_data(self, jaccard_decoder):
        """Test decoding with sparse addresses."""
        # Create sparse address
        address = (np.random.random(256) < 0.3).astype(np.uint8)
        activated = jaccard_decoder.decode(address)
        
        assert isinstance(activated, np.ndarray)
        # Should activate some locations with similar sparsity
        assert len(activated) > 0
    
    def test_decode_all_zeros(self, jaccard_decoder):
        """Test decoding with all-zero address."""
        address = np.zeros(256, dtype=np.uint8)
        activated = jaccard_decoder.decode(address)
        
        # Should activate all-zero locations if any
        assert isinstance(activated, np.ndarray)
    
    def test_decode_dense_data(self, jaccard_decoder):
        """Test decoding with dense addresses."""
        # Create dense address
        address = (np.random.random(256) < 0.8).astype(np.uint8)
        activated = jaccard_decoder.decode(address)
        
        assert isinstance(activated, np.ndarray)
    
    def test_similarity_threshold(self, jaccard_decoder):
        """Test that similarity threshold is respected."""
        address = (np.random.random(256) < 0.3).astype(np.uint8)
        activated = jaccard_decoder.decode(address)
        
        # Manually verify similarities
        for idx in activated:
            loc = jaccard_decoder.hard_locations[idx]
            intersection = np.sum(address & loc)
            union = np.sum(address | loc)
            if union > 0:
                similarity = intersection / union
                assert similarity >= jaccard_decoder.min_similarity - 1e-10
    
    def test_custom_similarity(self):
        """Test decoder with custom similarity threshold."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=500)
        locations = np.random.randint(0, 2, (50, 128))
        
        decoder = JaccardDecoder(config, locations, min_similarity=0.8)
        assert decoder.min_similarity == 0.8
    
    def test_expected_activations(self, jaccard_decoder):
        """Test expected activation estimation."""
        expected = jaccard_decoder.expected_activations()
        
        assert expected > 0
        assert expected < jaccard_decoder.config.num_hard_locations
        
        # This is a rough estimate for Jaccard
        assert expected < 20
    
    def test_warning_for_zero_locations(self):
        """Test warning when hard locations contain all zeros."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=700)
        locations = np.random.randint(0, 2, (50, 128))
        # Force some locations to be all zeros
        locations[0] = 0
        locations[1] = 0
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decoder = JaccardDecoder(config, locations)
            
            assert len(w) == 1
            assert "all-zero hard locations" in str(w[0].message)


class TestRandomDecoder:
    """Test random hash-based decoder."""
    
    @pytest.fixture
    def random_decoder(self):
        """Create a Random decoder for testing."""
        config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=10,  # Number of hash functions
            seed=42
        )
        locations = np.random.randint(0, 2, (100, 256))
        return RandomDecoder(config, locations)
    
    def test_initialization(self, random_decoder):
        """Test Random decoder initialization."""
        assert random_decoder.num_hashes == 10
        assert len(random_decoder.hash_projections) == 10
        
        # Each projection should be correct shape
        for proj in random_decoder.hash_projections:
            assert proj.shape == (256, 32)
    
    def test_decode_deterministic(self, random_decoder):
        """Test that decode is deterministic for same input."""
        address = np.random.randint(0, 2, 256)
        
        activated1 = random_decoder.decode(address)
        activated2 = random_decoder.decode(address)
        
        assert np.array_equal(activated1, activated2)
    
    def test_decode_fixed_size(self, random_decoder):
        """Test that activation count is roughly fixed."""
        # Test multiple addresses
        activation_counts = []
        for _ in range(10):
            address = np.random.randint(0, 2, 256)
            activated = random_decoder.decode(address)
            activation_counts.append(len(activated))
        
        # Should be close to num_hashes (allowing for collisions)
        assert all(5 <= count <= 10 for count in activation_counts)
    
    def test_expected_activations(self, random_decoder):
        """Test expected activation calculation."""
        expected = random_decoder.expected_activations()
        
        # Should account for collisions
        assert 0 < expected <= random_decoder.num_hashes
        assert expected > random_decoder.num_hashes * 0.5
    
    def test_custom_num_hashes(self):
        """Test decoder with custom number of hashes."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=5)
        locations = np.random.randint(0, 2, (50, 128))
        
        decoder = RandomDecoder(config, locations, num_hashes=20)
        assert decoder.num_hashes == 20
        assert len(decoder.hash_projections) == 20


class TestAdaptiveDecoder:
    """Test adaptive decoder."""
    
    @pytest.fixture
    def adaptive_decoder(self):
        """Create an Adaptive decoder for testing."""
        config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115,
            seed=42
        )
        rng = np.random.RandomState(config.seed)
        locations = rng.randint(0, 2, (100, 256))
        return AdaptiveDecoder(config, locations, target_activations=10)
    
    def test_initialization(self, adaptive_decoder):
        """Test Adaptive decoder initialization."""
        assert adaptive_decoder.target_activations == 10
        assert adaptive_decoder.adaptation_rate == 0.1
        assert len(adaptive_decoder.location_radii) == 100
        assert np.all(adaptive_decoder.location_radii == 115)
    
    def test_decode_adapts_count(self, adaptive_decoder):
        """Test that decoder adapts activation count."""
        address = np.random.randint(0, 2, 256)
        
        # Initial decode
        activated = adaptive_decoder.decode(address)
        initial_count = len(activated)
        
        # Should try to get close to target
        assert 5 <= len(activated) <= 20  # Within 2x of target
    
    def test_radius_adaptation(self, adaptive_decoder):
        """Test that radii adapt over time."""
        initial_radii = adaptive_decoder.location_radii.copy()
        
        # Decode several addresses
        for _ in range(20):
            address = np.random.randint(0, 2, 256)
            adaptive_decoder.decode(address)
        
        # Some radii should have changed
        assert not np.array_equal(initial_radii, adaptive_decoder.location_radii)
    
    def test_adapt_radii_method(self, adaptive_decoder):
        """Test explicit radius adaptation."""
        # Create some activation history
        for i in range(50):
            address = np.random.randint(0, 2, 256)
            adaptive_decoder.decode(address)
        
        # Adapt radii
        initial_radii = adaptive_decoder.location_radii.copy()
        adaptive_decoder.adapt_radii()
        
        # Check that adaptation occurred
        assert not np.array_equal(initial_radii, adaptive_decoder.location_radii)
        
        # Radii should be within valid range
        assert np.all(adaptive_decoder.location_radii >= 0)
        assert np.all(adaptive_decoder.location_radii <= 256)
    
    def test_expected_activations(self, adaptive_decoder):
        """Test expected activation calculation."""
        expected = adaptive_decoder.expected_activations()
        assert expected == adaptive_decoder.target_activations
    
    def test_get_adaptation_stats(self, adaptive_decoder):
        """Test adaptation statistics."""
        # Generate some history
        for _ in range(30):
            address = np.random.randint(0, 2, 256)
            adaptive_decoder.decode(address)
        
        stats = adaptive_decoder.get_adaptation_stats()
        
        assert 'mean_radius' in stats
        assert 'std_radius' in stats
        assert 'min_radius' in stats
        assert 'max_radius' in stats
        assert 'mean_frequency' in stats
        assert 'frequency_uniformity' in stats
        assert 'operation_count' in stats
        
        assert stats['operation_count'] == 30
        assert stats['min_radius'] <= stats['mean_radius'] <= stats['max_radius']
    
    def test_custom_adaptation_rate(self):
        """Test decoder with custom adaptation rate."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=57)
        locations = np.random.randint(0, 2, (50, 128))
        
        decoder = AdaptiveDecoder(config, locations, adaptation_rate=0.5)
        assert decoder.adaptation_rate == 0.5


class TestHierarchicalDecoder:
    """Test hierarchical decoder."""
    
    @pytest.fixture
    def hierarchical_decoder(self):
        """Create a Hierarchical decoder for testing."""
        config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115,
            seed=42
        )
        rng = np.random.RandomState(config.seed)
        locations = rng.randint(0, 2, (100, 256))
        return HierarchicalDecoder(config, locations, num_levels=3, branching_factor=4)
    
    def test_initialization(self, hierarchical_decoder):
        """Test Hierarchical decoder initialization."""
        assert hierarchical_decoder.num_levels == 3
        assert hierarchical_decoder.branching_factor == 4
        assert len(hierarchical_decoder.hierarchy) > 0
    
    def test_decode_returns_valid_indices(self, hierarchical_decoder):
        """Test that decode returns valid location indices."""
        address = np.random.randint(0, 2, 256)
        activated = hierarchical_decoder.decode(address)
        
        assert isinstance(activated, np.ndarray)
        assert len(activated) > 0
        assert np.all(activated >= 0)
        assert np.all(activated < 100)
    
    def test_hierarchical_structure(self, hierarchical_decoder):
        """Test that hierarchy is properly structured."""
        hierarchy = hierarchical_decoder.hierarchy
        
        # Each level should have centers, labels, and indices
        for level in hierarchy:
            assert 'centers' in level
            assert 'labels' in level
            assert 'indices' in level
            
            # Centers should be correct dimension
            assert level['centers'].shape[1] == 256
    
    def test_expected_activations(self, hierarchical_decoder):
        """Test expected activation estimation."""
        expected = hierarchical_decoder.expected_activations()
        
        assert expected > 0
        assert expected < hierarchical_decoder.config.num_hard_locations
    
    def test_visualize_hierarchy(self, hierarchical_decoder):
        """Test hierarchy visualization data."""
        viz_data = hierarchical_decoder.visualize_hierarchy()
        
        assert 'num_levels' in viz_data
        assert 'level_sizes' in viz_data
        assert 'level_centers' in viz_data
        
        assert viz_data['num_levels'] == len(hierarchical_decoder.hierarchy)
        assert len(viz_data['level_sizes']) == len(hierarchical_decoder.hierarchy)
    
    def test_decode_with_small_locations(self):
        """Test hierarchical decoder with very few locations."""
        config = DecoderConfig(dimension=128, num_hard_locations=10, activation_radius=57)
        locations = np.random.randint(0, 2, (10, 128))
        
        decoder = HierarchicalDecoder(config, locations, num_levels=2, branching_factor=3)
        
        address = np.random.randint(0, 2, 128)
        activated = decoder.decode(address)
        
        assert len(activated) > 0
        assert len(activated) <= 10


class TestLSHDecoder:
    """Test LSH-based decoder."""
    
    @pytest.fixture
    def lsh_decoder(self):
        """Create an LSH decoder for testing."""
        config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115,
            seed=42
        )
        rng = np.random.RandomState(config.seed)
        locations = rng.randint(0, 2, (100, 256))
        return LSHDecoder(config, locations, num_tables=5, hash_size=8)
    
    def test_initialization(self, lsh_decoder):
        """Test LSH decoder initialization."""
        assert lsh_decoder.num_tables == 5
        assert lsh_decoder.hash_size == 8
        assert len(lsh_decoder.hash_tables) == 5
        assert len(lsh_decoder.hash_functions) == 5
        
        # Each hash function should have correct shape
        for hyperplanes in lsh_decoder.hash_functions:
            assert hyperplanes.shape == (8, 256)
    
    def test_decode_returns_candidates(self, lsh_decoder):
        """Test that decode returns candidate locations."""
        address = np.random.randint(0, 2, 256)
        activated = lsh_decoder.decode(address)
        
        assert isinstance(activated, np.ndarray)
        assert len(activated) > 0
        assert np.all(activated >= 0)
        assert np.all(activated < 100)
    
    def test_hash_computation(self, lsh_decoder):
        """Test hash computation."""
        vector = np.random.randn(256)  # Bipolar vector
        hyperplanes = lsh_decoder.hash_functions[0]
        
        hash_value = lsh_decoder._compute_hash(vector, hyperplanes)
        
        assert isinstance(hash_value, int)
        assert 0 <= hash_value < 2**8
    
    def test_decode_with_no_candidates(self, lsh_decoder):
        """Test decode when no candidates found in hash tables."""
        # Mock empty hash tables
        lsh_decoder.hash_tables = [{} for _ in range(lsh_decoder.num_tables)]
        
        address = np.random.randint(0, 2, 256)
        activated = lsh_decoder.decode(address)
        
        # Should return random selection
        assert len(activated) > 0
        assert len(activated) <= 10
    
    def test_expected_activations(self, lsh_decoder):
        """Test expected activation estimation."""
        expected = lsh_decoder.expected_activations()
        
        assert expected > 0
        assert expected < lsh_decoder.config.num_hard_locations
    
    def test_get_hash_statistics(self, lsh_decoder):
        """Test hash table statistics."""
        stats = lsh_decoder.get_hash_statistics()
        
        assert 'num_tables' in stats
        assert 'hash_size' in stats
        assert 'table_stats' in stats
        
        assert stats['num_tables'] == 5
        assert stats['hash_size'] == 8
        assert len(stats['table_stats']) == 5
        
        # Each table should have statistics
        for table_stat in stats['table_stats']:
            assert 'num_buckets' in table_stat
            assert 'avg_bucket_size' in table_stat
            assert 'max_bucket_size' in table_stat
            assert 'empty_bucket_ratio' in table_stat


class TestCreateDecoder:
    """Test decoder factory function."""
    
    def test_create_hamming_decoder(self):
        """Test creating Hamming decoder via factory."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=57)
        locations = np.random.randint(0, 2, (50, 128))
        
        decoder = create_decoder('hamming', config, locations)
        assert isinstance(decoder, HammingDecoder)
    
    def test_create_all_decoders(self):
        """Test creating all decoder types."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=57)
        locations = np.random.randint(0, 2, (50, 128))
        
        decoder_types = {
            'hamming': HammingDecoder,
            'jaccard': JaccardDecoder,
            'random': RandomDecoder,
            'adaptive': AdaptiveDecoder,
            'hierarchical': HierarchicalDecoder,
            'lsh': LSHDecoder
        }
        
        for name, expected_class in decoder_types.items():
            decoder = create_decoder(name, config, locations)
            assert isinstance(decoder, expected_class)
    
    def test_create_decoder_with_kwargs(self):
        """Test creating decoder with additional parameters."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=57)
        locations = np.random.randint(0, 2, (50, 128))
        
        # Create adaptive decoder with custom parameters
        decoder = create_decoder(
            'adaptive', 
            config, 
            locations,
            target_activations=15,
            adaptation_rate=0.2
        )
        
        assert isinstance(decoder, AdaptiveDecoder)
        assert decoder.target_activations == 15
        assert decoder.adaptation_rate == 0.2
    
    def test_invalid_decoder_type(self):
        """Test that invalid decoder type raises error."""
        config = DecoderConfig(dimension=128, num_hard_locations=50, activation_radius=57)
        locations = np.random.randint(0, 2, (50, 128))
        
        with pytest.raises(ValueError, match="Unknown decoder type"):
            create_decoder('invalid', config, locations)


class TestDecoderIntegration:
    """Integration tests for decoders with SDM."""
    
    @pytest.mark.parametrize("decoder_type", ['hamming', 'jaccard', 'random', 'adaptive', 'hierarchical', 'lsh'])
    def test_decoder_with_sdm(self, decoder_type):
        """Test that all decoders work with SDM."""
        # Create SDM configuration
        sdm_config = SDMConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115,
            seed=42
        )
        sdm = SDM(sdm_config)
        
        # Create decoder
        decoder_config = DecoderConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115,
            seed=42
        )
        
        # For Jaccard, use appropriate threshold (similarity * 1000)
        if decoder_type == 'jaccard':
            decoder_config.activation_radius = 200
        
        decoder = create_decoder(decoder_type, decoder_config, sdm.hard_locations)
        
        # Test basic operations
        address = np.random.randint(0, 2, 256)
        activated = decoder.decode(address)
        
        assert len(activated) > 0
        assert len(activated) <= 100
        
        # Store and recall pattern
        data = np.random.randint(0, 2, 256)
        sdm.store(address, data)
        recalled = sdm.recall(address)
        
        assert recalled is not None
    
    def test_decoder_consistency(self):
        """Test that similar addresses activate overlapping locations."""
        config = DecoderConfig(dimension=256, num_hard_locations=100, activation_radius=115)
        locations = np.random.randint(0, 2, (100, 256))
        
        # Test with Hamming decoder
        decoder = HammingDecoder(config, locations)
        
        # Create similar addresses
        address1 = np.random.randint(0, 2, 256)
        address2 = address1.copy()
        # Flip a few bits
        flip_indices = np.random.choice(256, 10, replace=False)
        address2[flip_indices] = 1 - address2[flip_indices]
        
        activated1 = set(decoder.decode(address1))
        activated2 = set(decoder.decode(address2))
        
        # Should have significant overlap
        overlap = len(activated1 & activated2)
        assert overlap > 0
        assert overlap > len(activated1) * 0.5
    
    def test_decoder_performance_comparison(self):
        """Compare performance characteristics of different decoders."""
        config = DecoderConfig(dimension=512, num_hard_locations=200, activation_radius=230)
        locations = np.random.randint(0, 2, (200, 512))
        
        decoders = {
            'hamming': HammingDecoder(config, locations),
            'random': RandomDecoder(config, locations, num_hashes=20),
            'lsh': LSHDecoder(config, locations, num_tables=10)
        }
        
        # Test activation counts
        address = np.random.randint(0, 2, 512)
        
        results = {}
        for name, decoder in decoders.items():
            activated = decoder.decode(address)
            results[name] = len(activated)
        
        # Random should have most consistent count
        # Hamming should vary based on actual distances
        # LSH should be somewhere in between
        assert results['random'] <= 20  # Limited by num_hashes
        assert results['hamming'] > 0
        assert results['lsh'] > 0