"""
Tests for SDM utility functions.

This module contains comprehensive tests for all utility functions including:
- Pattern generation and manipulation
- Noise addition
- Capacity calculations
- Performance testing
- Data encoding
- Pattern similarity
- Save/load functionality
"""

import pytest
import numpy as np
import tempfile
import os
from typing import List
import time
from unittest.mock import Mock, patch

from cognitive_computing.sdm.utils import (
    add_noise,
    generate_random_patterns,
    compute_memory_capacity,
    analyze_activation_patterns,
    test_sdm_performance,
    calculate_pattern_similarity,
    create_orthogonal_patterns,
    PatternEncoder,
    save_sdm_state,
    load_sdm_state,
    PerformanceTestResult
)
from cognitive_computing.sdm.core import SDM, SDMConfig


class TestAddNoise:
    """Test noise addition functions."""
    
    def test_flip_noise(self):
        """Test random bit flip noise."""
        pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 32)  # 256 bits
        noise_level = 0.1
        
        noisy = add_noise(pattern, noise_level, noise_type='flip', seed=42)
        
        # Check that some bits flipped
        differences = np.sum(pattern != noisy)
        expected_flips = len(pattern) * noise_level
        
        # Should be close to expected (with some variance)
        assert 0.5 * expected_flips <= differences <= 1.5 * expected_flips
        
        # Check reproducibility with seed
        noisy2 = add_noise(pattern, noise_level, noise_type='flip', seed=42)
        assert np.array_equal(noisy, noisy2)
    
    def test_swap_noise(self):
        """Test bit swap noise."""
        pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 32)
        noise_level = 0.2
        
        noisy = add_noise(pattern, noise_level, noise_type='swap')
        
        # Total number of 1s should remain the same
        assert np.sum(pattern) == np.sum(noisy)
        
        # But positions should change
        assert not np.array_equal(pattern, noisy)
    
    def test_burst_noise(self):
        """Test burst error noise."""
        pattern = np.zeros(256, dtype=np.uint8)
        noise_level = 0.1  # 10% burst length
        
        noisy = add_noise(pattern, noise_level, noise_type='burst', seed=42)
        
        # Should have contiguous flipped bits
        burst_length = int(256 * noise_level)
        differences = pattern != noisy
        
        # Find the burst
        diff_indices = np.where(differences)[0]
        if len(diff_indices) > 0:
            # Check contiguity
            assert np.max(diff_indices) - np.min(diff_indices) + 1 == len(diff_indices)
            assert len(diff_indices) == burst_length
    
    def test_salt_pepper_noise(self):
        """Test salt and pepper noise."""
        pattern = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 32)
        noise_level = 0.2
        
        noisy = add_noise(pattern, noise_level, noise_type='salt_pepper', seed=42)
        
        # Some bits should be forced to 0 or 1
        differences = np.sum(pattern != noisy)
        assert differences > 0
        
        # Unlike flip noise, this can set bits to same value
        # so we just check that pattern changed
        assert not np.array_equal(pattern, noisy)
    
    def test_invalid_noise_level(self):
        """Test that invalid noise levels raise errors."""
        pattern = np.zeros(256)
        
        with pytest.raises(ValueError, match="Noise level must be in"):
            add_noise(pattern, -0.1, 'flip')
        
        with pytest.raises(ValueError, match="Noise level must be in"):
            add_noise(pattern, 1.5, 'flip')
    
    def test_invalid_noise_type(self):
        """Test that invalid noise type raises error."""
        pattern = np.zeros(256)
        
        with pytest.raises(ValueError, match="Unknown noise type"):
            add_noise(pattern, 0.1, 'invalid')


class TestGenerateRandomPatterns:
    """Test random pattern generation."""
    
    def test_basic_generation(self):
        """Test basic pattern generation."""
        num_patterns = 10
        dimension = 256
        
        addresses, data = generate_random_patterns(num_patterns, dimension)
        
        assert len(addresses) == num_patterns
        assert len(data) == num_patterns
        
        for addr, dat in zip(addresses, data):
            assert addr.shape == (dimension,)
            assert dat.shape == (dimension,)
            assert addr.dtype == np.uint8
            assert dat.dtype == np.uint8
            assert np.all((addr == 0) | (addr == 1))
            assert np.all((dat == 0) | (dat == 1))
    
    def test_sparsity(self):
        """Test pattern sparsity control."""
        num_patterns = 100
        dimension = 256
        sparsity = 0.3
        
        addresses, data = generate_random_patterns(
            num_patterns, dimension, sparsity=sparsity
        )
        
        # Check average sparsity
        addr_density = np.mean([np.mean(addr) for addr in addresses])
        data_density = np.mean([np.mean(dat) for dat in data])
        
        # Should be close to specified sparsity
        assert abs(addr_density - sparsity) < 0.1
        assert abs(data_density - sparsity) < 0.1
    
    def test_correlation(self):
        """Test correlation between addresses and data."""
        num_patterns = 50
        dimension = 256
        correlation = 0.7
        
        addresses, data = generate_random_patterns(
            num_patterns, dimension, correlation=correlation
        )
        
        # Measure actual correlation
        correlations = []
        for addr, dat in zip(addresses, data):
            # Correlation measured as fraction of matching bits
            matching = np.mean(addr == dat)
            correlations.append(matching)
        
        avg_correlation = np.mean(correlations)
        
        # Should be close to specified correlation
        assert abs(avg_correlation - correlation) < 0.1
    
    def test_reproducibility(self):
        """Test that seed produces reproducible patterns."""
        addresses1, data1 = generate_random_patterns(10, 256, seed=42)
        addresses2, data2 = generate_random_patterns(10, 256, seed=42)
        
        for a1, a2 in zip(addresses1, addresses2):
            assert np.array_equal(a1, a2)
        
        for d1, d2 in zip(data1, data2):
            assert np.array_equal(d1, d2)


class TestComputeMemoryCapacity:
    """Test memory capacity calculations."""
    
    def test_basic_capacity(self):
        """Test basic capacity calculation."""
        dimension = 1000
        num_locations = 1000
        activation_radius = 451  # Critical distance
        
        capacity = compute_memory_capacity(
            dimension, num_locations, activation_radius
        )
        
        assert 'kanerva_estimate' in capacity
        assert 'information_theoretic' in capacity
        assert 'sphere_packing' in capacity
        assert 'coverage_based' in capacity
        assert 'expected_activated' in capacity
        assert 'activation_probability' in capacity
        assert 'recommended_capacity' in capacity
        
        # Kanerva estimate should be ~0.15 * locations at critical distance
        assert abs(capacity['kanerva_estimate'] - 150) < 20
    
    def test_capacity_scaling(self):
        """Test capacity scaling with parameters."""
        dimension = 500
        num_locations = 1000
        
        # Capacity at critical distance
        critical_radius = int(0.451 * dimension)
        capacity_critical = compute_memory_capacity(
            dimension, num_locations, critical_radius
        )
        
        # Capacity at larger radius
        large_radius = int(0.6 * dimension)
        capacity_large = compute_memory_capacity(
            dimension, num_locations, large_radius
        )
        
        # Larger radius should give lower capacity
        assert capacity_large['kanerva_estimate'] < capacity_critical['kanerva_estimate']
    
    def test_activation_probability(self):
        """Test activation probability calculation."""
        dimension = 256
        num_locations = 100
        activation_radius = 115
        
        capacity = compute_memory_capacity(
            dimension, num_locations, activation_radius
        )
        
        prob = capacity['activation_probability']
        assert 0 < prob < 1
        
        # Expected activations should match probability
        expected = capacity['expected_activated']
        assert abs(expected - num_locations * prob) < 1
    
    def test_error_tolerance_effect(self):
        """Test effect of error tolerance on capacity."""
        dimension = 500
        num_locations = 500
        activation_radius = 225
        
        capacity_low_error = compute_memory_capacity(
            dimension, num_locations, activation_radius, error_tolerance=0.01
        )
        
        capacity_high_error = compute_memory_capacity(
            dimension, num_locations, activation_radius, error_tolerance=0.1
        )
        
        # Higher error tolerance might affect information theoretic bound
        assert capacity_low_error['information_theoretic'] != capacity_high_error['information_theoretic']


class TestAnalyzeActivationPatterns:
    """Test activation pattern analysis."""
    
    @pytest.fixture
    def test_sdm(self):
        """Create test SDM."""
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115,
            seed=42
        )
        return SDM(config)
    
    def test_basic_analysis(self, test_sdm):
        """Test basic activation analysis."""
        analysis = analyze_activation_patterns(test_sdm, sample_size=100, visualize=False)
        
        assert 'mean_activation_count' in analysis
        assert 'std_activation_count' in analysis
        assert 'min_activations' in analysis
        assert 'max_activations' in analysis
        assert 'mean_overlap' in analysis
        assert 'std_overlap' in analysis
        assert 'location_usage_mean' in analysis
        assert 'unused_locations' in analysis
        assert 'usage_uniformity' in analysis
        
        # Sanity checks
        assert analysis['mean_activation_count'] > 0
        assert analysis['min_activations'] <= analysis['mean_activation_count'] <= analysis['max_activations']
        assert 0 <= analysis['usage_uniformity'] <= 1
    
    def test_similarity_correlation(self, test_sdm):
        """Test address similarity vs activation overlap correlation."""
        analysis = analyze_activation_patterns(test_sdm, sample_size=50, visualize=False)
        
        if 'similarity_correlation' in analysis:
            corr = analysis['similarity_correlation']
            assert -1 <= corr <= 1
            # Should be positive - similar addresses activate similar locations
            assert corr > 0
    
    @pytest.mark.slow
    def test_with_visualization(self, test_sdm):
        """Test analysis with visualization (no display)."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        analysis = analyze_activation_patterns(test_sdm, sample_size=50, visualize=True)
        
        assert 'figure' in analysis
        assert analysis['figure'] is not None


class TestPerformanceTesting:
    """Test performance testing utilities."""
    
    def test_performance_test_result(self):
        """Test PerformanceTestResult dataclass."""
        result = PerformanceTestResult(
            pattern_count=100,
            dimension=1000,
            write_time_mean=0.001,
            write_time_std=0.0001,
            read_time_mean=0.0005,
            read_time_std=0.00005,
            recall_accuracy_mean=0.95,
            recall_accuracy_std=0.02,
            noise_tolerance={0.0: 1.0, 0.1: 0.9, 0.2: 0.8},
            capacity_utilization=0.75
        )
        
        assert result.pattern_count == 100
        assert result.recall_accuracy_mean == 0.95
        assert result.noise_tolerance[0.1] == 0.9
    
    def test_test_sdm_performance(self):
        """Test SDM performance testing function."""
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115
        )
        sdm = SDM(config)
        
        # Run quick performance test
        results = test_sdm_performance(
            sdm, 
            test_patterns=10,
            noise_levels=[0.0, 0.1, 0.2],
            progress=False
        )
        
        assert isinstance(results, PerformanceTestResult)
        assert results.pattern_count == 10
        assert results.dimension == 256
        assert results.write_time_mean > 0
        assert results.read_time_mean > 0
        assert 0 <= results.recall_accuracy_mean <= 1
        assert len(results.noise_tolerance) == 3
        assert 0 <= results.capacity_utilization <= 1


class TestPatternSimilarity:
    """Test pattern similarity calculations."""
    
    def test_hamming_similarity(self):
        """Test Hamming similarity calculation."""
        pattern1 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        pattern2 = np.array([1, 0, 1, 0, 0, 1, 0, 1])  # 4 differences
        
        similarity = calculate_pattern_similarity(pattern1, pattern2, metric='hamming')
        
        # Hamming similarity = 1 - (differences / length)
        expected = 1 - (4 / 8)
        assert similarity == expected
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        pattern1 = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        pattern2 = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        
        similarity = calculate_pattern_similarity(pattern1, pattern2, metric='jaccard')
        
        # Jaccard = intersection / union
        # intersection = 2 (positions 0 and 4)
        # union = 4 (positions 0, 1, 2, 4)
        expected = 2 / 4
        assert similarity == expected
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        pattern1 = np.array([1, 0, 1, 0])
        pattern2 = np.array([1, 1, 0, 0])
        
        similarity = calculate_pattern_similarity(pattern1, pattern2, metric='cosine')
        
        # Cosine similarity calculation
        dot_product = np.dot(pattern1, pattern2)  # 1
        norm1 = np.linalg.norm(pattern1)  # sqrt(2)
        norm2 = np.linalg.norm(pattern2)  # sqrt(2)
        expected = dot_product / (norm1 * norm2)  # 1/2
        
        assert abs(similarity - expected) < 1e-10
    
    def test_zero_patterns(self):
        """Test similarity with zero patterns."""
        pattern1 = np.zeros(8)
        pattern2 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        
        # Hamming similarity
        sim_hamming = calculate_pattern_similarity(pattern1, pattern2, metric='hamming')
        assert sim_hamming == 0.5  # Half the bits match
        
        # Jaccard similarity with all zeros
        sim_jaccard = calculate_pattern_similarity(pattern1, pattern2, metric='jaccard')
        assert sim_jaccard == 0.0  # No intersection
        
        # Cosine similarity with zero vector
        sim_cosine = calculate_pattern_similarity(pattern1, pattern2, metric='cosine')
        assert sim_cosine == 0.0
    
    def test_mutual_info(self):
        """Test mutual information calculation."""
        pattern1 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        pattern2 = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Same pattern
        
        similarity = calculate_pattern_similarity(pattern1, pattern2, metric='mutual_info')
        
        # Mutual info should be high for identical patterns
        assert similarity > 0
    
    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        pattern1 = np.zeros(8)
        pattern2 = np.ones(8)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_pattern_similarity(pattern1, pattern2, metric='invalid')
    
    def test_different_shapes(self):
        """Test that different shapes raise error."""
        pattern1 = np.zeros(8)
        pattern2 = np.zeros(16)
        
        with pytest.raises(ValueError, match="same shape"):
            calculate_pattern_similarity(pattern1, pattern2)


class TestCreateOrthogonalPatterns:
    """Test orthogonal pattern creation."""
    
    def test_basic_creation(self):
        """Test creating orthogonal patterns."""
        num_patterns = 5
        dimension = 128
        min_distance = 40
        
        patterns = create_orthogonal_patterns(num_patterns, dimension, min_distance)
        
        assert len(patterns) == num_patterns
        
        # Check all patterns are correct dimension
        for pattern in patterns:
            assert pattern.shape == (dimension,)
            assert pattern.dtype in [np.uint8, np.int64, np.int32]
        
        # Check minimum distance constraint
        for i in range(num_patterns):
            for j in range(i + 1, num_patterns):
                distance = np.sum(patterns[i] != patterns[j])
                assert distance >= min_distance
    
    def test_default_min_distance(self):
        """Test default minimum distance."""
        patterns = create_orthogonal_patterns(3, 90)
        
        # Default is dimension // 3 = 30
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                distance = np.sum(patterns[i] != patterns[j])
                assert distance >= 30
    
    def test_impossible_constraints(self):
        """Test warning when constraints are impossible."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Try to create too many patterns with large min distance
            patterns = create_orthogonal_patterns(
                num_patterns=10,
                dimension=64,
                min_distance=60  # Very restrictive
            )
            
            # Should warn and return fewer patterns
            assert len(w) > 0
            assert "Could only create" in str(w[0].message)
            assert len(patterns) < 10


class TestPatternEncoder:
    """Test PatternEncoder class."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return PatternEncoder(dimension=256)
    
    def test_encode_integer(self, encoder):
        """Test integer encoding."""
        # Positive integer
        encoded = encoder.encode_integer(42)
        assert encoded.shape == (256,)
        assert encoded.dtype == np.uint8
        assert np.all((encoded == 0) | (encoded == 1))
        
        # Negative integer
        encoded_neg = encoder.encode_integer(-42)
        assert encoded_neg.shape == (256,)
        
        # Different integers should give different encodings
        encoded2 = encoder.encode_integer(43)
        assert not np.array_equal(encoded, encoded2)
    
    def test_encode_integer_with_bits(self, encoder):
        """Test integer encoding with specified bits."""
        encoded = encoder.encode_integer(255, bits=8)
        assert encoded.shape == (256,)
        
        # First 8 bits should be all 1s for 255
        assert np.all(encoded[:8] == 1)
    
    def test_encode_float(self, encoder):
        """Test float encoding."""
        encoded = encoder.encode_float(3.14159, precision=16)
        assert encoded.shape == (256,)
        assert encoded.dtype == np.uint8
        
        # Different floats should give different encodings
        encoded2 = encoder.encode_float(2.71828, precision=16)
        assert not np.array_equal(encoded, encoded2)
    
    def test_encode_string_hash(self, encoder):
        """Test string encoding with hash method."""
        encoded = encoder.encode_string("Hello SDM", method='hash')
        assert encoded.shape == (256,)
        assert encoded.dtype == np.uint8
        
        # Same string should give same encoding
        encoded2 = encoder.encode_string("Hello SDM", method='hash')
        assert np.array_equal(encoded, encoded2)
        
        # Different string should give different encoding
        encoded3 = encoder.encode_string("Hello VSA", method='hash')
        assert not np.array_equal(encoded, encoded3)
    
    def test_encode_string_char(self, encoder):
        """Test string encoding with character method."""
        encoded = encoder.encode_string("AB", method='char')
        assert encoded.shape == (256,)
        
        # Should encode ASCII values
        # 'A' = 65 = 01000001, 'B' = 66 = 01000010
        # First 16 bits should represent these characters
    
    def test_encode_string_invalid_method(self, encoder):
        """Test string encoding with invalid method."""
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encoder.encode_string("test", method='invalid')
    
    def test_encode_vector_threshold(self, encoder):
        """Test vector encoding with threshold method."""
        vector = np.array([0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4])
        encoded = encoder.encode_vector(vector, method='threshold')
        
        assert encoded.shape == (256,)
        assert encoded.dtype == np.uint8
    
    def test_encode_vector_rank(self, encoder):
        """Test vector encoding with rank method."""
        vector = np.array([0.1, 0.5, 0.9, 0.2, 0.7])
        encoded = encoder.encode_vector(vector, method='rank')
        
        assert encoded.shape == (256,)
    
    def test_encode_vector_random_projection(self, encoder):
        """Test vector encoding with random projection."""
        vector = np.random.randn(50)
        encoded = encoder.encode_vector(vector, method='random_projection')
        
        assert encoded.shape == (256,)
        
        # Should be deterministic with fixed seed
        encoded2 = encoder.encode_vector(vector, method='random_projection')
        assert np.array_equal(encoded, encoded2)
    
    def test_encode_vector_invalid_method(self, encoder):
        """Test vector encoding with invalid method."""
        vector = np.random.randn(10)
        
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encoder.encode_vector(vector, method='invalid')


class TestSaveLoadSDM:
    """Test SDM save/load functionality."""
    
    @pytest.fixture
    def sdm_with_data(self):
        """Create SDM with some stored patterns."""
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115,
            seed=42
        )
        sdm = SDM(config)
        
        # Store some patterns
        addresses, data = generate_random_patterns(5, 256)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        return sdm
    
    def test_save_sdm_state(self, sdm_with_data):
        """Test saving SDM state."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            save_sdm_state(sdm_with_data, tmp.name, include_patterns=True)
            
            # File should exist and have content
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_save_without_patterns(self, sdm_with_data):
        """Test saving SDM without stored patterns."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            save_sdm_state(sdm_with_data, tmp.name, include_patterns=False)
            
            # File should be smaller without patterns
            size_without = os.path.getsize(tmp.name)
            
            save_sdm_state(sdm_with_data, tmp.name, include_patterns=True)
            size_with = os.path.getsize(tmp.name)
            
            assert size_with > size_without
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_load_sdm_state(self, sdm_with_data):
        """Test loading SDM state."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            # Save state
            save_sdm_state(sdm_with_data, tmp.name, include_patterns=True)
            
            # Load state
            loaded_sdm = load_sdm_state(tmp.name)
            
            # Verify configuration matches
            assert loaded_sdm.config.dimension == sdm_with_data.config.dimension
            assert loaded_sdm.config.num_hard_locations == sdm_with_data.config.num_hard_locations
            assert loaded_sdm.config.activation_radius == sdm_with_data.config.activation_radius
            
            # Verify hard locations match
            assert np.array_equal(loaded_sdm.hard_locations, sdm_with_data.hard_locations)
            
            # Verify stored patterns match
            assert len(loaded_sdm._stored_addresses) == len(sdm_with_data._stored_addresses)
            
            # Test recall works
            if len(loaded_sdm._stored_addresses) > 0:
                test_addr = loaded_sdm._stored_addresses[0]
                recalled = loaded_sdm.recall(test_addr)
                assert recalled is not None
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_load_with_custom_class(self, sdm_with_data):
        """Test loading with custom SDM class."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            save_sdm_state(sdm_with_data, tmp.name)
            
            # Mock custom class
            from cognitive_computing.sdm.core import SDM
            loaded_sdm = load_sdm_state(tmp.name, sdm_class=SDM)
            
            assert isinstance(loaded_sdm, SDM)
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_save_load_binary_storage(self):
        """Test save/load with binary storage."""
        config = SDMConfig(
            dimension=128,
            num_hard_locations=30,
            activation_radius=57,
            storage_method="binary"
        )
        sdm = SDM(config)
        
        # Store pattern
        addr = np.random.randint(0, 2, 128)
        data = np.random.randint(0, 2, 128)
        sdm.store(addr, data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            save_sdm_state(sdm, tmp.name)
            loaded_sdm = load_sdm_state(tmp.name)
            
            # Verify binary storage
            assert loaded_sdm.config.storage_method == "binary"
            assert loaded_sdm.binary_storage is not None
            assert loaded_sdm.counters is None
            
            # Cleanup
            os.unlink(tmp.name)


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_full_workflow(self):
        """Test complete workflow using utilities."""
        # Generate patterns
        addresses, data = generate_random_patterns(20, 256, sparsity=0.3)
        
        # Create and test SDM
        config = SDMConfig(
            dimension=256,
            num_hard_locations=100,
            activation_radius=115
        )
        sdm = SDM(config)
        
        # Store patterns
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Test performance
        results = test_sdm_performance(sdm, test_patterns=10, progress=False)
        assert results.recall_accuracy_mean > 0.8
        
        # Analyze patterns
        analysis = analyze_activation_patterns(sdm, sample_size=50, visualize=False)
        assert analysis['mean_activation_count'] > 0
        
        # Test with noise
        test_addr = addresses[0]
        noisy_addr = add_noise(test_addr, 0.1, 'flip')
        recalled = sdm.recall(noisy_addr)
        assert recalled is not None
        
        # Calculate similarity
        if recalled is not None:
            similarity = calculate_pattern_similarity(recalled, data[0], 'hamming')
            assert similarity > 0.7
    
    def test_encoder_with_sdm(self):
        """Test pattern encoder with SDM storage."""
        encoder = PatternEncoder(dimension=512)
        sdm = SDM(SDMConfig(dimension=512, num_hard_locations=200, activation_radius=230))
        
        # Encode and store different data types
        # Integer
        int_encoded = encoder.encode_integer(42)
        sdm.store(int_encoded, int_encoded)
        
        # String
        str_encoded = encoder.encode_string("cognitive computing")
        sdm.store(str_encoded, str_encoded)
        
        # Float
        float_encoded = encoder.encode_float(3.14159)
        sdm.store(float_encoded, float_encoded)
        
        # Vector
        vector = np.random.randn(20)
        vec_encoded = encoder.encode_vector(vector)
        sdm.store(vec_encoded, vec_encoded)
        
        # Test recall
        recalled_int = sdm.recall(int_encoded)
        assert recalled_int is not None
        assert np.mean(recalled_int == int_encoded) > 0.9
    
    def test_orthogonal_patterns_with_sdm(self):
        """Test orthogonal patterns in SDM."""
        # Create orthogonal patterns
        patterns = create_orthogonal_patterns(10, 256, min_distance=80)
        
        # Store in SDM
        config = SDMConfig(dimension=256, num_hard_locations=100, activation_radius=115)
        sdm = SDM(config)
        
        for i, pattern in enumerate(patterns):
            # Use pattern as both address and data
            sdm.store(pattern, pattern)
        
        # Test recall - should have minimal interference
        accuracies = []
        for pattern in patterns:
            recalled = sdm.recall(pattern)
            if recalled is not None:
                accuracy = np.mean(recalled == pattern)
                accuracies.append(accuracy)
        
        # Orthogonal patterns should have high recall accuracy
        assert np.mean(accuracies) > 0.95


@pytest.mark.parametrize("noise_type", ['flip', 'swap', 'burst', 'salt_pepper'])
class TestNoiseTypes:
    """Parameterized tests for all noise types."""
    
    def test_noise_preserves_dimension(self, noise_type):
        """Test that noise preserves pattern dimension."""
        pattern = np.random.randint(0, 2, 256)
        noisy = add_noise(pattern, 0.1, noise_type)
        
        assert noisy.shape == pattern.shape
        assert noisy.dtype == pattern.dtype
    
    def test_zero_noise_unchanged(self, noise_type):
        """Test that zero noise leaves pattern unchanged."""
        pattern = np.random.randint(0, 2, 256)
        noisy = add_noise(pattern, 0.0, noise_type, seed=42)
        
        assert np.array_equal(pattern, noisy)
    
    def test_noise_reproducibility(self, noise_type):
        """Test that noise is reproducible with seed."""
        pattern = np.random.randint(0, 2, 256)
        
        noisy1 = add_noise(pattern, 0.2, noise_type, seed=42)
        noisy2 = add_noise(pattern, 0.2, noise_type, seed=42)
        
        assert np.array_equal(noisy1, noisy2)