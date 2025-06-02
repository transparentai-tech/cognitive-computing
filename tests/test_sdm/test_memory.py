"""
Tests for SDM memory module components.

This module contains comprehensive tests for:
- HardLocation class
- MemoryContents class
- MemoryStatistics class
- MemoryOptimizer class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import warnings
from typing import List
import time

from cognitive_computing.sdm.memory import (
    HardLocation,
    MemoryContents,
    MemoryStatistics,
    MemoryOptimizer
)
from cognitive_computing.sdm.core import SDM, SDMConfig
from cognitive_computing.sdm.utils import generate_random_patterns


class TestHardLocation:
    """Test HardLocation class."""
    
    def test_initialization_counters(self):
        """Test initialization with counter storage."""
        address = np.random.randint(0, 2, 256)
        loc = HardLocation(
            index=0,
            address=address,
            dimension=256,
            storage_type="counters"
        )
        
        assert loc.index == 0
        assert np.array_equal(loc.address, address)
        assert loc.dimension == 256
        assert loc.storage_type == "counters"
        assert loc.counters is not None
        assert loc.binary_data is None
        assert np.all(loc.counters == 0)
        assert loc.counters.shape == (256,)
        assert loc.access_count == 0
        assert loc.write_count == 0
    
    def test_initialization_binary(self):
        """Test initialization with binary storage."""
        address = np.random.randint(0, 2, 256)
        loc = HardLocation(
            index=1,
            address=address,
            dimension=256,
            storage_type="binary"
        )
        
        assert loc.storage_type == "binary"
        assert loc.binary_data is not None
        assert loc.counters is None
        assert np.all(loc.binary_data == 0)
        assert loc.binary_data.shape == (256,)
    
    def test_invalid_storage_type(self):
        """Test that invalid storage type raises error."""
        with pytest.raises(ValueError, match="Invalid storage_type"):
            HardLocation(
                index=0,
                address=np.zeros(256),
                dimension=256,
                storage_type="invalid"
            )
    
    def test_write_counters(self):
        """Test writing data to counter-based location."""
        loc = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="counters"
        )
        
        # Write bipolar data
        data = np.array([1, -1, 1, -1] * 64)  # Bipolar pattern
        loc.write(data, timestamp=100)
        
        assert loc.write_count == 1
        assert loc.access_count == 1
        assert loc.last_access_time == 100
        assert np.array_equal(loc.counters, data)
        assert 100 in loc.activation_history
        
        # Write again
        loc.write(data, timestamp=200)
        assert loc.write_count == 2
        assert np.array_equal(loc.counters, data * 2)
    
    def test_write_binary(self):
        """Test writing data to binary location."""
        loc = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="binary"
        )
        
        # Write binary data
        data1 = np.array([1, 0, 1, 0] * 64, dtype=np.uint8)
        loc.write(data1, timestamp=100)
        
        assert loc.write_count == 1
        assert np.array_equal(loc.binary_data, data1)
        
        # Write more data (OR operation)
        data2 = np.array([0, 1, 0, 1] * 64, dtype=np.uint8)
        loc.write(data2, timestamp=200)
        
        expected = np.logical_or(data1, data2).astype(np.uint8)
        assert np.array_equal(loc.binary_data, expected)
    
    def test_read(self):
        """Test reading data from location."""
        # Counter storage
        loc_counter = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="counters"
        )
        data_counter = np.array([5, -3, 2, -1] * 64)
        loc_counter.counters = data_counter
        
        read_data = loc_counter.read(timestamp=300)
        assert np.array_equal(read_data, data_counter)
        assert loc_counter.access_count == 1
        assert loc_counter.last_access_time == 300
        
        # Binary storage
        loc_binary = HardLocation(
            index=1,
            address=np.zeros(256),
            dimension=256,
            storage_type="binary"
        )
        data_binary = np.array([1, 0, 1, 0] * 64, dtype=np.uint8)
        loc_binary.binary_data = data_binary
        
        read_data = loc_binary.read(timestamp=400)
        assert np.array_equal(read_data, data_binary)
    
    def test_saturation_level(self):
        """Test saturation level calculation."""
        loc = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="counters"
        )
        
        # No saturation
        loc.counters = np.array([10, -10, 5, -5] * 64)
        assert loc.get_saturation_level(max_value=127) == 0.0
        
        # Some saturation
        loc.counters[0:10] = 127
        loc.counters[10:20] = -127
        saturation = loc.get_saturation_level(max_value=127)
        assert saturation == 20 / 256
        
        # Binary storage should return 0
        loc_binary = HardLocation(
            index=1,
            address=np.zeros(256),
            dimension=256,
            storage_type="binary"
        )
        assert loc_binary.get_saturation_level() == 0.0
    
    def test_bit_density(self):
        """Test bit density calculation."""
        loc_binary = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="binary"
        )
        
        # All zeros
        assert loc_binary.get_bit_density() == 0.0
        
        # Half ones
        loc_binary.binary_data[::2] = 1
        assert loc_binary.get_bit_density() == 0.5
        
        # All ones
        loc_binary.binary_data[:] = 1
        assert loc_binary.get_bit_density() == 1.0
        
        # Counter storage should return 0
        loc_counter = HardLocation(
            index=1,
            address=np.zeros(256),
            dimension=256,
            storage_type="counters"
        )
        assert loc_counter.get_bit_density() == 0.0
    
    def test_entropy(self):
        """Test entropy calculation."""
        # Counter storage
        loc_counter = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="counters"
        )
        
        # All zeros - no entropy
        assert loc_counter.get_entropy() == 0.0
        
        # Uniform distribution - high entropy
        loc_counter.counters = np.ones(256)
        entropy = loc_counter.get_entropy()
        assert entropy > 0
        
        # Binary storage
        loc_binary = HardLocation(
            index=1,
            address=np.zeros(256),
            dimension=256,
            storage_type="binary"
        )
        
        # All zeros or all ones - no entropy
        assert loc_binary.get_entropy() == 0.0
        
        loc_binary.binary_data[:] = 1
        assert loc_binary.get_entropy() == 0.0
        
        # Half ones - maximum binary entropy
        loc_binary.binary_data[:128] = 0
        loc_binary.binary_data[128:] = 1
        entropy = loc_binary.get_entropy()
        assert 0.9 < entropy < 1.1  # Close to 1.0
    
    def test_reset(self):
        """Test resetting location."""
        loc = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type="counters"
        )
        
        # Add some data and history
        data = np.array([5, -3, 2, -1] * 64)
        loc.write(data, timestamp=100)
        loc.read(timestamp=200)
        
        assert loc.access_count > 0
        assert len(loc.activation_history) > 0
        
        # Reset
        loc.reset()
        
        assert np.all(loc.counters == 0)
        assert loc.access_count == 0
        assert loc.write_count == 0
        assert loc.last_access_time == 0
        assert len(loc.activation_history) == 0


class TestMemoryContents:
    """Test MemoryContents class."""
    
    @pytest.fixture
    def test_sdm(self):
        """Create a test SDM with some patterns."""
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115,
            seed=42
        )
        sdm = SDM(config)
        
        # Store some patterns
        addresses, data = generate_random_patterns(10, 256)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        return sdm
    
    def test_initialization(self, test_sdm):
        """Test MemoryContents initialization."""
        contents = MemoryContents(test_sdm)
        
        assert len(contents.hard_locations) == test_sdm.config.num_hard_locations
        assert all(isinstance(loc, HardLocation) for loc in contents.hard_locations)
    
    def test_memory_map(self, test_sdm):
        """Test memory map generation."""
        contents = MemoryContents(test_sdm)
        maps = contents.get_memory_map()
        
        assert 'usage_map' in maps
        assert 'entropy_map' in maps
        assert len(maps['usage_map']) == test_sdm.config.num_hard_locations
        
        # Check storage-specific maps
        if test_sdm.config.storage_method == "counters":
            assert 'saturation_map' in maps
            assert 'avg_counter_magnitude' in maps
        else:
            assert 'density_map' in maps
        
        # Usage map should have some non-zero values
        assert np.any(maps['usage_map'] > 0)
    
    def test_pattern_distribution_analysis(self, test_sdm):
        """Test pattern distribution analysis."""
        contents = MemoryContents(test_sdm)
        analysis = contents.analyze_pattern_distribution(sample_size=100)
        
        assert 'mean_activation_count' in analysis
        assert 'std_activation_count' in analysis
        assert 'mean_overlap' in analysis
        assert 'activation_uniformity' in analysis
        assert 'activation_deviation' in analysis
        
        # Sanity checks
        assert analysis['mean_activation_count'] > 0
        assert analysis['mean_activation_count'] < test_sdm.config.num_hard_locations
        assert 0 <= analysis['activation_uniformity'] <= 1
    
    def test_find_similar_locations(self, test_sdm):
        """Test finding similar locations."""
        contents = MemoryContents(test_sdm)
        
        # For counters, similarity is based on cosine similarity
        # For binary, it's based on Jaccard similarity
        similar_pairs = contents.find_similar_locations(threshold=0.5)
        
        # Check format
        for pair in similar_pairs:
            assert len(pair) == 3  # (loc1_idx, loc2_idx, similarity)
            assert 0 <= pair[0] < test_sdm.config.num_hard_locations
            assert 0 <= pair[1] < test_sdm.config.num_hard_locations
            assert 0 <= pair[2] <= 1
        
        # Should be sorted by similarity
        if len(similar_pairs) > 1:
            similarities = [pair[2] for pair in similar_pairs]
            assert similarities == sorted(similarities, reverse=True)
    
    def test_capacity_estimate(self, test_sdm):
        """Test capacity estimation."""
        contents = MemoryContents(test_sdm)
        capacity = contents.get_capacity_estimate()
        
        assert 'theoretical_capacity' in capacity
        assert 'patterns_stored' in capacity
        assert 'location_utilization' in capacity
        assert 'capacity_used_estimate' in capacity
        assert 'remaining_capacity_estimate' in capacity
        assert 'average_recall_error' in capacity
        assert 'signal_to_noise_ratio_db' in capacity
        assert 'recommended_max_patterns' in capacity
        
        # Sanity checks
        assert capacity['patterns_stored'] == 10  # We stored 10 patterns
        assert 0 <= capacity['location_utilization'] <= 1
        assert 0 <= capacity['capacity_used_estimate'] <= 1
        assert capacity['remaining_capacity_estimate'] >= 0
    
    def test_expected_activation_calculation(self, test_sdm):
        """Test expected activation calculation."""
        contents = MemoryContents(test_sdm)
        expected = contents._compute_expected_activation()
        
        assert expected > 0
        assert expected < test_sdm.config.num_hard_locations
        
        # Should be reasonable for the parameters
        # With 50 locations and radius ~115 in 256D space
        assert 1 < expected < 30


class TestMemoryStatistics:
    """Test MemoryStatistics class."""
    
    @pytest.fixture
    def stats_sdm(self):
        """Create SDM with some operation history."""
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115,
            seed=42
        )
        sdm = SDM(config)
        return sdm
    
    def test_initialization(self, stats_sdm):
        """Test MemoryStatistics initialization."""
        stats = MemoryStatistics(stats_sdm)
        
        assert stats.sdm is stats_sdm
        assert isinstance(stats.contents, MemoryContents)
        assert len(stats.operation_history) == 0
        assert len(stats.performance_history) == 0
    
    def test_record_operation(self, stats_sdm):
        """Test operation recording."""
        stats = MemoryStatistics(stats_sdm)
        
        # Record some operations
        stats.record_operation('store', True, {'pattern_id': 1})
        stats.record_operation('recall', True, {'pattern_id': 1})
        stats.record_operation('recall', False, {'pattern_id': 2})
        
        assert len(stats.operation_history) == 3
        assert stats.operation_history[0]['operation'] == 'store'
        assert stats.operation_history[1]['success'] == True
        assert stats.operation_history[2]['success'] == False
    
    def test_temporal_patterns(self, stats_sdm):
        """Test temporal pattern analysis."""
        stats = MemoryStatistics(stats_sdm)
        
        # Need enough operations for analysis
        for i in range(200):
            if i % 3 == 0:
                stats.record_operation('store', True)
            else:
                stats.record_operation('recall', i % 5 != 0)
        
        patterns = stats.analyze_temporal_patterns(window_size=50)
        
        assert 'store_rates' in patterns
        assert 'recall_rates' in patterns
        assert 'success_rates' in patterns
        assert 'operation_balance' in patterns
        
        # Check shapes
        assert len(patterns['store_rates']) == 151  # 200 - 50 + 1
        assert np.all(patterns['store_rates'] >= 0)
        assert np.all(patterns['store_rates'] <= 1)
        
        # Store rate should be approximately 1/3
        assert 0.2 < np.mean(patterns['store_rates']) < 0.4
    
    def test_temporal_patterns_insufficient_data(self, stats_sdm):
        """Test temporal patterns with insufficient data."""
        stats = MemoryStatistics(stats_sdm)
        
        # Record too few operations
        for i in range(10):
            stats.record_operation('store', True)
        
        patterns = stats.analyze_temporal_patterns(window_size=50)
        assert 'error' in patterns
    
    def test_correlation_matrix(self, stats_sdm):
        """Test correlation matrix computation."""
        stats = MemoryStatistics(stats_sdm)
        
        # Store some patterns first
        addresses, data = generate_random_patterns(5, 256)
        for addr, dat in zip(addresses, data):
            stats_sdm.store(addr, dat)
        
        # Compute correlation
        corr_matrix = stats.compute_correlation_matrix(sample_size=10)
        
        # With sparse data, many locations may have zero variance and get filtered
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
        assert corr_matrix.shape[0] >= 1  # At least 1x1
        assert corr_matrix.shape[0] <= 10  # At most 10x10
        
        # Check properties of correlation matrix
        assert np.all(np.abs(np.diag(corr_matrix) - 1.0) < 1e-10)  # Self-correlation is ~1
        assert np.all(corr_matrix >= -1.001)  # Allow small numerical error
        assert np.all(corr_matrix <= 1.001)
        assert np.allclose(corr_matrix, corr_matrix.T, atol=1e-10)  # Symmetric
    
    def test_recall_quality_analysis(self, stats_sdm):
        """Test recall quality analysis."""
        stats = MemoryStatistics(stats_sdm)
        
        # Store patterns
        addresses, data = generate_random_patterns(20, 256)
        for addr, dat in zip(addresses, data):
            stats_sdm.store(addr, dat)
        
        quality = stats.analyze_recall_quality(
            test_size=10,
            noise_levels=[0.0, 0.1, 0.2]
        )
        
        assert 'noise_levels' in quality
        assert 'recall_accuracies' in quality
        assert 'recall_success_rates' in quality
        assert 'bit_error_rates' in quality
        
        # Accuracy should decrease with noise
        accuracies = quality['recall_accuracies']
        assert len(accuracies) == 3
        assert accuracies[0] >= accuracies[1] >= accuracies[2]
        
        # Perfect recall at zero noise
        assert accuracies[0] > 0.9
    
    def test_recall_quality_empty_memory(self, stats_sdm):
        """Test recall quality analysis with empty memory."""
        stats = MemoryStatistics(stats_sdm)
        
        quality = stats.analyze_recall_quality()
        assert 'error' in quality
    
    def test_generate_report(self, stats_sdm):
        """Test comprehensive report generation."""
        stats = MemoryStatistics(stats_sdm)
        
        # Add some data and operations
        addresses, data = generate_random_patterns(10, 256)
        for i, (addr, dat) in enumerate(zip(addresses, data)):
            stats_sdm.store(addr, dat)
            stats.record_operation('store', True, {'pattern_id': i})
        
        # Generate report
        report = stats.generate_report()
        
        assert 'configuration' in report
        assert 'basic_stats' in report
        assert 'capacity_analysis' in report
        assert 'distribution_analysis' in report
        assert 'memory_maps' in report
        
        # Check configuration
        assert report['configuration']['dimension'] == 256
        assert report['configuration']['num_hard_locations'] == 50
        
        # Check basic stats
        assert report['basic_stats']['num_patterns_stored'] == 10
    
    @pytest.mark.slow
    def test_plot_analysis(self, stats_sdm):
        """Test plot generation (without display)."""
        stats = MemoryStatistics(stats_sdm)
        
        # Add some data
        addresses, data = generate_random_patterns(5, 256)
        for addr, dat in zip(addresses, data):
            stats_sdm.store(addr, dat)
        
        # Generate plots
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig = stats.plot_analysis(figsize=(12, 8))
        assert fig is not None
        
        # Check that subplots were created
        axes = fig.get_axes()
        assert len(axes) >= 6


class TestMemoryOptimizer:
    """Test MemoryOptimizer static methods."""
    
    def test_find_optimal_radius(self):
        """Test optimal radius finding."""
        # Test with specific parameters
        optimal = MemoryOptimizer.find_optimal_radius(
            dimension=1000,
            num_locations=1000,
            target_activation=32  # sqrt(1000) ≈ 32
        )
        
        # Should be close to critical distance
        critical = int(0.451 * 1000)
        assert abs(optimal - critical) < 50
        
        # Test with custom target
        optimal_custom = MemoryOptimizer.find_optimal_radius(
            dimension=500,
            num_locations=100,
            target_activation=5
        )
        
        assert optimal_custom > 0
        assert optimal_custom < 500
    
    def test_estimate_required_locations(self):
        """Test required locations estimation."""
        # Test at critical distance
        required = MemoryOptimizer.estimate_required_locations(
            dimension=1000,
            capacity=150,  # Want to store 150 patterns
            activation_radius=451  # Critical distance
        )
        
        # Should be approximately capacity / 0.15
        expected = int(150 / 0.15)
        assert abs(required - expected) < 100
        
        # Test with larger radius
        required_large = MemoryOptimizer.estimate_required_locations(
            dimension=1000,
            capacity=150,
            activation_radius=600  # Larger than critical
        )
        
        assert required_large > required  # Need more locations
    
    def test_analyze_parameter_space(self):
        """Test parameter space analysis."""
        results = MemoryOptimizer.analyze_parameter_space(
            dimension_range=(100, 300),
            location_range=(50, 150),
            samples=3
        )
        
        assert len(results) == 9  # 3 dimensions × 3 location counts
        
        for result in results:
            assert 'dimension' in result
            assert 'num_locations' in result
            assert 'optimal_radius' in result
            assert 'estimated_capacity' in result
            assert 'bits_per_pattern' in result
            assert 'efficiency' in result
            
            # Sanity checks
            assert result['optimal_radius'] > 0
            assert result['optimal_radius'] <= result['dimension']
            assert result['estimated_capacity'] > 0
            assert result['bits_per_pattern'] > 0
            assert 0 <= result['efficiency'] <= 1


class TestMemoryIntegration:
    """Integration tests for memory components."""
    
    def test_full_memory_analysis_workflow(self):
        """Test complete memory analysis workflow."""
        # Create SDM
        config = SDMConfig(
            dimension=512,
            num_hard_locations=200,
            activation_radius=230,
            seed=42
        )
        sdm = SDM(config)
        
        # Store patterns
        addresses, data = generate_random_patterns(30, 512)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Create analysis components
        contents = MemoryContents(sdm)
        stats = MemoryStatistics(sdm)
        
        # Run various analyses
        memory_map = contents.get_memory_map()
        distribution = contents.analyze_pattern_distribution()
        capacity = contents.get_capacity_estimate()
        
        # Verify results are consistent
        assert capacity['patterns_stored'] == 30
        assert distribution['mean_activation_count'] > 0
        assert np.sum(memory_map['usage_map'] > 0) > 0
        
        # Generate report
        report = stats.generate_report()
        assert report['basic_stats']['num_patterns_stored'] == 30
    
    def test_optimizer_with_real_sdm(self):
        """Test optimizer recommendations with real SDM."""
        # Use optimizer to find parameters
        optimal_radius = MemoryOptimizer.find_optimal_radius(
            dimension=500,
            num_locations=500,
            target_activation=22  # sqrt(500)
        )
        
        # Create SDM with recommended parameters
        config = SDMConfig(
            dimension=500,
            num_hard_locations=500,
            activation_radius=optimal_radius
        )
        sdm = SDM(config)
        
        # Test that it works well
        addresses, data = generate_random_patterns(50, 500)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Check recall accuracy
        accuracies = []
        for i in range(10):
            recalled = sdm.recall(addresses[i])
            if recalled is not None:
                accuracy = np.mean(recalled == data[i])
                accuracies.append(accuracy)
        
        # Should have good accuracy with optimal parameters
        assert np.mean(accuracies) > 0.9
    
    def test_memory_saturation_detection(self):
        """Test detection of memory saturation."""
        # Create small SDM with counter storage
        config = SDMConfig(
            dimension=100,
            num_hard_locations=20,
            activation_radius=45,
            storage_method="counters",
            saturation_value=10  # Low for testing
        )
        sdm = SDM(config)
        
        # Repeatedly store same patterns to cause saturation
        address = np.random.randint(0, 2, 100)
        data = np.random.randint(0, 2, 100)
        
        for _ in range(50):
            sdm.store(address, data)
        
        # Check saturation
        contents = MemoryContents(sdm)
        memory_map = contents.get_memory_map()
        
        # Should have high saturation
        assert np.mean(memory_map['saturation_map']) > 0.5
        
        # Check capacity estimate reflects saturation
        capacity = contents.get_capacity_estimate()
        assert capacity['capacity_used_estimate'] > 0.5


@pytest.mark.parametrize("storage_method", ["counters", "binary"])
class TestMemoryStorageMethods:
    """Test memory components with different storage methods."""
    
    def test_hard_location_operations(self, storage_method):
        """Test HardLocation with different storage methods."""
        loc = HardLocation(
            index=0,
            address=np.zeros(256),
            dimension=256,
            storage_type=storage_method
        )
        
        # Write and read
        if storage_method == "counters":
            data = np.random.randint(-5, 5, 256)
        else:
            data = np.random.randint(0, 2, 256, dtype=np.uint8)
        
        loc.write(data, timestamp=100)
        read_data = loc.read(timestamp=200)
        
        if storage_method == "counters":
            assert np.array_equal(read_data, data)
        else:
            # Binary uses OR operation
            assert np.all(read_data >= data)
    
    def test_memory_analysis_consistency(self, storage_method):
        """Test that memory analysis works with both storage methods."""
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115,
            storage_method=storage_method
        )
        sdm = SDM(config)
        
        # Store patterns
        addresses, data = generate_random_patterns(10, 256)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Run analysis
        contents = MemoryContents(sdm)
        stats = MemoryStatistics(sdm)
        
        # Should work regardless of storage method
        memory_map = contents.get_memory_map()
        capacity = contents.get_capacity_estimate()
        report = stats.generate_report()
        
        assert 'usage_map' in memory_map
        assert capacity['patterns_stored'] == 10
        assert report['configuration']['storage_method'] == storage_method