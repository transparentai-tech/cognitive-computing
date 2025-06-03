"""
Tests for HRR utility functions.

Tests vector generation, analysis functions, and performance utilities.
"""

import pytest
import numpy as np
import time

from cognitive_computing.hrr import create_hrr, HRRConfig
from cognitive_computing.hrr.utils import (
    # Vector generation
    generate_random_vector,
    generate_unitary_vector,
    generate_orthogonal_set,
    # Analysis
    analyze_binding_capacity,
    measure_crosstalk,
    test_associative_capacity,
    # Conversion
    to_complex,
    from_complex,
    # Performance
    benchmark_hrr_performance,
    # Utilities
    create_cleanup_memory,
    compare_storage_methods,
    HRRAnalysisResult,
    HRRPerformanceResult
)
from cognitive_computing.hrr.operations import CircularConvolution


class TestVectorGeneration:
    """Test vector generation functions."""
    
    def test_generate_random_vector_gaussian(self):
        """Test Gaussian vector generation."""
        v = generate_random_vector(1000, method="gaussian", seed=42)
        
        assert v.shape == (1000,)
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-6  # Normalized
        
        # After normalization, values will be much smaller
        # For a unit vector in 1000 dimensions, typical values are around 1/sqrt(1000) ~ 0.03
        assert -0.2 < np.min(v) < 0
        assert 0 < np.max(v) < 0.2
        assert np.abs(np.mean(v)) < 0.01  # Centered around 0
    
    def test_generate_random_vector_binary(self):
        """Test binary vector generation."""
        v = generate_random_vector(1000, method="binary", seed=42)
        
        assert v.shape == (1000,)
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-6  # Normalized
        
        # Should only contain -1 and 1 (after normalization)
        unique_vals = np.unique(v * np.sqrt(1000))
        assert len(unique_vals) == 2
        assert np.allclose(sorted(unique_vals), [-1, 1], atol=1e-10)
    
    def test_generate_random_vector_ternary(self):
        """Test ternary vector generation."""
        v = generate_random_vector(1000, method="ternary", seed=42)
        
        assert v.shape == (1000,)
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-6  # Normalized
        
        # Original values should be from {-1, 0, 1}
        # After normalization, we can't check exact values
        # But we can check that there are at most 3 distinct values
        # (accounting for scaling)
        assert len(np.unique(np.round(v * 100))) <= 3
    
    def test_generate_random_vector_sparse(self):
        """Test sparse vector generation."""
        v = generate_random_vector(1000, method="sparse", seed=42)
        
        assert v.shape == (1000,)
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-6  # Normalized
        
        # Should be sparse (most elements near zero)
        n_nonzero = np.sum(np.abs(v) > 1e-10)
        assert n_nonzero < 200  # Less than 20% non-zero
    
    def test_generate_random_vector_invalid_method(self):
        """Test invalid generation method."""
        with pytest.raises(ValueError, match="Unknown method"):
            generate_random_vector(100, method="invalid")
    
    def test_generate_unitary_vector(self):
        """Test unitary vector generation."""
        v = generate_unitary_vector(512, seed=42)
        
        assert v.shape == (512,)
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-6
        
        # Test self-inverse property
        # v correlated with itself should give delta function
        corr = CircularConvolution.correlate(v, v)
        
        # Peak should be at position 0
        assert np.argmax(np.abs(corr)) == 0
        
        # Peak should be close to 1
        assert np.abs(corr[0] - 1.0) < 0.1
    
    def test_generate_orthogonal_set_gram_schmidt(self):
        """Test orthogonal set generation with Gram-Schmidt."""
        vectors = generate_orthogonal_set(100, 5, method="gram_schmidt", seed=42)
        
        assert vectors.shape == (5, 100)
        
        # Check orthogonality
        for i in range(5):
            for j in range(5):
                dot = np.dot(vectors[i], vectors[j])
                if i == j:
                    assert np.abs(dot - 1.0) < 1e-10  # Normalized
                else:
                    assert np.abs(dot) < 1e-10  # Orthogonal
    
    def test_generate_orthogonal_set_hadamard(self):
        """Test orthogonal set generation with Hadamard matrix."""
        # Must use power of 2 dimension
        vectors = generate_orthogonal_set(64, 4, method="hadamard", seed=42)
        
        assert vectors.shape == (4, 64)
        
        # Check orthogonality
        for i in range(4):
            for j in range(4):
                dot = np.dot(vectors[i], vectors[j])
                if i == j:
                    assert np.abs(dot - 1.0) < 1e-10
                else:
                    assert np.abs(dot) < 1e-10
    
    def test_generate_orthogonal_set_hadamard_invalid_dim(self):
        """Test Hadamard with non-power-of-2 dimension."""
        with pytest.raises(ValueError, match="power of 2"):
            generate_orthogonal_set(100, 4, method="hadamard")
    
    def test_generate_orthogonal_set_too_many(self):
        """Test requesting too many orthogonal vectors."""
        with pytest.raises(ValueError, match="Cannot generate"):
            generate_orthogonal_set(10, 20)  # 20 vectors in 10D space


class TestAnalysisFunctions:
    """Test HRR analysis functions."""
    
    def test_analyze_binding_capacity(self):
        """Test binding capacity analysis."""
        hrr = create_hrr(dimension=2048, seed=42)
        
        # Test with small number of pairs
        results = analyze_binding_capacity(hrr, n_pairs=5, n_trials=3)
        
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert "mean_similarity" in results
        assert "capacity_estimate" in results
        
        # With 5 pairs, accuracy should be high
        assert results["mean_accuracy"] > 0.8
        assert results["mean_similarity"] > 0.5
    
    def test_analyze_binding_capacity_with_noise(self):
        """Test binding capacity with noisy queries."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        # Test with noise
        results = analyze_binding_capacity(hrr, n_pairs=3, n_trials=3, 
                                         noise_level=0.2)
        
        # Should still work but with lower accuracy
        assert results["mean_accuracy"] > 0.5
        assert results["mean_accuracy"] < 1.0
    
    def test_measure_crosstalk(self):
        """Test crosstalk measurement."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        # Generate random vectors
        vectors = [hrr.generate_vector() for _ in range(5)]
        
        crosstalk = measure_crosstalk(hrr, vectors)
        
        assert 0 <= crosstalk <= 1
        
        # For random vectors, crosstalk should be relatively low
        assert crosstalk < 0.3
    
    def test_measure_crosstalk_single_vector(self):
        """Test crosstalk with single vector."""
        hrr = create_hrr(dimension=512)
        
        vectors = [hrr.generate_vector()]
        crosstalk = measure_crosstalk(hrr, vectors)
        
        assert crosstalk == 0.0  # No crosstalk with single vector
    
    def test_test_associative_capacity(self):
        """Test associative memory capacity testing."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        results = test_associative_capacity(hrr, n_items=10)
        
        assert results["n_items"] == 10
        assert "accuracy" in results
        assert "mean_similarity" in results
        assert "total_store_time" in results
        assert "mean_recall_time" in results
        assert "items_per_second_store" in results
        assert "items_per_second_recall" in results
        
        # Should have reasonable performance
        assert results["accuracy"] > 0.5
        assert results["mean_similarity"] > 0.3
        assert results["items_per_second_store"] > 100
    
    def test_test_associative_capacity_custom_dimension(self):
        """Test associative capacity with custom item dimension."""
        hrr = create_hrr(dimension=2048, seed=42)
        
        # Use smaller item dimension
        results = test_associative_capacity(hrr, n_items=5, item_dimension=512)
        
        assert results["n_items"] == 5
        # Should still work but dimensions won't match HRR dimension


class TestConversionUtilities:
    """Test vector conversion utilities."""
    
    def test_to_complex_from_real(self):
        """Test converting real vector to complex."""
        v_real = np.random.randn(128)
        v_complex = to_complex(v_real)
        
        # Should have half the dimension (plus DC)
        assert v_complex.shape == (65,)  # 128/2 + 1
        assert np.iscomplexobj(v_complex)
        
        # Should preserve information (can convert back)
        v_reconstructed = from_complex(v_complex, 128)
        assert np.allclose(v_real, v_reconstructed, rtol=1e-10)
    
    def test_to_complex_already_complex(self):
        """Test converting complex vector (no-op)."""
        v = np.random.randn(64) + 1j * np.random.randn(64)
        v_result = to_complex(v)
        
        assert np.array_equal(v, v_result)
    
    def test_from_complex_to_real(self):
        """Test converting complex to real."""
        # Create complex vector
        v_complex = np.random.randn(65) + 1j * np.random.randn(65)
        
        # Convert to real
        v_real = from_complex(v_complex, 128)
        
        assert v_real.shape == (128,)
        assert not np.iscomplexobj(v_real)
    
    def test_from_complex_already_real(self):
        """Test converting real vector (no-op)."""
        v = np.random.randn(128)
        v_result = from_complex(v, 128)
        
        assert np.array_equal(v, v_result)
    
    def test_complex_conversion_roundtrip(self):
        """Test roundtrip conversion real->complex->real."""
        for dim in [64, 128, 256, 512]:
            v_original = np.random.randn(dim)
            v_complex = to_complex(v_original)
            v_recovered = from_complex(v_complex, dim)
            
            assert np.allclose(v_original, v_recovered, rtol=1e-10)


class TestPerformanceBenchmarking:
    """Test performance benchmarking functions."""
    
    def test_benchmark_hrr_performance(self):
        """Test HRR performance benchmarking."""
        results = benchmark_hrr_performance(
            dimension=512,
            n_operations=100,
            storage_method="real"
        )
        
        assert isinstance(results, HRRPerformanceResult)
        assert results.dimension == 512
        assert results.n_operations == 100
        
        # Check timing results
        assert results.bind_time_mean > 0
        assert results.unbind_time_mean > 0
        assert results.bundle_time_mean > 0
        assert results.operations_per_second > 0
        
        # Operations should be fast
        assert results.bind_time_mean < 0.01  # Less than 10ms
        assert results.unbind_time_mean < 0.01
        assert results.operations_per_second > 100
    
    def test_benchmark_hrr_performance_complex(self):
        """Test benchmarking with complex storage."""
        results = benchmark_hrr_performance(
            dimension=512,
            n_operations=50,
            storage_method="complex"
        )
        
        assert results.dimension == 512
        assert results.operations_per_second > 0


class TestUtilityFunctions:
    """Test utility helper functions."""
    
    def test_create_cleanup_memory(self):
        """Test cleanup memory creation helper."""
        hrr = create_hrr(dimension=512, seed=42)
        
        # Create items
        items = {
            "apple": hrr.generate_vector(),
            "banana": hrr.generate_vector(),
            "orange": hrr.generate_vector()
        }
        
        # Create cleanup memory
        cleanup = create_cleanup_memory(hrr, items, threshold=0.4)
        
        assert cleanup.config.threshold == 0.4
        assert cleanup.dimension == 512
        assert cleanup.size == 3
        
        # Check items were added
        for name in items:
            assert cleanup.get_item(name) is not None
    
    def test_compare_storage_methods(self):
        """Test storage method comparison."""
        results = compare_storage_methods(dimension=512, n_items=20)
        
        assert "real" in results
        assert "complex" in results
        
        # Check that both methods have results
        for method in ["real", "complex"]:
            assert "accuracy" in results[method]
            assert "mean_similarity" in results[method]
            assert "bind_time" in results[method]
            assert "unbind_time" in results[method]
            assert "ops_per_second" in results[method]
            
            # Both should work reasonably well
            assert results[method]["accuracy"] > 0.5
            assert results[method]["ops_per_second"] > 100


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_dimension_vector(self):
        """Test handling of zero-dimension vectors."""
        # Most functions should handle this gracefully or error
        with pytest.raises(ValueError):
            generate_random_vector(0)
    
    def test_analyze_binding_zero_pairs(self):
        """Test analyzing with zero pairs."""
        hrr = create_hrr(dimension=512)
        
        # Should handle gracefully
        results = analyze_binding_capacity(hrr, n_pairs=0, n_trials=1)
        
        # Results should indicate no capacity
        assert results["mean_accuracy"] == 0 or np.isnan(results["mean_accuracy"])
    
    def test_crosstalk_empty_list(self):
        """Test crosstalk with empty vector list."""
        hrr = create_hrr(dimension=512)
        
        crosstalk = measure_crosstalk(hrr, [])
        assert crosstalk == 0.0
    
    def test_associative_capacity_zero_items(self):
        """Test associative capacity with zero items."""
        hrr = create_hrr(dimension=512)
        
        results = test_associative_capacity(hrr, n_items=0)
        
        assert results["n_items"] == 0
        assert results["accuracy"] == 0


class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Create HRR
        hrr = create_hrr(dimension=1024, seed=42)
        
        # Generate orthogonal vectors
        roles = generate_orthogonal_set(1024, 5, method="gram_schmidt", seed=42)
        
        # Generate random fillers
        fillers = [generate_random_vector(1024, method="gaussian", seed=i) 
                  for i in range(5)]
        
        # Test binding capacity
        results = analyze_binding_capacity(hrr, n_pairs=5, n_trials=2)
        assert results["mean_accuracy"] > 0.8
        
        # Create cleanup memory
        items = {f"item_{i}": fillers[i] for i in range(5)}
        cleanup = create_cleanup_memory(hrr, items)
        assert cleanup.size == 5
        
        # Test performance
        perf = benchmark_hrr_performance(dimension=1024, n_operations=50)
        assert perf.operations_per_second > 100
    
    def test_complex_vs_real_analysis(self):
        """Test analysis with both storage methods."""
        dimension = 1024
        
        for storage_method in ["real", "complex"]:
            # Create HRR
            config = HRRConfig(
                dimension=dimension,
                storage_method=storage_method,
                seed=42
            )
            hrr = create_hrr(
                dimension=dimension,
                storage_method=storage_method,
                seed=42
            )
            
            # Test operations
            v1 = hrr.generate_vector()
            v2 = hrr.generate_vector()
            
            # Binding should work
            bound = hrr.bind(v1, v2)
            retrieved = hrr.unbind(bound, v1)
            
            similarity = hrr.similarity(retrieved, v2)
            assert similarity > 0.9