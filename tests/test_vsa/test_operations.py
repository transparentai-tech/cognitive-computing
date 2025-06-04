"""
Tests for VSA operations.

Tests permutation, thinning, bundling, normalization, and other
VSA-specific operations.
"""

import pytest
import numpy as np
from typing import List

from cognitive_computing.vsa.operations import (
    permute, inverse_permute, generate_permutation,
    thin, thicken, bundle, normalize_vector,
    measure_sparsity, create_sparse_vector
)
from cognitive_computing.vsa.core import VectorType


class TestPermutationOperations:
    """Test permutation operations."""
    
    def test_cyclic_shift_forward(self):
        """Test forward cyclic shift."""
        vec = np.array([1, 2, 3, 4, 5])
        
        # Shift by 1
        shifted1 = permute(vec, shift=1)
        assert np.array_equal(shifted1, np.array([5, 1, 2, 3, 4]))
        
        # Shift by 2
        shifted2 = permute(vec, shift=2)
        assert np.array_equal(shifted2, np.array([4, 5, 1, 2, 3]))
        
    def test_cyclic_shift_backward(self):
        """Test backward cyclic shift."""
        vec = np.array([1, 2, 3, 4, 5])
        
        # Shift by -1
        shifted = permute(vec, shift=-1)
        assert np.array_equal(shifted, np.array([2, 3, 4, 5, 1]))
        
    def test_cyclic_shift_wraparound(self):
        """Test cyclic shift with wraparound."""
        vec = np.array([1, 2, 3, 4, 5])
        
        # Shift by length should return original
        shifted = permute(vec, shift=5)
        assert np.array_equal(shifted, vec)
        
        # Shift by multiple of length
        shifted = permute(vec, shift=15)  # 3 * 5
        assert np.array_equal(shifted, vec)
        
    def test_permute_array_types(self):
        """Test permutation with different array types."""
        # Binary
        binary = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
        perm_binary = permute(binary, shift=1)
        assert np.array_equal(perm_binary, np.array([1, 0, 1, 0, 1]))
        
        # Bipolar
        bipolar = np.array([1, -1, 1, -1, 1], dtype=np.float32)
        perm_bipolar = permute(bipolar, shift=2)
        assert np.array_equal(perm_bipolar, np.array([-1, 1, 1, -1, 1]))
        
        # Complex
        complex_vec = np.exp(1j * np.linspace(0, 2*np.pi, 5, endpoint=False))
        perm_complex = permute(complex_vec, shift=1)
        expected = np.roll(complex_vec, 1)
        assert np.allclose(perm_complex, expected)
        
    def test_custom_permutation(self):
        """Test permutation with custom indices."""
        vec = np.array([1, -1, 1, -1])
        perm_indices = np.array([2, 3, 0, 1])  # Swap pairs
        
        permuted = permute(vec, permutation=perm_indices)
        expected = np.array([1, -1, 1, -1])
        assert np.array_equal(permuted, expected)
        
    def test_inverse_permute(self):
        """Test inverse permutation."""
        vec = np.array([1, -1, 1, -1, 1])
        
        # Shift permutation
        permuted = permute(vec, shift=2)
        recovered = inverse_permute(permuted, shift=2)
        assert np.array_equal(recovered, vec)
        
        # Custom permutation
        perm_indices = np.array([4, 0, 3, 1, 2])
        permuted = permute(vec, permutation=perm_indices)
        recovered = inverse_permute(permuted, permutation=perm_indices)
        assert np.array_equal(recovered, vec)
        
    def test_generate_permutation(self):
        """Test permutation generation."""
        # Test different sizes
        for size in [5, 10, 100]:
            perm = generate_permutation(size, seed=42)
            assert len(perm) == size
            assert set(perm) == set(range(size))  # All indices present
            
        # Test reproducibility with seed
        perm1 = generate_permutation(100, seed=42)
        perm2 = generate_permutation(100, seed=42)
        assert np.array_equal(perm1, perm2)
        
        # Test different seeds give different results
        perm3 = generate_permutation(100, seed=123)
        assert not np.array_equal(perm1, perm3)
        
    def test_generate_permutation_types(self):
        """Test different permutation types."""
        # Cyclic permutation
        cyclic = generate_permutation(5, permutation_type="cyclic")
        assert np.array_equal(cyclic, np.array([4, 0, 1, 2, 3]))
        
        # Block permutation
        block = generate_permutation(6, permutation_type="block")
        assert np.array_equal(block, np.array([3, 4, 5, 0, 1, 2]))
        
        # Hierarchical (bit-reversal) for power of 2
        hier = generate_permutation(8, permutation_type="hierarchical")
        # Bit reversal of 0,1,2,3,4,5,6,7 is 0,4,2,6,1,5,3,7
        expected = np.array([0, 4, 2, 6, 1, 5, 3, 7])
        assert np.array_equal(hier, expected)


class TestThinningOperations:
    """Test thinning and thickening operations."""
    
    def test_thin_array(self):
        """Test thinning arrays."""
        # Start with a vector with varying magnitudes
        vec = np.array([0.1, -0.2, 2, 0.5, -0.3, 3, 0.8, -0.7, 4, 0.9], dtype=np.float32)
        
        # Thin to 50% sparsity
        thinned = thin(vec, sparsity=0.5)
        sparsity = measure_sparsity(thinned)
        assert sparsity >= 0.45  # Allow some tolerance
        
        # Should preserve largest magnitude values when using magnitude method
        thinned_mag = thin(vec, sparsity=0.5, method="magnitude")
        nonzero_thinned = thinned_mag != 0
        # Check that largest values are kept (4, 3, 2, and maybe 0.9, 0.8)
        assert thinned_mag[8] == 4  # Largest value
        assert thinned_mag[5] == 3  # Second largest
        assert thinned_mag[2] == 2  # Third largest
        
    def test_thin_methods(self):
        """Test different thinning methods."""
        vec = np.array([5, 1, -5, 0, 3, -1, 4, 0, -4, 2], dtype=np.float32)
        
        # Threshold method
        thinned_thresh = thin(vec, sparsity=0.5, method="threshold")
        assert measure_sparsity(thinned_thresh) >= 0.5
        
        # Random method
        thinned_random = thin(vec, sparsity=0.5, method="random")
        assert measure_sparsity(thinned_random) >= 0.5
        
        # Magnitude method - should keep largest values
        thinned_mag = thin(vec, sparsity=0.5, method="magnitude")
        assert measure_sparsity(thinned_mag) >= 0.5
        # Should keep 5, -5, 4, -4, 3 (the 5 largest magnitude values)
        assert thinned_mag[0] == 5  # Largest positive
        assert thinned_mag[2] == -5  # Largest negative
        
    def test_thicken_array(self):
        """Test thickening arrays."""
        # Start with sparse vector
        data = np.array([1, 0, 0, -1, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        initial_density = 1 - measure_sparsity(data)
        assert abs(initial_density - 0.3) < 0.001  # Allow floating point tolerance
        
        # Thicken to 50% density
        thickened = thicken(data, density=0.5)
        final_density = 1 - measure_sparsity(thickened)
        assert final_density >= 0.5
        
        # Original non-zeros should be preserved
        original_nonzero = data != 0
        assert np.all(thickened[original_nonzero] == data[original_nonzero])
        
    def test_thicken_methods(self):
        """Test different thickening methods."""
        sparse_vec = np.array([1, 0, 0, -1, 0], dtype=np.float32)
        
        # Random method
        thick_random = thicken(sparse_vec, density=0.8, method="random")
        assert 1 - measure_sparsity(thick_random) >= 0.8
        
        # Interpolate method
        thick_interp = thicken(sparse_vec, density=0.8, method="interpolate")
        assert 1 - measure_sparsity(thick_interp) >= 0.8
        
        # Duplicate method
        thick_dup = thicken(sparse_vec, density=0.8, method="duplicate")
        assert 1 - measure_sparsity(thick_dup) >= 0.8
        # New values should be copies of existing non-zeros
        new_values = thick_dup[(sparse_vec == 0) & (thick_dup != 0)]
        assert all(v in [1, -1] for v in new_values)


class TestBundlingOperations:
    """Test bundling and superposition operations."""
    
    def test_bundle_binary_arrays(self):
        """Test bundling binary arrays."""
        vecs = [
            np.array([0, 1, 0, 1], dtype=np.uint8),
            np.array([1, 1, 0, 0], dtype=np.uint8),
            np.array([0, 0, 1, 1], dtype=np.uint8)
        ]
        
        bundled = bundle(vecs, method="majority", vector_type=VectorType.BINARY)
        # The majority method with binary type converts sum > 0 to 1
        # Sum: [0+1+0=1, 1+1+0=2, 0+0+1=1, 1+0+1=2]
        # All positive sums become 1 in binary
        expected = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert np.array_equal(bundled, expected)
        
    def test_bundle_bipolar_arrays(self):
        """Test bundling bipolar arrays."""
        vecs = [
            np.array([1, -1, 1, -1], dtype=np.float32),
            np.array([1, 1, -1, -1], dtype=np.float32),
            np.array([-1, 1, 1, -1], dtype=np.float32)
        ]
        
        bundled = bundle(vecs, method="majority", vector_type=VectorType.BIPOLAR)
        # Sum and sign: [1+1-1=1→1, -1+1+1=1→1, 1-1+1=1→1, -1-1-1=-3→-1]
        expected = np.array([1, 1, 1, -1], dtype=np.float32)
        assert np.array_equal(bundled, expected)
        
    def test_bundle_with_weights(self):
        """Test weighted bundling."""
        vecs = [
            np.array([1, -1, 1, -1], dtype=np.float32),
            np.array([-1, 1, -1, 1], dtype=np.float32)
        ]
        weights = [0.8, 0.2]
        
        bundled = bundle(vecs, weights=weights, method="majority")
        # Weighted sum: [0.8*1+0.2*(-1)=0.6→1, 0.8*(-1)+0.2*1=-0.6→-1, ...]
        expected = np.array([1, -1, 1, -1], dtype=np.float32)
        assert np.array_equal(bundled, expected)
        
    def test_bundle_methods(self):
        """Test different bundling methods."""
        vecs = [
            np.array([1.0, -1.0, 1.0, -1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0]),
            np.array([1.0, 1.0, -1.0, -1.0])
        ]
        
        # Sum method
        bundled_sum = bundle(vecs, method="sum", normalize=False)
        expected_sum = np.array([1.0, 1.0, -1.0, -1.0])
        assert np.array_equal(bundled_sum, expected_sum)
        
        # Average method
        bundled_avg = bundle(vecs, method="average", normalize=False)
        expected_avg = np.array([1/3, 1/3, -1/3, -1/3])
        assert np.allclose(bundled_avg, expected_avg)
        
        # Sample method (stochastic, just check shape)
        bundled_sample = bundle(vecs, method="sample")
        assert bundled_sample.shape == vecs[0].shape
        
    def test_bundle_empty_list(self):
        """Test bundling empty list."""
        with pytest.raises(ValueError, match="Cannot bundle empty list"):
            bundle([])
            
    def test_bundle_mismatched_dimensions(self):
        """Test bundling mismatched dimensions."""
        vecs = [
            np.array([1, -1, 1, -1]),
            np.array([1, -1, 1])
        ]
        
        # Note: The actual implementation might not check this explicitly
        # but it will fail when trying to sum/average arrays of different sizes
        with pytest.raises(ValueError):
            bundle(vecs)


class TestNormalizationOperations:
    """Test normalization operations."""
    
    def test_normalize_binary(self):
        """Test normalizing binary vectors."""
        vec = np.array([0.3, 0.7, 0.1, 0.9, 0.6])
        normalized = normalize_vector(vec, vector_type=VectorType.BINARY)
        
        # Should threshold at 0.5
        expected = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
        assert np.array_equal(normalized, expected)
        
    def test_normalize_bipolar(self):
        """Test normalizing bipolar vectors."""
        # Create non-normalized bipolar (from operations)
        data = np.array([2.0, -3.0, 1.5, -0.5])
        
        normalized = normalize_vector(data, vector_type=VectorType.BIPOLAR)
        assert np.all(np.isin(normalized, [-1, 1]))
        expected = np.array([1, -1, 1, -1], dtype=np.float32)
        assert np.array_equal(normalized, expected)
        
    def test_normalize_ternary(self):
        """Test normalizing ternary vectors."""
        vec = np.array([2, 0.1, -3, 0, 1])
        normalized = normalize_vector(vec, vector_type=VectorType.TERNARY)
        
        # Should threshold into -1, 0, 1
        expected = np.array([1, 0, -1, 0, 1], dtype=np.float32)
        assert np.array_equal(normalized, expected)
        
    def test_normalize_complex(self):
        """Test normalizing complex vectors."""
        # Create non-unit complex vector
        vec = np.array([1+1j, 2+0j, 0+3j, -1-1j])
        normalized = normalize_vector(vec, vector_type=VectorType.COMPLEX)
        
        # All elements should have unit magnitude
        mags = np.abs(normalized)
        assert np.allclose(mags, 1.0)
        
    def test_normalize_methods(self):
        """Test different normalization methods."""
        vec = np.array([3.0, 4.0, 0.0])
        
        # L2 norm
        norm_l2 = normalize_vector(vec, method="L2")
        assert np.allclose(np.linalg.norm(norm_l2), 1.0)
        
        # L1 norm
        norm_l1 = normalize_vector(vec, method="L1")
        assert np.allclose(np.sum(np.abs(norm_l1)), 1.0)
        
        # Max norm
        norm_max = normalize_vector(vec, method="max")
        assert np.allclose(np.max(np.abs(norm_max)), 1.0)
        
        # Sign norm
        norm_sign = normalize_vector(vec, method="sign")
        expected = np.array([1, 1, 0])
        assert np.array_equal(norm_sign, expected)
        
    def test_normalize_zero_vector(self):
        """Test normalizing zero vector."""
        # Zero vector
        vec = np.zeros(5)
        
        # Should handle gracefully
        normalized = normalize_vector(vec, method="L2")
        assert np.array_equal(normalized, vec)  # Returns original


class TestSparseVectorCreation:
    """Test sparse vector creation."""
    
    def test_create_sparse_binary(self):
        """Test creating sparse binary vectors."""
        vec = create_sparse_vector(100, 10, vector_type=VectorType.BINARY, seed=42)
        
        assert len(vec) == 100
        assert np.count_nonzero(vec) == 10
        assert vec.dtype == np.uint8
        assert np.all(np.isin(vec, [0, 1]))
        
    def test_create_sparse_bipolar(self):
        """Test creating sparse bipolar vectors."""
        vec = create_sparse_vector(100, 20, vector_type=VectorType.BIPOLAR, seed=42)
        
        assert len(vec) == 100
        assert np.count_nonzero(vec) == 20
        assert np.all(np.isin(vec[vec != 0], [-1, 1]))
        
    def test_create_sparse_ternary(self):
        """Test creating sparse ternary vectors."""
        vec = create_sparse_vector(100, 15, vector_type=VectorType.TERNARY, seed=42)
        
        assert len(vec) == 100
        assert np.count_nonzero(vec) == 15
        assert np.all(np.isin(vec, [-1, 0, 1]))
        
    def test_create_sparse_complex(self):
        """Test creating sparse complex vectors."""
        vec = create_sparse_vector(50, 5, vector_type=VectorType.COMPLEX, seed=42)
        
        assert len(vec) == 50
        assert np.count_nonzero(vec) == 5
        assert vec.dtype == np.complex64
        
        # Non-zero elements should have unit magnitude
        nonzero_mags = np.abs(vec[vec != 0])
        assert np.allclose(nonzero_mags, 1.0)
        
    def test_sparse_reproducibility(self):
        """Test reproducibility with seed."""
        vec1 = create_sparse_vector(100, 10, seed=42)
        vec2 = create_sparse_vector(100, 10, seed=42)
        assert np.array_equal(vec1, vec2)
        
        vec3 = create_sparse_vector(100, 10, seed=123)
        assert not np.array_equal(vec1, vec3)


class TestAdvancedOperations:
    """Test advanced VSA operations."""
    
    def test_permutation_composition(self):
        """Test composing multiple permutations."""
        vec = np.array([1, -1, 1, -1, 1])
        
        # Apply multiple permutations
        perm1 = permute(vec, shift=1)
        perm2 = permute(perm1, shift=2)
        
        # Should be equivalent to single permutation
        perm_combined = permute(vec, shift=3)
        assert np.array_equal(perm2, perm_combined)
        
    def test_thinning_preservation(self):
        """Test that thinning preserves important values."""
        # Create vector with clear important values
        vec = np.array([5, 1, -5, 0, 3, -1, 4, 0, -4, 2], dtype=np.float32)
        
        # Thin to 50% sparsity using magnitude method
        thinned = thin(vec, sparsity=0.5, method="magnitude")
        
        # Should keep largest magnitude values (5, -5, 4, -4, 3)
        assert thinned[0] == 5
        assert thinned[2] == -5
        assert thinned[6] == 4
        assert thinned[8] == -4
        assert thinned[4] == 3
        
    def test_bundle_large_set(self):
        """Test bundling large set of vectors."""
        # Create 100 random bipolar vectors
        np.random.seed(42)
        vecs = [np.random.choice([-1, 1], 1000).astype(np.float32)
                for i in range(100)]
        
        bundled = bundle(vecs, method="majority", vector_type=VectorType.BIPOLAR)
        assert len(bundled) == 1000
        
        # Should have some similarity to all inputs
        similarities = [np.corrcoef(bundled, v)[0, 1] for v in vecs]
        assert all(s > -0.2 for s in similarities)  # Not negatively correlated
        assert np.mean(similarities) > 0  # Positive average similarity
        
    def test_iterative_thinning(self):
        """Test iterative thinning process."""
        np.random.seed(42)
        # Start with a dense vector (all non-zero)
        vec = create_sparse_vector(100, 100, vector_type=VectorType.TERNARY, seed=42)
        
        # Progressively thin
        sparsities = [0.3, 0.5, 0.7, 0.9]
        current = vec
        
        for target_sparsity in sparsities:
            current = thin(current, sparsity=target_sparsity, method="magnitude")
            actual_sparsity = measure_sparsity(current)
            # Check that we're at least close to target (within 10%)
            assert actual_sparsity >= target_sparsity - 0.1
            
        # Should have very few non-zeros left
        assert np.sum(current != 0) <= 10


class TestOperationProperties:
    """Test mathematical properties of operations."""
    
    def test_permutation_preserves_norm(self):
        """Test that permutation preserves vector norm."""
        np.random.seed(42)
        vectors = [
            np.random.choice([-1, 1], 100).astype(np.float32),
            create_sparse_vector(100, 20, VectorType.TERNARY, seed=42),
            create_sparse_vector(100, 100, VectorType.COMPLEX, seed=42)
        ]
        
        for vec in vectors:
            original_norm = np.linalg.norm(vec)
            permuted = permute(vec, shift=37)
            permuted_norm = np.linalg.norm(permuted)
            
            assert abs(original_norm - permuted_norm) < 1e-10
            
    def test_bundle_commutativity(self):
        """Test that bundling is order-independent for certain methods."""
        np.random.seed(42)
        vecs1 = [
            np.random.choice([-1, 1], 50).astype(np.float32),
            np.random.choice([-1, 1], 50).astype(np.float32),
            np.random.choice([-1, 1], 50).astype(np.float32)
        ]
        
        vecs2 = [vecs1[2], vecs1[0], vecs1[1]]  # Reordered
        
        # Sum and average should be commutative
        bundled1 = bundle(vecs1, method="sum", normalize=False)
        bundled2 = bundle(vecs2, method="sum", normalize=False)
        assert np.allclose(bundled1, bundled2)
        
        bundled1 = bundle(vecs1, method="average", normalize=False)
        bundled2 = bundle(vecs2, method="average", normalize=False)
        assert np.allclose(bundled1, bundled2)
        
    def test_inverse_permutation_identity(self):
        """Test that inverse permutation gives identity."""
        np.random.seed(42)
        vec = np.random.randn(100)
        
        # Test with shift
        for shift in [1, 5, 17, 50]:
            permuted = permute(vec, shift=shift)
            recovered = inverse_permute(permuted, shift=shift)
            assert np.allclose(recovered, vec)
            
        # Test with random permutation
        perm_indices = generate_permutation(100, seed=42)
        permuted = permute(vec, permutation=perm_indices)
        recovered = inverse_permute(permuted, permutation=perm_indices)
        assert np.allclose(recovered, vec)


class TestEdgeCases:
    """Test edge cases for operations."""
    
    def test_single_element_operations(self):
        """Test operations on single-element vectors."""
        vec = np.array([1.0])
        
        # Permutation should return same
        permuted = permute(vec, shift=1)
        assert np.array_equal(permuted, vec)
        
        # Bundle single vector
        bundled = bundle([vec])
        assert np.array_equal(bundled, vec)
        
    def test_extreme_sparsity(self):
        """Test extreme sparsity values."""
        # Start with a dense vector 
        vec = create_sparse_vector(100, 100, VectorType.TERNARY, seed=42)
        
        # Thin to 99% sparsity
        very_sparse = thin(vec, sparsity=0.99, method="magnitude")
        assert measure_sparsity(very_sparse) >= 0.98  # Allow small tolerance
        assert np.sum(very_sparse != 0) >= 1  # At least one non-zero
        
        # Start with a sparse vector for thickening test
        sparse_vec = create_sparse_vector(100, 10, VectorType.TERNARY, seed=42)
        
        # Thicken to 100% density (all non-zero)
        very_dense = thicken(sparse_vec, density=1.0)
        assert measure_sparsity(very_dense) == 0.0
        
    def test_empty_permutation(self):
        """Test permutation edge cases."""
        # Empty array
        empty = np.array([])
        permuted = permute(empty, shift=1)
        assert len(permuted) == 0
        
    def test_zero_thin(self):
        """Test thinning with zero sparsity."""
        vec = np.array([1, -1, 1, -1])
        thinned = thin(vec, sparsity=0.0)
        assert np.array_equal(thinned, vec)
        
    def test_full_thin(self):
        """Test thinning with full sparsity."""
        vec = np.array([1, -1, 1, -1])
        thinned = thin(vec, sparsity=1.0)
        assert np.array_equal(thinned, np.zeros_like(vec))
        
    def test_large_permutation_shift(self):
        """Test permutation with very large shift."""
        vec = np.random.randn(10)
        
        # Shift by multiple of length plus offset
        large_shift = 12345  # = 1234 * 10 + 5
        permuted = permute(vec, shift=large_shift)
        
        # Should be same as shift by 5
        expected = permute(vec, shift=5)
        assert np.allclose(permuted, expected)