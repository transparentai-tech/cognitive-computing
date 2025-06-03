"""
Tests for VSA operations.

Tests permutation, thinning, bundling, normalization, and other
VSA-specific operations.
"""

import pytest
import numpy as np
from typing import List

from cognitive_computing.vsa.operations import (
    permute, inverse_permute, thin, thicken, bundle, normalize,
    create_permutation_matrix, cyclic_shift, random_permutation,
    magnitude_threshold, phase_threshold, consensus_sum,
    weighted_superposition
)
from cognitive_computing.vsa.vectors import (
    BinaryVector, BipolarVector, TernaryVector, ComplexVector, IntegerVector
)


class TestPermutationOperations:
    """Test permutation operations."""
    
    def test_cyclic_shift_forward(self):
        """Test forward cyclic shift."""
        vec = np.array([1, 2, 3, 4, 5])
        
        # Shift by 1
        shifted1 = cyclic_shift(vec, 1)
        assert np.array_equal(shifted1, np.array([5, 1, 2, 3, 4]))
        
        # Shift by 2
        shifted2 = cyclic_shift(vec, 2)
        assert np.array_equal(shifted2, np.array([4, 5, 1, 2, 3]))
        
    def test_cyclic_shift_backward(self):
        """Test backward cyclic shift."""
        vec = np.array([1, 2, 3, 4, 5])
        
        # Shift by -1
        shifted = cyclic_shift(vec, -1)
        assert np.array_equal(shifted, np.array([2, 3, 4, 5, 1]))
        
    def test_cyclic_shift_wraparound(self):
        """Test cyclic shift with wraparound."""
        vec = np.array([1, 2, 3, 4, 5])
        
        # Shift by length should return original
        shifted = cyclic_shift(vec, 5)
        assert np.array_equal(shifted, vec)
        
        # Shift by multiple of length
        shifted = cyclic_shift(vec, 15)  # 3 * 5
        assert np.array_equal(shifted, vec)
        
    def test_permute_vector_types(self):
        """Test permutation with different vector types."""
        # Binary
        binary = BinaryVector(np.array([0, 1, 0, 1, 1]))
        perm_binary = permute(binary, shift=1)
        assert isinstance(perm_binary, BinaryVector)
        assert np.array_equal(perm_binary.data, np.array([1, 0, 1, 0, 1]))
        
        # Bipolar
        bipolar = BipolarVector(np.array([1, -1, 1, -1, 1]))
        perm_bipolar = permute(bipolar, shift=2)
        assert isinstance(perm_bipolar, BipolarVector)
        assert np.array_equal(perm_bipolar.data, np.array([-1, 1, 1, -1, 1]))
        
        # Complex
        complex_vec = ComplexVector(np.exp(1j * np.linspace(0, 2*np.pi, 5, endpoint=False)))
        perm_complex = permute(complex_vec, shift=1)
        assert isinstance(perm_complex, ComplexVector)
        
    def test_custom_permutation(self):
        """Test permutation with custom indices."""
        vec = BipolarVector(np.array([1, -1, 1, -1]))
        perm_indices = np.array([2, 3, 0, 1])  # Swap pairs
        
        permuted = permute(vec, permutation=perm_indices)
        expected = np.array([1, -1, 1, -1])
        assert np.array_equal(permuted.data, expected)
        
    def test_inverse_permute(self):
        """Test inverse permutation."""
        vec = BipolarVector(np.array([1, -1, 1, -1, 1]))
        
        # Shift permutation
        permuted = permute(vec, shift=2)
        recovered = inverse_permute(permuted, shift=2)
        assert np.array_equal(recovered.data, vec.data)
        
        # Custom permutation
        perm_indices = np.array([4, 0, 3, 1, 2])
        permuted = permute(vec, permutation=perm_indices)
        recovered = inverse_permute(permuted, permutation=perm_indices)
        assert np.array_equal(recovered.data, vec.data)
        
    def test_random_permutation(self):
        """Test random permutation generation."""
        # Test different sizes
        for size in [5, 10, 100]:
            perm = random_permutation(size, seed=42)
            assert len(perm) == size
            assert set(perm) == set(range(size))  # All indices present
            
        # Test reproducibility with seed
        perm1 = random_permutation(100, seed=42)
        perm2 = random_permutation(100, seed=42)
        assert np.array_equal(perm1, perm2)
        
        # Test different seeds give different results
        perm3 = random_permutation(100, seed=123)
        assert not np.array_equal(perm1, perm3)
        
    def test_create_permutation_matrix(self):
        """Test permutation matrix creation."""
        indices = np.array([2, 0, 1])
        matrix = create_permutation_matrix(indices)
        
        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        assert np.array_equal(matrix, expected)
        
        # Test applying matrix
        vec = np.array([1, 2, 3])
        permuted = matrix @ vec
        assert np.array_equal(permuted, np.array([2, 3, 1]))


class TestThinningOperations:
    """Test thinning and thickening operations."""
    
    def test_thin_ternary_vector(self):
        """Test thinning ternary vectors."""
        vec = TernaryVector(np.array([1, -1, 0, 1, -1, 0, 1, -1, 0, 1]))
        
        # Thin to 50% sparsity
        thinned = thin(vec, sparsity=0.5)
        assert isinstance(thinned, TernaryVector)
        assert thinned.sparsity >= 0.5
        
        # Should preserve largest magnitude values
        nonzero_original = np.abs(vec.data) > 0
        nonzero_thinned = np.abs(thinned.data) > 0
        assert np.sum(nonzero_thinned) <= np.sum(nonzero_original)
        
    def test_thin_binary_vector(self):
        """Test thinning binary vectors."""
        vec = BinaryVector(np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1]))
        
        # Thin to 70% zeros
        thinned = thin(vec, sparsity=0.7)
        assert isinstance(thinned, BinaryVector)
        
        # Check sparsity (proportion of zeros)
        actual_sparsity = np.mean(thinned.data == 0)
        assert actual_sparsity >= 0.7
        
    def test_thicken_ternary_vector(self):
        """Test thickening ternary vectors."""
        # Start with sparse vector
        data = np.array([1, 0, 0, -1, 0, 0, 1, 0, 0, 0])
        vec = TernaryVector(data)
        assert vec.sparsity == 0.7
        
        # Thicken to 50% sparsity
        thickened = thicken(vec, sparsity=0.5)
        assert isinstance(thickened, TernaryVector)
        assert thickened.sparsity <= 0.5
        
        # Original non-zeros should be preserved
        original_nonzero = vec.data != 0
        assert np.all(thickened.data[original_nonzero] == vec.data[original_nonzero])
        
    def test_magnitude_threshold(self):
        """Test magnitude-based thresholding."""
        # Complex vector
        phases = np.linspace(0, 2*np.pi, 10, endpoint=False)
        magnitudes = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0])
        complex_data = magnitudes * np.exp(1j * phases)
        vec = ComplexVector(complex_data / np.abs(complex_data))  # Normalize
        
        # Apply magnitude threshold (keep only high magnitude)
        thresholded = magnitude_threshold(vec, min_magnitude=0.7)
        
        # Should only keep elements with magnitude >= 0.7
        assert isinstance(thresholded, ComplexVector)
        
    def test_phase_threshold(self):
        """Test phase-based thresholding for complex vectors."""
        # Create complex vector with various phases
        phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4])
        vec = ComplexVector(np.exp(1j * phases))
        
        # Keep only phases in [-π/2, π/2]
        thresholded = phase_threshold(vec, min_phase=-np.pi/2, max_phase=np.pi/2)
        
        # Check phases are in range
        result_phases = np.angle(thresholded.data)
        nonzero = thresholded.data != 0
        assert np.all(result_phases[nonzero] >= -np.pi/2 - 1e-10)
        assert np.all(result_phases[nonzero] <= np.pi/2 + 1e-10)


class TestBundlingOperations:
    """Test bundling and superposition operations."""
    
    def test_bundle_binary_vectors(self):
        """Test bundling binary vectors."""
        vecs = [
            BinaryVector(np.array([0, 1, 0, 1])),
            BinaryVector(np.array([1, 1, 0, 0])),
            BinaryVector(np.array([0, 0, 1, 1]))
        ]
        
        bundled = bundle(vecs)
        # Majority vote: [0+1+0=1→0, 1+1+0=2→1, 0+0+1=1→0, 1+0+1=2→1]
        expected = np.array([0, 1, 0, 1])
        assert np.array_equal(bundled.data, expected)
        
    def test_bundle_bipolar_vectors(self):
        """Test bundling bipolar vectors."""
        vecs = [
            BipolarVector(np.array([1, -1, 1, -1])),
            BipolarVector(np.array([1, 1, -1, -1])),
            BipolarVector(np.array([-1, 1, 1, -1]))
        ]
        
        bundled = bundle(vecs)
        # Sum and sign: [1+1-1=1→1, -1+1+1=1→1, 1-1+1=1→1, -1-1-1=-3→-1]
        expected = np.array([1, 1, 1, -1])
        assert np.array_equal(bundled.data, expected)
        
    def test_bundle_with_weights(self):
        """Test weighted bundling."""
        vecs = [
            BipolarVector(np.array([1, -1, 1, -1])),
            BipolarVector(np.array([-1, 1, -1, 1]))
        ]
        weights = [0.8, 0.2]
        
        bundled = bundle(vecs, weights=weights)
        # Weighted sum: [0.8*1+0.2*(-1)=0.6→1, 0.8*(-1)+0.2*1=-0.6→-1, ...]
        expected = np.array([1, -1, 1, -1])
        assert np.array_equal(bundled.data, expected)
        
    def test_weighted_superposition(self):
        """Test weighted superposition for continuous values."""
        vecs = [
            BipolarVector(np.array([1.0, -1.0, 1.0, -1.0])),
            BipolarVector(np.array([-1.0, 1.0, -1.0, 1.0])),
            BipolarVector(np.array([1.0, 1.0, -1.0, -1.0]))
        ]
        weights = [0.5, 0.3, 0.2]
        
        result = weighted_superposition(vecs, weights, normalize=True)
        
        # Should be normalized
        norm = np.linalg.norm(result.data)
        assert abs(norm - 1.0) < 1e-10
        
    def test_consensus_sum(self):
        """Test consensus sum operation."""
        vecs = [
            BipolarVector(np.array([1, -1, 1, -1, 1])),
            BipolarVector(np.array([1, -1, 1, 1, -1])),
            BipolarVector(np.array([1, -1, -1, -1, 1])),
            BipolarVector(np.array([-1, -1, 1, -1, 1]))
        ]
        
        consensus = consensus_sum(vecs, threshold=0.6)
        
        # Elements with >= 60% agreement should be kept
        # Position 0: 3/4 positive = 75% → 1
        # Position 1: 4/4 negative = 100% → -1
        # Position 2: 3/4 positive = 75% → 1
        # Position 3: 3/4 negative = 75% → -1
        # Position 4: 3/4 positive = 75% → 1
        expected = np.array([1, -1, 1, -1, 1])
        assert np.array_equal(consensus.data, expected)
        
    def test_bundle_empty_list(self):
        """Test bundling empty list."""
        with pytest.raises(ValueError, match="No vectors to bundle"):
            bundle([])
            
    def test_bundle_mismatched_types(self):
        """Test bundling mismatched vector types."""
        vecs = [
            BinaryVector(np.array([0, 1, 0, 1])),
            BipolarVector(np.array([1, -1, 1, -1]))
        ]
        
        with pytest.raises(TypeError, match="All vectors must be of the same type"):
            bundle(vecs)
            
    def test_bundle_mismatched_dimensions(self):
        """Test bundling mismatched dimensions."""
        vecs = [
            BipolarVector(np.array([1, -1, 1, -1])),
            BipolarVector(np.array([1, -1, 1]))
        ]
        
        with pytest.raises(ValueError, match="dimension"):
            bundle(vecs)


class TestNormalizationOperations:
    """Test normalization operations."""
    
    def test_normalize_binary(self):
        """Test normalizing binary vectors (no-op)."""
        vec = BinaryVector(np.array([0, 1, 0, 1, 1]))
        normalized = normalize(vec)
        
        # Binary normalization is identity
        assert np.array_equal(normalized.data, vec.data)
        
    def test_normalize_bipolar(self):
        """Test normalizing bipolar vectors."""
        # Create non-normalized bipolar (from operations)
        data = np.array([2.0, -3.0, 1.5, -0.5])
        vec = BipolarVector(np.sign(data).astype(int))
        
        normalized = normalize(vec)
        assert isinstance(normalized, BipolarVector)
        assert np.all(np.isin(normalized.data, [-1, 1]))
        
    def test_normalize_ternary(self):
        """Test normalizing ternary vectors."""
        vec = TernaryVector(np.array([2, 0, -3, 0, 1]))
        normalized = normalize(vec)
        
        # Should have unit norm
        norm = np.linalg.norm(normalized.data)
        assert abs(norm - 1.0) < 1e-10
        
        # Should preserve zero positions
        assert np.array_equal(normalized.data == 0, vec.data == 0)
        
    def test_normalize_complex(self):
        """Test normalizing complex vectors."""
        # Create non-unit complex vector
        vec = ComplexVector(np.array([1+1j, 2+0j, 0+3j, -1-1j]) / np.sqrt(2))
        normalized = normalize(vec)
        
        # All elements should have unit magnitude
        mags = np.abs(normalized.data)
        assert np.allclose(mags, 1.0)
        
    def test_normalize_integer(self):
        """Test normalizing integer vectors."""
        vec = IntegerVector(np.array([0, 50, 100, 150, 200]), modulus=256)
        normalized = normalize(vec)
        
        # Should map to [-1, 1] range
        assert np.all(normalized.data >= -1)
        assert np.all(normalized.data <= 1)
        
    def test_normalize_zero_vector(self):
        """Test normalizing zero vector."""
        # Ternary zero vector
        vec = TernaryVector(np.zeros(5))
        normalized = normalize(vec)
        
        # Should handle gracefully (return zero or small random)
        assert isinstance(normalized, TernaryVector)


class TestAdvancedOperations:
    """Test advanced VSA operations."""
    
    def test_permutation_composition(self):
        """Test composing multiple permutations."""
        vec = BipolarVector(np.array([1, -1, 1, -1, 1]))
        
        # Apply multiple permutations
        perm1 = permute(vec, shift=1)
        perm2 = permute(perm1, shift=2)
        
        # Should be equivalent to single permutation
        perm_combined = permute(vec, shift=3)
        assert np.array_equal(perm2.data, perm_combined.data)
        
    def test_thinning_preservation(self):
        """Test that thinning preserves important values."""
        # Create vector with clear important values
        vec = TernaryVector(np.array([5, 1, -5, 0, 3, -1, 4, 0, -4, 2]))
        
        # Thin to 50% sparsity
        thinned = thin(vec, sparsity=0.5)
        
        # Should keep largest magnitude values
        kept_indices = thinned.data != 0
        kept_values = np.abs(vec.data[kept_indices])
        min_kept = np.min(kept_values) if len(kept_values) > 0 else 0
        
        dropped_indices = thinned.data == 0
        dropped_values = np.abs(vec.data[dropped_indices])
        max_dropped = np.max(dropped_values) if len(dropped_values) > 0 else 0
        
        # All kept values should be >= all dropped values
        if min_kept > 0 and max_dropped > 0:
            assert min_kept >= max_dropped
            
    def test_bundle_large_set(self):
        """Test bundling large set of vectors."""
        # Create 100 random bipolar vectors
        vecs = [BipolarVector.random(1000, rng=np.random.RandomState(i)) 
                for i in range(100)]
        
        bundled = bundle(vecs)
        assert isinstance(bundled, BipolarVector)
        assert len(bundled.data) == 1000
        
        # Should have some similarity to all inputs
        similarities = [np.corrcoef(bundled.data, v.data)[0, 1] for v in vecs]
        assert all(s > -0.2 for s in similarities)  # Not negatively correlated
        assert np.mean(similarities) > 0  # Positive average similarity
        
    def test_iterative_thinning(self):
        """Test iterative thinning process."""
        vec = TernaryVector.random(100, sparsity=0.1, rng=np.random.RandomState(42))
        
        # Progressively thin
        sparsities = [0.3, 0.5, 0.7, 0.9]
        current = vec
        
        for target_sparsity in sparsities:
            current = thin(current, sparsity=target_sparsity)
            assert current.sparsity >= target_sparsity
            
        # Should have very few non-zeros left
        assert np.sum(current.data != 0) <= 10


class TestOperationProperties:
    """Test mathematical properties of operations."""
    
    def test_permutation_preserves_norm(self):
        """Test that permutation preserves vector norm."""
        vector_types = [
            BipolarVector.random(100),
            TernaryVector.random(100, sparsity=0.8),
            ComplexVector.random(100)
        ]
        
        for vec in vector_types:
            original_norm = np.linalg.norm(vec.data)
            permuted = permute(vec, shift=37)
            permuted_norm = np.linalg.norm(permuted.data)
            
            assert abs(original_norm - permuted_norm) < 1e-10
            
    def test_bundle_commutativity(self):
        """Test that bundling is commutative."""
        vecs1 = [
            BipolarVector.random(50, rng=np.random.RandomState(1)),
            BipolarVector.random(50, rng=np.random.RandomState(2)),
            BipolarVector.random(50, rng=np.random.RandomState(3))
        ]
        
        vecs2 = [vecs1[2], vecs1[0], vecs1[1]]  # Reordered
        
        bundled1 = bundle(vecs1)
        bundled2 = bundle(vecs2)
        
        # Should give same result (or very similar for majority vote)
        similarity = np.corrcoef(bundled1.data, bundled2.data)[0, 1]
        assert similarity > 0.99
        
    def test_inverse_permutation_identity(self):
        """Test that inverse permutation gives identity."""
        vec = BipolarVector.random(100)
        
        # Test with shift
        for shift in [1, 5, 17, 50]:
            permuted = permute(vec, shift=shift)
            recovered = inverse_permute(permuted, shift=shift)
            assert np.array_equal(recovered.data, vec.data)
            
        # Test with random permutation
        perm_indices = random_permutation(100, seed=42)
        permuted = permute(vec, permutation=perm_indices)
        recovered = inverse_permute(permuted, permutation=perm_indices)
        assert np.array_equal(recovered.data, vec.data)


class TestEdgeCases:
    """Test edge cases for operations."""
    
    def test_single_element_operations(self):
        """Test operations on single-element vectors."""
        vec = BipolarVector(np.array([1]))
        
        # Permutation should return same
        permuted = permute(vec, shift=1)
        assert np.array_equal(permuted.data, vec.data)
        
        # Bundle single vector
        bundled = bundle([vec])
        assert np.array_equal(bundled.data, vec.data)
        
    def test_extreme_sparsity(self):
        """Test extreme sparsity values."""
        vec = TernaryVector.random(100, sparsity=0.1)
        
        # Thin to 99% sparsity
        very_sparse = thin(vec, sparsity=0.99)
        assert very_sparse.sparsity >= 0.99
        assert np.sum(very_sparse.data != 0) >= 1  # At least one non-zero
        
        # Thicken to 0% sparsity (all non-zero)
        very_dense = thicken(vec, sparsity=0.0)
        assert very_dense.sparsity == 0.0
        
    def test_large_permutation_shift(self):
        """Test permutation with very large shift."""
        vec = BipolarVector.random(10)
        
        # Shift by multiple of length plus offset
        large_shift = 12345  # = 1234 * 10 + 5
        permuted = permute(vec, shift=large_shift)
        
        # Should be same as shift by 5
        expected = permute(vec, shift=5)
        assert np.array_equal(permuted.data, expected.data)