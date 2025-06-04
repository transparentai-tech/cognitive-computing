"""Tests for HDC operations."""

import pytest
import numpy as np

from cognitive_computing.hdc.operations import (
    BundlingMethod,
    PermutationMethod,
    bind_hypervectors,
    bundle_hypervectors,
    permute_hypervector,
    similarity,
    noise_hypervector,
    thin_hypervector,
    segment_hypervector,
    concatenate_hypervectors,
    power_hypervector,
    normalize_hypervector,
    protect_sequence,
)


class TestBindOperations:
    """Test binding operations."""
    
    def test_bind_binary(self):
        """Test binary binding (XOR)."""
        a = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        b = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
        
        bound = bind_hypervectors(a, b, hypervector_type="binary")
        expected = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
        assert np.array_equal(bound, expected)
        
        # Test self-inverse property
        unbound = bind_hypervectors(bound, b, hypervector_type="binary")
        assert np.array_equal(unbound, a)
        
    def test_bind_bipolar(self):
        """Test bipolar binding (multiplication)."""
        a = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        b = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        
        bound = bind_hypervectors(a, b, hypervector_type="bipolar")
        expected = np.array([1, -1, -1, 1, 1], dtype=np.int8)
        assert np.array_equal(bound, expected)
        
        # Test self-inverse property
        unbound = bind_hypervectors(bound, b, hypervector_type="bipolar")
        assert np.array_equal(unbound, a)
        
    def test_bind_shape_mismatch(self):
        """Test binding with mismatched shapes."""
        a = np.ones(5)
        b = np.ones(6)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            bind_hypervectors(a, b)
            
    def test_bind_unknown_type(self):
        """Test binding with unknown type."""
        a = np.ones(5)
        b = np.ones(5)
        
        with pytest.raises(ValueError, match="Unknown hypervector type"):
            bind_hypervectors(a, b, hypervector_type="unknown")


class TestBundleOperations:
    """Test bundling operations."""
    
    def test_bundle_majority_binary(self):
        """Test majority bundling for binary vectors."""
        vectors = [
            np.array([1, 0, 1, 0, 1], dtype=np.uint8),
            np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1, 0, 1], dtype=np.uint8),
        ]
        
        bundled = bundle_hypervectors(
            vectors,
            method=BundlingMethod.MAJORITY,
            hypervector_type="binary"
        )
        
        # Majority: [2, 2, 2, 0, 3] -> [1, ?, 1, 0, 1] (? is tie)
        assert bundled[0] == 1
        assert bundled[2] == 1
        assert bundled[3] == 0
        assert bundled[4] == 1
        
    def test_bundle_majority_bipolar(self):
        """Test majority bundling for bipolar vectors."""
        vectors = [
            np.array([1, -1, 1, -1, 1], dtype=np.int8),
            np.array([1, 1, -1, -1, 1], dtype=np.int8),
            np.array([-1, 1, 1, -1, 1], dtype=np.int8),
        ]
        
        bundled = bundle_hypervectors(
            vectors,
            method=BundlingMethod.MAJORITY,
            hypervector_type="bipolar"
        )
        
        # Sum: [1, 1, 1, -3, 3] -> [1, 1, 1, -1, 1]
        expected = np.array([1, 1, 1, -1, 1], dtype=np.int8)
        assert np.array_equal(bundled, expected)
        
    def test_bundle_average(self):
        """Test average bundling."""
        vectors = [
            np.array([1, 0, 1, 0, 1], dtype=np.uint8),
            np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1, 1, 1], dtype=np.uint8),
        ]
        
        bundled = bundle_hypervectors(
            vectors,
            method=BundlingMethod.AVERAGE,
            hypervector_type="binary"
        )
        
        # Average: [2/3, 2/3, 2/3, 1/3, 1] -> [1, 1, 1, 0, 1]
        expected = np.array([1, 1, 1, 0, 1], dtype=np.uint8)
        assert np.array_equal(bundled, expected)
        
    def test_bundle_sample(self):
        """Test sample bundling."""
        np.random.seed(42)
        vectors = [
            np.array([1, 0, 1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 0, 1, 0], dtype=np.uint8),
        ]
        
        bundled = bundle_hypervectors(
            vectors,
            method=BundlingMethod.SAMPLE,
            hypervector_type="binary"
        )
        
        # Each element should come from one of the input vectors
        assert bundled.shape == (5,)
        for i in range(5):
            assert bundled[i] in [vectors[0][i], vectors[1][i]]
            
    def test_bundle_weighted(self):
        """Test weighted bundling."""
        vectors = [
            np.array([1, -1, 1, -1], dtype=np.int8),
            np.array([-1, 1, -1, 1], dtype=np.int8),
        ]
        weights = np.array([0.7, 0.3])
        
        bundled = bundle_hypervectors(
            vectors,
            method=BundlingMethod.WEIGHTED,
            hypervector_type="bipolar",
            weights=weights
        )
        
        # Weighted sum: 0.7*[1,-1,1,-1] + 0.3*[-1,1,-1,1] = [0.4, -0.4, 0.4, -0.4]
        expected = np.array([1, -1, 1, -1], dtype=np.int8)
        assert np.array_equal(bundled, expected)
        
    def test_bundle_empty(self):
        """Test bundling empty list."""
        with pytest.raises(ValueError, match="Cannot bundle empty"):
            bundle_hypervectors([])
            
    def test_bundle_dimension_mismatch(self):
        """Test bundling with dimension mismatch."""
        vectors = [
            np.ones(5),
            np.ones(6)
        ]
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            bundle_hypervectors(vectors)
            
    def test_bundle_weighted_no_weights(self):
        """Test weighted bundling without weights."""
        vectors = [np.ones(5), np.ones(5)]
        
        with pytest.raises(ValueError, match="Weights required"):
            bundle_hypervectors(vectors, method=BundlingMethod.WEIGHTED)


class TestPermuteOperations:
    """Test permutation operations."""
    
    def test_permute_cyclic(self):
        """Test cyclic permutation."""
        hv = np.array([1, 2, 3, 4, 5])
        
        # Shift right by 1
        permuted = permute_hypervector(hv, method=PermutationMethod.CYCLIC, shift=1)
        expected = np.array([5, 1, 2, 3, 4])
        assert np.array_equal(permuted, expected)
        
        # Shift left by 1
        permuted = permute_hypervector(hv, method=PermutationMethod.CYCLIC, shift=-1)
        expected = np.array([2, 3, 4, 5, 1])
        assert np.array_equal(permuted, expected)
        
    def test_permute_random(self):
        """Test random permutation."""
        np.random.seed(42)
        hv = np.arange(10)
        
        permuted = permute_hypervector(hv, method=PermutationMethod.RANDOM)
        
        # Should be a permutation (same elements, different order)
        assert len(permuted) == len(hv)
        assert set(permuted) == set(hv)
        assert not np.array_equal(permuted, hv)  # Should be different
        
    def test_permute_random_custom(self):
        """Test random permutation with custom permutation."""
        hv = np.array([1, 2, 3, 4, 5])
        perm = np.array([4, 3, 2, 1, 0])
        
        permuted = permute_hypervector(
            hv,
            method=PermutationMethod.RANDOM,
            permutation=perm
        )
        expected = np.array([5, 4, 3, 2, 1])
        assert np.array_equal(permuted, expected)
        
    def test_permute_block(self):
        """Test block permutation."""
        hv = np.array([1, 2, 3, 4, 5, 6])
        
        permuted = permute_hypervector(
            hv,
            method=PermutationMethod.BLOCK,
            block_size=2,
            shift=1
        )
        # Blocks: [1,2], [3,4], [5,6] -> shift -> [5,6], [1,2], [3,4]
        expected = np.array([5, 6, 1, 2, 3, 4])
        assert np.array_equal(permuted, expected)
        
    def test_permute_block_invalid(self):
        """Test block permutation with invalid block size."""
        hv = np.ones(7)
        
        with pytest.raises(ValueError, match="Block size required"):
            permute_hypervector(hv, method=PermutationMethod.BLOCK)
            
        with pytest.raises(ValueError, match="divisible by block size"):
            permute_hypervector(hv, method=PermutationMethod.BLOCK, block_size=3)
            
    def test_permute_inverse(self):
        """Test inverse permutation."""
        hv = np.array([1, 2, 3, 4, 5])
        perm = np.array([4, 3, 2, 1, 0])
        
        # Apply permutation
        permuted = permute_hypervector(
            hv,
            method=PermutationMethod.RANDOM,
            permutation=perm
        )
        
        # Apply inverse
        restored = permute_hypervector(
            permuted,
            method=PermutationMethod.INVERSE,
            permutation=perm
        )
        
        assert np.array_equal(restored, hv)
        
    def test_permute_inverse_no_perm(self):
        """Test inverse permutation without permutation array."""
        hv = np.ones(5)
        
        with pytest.raises(ValueError, match="Permutation array required"):
            permute_hypervector(hv, method=PermutationMethod.INVERSE)


class TestSimilarityOperations:
    """Test similarity operations."""
    
    def test_similarity_cosine(self):
        """Test cosine similarity."""
        a = np.array([1, 0, 1, 0])
        b = np.array([1, 0, 1, 0])
        
        # Same vectors
        assert abs(similarity(a, b, metric="cosine") - 1.0) < 1e-10
        
        # Orthogonal vectors
        a = np.array([1, 0])
        b = np.array([0, 1])
        assert abs(similarity(a, b, metric="cosine")) < 1e-10
        
        # Opposite vectors
        a = np.array([1, 1])
        b = np.array([-1, -1])
        assert abs(similarity(a, b, metric="cosine") + 1.0) < 1e-10
        
    def test_similarity_hamming(self):
        """Test Hamming similarity."""
        a = np.array([1, 0, 1, 0])
        b = np.array([1, 0, 1, 0])
        
        # Same vectors
        assert similarity(a, b, metric="hamming") == 1.0
        
        # Opposite vectors
        b = np.array([0, 1, 0, 1])
        assert similarity(a, b, metric="hamming") == -1.0
        
        # Half different
        b = np.array([1, 0, 0, 1])
        assert similarity(a, b, metric="hamming") == 0.0
        
    def test_similarity_euclidean(self):
        """Test Euclidean similarity."""
        a = np.array([1, 0, 1, 0])
        b = np.array([1, 0, 1, 0])
        
        # Same vectors
        assert similarity(a, b, metric="euclidean") == 1.0
        
        # Different vectors
        b = np.array([0, 1, 0, 1])
        sim = similarity(a, b, metric="euclidean")
        assert 0 < sim < 1
        
    def test_similarity_jaccard(self):
        """Test Jaccard similarity."""
        a = np.array([1, 0, 1, 0])
        b = np.array([1, 1, 0, 0])
        
        # Intersection: 1, Union: 3 -> 1/3
        assert abs(similarity(a, b, metric="jaccard") - 1/3) < 1e-6
        
        # Same vectors
        assert similarity(a, a, metric="jaccard") == 1.0
        
        # No overlap
        a = np.array([1, 1, 0, 0])
        b = np.array([0, 0, 1, 1])
        assert similarity(a, b, metric="jaccard") == 0.0
        
    def test_similarity_shape_mismatch(self):
        """Test similarity with shape mismatch."""
        a = np.ones(5)
        b = np.ones(6)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            similarity(a, b)
            
    def test_similarity_unknown_metric(self):
        """Test similarity with unknown metric."""
        a = np.ones(5)
        b = np.ones(5)
        
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            similarity(a, b, metric="unknown")


class TestNoiseOperations:
    """Test noise operations."""
    
    def test_noise_binary(self):
        """Test noise on binary vectors."""
        np.random.seed(42)
        hv = np.ones(100, dtype=np.uint8)
        
        # Add 10% noise
        noisy = noise_hypervector(hv, noise_level=0.1, hypervector_type="binary")
        
        # Should have approximately 10 flips
        n_flips = np.sum(noisy != hv)
        assert 5 < n_flips < 15
        
        # Flipped positions should be 0
        assert np.all(noisy[noisy != hv] == 0)
        
    def test_noise_bipolar(self):
        """Test noise on bipolar vectors."""
        np.random.seed(42)
        hv = np.ones(100, dtype=np.int8)
        
        # Add 20% noise
        noisy = noise_hypervector(hv, noise_level=0.2, hypervector_type="bipolar")
        
        # Should have approximately 20 flips
        n_flips = np.sum(noisy != hv)
        assert 15 < n_flips < 25
        
        # Flipped positions should be -1
        assert np.all(noisy[noisy != hv] == -1)
        
    def test_noise_zero_level(self):
        """Test zero noise level."""
        hv = np.ones(10)
        noisy = noise_hypervector(hv, noise_level=0.0)
        assert np.array_equal(noisy, hv)
        
    def test_noise_invalid_level(self):
        """Test invalid noise level."""
        hv = np.ones(10)
        
        with pytest.raises(ValueError, match="Noise level must be"):
            noise_hypervector(hv, noise_level=-0.1)
            
        with pytest.raises(ValueError, match="Noise level must be"):
            noise_hypervector(hv, noise_level=1.1)


class TestThinningOperations:
    """Test thinning operations."""
    
    def test_thin_hypervector(self):
        """Test hypervector thinning."""
        np.random.seed(42)
        hv = np.ones(100)
        
        # Make 30% sparse
        sparse = thin_hypervector(hv, sparsity=0.3)
        
        # Should have approximately 30 zeros
        n_zeros = np.sum(sparse == 0)
        assert 25 < n_zeros < 35
        
        # Non-zero elements should be unchanged
        assert np.all(sparse[sparse != 0] == 1)
        
    def test_thin_invalid_sparsity(self):
        """Test invalid sparsity values."""
        hv = np.ones(10)
        
        with pytest.raises(ValueError, match="Sparsity must be"):
            thin_hypervector(hv, sparsity=-0.1)
            
        with pytest.raises(ValueError, match="Sparsity must be"):
            thin_hypervector(hv, sparsity=1.1)


class TestSegmentOperations:
    """Test segmentation operations."""
    
    def test_segment_hypervector(self):
        """Test hypervector segmentation."""
        hv = np.arange(12)
        
        segments = segment_hypervector(hv, n_segments=3)
        
        assert len(segments) == 3
        assert np.array_equal(segments[0], [0, 1, 2, 3])
        assert np.array_equal(segments[1], [4, 5, 6, 7])
        assert np.array_equal(segments[2], [8, 9, 10, 11])
        
    def test_segment_indivisible(self):
        """Test segmentation with indivisible dimension."""
        hv = np.ones(10)
        
        with pytest.raises(ValueError, match="not divisible"):
            segment_hypervector(hv, n_segments=3)
            
    def test_concatenate_hypervectors(self):
        """Test hypervector concatenation."""
        segments = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        
        concatenated = concatenate_hypervectors(segments)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert np.array_equal(concatenated, expected)
        
    def test_concatenate_empty(self):
        """Test concatenating empty list."""
        with pytest.raises(ValueError, match="Cannot concatenate empty"):
            concatenate_hypervectors([])


class TestPowerOperations:
    """Test power operations."""
    
    def test_power_binary(self):
        """Test power operation for binary vectors."""
        hv = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        
        # XOR binding function
        def xor_bind(a, b):
            return np.bitwise_xor(a, b)
            
        # Power 0 should give identity (all ones for XOR)
        result = power_hypervector(hv, 0, xor_bind)
        assert np.all(result == 1)
        
        # Power 1 should give original
        result = power_hypervector(hv, 1, xor_bind)
        assert np.array_equal(result, hv)
        
        # Power 2 should give identity (XOR is self-inverse)
        result = power_hypervector(hv, 2, xor_bind)
        assert np.all(result == 0)  # hv XOR hv = 0
        
    def test_power_bipolar(self):
        """Test power operation for bipolar vectors."""
        hv = np.array([1, -1, 1, -1], dtype=np.int8)
        
        # Multiplication binding
        def mult_bind(a, b):
            return a * b
            
        # Power 2
        result = power_hypervector(hv, 2, mult_bind)
        expected = np.ones(4, dtype=np.int8)  # All 1s since (-1)^2 = 1
        assert np.array_equal(result, expected)


class TestNormalizeOperations:
    """Test normalization operations."""
    
    def test_normalize_binary(self):
        """Test binary normalization."""
        # Test with values that need clipping and rounding
        hv = np.array([-0.5, 0.3, 0.8, 1.2, 2.0])
        
        normalized = normalize_hypervector(hv, hypervector_type="binary")
        # Clipping to [0, 1]: [0, 0.3, 0.8, 1, 1]
        # Converting to uint8 truncates: [0, 0, 0, 1, 1]
        expected = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        assert np.array_equal(normalized, expected)
        
    def test_normalize_bipolar(self):
        """Test bipolar normalization."""
        hv = np.array([0.5, -0.5, 0, 2, -2])
        
        normalized = normalize_hypervector(hv, hypervector_type="bipolar")
        expected = np.array([1, -1, 1, 1, -1], dtype=np.int8)
        assert np.array_equal(normalized, expected)
        
    def test_normalize_ternary(self):
        """Test ternary normalization."""
        hv = np.array([0.7, -0.7, 0.3, -0.3, 0])
        
        normalized = normalize_hypervector(hv, hypervector_type="ternary")
        expected = np.array([1, -1, 0, 0, 0], dtype=np.int8)
        assert np.array_equal(normalized, expected)


class TestSequenceProtection:
    """Test sequence protection."""
    
    def test_protect_sequence(self):
        """Test basic sequence protection."""
        np.random.seed(42)
        
        # Create a sequence
        sequence = [
            np.array([1, -1, 1, -1], dtype=np.int8),
            np.array([-1, 1, -1, 1], dtype=np.int8),
            np.array([1, 1, -1, -1], dtype=np.int8),
        ]
        
        protected = protect_sequence(sequence)
        
        # Result should have same dimension
        assert protected.shape == (4,)
        assert protected.dtype == np.int8
        
    def test_protect_sequence_custom_positions(self):
        """Test sequence protection with custom position vectors."""
        sequence = [
            np.array([1, -1, 1, -1], dtype=np.int8),
            np.array([-1, 1, -1, 1], dtype=np.int8),
        ]
        
        position_vectors = [
            np.array([1, 1, 1, 1], dtype=np.int8),
            np.array([-1, -1, -1, -1], dtype=np.int8),
        ]
        
        protected = protect_sequence(sequence, position_vectors)
        
        # Check manual calculation
        # First: [1,-1,1,-1] * [1,1,1,1] = [1,-1,1,-1]
        # Second: [-1,1,-1,1] * [-1,-1,-1,-1] = [1,-1,1,-1]
        # Bundle: majority of [[1,-1,1,-1], [1,-1,1,-1]] = [1,-1,1,-1]
        expected = np.array([1, -1, 1, -1], dtype=np.int8)
        assert np.array_equal(protected, expected)
        
    def test_protect_empty_sequence(self):
        """Test protecting empty sequence."""
        with pytest.raises(ValueError, match="Cannot protect empty"):
            protect_sequence([])
            
    def test_protect_sequence_position_mismatch(self):
        """Test sequence protection with position vector mismatch."""
        sequence = [np.ones(5), np.ones(5)]
        position_vectors = [np.ones(5)]  # Only one position vector
        
        with pytest.raises(ValueError, match="Number of position vectors"):
            protect_sequence(sequence, position_vectors)