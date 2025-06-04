"""Tests for hypervector types and operations."""

import pytest
import numpy as np

from cognitive_computing.hdc.hypervectors import (
    BinaryHypervector,
    BipolarHypervector,
    TernaryHypervector,
    LevelHypervector,
    generate_orthogonal_hypervectors,
    fractional_binding,
    protect_hypervector,
    unprotect_hypervector,
)


class TestBinaryHypervector:
    """Test binary hypervector implementation."""
    
    def test_random(self):
        """Test random binary hypervector generation."""
        hv = BinaryHypervector(dimension=1000, seed=42)
        
        vec = hv.random()
        assert vec.shape == (1000,)
        assert vec.dtype == np.uint8
        assert np.all(np.isin(vec, [0, 1]))
        
        # Check approximate uniformity
        assert 0.4 < np.mean(vec) < 0.6
        
    def test_zero(self):
        """Test zero hypervector generation."""
        hv = BinaryHypervector(dimension=1000, seed=42)
        
        zero = hv.zero()
        assert zero.shape == (1000,)
        assert np.all(np.isin(zero, [0, 1]))
        
    def test_bind(self):
        """Test binary binding (XOR)."""
        hv = BinaryHypervector(dimension=1000, seed=42)
        
        a = hv.random()
        b = hv.random()
        
        bound = hv.bind(a, b)
        assert bound.shape == (1000,)
        assert np.all(np.isin(bound, [0, 1]))
        
        # Test self-inverse property
        unbound = hv.bind(bound, a)
        assert np.array_equal(unbound, b)
        
    def test_bundle(self):
        """Test binary bundling."""
        hv = BinaryHypervector(dimension=1000, seed=42)
        
        vectors = [hv.random() for _ in range(5)]
        bundled = hv.bundle(vectors)
        
        assert bundled.shape == (1000,)
        assert np.all(np.isin(bundled, [0, 1]))
        
        # Empty list should raise error
        with pytest.raises(ValueError, match="Cannot bundle empty"):
            hv.bundle([])
            
    def test_similarity(self):
        """Test binary similarity measure."""
        hv = BinaryHypervector(dimension=1000, seed=42)
        
        # Same vector
        vec = hv.random()
        assert hv.similarity(vec, vec) == 1.0
        
        # Opposite vector
        opposite = 1 - vec
        assert hv.similarity(vec, opposite) == -1.0
        
        # Random vectors should have low similarity
        a = hv.random()
        b = hv.random()
        sim = hv.similarity(a, b)
        assert -0.2 < sim < 0.2


class TestBipolarHypervector:
    """Test bipolar hypervector implementation."""
    
    def test_random(self):
        """Test random bipolar hypervector generation."""
        hv = BipolarHypervector(dimension=1000, seed=42)
        
        vec = hv.random()
        assert vec.shape == (1000,)
        assert np.all(np.isin(vec, [-1, 1]))
        
        # Check approximate balance
        assert -0.1 < np.mean(vec) < 0.1
        
    def test_bind(self):
        """Test bipolar binding (multiplication)."""
        hv = BipolarHypervector(dimension=1000, seed=42)
        
        a = hv.random()
        b = hv.random()
        
        bound = hv.bind(a, b)
        assert bound.shape == (1000,)
        assert np.all(np.isin(bound, [-1, 1]))
        
        # Test self-inverse property
        unbound = hv.bind(bound, a)
        assert np.array_equal(unbound, b)
        
    def test_bundle(self):
        """Test bipolar bundling."""
        hv = BipolarHypervector(dimension=1000, seed=42)
        
        vectors = [hv.random() for _ in range(5)]
        bundled = hv.bundle(vectors)
        
        assert bundled.shape == (1000,)
        assert bundled.dtype == np.int8
        assert np.all(np.isin(bundled, [-1, 1]))
        
    def test_similarity(self):
        """Test bipolar similarity (cosine)."""
        hv = BipolarHypervector(dimension=1000, seed=42)
        
        # Same vector
        vec = hv.random()
        assert hv.similarity(vec, vec) == 1.0
        
        # Opposite vector
        opposite = -vec
        assert hv.similarity(vec, opposite) == -1.0
        
        # Orthogonal vectors
        a = hv.random()
        b = hv.random()
        sim = hv.similarity(a, b)
        assert -0.1 < sim < 0.1


class TestTernaryHypervector:
    """Test ternary hypervector implementation."""
    
    def test_random(self):
        """Test random ternary hypervector generation."""
        hv = TernaryHypervector(dimension=1000, sparsity=0.3, seed=42)
        
        vec = hv.random()
        assert vec.shape == (1000,)
        assert vec.dtype == np.int8
        assert np.all(np.isin(vec, [-1, 0, 1]))
        
        # Check sparsity
        n_nonzero = np.sum(vec != 0)
        expected = 1000 * 0.3
        assert abs(n_nonzero - expected) < 50
        
    def test_zero(self):
        """Test zero ternary hypervector."""
        hv = TernaryHypervector(dimension=1000, sparsity=0.3, seed=42)
        
        zero = hv.zero()
        assert zero.shape == (1000,)
        assert np.all(zero == 0)
        
    def test_bind(self):
        """Test ternary binding."""
        hv = TernaryHypervector(dimension=1000, sparsity=0.3, seed=42)
        
        a = hv.random()
        b = hv.random()
        
        bound = hv.bind(a, b)
        assert bound.shape == (1000,)
        # Bound vector may have values outside {-1,0,1}
        
    def test_bundle(self):
        """Test ternary bundling."""
        hv = TernaryHypervector(dimension=1000, sparsity=0.3, seed=42)
        
        vectors = [hv.random() for _ in range(5)]
        bundled = hv.bundle(vectors)
        
        assert bundled.shape == (1000,)
        assert bundled.dtype == np.int8
        assert np.all(np.isin(bundled, [-1, 0, 1]))
        
    def test_similarity(self):
        """Test ternary similarity."""
        hv = TernaryHypervector(dimension=1000, sparsity=0.3, seed=42)
        
        # Same non-zero vector should have similarity 1
        vec = np.array([1, -1, 0, 0, 1, -1, 0, 0], dtype=np.int8)
        # Normalized self-similarity: dot(v,v) / (||v|| * ||v||) = ||v||^2 / ||v||^2 = 1
        assert abs(hv.similarity(vec, vec) - 1.0) < 1e-6
        
        # Opposite vectors
        vec1 = np.array([1, -1, 0, 0, 1, -1, 0, 0], dtype=np.int8)
        vec2 = np.array([-1, 1, 0, 0, -1, 1, 0, 0], dtype=np.int8)
        assert abs(hv.similarity(vec1, vec2) + 1.0) < 1e-6
        
        # Zero vector with non-zero vector
        zero = hv.zero()
        vec = hv.random()
        if np.linalg.norm(vec) > 0:  # Only if vec is non-zero
            assert hv.similarity(vec, zero) == 0.0


class TestLevelHypervector:
    """Test level hypervector implementation."""
    
    def test_random(self):
        """Test random level hypervector generation."""
        hv = LevelHypervector(dimension=1000, levels=5, seed=42)
        
        vec = hv.random()
        assert vec.shape == (1000,)
        assert vec.dtype == np.int8
        assert np.all((0 <= vec) & (vec < 5))
        
        # Check approximate uniformity
        for level in range(5):
            count = np.sum(vec == level)
            assert 150 < count < 250
            
    def test_zero(self):
        """Test zero level hypervector (middle level)."""
        hv = LevelHypervector(dimension=1000, levels=5, seed=42)
        
        zero = hv.zero()
        assert zero.shape == (1000,)
        assert np.all(zero == 2)  # Middle level for 5 levels
        
    def test_bind(self):
        """Test level binding (modular addition)."""
        hv = LevelHypervector(dimension=1000, levels=5, seed=42)
        
        a = np.full(1000, 2, dtype=np.int8)
        b = np.full(1000, 3, dtype=np.int8)
        
        bound = hv.bind(a, b)
        assert np.all(bound == 0)  # (2 + 3) % 5 = 0
        
    def test_bundle(self):
        """Test level bundling."""
        hv = LevelHypervector(dimension=1000, levels=5, seed=42)
        
        vectors = [hv.random() for _ in range(5)]
        bundled = hv.bundle(vectors)
        
        assert bundled.shape == (1000,)
        assert np.all((0 <= bundled) & (bundled < 5))
        
    def test_similarity(self):
        """Test level similarity."""
        hv = LevelHypervector(dimension=1000, levels=5, seed=42)
        
        # Same vector
        vec = hv.random()
        assert hv.similarity(vec, vec) == 1.0
        
        # Different vectors
        a = np.zeros(1000, dtype=np.int8)
        b = np.ones(1000, dtype=np.int8)
        assert hv.similarity(a, b) == 0.0


class TestOrthogonalGeneration:
    """Test orthogonal hypervector generation."""
    
    def test_binary_orthogonal(self):
        """Test binary orthogonal generation."""
        vectors = generate_orthogonal_hypervectors(
            dimension=1000,
            n_vectors=5,
            hypervector_type="binary",
            seed=42
        )
        
        assert len(vectors) == 5
        for vec in vectors:
            assert vec.shape == (1000,)
            assert vec.dtype == np.uint8
            assert np.all(np.isin(vec, [0, 1]))
            
    def test_bipolar_orthogonal_small(self):
        """Test bipolar orthogonal generation for small sets."""
        vectors = generate_orthogonal_hypervectors(
            dimension=1000,
            n_vectors=10,  # Small enough for Gram-Schmidt
            hypervector_type="bipolar",
            seed=42
        )
        
        assert len(vectors) == 10
        
        # Check near-orthogonality
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dot = np.dot(vectors[i], vectors[j])
                similarity = dot / 1000
                assert abs(similarity) < 0.15  # Near orthogonal
                
    def test_bipolar_orthogonal_large(self):
        """Test bipolar orthogonal generation for large sets."""
        vectors = generate_orthogonal_hypervectors(
            dimension=1000,
            n_vectors=200,  # Too large for Gram-Schmidt
            hypervector_type="bipolar",
            seed=42
        )
        
        assert len(vectors) == 200
        
        # Random vectors in high dimensions are nearly orthogonal
        # Sample a few pairs
        for _ in range(10):
            i, j = np.random.choice(200, 2, replace=False)
            dot = np.dot(vectors[i], vectors[j])
            similarity = dot / 1000
            assert abs(similarity) < 0.2
            
    def test_invalid_parameters(self):
        """Test invalid parameters."""
        # More vectors than dimensions
        with pytest.raises(ValueError, match="Cannot generate"):
            generate_orthogonal_hypervectors(
                dimension=100,
                n_vectors=101,
                hypervector_type="bipolar"
            )
            
        # Invalid type
        with pytest.raises(ValueError, match="Unsupported hypervector type"):
            generate_orthogonal_hypervectors(
                dimension=100,
                n_vectors=10,
                hypervector_type="invalid"
            )


class TestFractionalBinding:
    """Test fractional binding operations."""
    
    def test_fractional_binding_bipolar(self):
        """Test fractional binding for bipolar vectors."""
        np.random.seed(42)
        a = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
        b = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
        
        # Weight 0 should return a
        result = fractional_binding(a, b, weight=0.0, hypervector_type="bipolar")
        assert np.array_equal(result, a)
        
        # Weight 1 should return a*b
        result = fractional_binding(a, b, weight=1.0, hypervector_type="bipolar")
        expected = a * b
        assert np.array_equal(result, expected)
        
        # Intermediate weight - can't test exact values due to randomness
        result = fractional_binding(a, b, weight=0.5, hypervector_type="bipolar")
        assert result.shape == a.shape
        assert np.all(np.isin(result, [-1, 1]))
        
    def test_fractional_binding_binary(self):
        """Test fractional binding for binary vectors."""
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        b = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
        
        # Weight 0
        result = fractional_binding(a, b, weight=0.0, hypervector_type="binary")
        assert np.array_equal(result, a)
        
        # Weight 1
        result = fractional_binding(a, b, weight=1.0, hypervector_type="binary")
        expected = np.bitwise_xor(a, b)
        assert np.array_equal(result, expected)
        
    def test_invalid_weight(self):
        """Test invalid weight values."""
        a = np.ones(10, dtype=np.int8)
        b = np.ones(10, dtype=np.int8)
        
        with pytest.raises(ValueError, match="Weight must be in"):
            fractional_binding(a, b, weight=-0.1)
            
        with pytest.raises(ValueError, match="Weight must be in"):
            fractional_binding(a, b, weight=1.1)
            
    def test_unsupported_type(self):
        """Test unsupported hypervector type."""
        a = np.ones(10)
        b = np.ones(10)
        
        with pytest.raises(ValueError, match="Unsupported type"):
            fractional_binding(a, b, weight=1.0, hypervector_type="ternary")


class TestProtection:
    """Test hypervector protection/unprotection."""
    
    def test_protect_unprotect(self):
        """Test protection and unprotection."""
        np.random.seed(42)
        original = np.random.randint(0, 2, size=100, dtype=np.uint8)
        
        # Protect
        protected, perm = protect_hypervector(original, n_protections=1)
        
        assert protected.shape == original.shape
        assert not np.array_equal(protected, original)  # Should be different
        
        # Unprotect
        recovered = unprotect_hypervector(protected, perm, n_protections=1)
        assert np.array_equal(recovered, original)
        
    def test_multiple_protections(self):
        """Test multiple protection layers."""
        np.random.seed(42)
        original = np.random.randint(0, 2, size=100, dtype=np.uint8)
        
        # Multiple protections
        protected, perm = protect_hypervector(original, n_protections=3)
        
        # Should still recover original
        recovered = unprotect_hypervector(protected, perm, n_protections=3)
        assert np.array_equal(recovered, original)
        
    def test_custom_permutation(self):
        """Test with custom permutation."""
        original = np.arange(10, dtype=np.uint8)
        perm = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        
        protected, _ = protect_hypervector(original, n_protections=1, permutation=perm)
        expected = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint8)
        assert np.array_equal(protected, expected)