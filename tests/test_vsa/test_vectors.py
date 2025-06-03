"""
Tests for VSA vector types.

Tests all vector type implementations including Binary, Bipolar,
Ternary, Complex, and Integer vectors.
"""

import pytest
import numpy as np
from typing import Type

from cognitive_computing.vsa.vectors import (
    VSAVector, BinaryVector, BipolarVector, TernaryVector,
    ComplexVector, IntegerVector
)


class TestVSAVectorBase:
    """Test base VSAVector functionality."""
    
    def test_abstract_base_class(self):
        """Test that VSAVector cannot be instantiated."""
        with pytest.raises(TypeError):
            VSAVector(np.array([1, 2, 3]))
            
    def test_vector_interface(self):
        """Test that all vector types implement required methods."""
        vector_types = [
            BinaryVector, BipolarVector, TernaryVector,
            ComplexVector, IntegerVector
        ]
        
        required_methods = ['similarity', 'normalize', 'to_bipolar']
        
        for vtype in vector_types:
            for method in required_methods:
                assert hasattr(vtype, method)


class TestBinaryVector:
    """Test binary vector implementation."""
    
    def test_creation(self):
        """Test binary vector creation."""
        data = np.array([0, 1, 0, 1, 1])
        vec = BinaryVector(data)
        assert np.array_equal(vec.data, data)
        assert vec.dimension == 5
        
    def test_validation(self):
        """Test binary vector validation."""
        # Invalid values
        with pytest.raises(ValueError, match="Binary vector must contain only 0 and 1"):
            BinaryVector(np.array([0, 1, 2]))
            
        with pytest.raises(ValueError, match="Binary vector must contain only 0 and 1"):
            BinaryVector(np.array([0.5, 1.0]))
            
    def test_similarity(self):
        """Test binary vector similarity (Hamming)."""
        vec1 = BinaryVector(np.array([0, 1, 0, 1]))
        vec2 = BinaryVector(np.array([0, 1, 1, 1]))
        vec3 = BinaryVector(np.array([1, 0, 1, 0]))
        
        # Self-similarity should be 1
        assert vec1.similarity(vec1) == 1.0
        
        # 3 out of 4 match
        assert vec1.similarity(vec2) == 0.75
        
        # Exact opposite
        assert vec1.similarity(vec3) == 0.0
        
    def test_normalize(self):
        """Test binary vector normalization (no-op)."""
        vec = BinaryVector(np.array([0, 1, 0, 1]))
        normalized = vec.normalize()
        assert np.array_equal(normalized.data, vec.data)
        
    def test_to_bipolar(self):
        """Test conversion to bipolar."""
        vec = BinaryVector(np.array([0, 1, 0, 1]))
        bipolar = vec.to_bipolar()
        expected = np.array([-1, 1, -1, 1])
        assert np.array_equal(bipolar, expected)
        
    def test_random_generation(self):
        """Test random binary vector generation."""
        vec = BinaryVector.random(dimension=100, rng=np.random.RandomState(42))
        assert vec.dimension == 100
        assert np.all(np.isin(vec.data, [0, 1]))
        # Should be roughly balanced
        assert 0.3 < np.mean(vec.data) < 0.7


class TestBipolarVector:
    """Test bipolar vector implementation."""
    
    def test_creation(self):
        """Test bipolar vector creation."""
        data = np.array([-1, 1, -1, 1, 1])
        vec = BipolarVector(data)
        assert np.array_equal(vec.data, data)
        assert vec.dimension == 5
        
    def test_validation(self):
        """Test bipolar vector validation."""
        # Invalid values
        with pytest.raises(ValueError, match="Bipolar vector must contain only -1 and 1"):
            BipolarVector(np.array([-1, 0, 1]))
            
        with pytest.raises(ValueError, match="Bipolar vector must contain only -1 and 1"):
            BipolarVector(np.array([-1, 2, 1]))
            
    def test_similarity(self):
        """Test bipolar vector similarity (cosine)."""
        vec1 = BipolarVector(np.array([-1, 1, -1, 1]))
        vec2 = BipolarVector(np.array([-1, 1, 1, 1]))
        vec3 = BipolarVector(np.array([1, -1, 1, -1]))
        
        # Self-similarity should be 1
        assert abs(vec1.similarity(vec1) - 1.0) < 1e-10
        
        # 3 out of 4 match: similarity = (3 - 1) / 4 = 0.5
        assert abs(vec1.similarity(vec2) - 0.5) < 1e-10
        
        # Exact opposite
        assert abs(vec1.similarity(vec3) - (-1.0)) < 1e-10
        
    def test_normalize(self):
        """Test bipolar vector normalization."""
        # Already normalized for bipolar
        vec = BipolarVector(np.array([-1, 1, -1, 1]))
        normalized = vec.normalize()
        assert np.array_equal(normalized.data, vec.data)
        
    def test_to_bipolar(self):
        """Test conversion to bipolar (identity)."""
        vec = BipolarVector(np.array([-1, 1, -1, 1]))
        bipolar = vec.to_bipolar()
        assert np.array_equal(bipolar, vec.data)
        
    def test_random_generation(self):
        """Test random bipolar vector generation."""
        vec = BipolarVector.random(dimension=100, rng=np.random.RandomState(42))
        assert vec.dimension == 100
        assert np.all(np.isin(vec.data, [-1, 1]))
        # Should be roughly balanced
        assert -0.2 < np.mean(vec.data) < 0.2


class TestTernaryVector:
    """Test ternary vector implementation."""
    
    def test_creation(self):
        """Test ternary vector creation."""
        data = np.array([-1, 0, 1, 0, -1])
        vec = TernaryVector(data)
        assert np.array_equal(vec.data, data)
        assert vec.dimension == 5
        
    def test_validation(self):
        """Test ternary vector validation."""
        # Invalid values
        with pytest.raises(ValueError, match="Ternary vector must contain only -1, 0, and 1"):
            TernaryVector(np.array([-1, 0, 2]))
            
        with pytest.raises(ValueError, match="Ternary vector must contain only -1, 0, and 1"):
            TernaryVector(np.array([-2, 0, 1]))
            
    def test_similarity(self):
        """Test ternary vector similarity."""
        vec1 = TernaryVector(np.array([-1, 0, 1, 0, -1]))
        vec2 = TernaryVector(np.array([-1, 0, 1, 1, -1]))
        vec3 = TernaryVector(np.array([1, 0, -1, 0, 1]))
        vec4 = TernaryVector(np.array([0, 0, 0, 0, 0]))
        
        # Self-similarity should be 1
        assert abs(vec1.similarity(vec1) - 1.0) < 1e-10
        
        # Mostly similar
        sim12 = vec1.similarity(vec2)
        assert 0.5 < sim12 < 1.0
        
        # Opposite non-zero values
        sim13 = vec1.similarity(vec3)
        assert sim13 < 0
        
        # All zeros should have zero similarity
        assert vec4.similarity(vec1) == 0.0
        
    def test_normalize(self):
        """Test ternary vector normalization."""
        vec = TernaryVector(np.array([-1, 0, 1, 0, -1]))
        normalized = vec.normalize()
        # Should be unit length
        norm = np.linalg.norm(normalized.data)
        assert abs(norm - 1.0) < 1e-6  # Relax tolerance for float32
        
    def test_to_bipolar(self):
        """Test conversion to bipolar."""
        vec = TernaryVector(np.array([-1, 0, 1, 0, -1]))
        bipolar = vec.to_bipolar()
        # Zeros should become 1 or -1 (implementation specific)
        assert np.all(np.isin(bipolar, [-1, 1]))
        # Non-zero values should be preserved
        assert bipolar[0] == -1
        assert bipolar[2] == 1
        assert bipolar[4] == -1
        
    def test_sparsity(self):
        """Test ternary vector sparsity property."""
        vec = TernaryVector(np.array([-1, 0, 1, 0, -1, 0, 0, 0, 1, 0]))
        assert vec.sparsity == 0.6  # 6 zeros out of 10
        
    def test_random_generation(self):
        """Test random ternary vector generation."""
        vec = TernaryVector.random(
            dimension=100, 
            sparsity=0.8,
            rng=np.random.RandomState(42)
        )
        assert vec.dimension == 100
        assert np.all(np.isin(vec.data, [-1, 0, 1]))
        # Check sparsity
        actual_sparsity = np.mean(vec.data == 0)
        assert 0.7 < actual_sparsity < 0.9


class TestComplexVector:
    """Test complex vector implementation."""
    
    def test_creation(self):
        """Test complex vector creation."""
        phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        data = np.exp(1j * phases)
        vec = ComplexVector(data)
        assert np.allclose(vec.data, data)
        assert vec.dimension == 4
        
    def test_validation(self):
        """Test complex vector validation."""
        # Non-unit magnitude
        with pytest.raises(ValueError, match="Complex vector components must have unit magnitude"):
            ComplexVector(np.array([2+0j, 0+1j]))
            
        # Real numbers (not on unit circle)
        with pytest.raises(ValueError, match="Complex vector components must have unit magnitude"):
            ComplexVector(np.array([0.5, 0.7]))
            
    def test_similarity(self):
        """Test complex vector similarity."""
        vec1 = ComplexVector(np.exp(1j * np.array([0, np.pi/2, np.pi, 3*np.pi/2])))
        vec2 = ComplexVector(np.exp(1j * np.array([0, np.pi/2, np.pi, 3*np.pi/2])))
        vec3 = ComplexVector(np.exp(1j * np.array([np.pi, 3*np.pi/2, 0, np.pi/2])))
        
        # Self-similarity should be 1
        assert abs(vec1.similarity(vec1) - 1.0) < 1e-10
        
        # Same vector
        assert abs(vec1.similarity(vec2) - 1.0) < 1e-10
        
        # Orthogonal phases
        sim13 = vec1.similarity(vec3)
        assert abs(sim13) <= 1.0  # Similarity is in [-1, 1]
        
    def test_normalize(self):
        """Test complex vector normalization."""
        # Create slightly non-normalized vector
        data = np.array([1+0.1j, 0+0.9j, -0.8-0.5j, 0.6+0.7j])
        vec = ComplexVector(data / np.abs(data))  # Make unit magnitude
        normalized = vec.normalize()
        
        # Check all components have unit magnitude
        mags = np.abs(normalized.data)
        assert np.allclose(mags, 1.0)
        
    def test_to_bipolar(self):
        """Test conversion to bipolar."""
        phases = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
        vec = ComplexVector(np.exp(1j * phases))
        bipolar = vec.to_bipolar()
        
        # Based on real part sign
        expected = np.array([1, 1, -1, -1, -1, 1])
        assert np.array_equal(bipolar, expected)
        
    def test_random_generation(self):
        """Test random complex vector generation."""
        vec = ComplexVector.random(dimension=100, rng=np.random.RandomState(42))
        assert vec.dimension == 100
        # All components should have unit magnitude
        mags = np.abs(vec.data)
        assert np.allclose(mags, 1.0)
        # Phases should be uniformly distributed
        phases = np.angle(vec.data)
        assert phases.min() >= -np.pi
        assert phases.max() <= np.pi


class TestIntegerVector:
    """Test integer vector implementation."""
    
    def test_creation(self):
        """Test integer vector creation."""
        data = np.array([0, 1, 2, 3, 4])
        vec = IntegerVector(data, modulus=5)
        assert np.array_equal(vec.data, data)
        assert vec.dimension == 5
        assert vec.modulus == 5
        
    def test_validation(self):
        """Test integer vector validation."""
        # Values outside modulus range
        with pytest.raises(ValueError, match="Integer vector values must be in range"):
            IntegerVector(np.array([0, 1, 5]), modulus=5)
            
        with pytest.raises(ValueError, match="Integer vector values must be in range"):
            IntegerVector(np.array([-1, 0, 1]), modulus=5)
            
        # Invalid modulus
        with pytest.raises(ValueError, match="Modulus must be at least 2"):
            IntegerVector(np.array([0, 1]), modulus=1)
            
    def test_similarity(self):
        """Test integer vector similarity."""
        vec1 = IntegerVector(np.array([0, 1, 2, 3, 4]), modulus=5)
        vec2 = IntegerVector(np.array([0, 1, 2, 3, 4]), modulus=5)
        vec3 = IntegerVector(np.array([1, 2, 3, 4, 0]), modulus=5)
        vec4 = IntegerVector(np.array([0, 0, 0, 0, 0]), modulus=5)
        
        # Self-similarity should be 1
        assert abs(vec1.similarity(vec1) - 1.0) < 1e-10
        
        # Same vector
        assert abs(vec1.similarity(vec2) - 1.0) < 1e-10
        
        # Shifted vector (may be orthogonal)
        sim13 = vec1.similarity(vec3)
        assert -1 <= sim13 <= 1
        
        # Check with constant vector
        sim14 = vec1.similarity(vec4)
        assert sim14 < 0.5
        
    def test_normalize(self):
        """Test integer vector normalization."""
        vec = IntegerVector(np.array([0, 1, 2, 3, 4]), modulus=5)
        normalized = vec.normalize()
        # For integer vectors, normalize maps to [-1, 1] range
        assert normalized.data.min() >= -1
        assert normalized.data.max() <= 1
        
    def test_to_bipolar(self):
        """Test conversion to bipolar."""
        vec = IntegerVector(np.array([0, 1, 2, 3, 4]), modulus=5)
        bipolar = vec.to_bipolar()
        
        # Values < modulus/2 -> -1, others -> 1
        expected = np.array([-1, -1, -1, 1, 1])
        assert np.array_equal(bipolar, expected)
        
    def test_modular_arithmetic(self):
        """Test modular arithmetic properties."""
        vec1 = IntegerVector(np.array([3, 4, 2, 1]), modulus=5)
        vec2 = IntegerVector(np.array([2, 3, 4, 3]), modulus=5)
        
        # Addition should wrap around
        result = (vec1.data + vec2.data) % vec1.modulus
        expected = np.array([0, 2, 1, 4])
        assert np.array_equal(result, expected)
        
    def test_random_generation(self):
        """Test random integer vector generation."""
        vec = IntegerVector.random(
            dimension=100,
            modulus=256,
            rng=np.random.RandomState(42)
        )
        assert vec.dimension == 100
        assert vec.modulus == 256
        assert np.all(vec.data >= 0)
        assert np.all(vec.data < 256)
        # Should be roughly uniform
        assert 100 < vec.data.mean() < 156


class TestVectorConversions:
    """Test conversions between vector types."""
    
    def test_binary_to_others(self):
        """Test converting binary vectors to other types."""
        binary = BinaryVector(np.array([0, 1, 0, 1, 1, 0]))
        
        # To bipolar
        bipolar = BipolarVector.from_binary(binary.data)
        expected_bipolar = np.array([-1, 1, -1, 1, 1, -1])
        assert np.array_equal(bipolar.data, expected_bipolar)
        
        # To ternary (with sparsification)
        ternary_data = binary.to_bipolar()
        ternary_data[::3] = 0  # Make sparse
        ternary = TernaryVector(ternary_data)
        assert ternary.sparsity > 0
        
    def test_bipolar_to_others(self):
        """Test converting bipolar vectors to other types."""
        bipolar = BipolarVector(np.array([-1, 1, -1, 1, 1, -1]))
        
        # To binary
        binary = BinaryVector((bipolar.data > 0).astype(int))
        expected_binary = np.array([0, 1, 0, 1, 1, 0])
        assert np.array_equal(binary.data, expected_binary)
        
        # To ternary (with sparsification)
        ternary_data = bipolar.data.copy()
        ternary_data[::3] = 0  # Make sparse
        ternary = TernaryVector(ternary_data)
        assert ternary.sparsity > 0
        
    def test_complex_to_bipolar(self):
        """Test converting complex vectors to bipolar."""
        phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4])
        complex_vec = ComplexVector(np.exp(1j * phases))
        
        bipolar = complex_vec.to_bipolar()
        # Based on real part sign (cos(phase))
        # Note: cos(π/2) ≈ 0, which np.sign treats as +1
        expected = np.array([1, 1, 1, -1, -1, -1])  # cos(phase) sign
        assert np.array_equal(bipolar, expected)


class TestVectorProperties:
    """Test mathematical properties of vectors."""
    
    @pytest.mark.parametrize("vector_class,dimension", [
        (BinaryVector, 100),
        (BipolarVector, 100),
        (TernaryVector, 100),
        (ComplexVector, 100),
        (IntegerVector, 100)
    ])
    def test_self_similarity(self, vector_class, dimension):
        """Test that self-similarity is always 1."""
        if vector_class == TernaryVector:
            vec = vector_class.random(dimension, sparsity=0.8)
        elif vector_class == IntegerVector:
            vec = vector_class.random(dimension, modulus=256)
        else:
            vec = vector_class.random(dimension)
            
        assert abs(vec.similarity(vec) - 1.0) < 1e-10
        
    @pytest.mark.parametrize("vector_class", [
        BinaryVector, BipolarVector, TernaryVector, ComplexVector, IntegerVector
    ])
    def test_random_orthogonality(self, vector_class):
        """Test that random vectors are approximately orthogonal."""
        dimension = 1000
        n_vectors = 10
        rng = np.random.RandomState(42)
        
        # Generate random vectors
        vectors = []
        for _ in range(n_vectors):
            if vector_class == TernaryVector:
                vec = vector_class.random(dimension, sparsity=0.9, rng=rng)
            elif vector_class == IntegerVector:
                vec = vector_class.random(dimension, modulus=256, rng=rng)
            else:
                vec = vector_class.random(dimension, rng=rng)
            vectors.append(vec)
            
        # Check pairwise similarities
        similarities = []
        for i in range(n_vectors):
            for j in range(i+1, n_vectors):
                sim = vectors[i].similarity(vectors[j])
                similarities.append(sim)
                
        # Average similarity should be near expected value
        avg_sim = np.mean(similarities)
        
        # Binary vectors have expected similarity of 0.5
        if vector_class == BinaryVector:
            assert abs(avg_sim - 0.5) < 0.1
        else:
            # Other vectors have expected similarity near 0
            assert abs(avg_sim) < 0.1
        
        # Check similarity distribution
        if vector_class == BinaryVector:
            # Binary similarities are centered around 0.5
            assert np.percentile(np.abs(np.array(similarities) - 0.5), 90) < 0.2
        else:
            # Other similarities should be small
            assert np.percentile(np.abs(similarities), 90) < 0.2


class TestVectorEdgeCases:
    """Test edge cases for vector types."""
    
    def test_single_element_vectors(self):
        """Test vectors with single element."""
        binary = BinaryVector(np.array([1]))
        assert binary.dimension == 1
        assert binary.similarity(binary) == 1.0
        
        bipolar = BipolarVector(np.array([1]))
        assert bipolar.dimension == 1
        assert bipolar.similarity(bipolar) == 1.0
        
    def test_large_vectors(self):
        """Test vectors with large dimensions."""
        dimension = 10000
        
        binary = BinaryVector.random(dimension)
        assert binary.dimension == dimension
        
        bipolar = BipolarVector.random(dimension)
        assert bipolar.dimension == dimension
        
        # Operations should still work
        sim = binary.similarity(binary)
        assert abs(sim - 1.0) < 1e-10
        
    def test_all_same_value_vectors(self):
        """Test vectors with all same values."""
        # All zeros binary
        binary = BinaryVector(np.zeros(100, dtype=int))
        assert binary.similarity(binary) == 1.0
        
        # All ones binary
        binary_ones = BinaryVector(np.ones(100, dtype=int))
        assert binary.similarity(binary_ones) == 0.0
        
        # All positive bipolar
        bipolar = BipolarVector(np.ones(100, dtype=int))
        assert bipolar.similarity(bipolar) == 1.0
        
    def test_maximum_sparsity_ternary(self):
        """Test ternary vector with maximum sparsity."""
        # All zeros
        ternary = TernaryVector(np.zeros(100))
        assert ternary.sparsity == 1.0
        
        # Single non-zero
        data = np.zeros(100)
        data[50] = 1
        ternary_sparse = TernaryVector(data)
        assert ternary_sparse.sparsity == 0.99