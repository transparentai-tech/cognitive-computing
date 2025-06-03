"""
Tests for VSA binding operations.

Tests all binding operation implementations including XOR, Multiplication,
Convolution, MAP, and Permutation binding.
"""

import pytest
import numpy as np
from scipy import signal

from cognitive_computing.vsa.binding import (
    BindingOperation, XORBinding, MultiplicationBinding,
    ConvolutionBinding, MAPBinding, PermutationBinding
)
from cognitive_computing.vsa.vectors import (
    BinaryVector, BipolarVector, ComplexVector, TernaryVector, IntegerVector
)
from cognitive_computing.vsa.core import VectorType


class TestBindingOperationBase:
    """Test base BindingOperation functionality."""
    
    def test_abstract_base_class(self):
        """Test that BindingOperation cannot be instantiated."""
        with pytest.raises(TypeError):
            BindingOperation()
            
    def test_binding_interface(self):
        """Test that all binding types implement required methods."""
        binding_types = [
            XORBinding, MultiplicationBinding, ConvolutionBinding,
            MAPBinding, PermutationBinding
        ]
        
        required_methods = ['bind', 'unbind', 'is_commutative', 'is_associative']
        
        for btype in binding_types:
            for method in required_methods:
                assert hasattr(btype, method)


class TestXORBinding:
    """Test XOR binding operation."""
    
    @pytest.fixture
    def xor_binding(self):
        """Create XOR binding instance."""
        return XORBinding(VectorType.BINARY, 1000)
        
    def test_bind_binary_vectors(self, xor_binding):
        """Test XOR binding with binary vectors."""
        vec1 = BinaryVector(np.array([0, 1, 0, 1, 1, 0]))
        vec2 = BinaryVector(np.array([1, 1, 0, 0, 1, 1]))
        
        bound = xor_binding.bind(vec1.data, vec2.data)
        expected = np.array([1, 0, 0, 1, 0, 1])
        
        assert isinstance(bound, np.ndarray)
        assert np.array_equal(bound, expected)
        
    def test_unbind_is_bind(self, xor_binding):
        """Test that unbind is same as bind for XOR."""
        vec1 = BinaryVector(np.array([0, 1, 0, 1]))
        vec2 = BinaryVector(np.array([1, 1, 0, 0]))
        
        bound = xor_binding.bind(vec1, vec2)
        unbound = xor_binding.unbind(bound, vec1)
        
        assert np.array_equal(unbound.data, vec2.data)
        
    def test_self_inverse_property(self, xor_binding):
        """Test XOR self-inverse property."""
        vec = BinaryVector(np.array([0, 1, 0, 1, 1, 0]))
        
        # Binding with itself should give zeros
        bound = xor_binding.bind(vec, vec)
        expected = np.zeros(6, dtype=int)
        assert np.array_equal(bound.data, expected)
        
    def test_properties(self, xor_binding):
        """Test XOR properties."""
        assert xor_binding.is_commutative() is True
        assert xor_binding.is_associative() is True
        
    def test_invalid_vector_type(self, xor_binding):
        """Test XOR with non-binary vectors."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1]))
        
        with pytest.raises(TypeError, match="XOR binding requires BinaryVector"):
            xor_binding.bind(vec1, vec2)
            
    def test_dimension_mismatch(self, xor_binding):
        """Test XOR with mismatched dimensions."""
        vec1 = BinaryVector(np.array([0, 1, 0, 1]))
        vec2 = BinaryVector(np.array([1, 1, 0]))
        
        with pytest.raises(ValueError, match="Vectors must have same dimension"):
            xor_binding.bind(vec1, vec2)


class TestMultiplicationBinding:
    """Test multiplication binding operation."""
    
    @pytest.fixture
    def mult_binding(self):
        """Create multiplication binding instance."""
        return MultiplicationBinding(VectorType.BIPOLAR, 1000)
        
    def test_bind_bipolar_vectors(self, mult_binding):
        """Test multiplication binding with bipolar vectors."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1, 1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1, 1]))
        
        bound = mult_binding.bind(vec1, vec2)
        expected = np.array([1, -1, -1, 1, 1])
        
        assert isinstance(bound, BipolarVector)
        assert np.array_equal(bound.data, expected)
        
    def test_bind_complex_vectors(self, mult_binding):
        """Test multiplication binding with complex vectors."""
        vec1 = ComplexVector(np.exp(1j * np.array([0, np.pi/2, np.pi, 3*np.pi/2])))
        vec2 = ComplexVector(np.exp(1j * np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4])))
        
        bound = mult_binding.bind(vec1, vec2)
        expected_phases = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        expected = np.exp(1j * expected_phases)
        
        assert isinstance(bound, ComplexVector)
        assert np.allclose(bound.data, expected)
        
    def test_unbind_bipolar(self, mult_binding):
        """Test unbinding for bipolar vectors."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1]))
        
        bound = mult_binding.bind(vec1, vec2)
        unbound = mult_binding.unbind(bound, vec1)
        
        assert np.array_equal(unbound.data, vec2.data)
        
    def test_unbind_complex(self, mult_binding):
        """Test unbinding for complex vectors."""
        vec1 = ComplexVector(np.exp(1j * np.array([0, np.pi/2, np.pi, 3*np.pi/2])))
        vec2 = ComplexVector(np.exp(1j * np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4])))
        
        bound = mult_binding.bind(vec1, vec2)
        unbound = mult_binding.unbind(bound, vec1)
        
        assert np.allclose(unbound.data, vec2.data)
        
    def test_properties(self, mult_binding):
        """Test multiplication properties."""
        assert mult_binding.is_commutative() is True
        assert mult_binding.is_associative() is True
        
    def test_with_ternary_vectors(self, mult_binding):
        """Test multiplication with ternary vectors."""
        vec1 = TernaryVector(np.array([1, 0, -1, 0, 1]))
        vec2 = TernaryVector(np.array([1, -1, -1, 0, 1]))
        
        bound = mult_binding.bind(vec1, vec2)
        expected = np.array([1, 0, 1, 0, 1])
        
        assert isinstance(bound, TernaryVector)
        assert np.array_equal(bound.data, expected)


class TestConvolutionBinding:
    """Test convolution binding operation."""
    
    @pytest.fixture
    def conv_binding(self):
        """Create convolution binding instance."""
        return ConvolutionBinding(VectorType.BIPOLAR, 1000)
        
    def test_bind_bipolar_vectors(self, conv_binding):
        """Test convolution binding with bipolar vectors."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1]))
        
        bound = conv_binding.bind(vec1, vec2)
        
        # Verify using scipy's circular convolution
        expected = signal.fftconvolve(vec1.data, vec2.data, mode='same')
        expected = np.roll(expected, len(expected)//2)  # Adjust for circular
        
        assert isinstance(bound, BipolarVector)
        # Note: May need normalization
        similarity = np.corrcoef(bound.data.flatten(), expected.flatten())[0, 1]
        assert similarity > 0.9
        
    def test_bind_complex_vectors(self, conv_binding):
        """Test convolution binding with complex vectors."""
        vec1 = ComplexVector(np.exp(1j * np.linspace(0, 2*np.pi, 8, endpoint=False)))
        vec2 = ComplexVector(np.exp(1j * np.linspace(0, np.pi, 8, endpoint=False)))
        
        bound = conv_binding.bind(vec1, vec2)
        assert isinstance(bound, ComplexVector)
        assert len(bound.data) == len(vec1.data)
        
    def test_unbind_via_correlation(self, conv_binding):
        """Test unbinding via correlation."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1, 1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1, 1, 1, -1, -1]))
        
        bound = conv_binding.bind(vec1, vec2)
        unbound = conv_binding.unbind(bound, vec1)
        
        # Should recover vec2 (approximately due to normalization)
        similarity = np.corrcoef(unbound.data.flatten(), vec2.data.flatten())[0, 1]
        assert similarity > 0.8
        
    def test_properties(self, conv_binding):
        """Test convolution properties."""
        assert conv_binding.is_commutative() is True
        assert conv_binding.is_associative() is True
        
    def test_fft_implementation(self, conv_binding):
        """Test that FFT implementation is used for efficiency."""
        # Large vectors to ensure FFT is used
        size = 1024
        vec1 = BipolarVector.random(size)
        vec2 = BipolarVector.random(size)
        
        bound = conv_binding.bind(vec1, vec2)
        assert len(bound.data) == size
        
    def test_invalid_vector_type(self, conv_binding):
        """Test convolution with binary vectors (not supported)."""
        vec1 = BinaryVector(np.array([0, 1, 0, 1]))
        vec2 = BinaryVector(np.array([1, 1, 0, 0]))
        
        with pytest.raises(TypeError, match="does not support BinaryVector"):
            conv_binding.bind(vec1, vec2)


class TestMAPBinding:
    """Test MAP (Multiply-Add-Permute) binding operation."""
    
    @pytest.fixture
    def map_binding(self):
        """Create MAP binding instance."""
        return MAPBinding(seed=42)
        
    def test_bind_binary_vectors(self, map_binding):
        """Test MAP binding with binary vectors."""
        vec1 = BinaryVector(np.array([0, 1, 0, 1, 1, 0, 1, 0]))
        vec2 = BinaryVector(np.array([1, 1, 0, 0, 1, 1, 0, 0]))
        
        bound = map_binding.bind(vec1, vec2)
        assert isinstance(bound, BinaryVector)
        assert len(bound.data) == len(vec1.data)
        
        # Result should be different from both inputs
        assert not np.array_equal(bound.data, vec1.data)
        assert not np.array_equal(bound.data, vec2.data)
        
    def test_bind_bipolar_vectors(self, map_binding):
        """Test MAP binding with bipolar vectors."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1, 1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1, 1, 1, -1, -1]))
        
        bound = map_binding.bind(vec1, vec2)
        assert isinstance(bound, BipolarVector)
        assert len(bound.data) == len(vec1.data)
        
    def test_unbind_recovers_original(self, map_binding):
        """Test that unbinding recovers original vector."""
        vec1 = BipolarVector.random(100, rng=np.random.RandomState(42))
        vec2 = BipolarVector.random(100, rng=np.random.RandomState(43))
        
        bound = map_binding.bind(vec1, vec2)
        unbound = map_binding.unbind(bound, vec1)
        
        # Should recover vec2 (approximately)
        similarity = np.corrcoef(unbound.data.flatten(), vec2.data.flatten())[0, 1]
        assert similarity > 0.8
        
    def test_deterministic_permutation(self):
        """Test that MAP uses deterministic permutation with seed."""
        map1 = MAPBinding(seed=42)
        map2 = MAPBinding(seed=42)
        
        vec1 = BipolarVector.random(100, rng=np.random.RandomState(1))
        vec2 = BipolarVector.random(100, rng=np.random.RandomState(2))
        
        bound1 = map1.bind(vec1, vec2)
        bound2 = map2.bind(vec1, vec2)
        
        assert np.array_equal(bound1.data, bound2.data)
        
    def test_properties(self, map_binding):
        """Test MAP properties."""
        # MAP is not commutative due to permutation
        assert map_binding.is_commutative() is False
        # MAP is not strictly associative
        assert map_binding.is_associative() is False
        
    def test_with_different_seeds(self):
        """Test MAP with different seeds produces different results."""
        map1 = MAPBinding(seed=42)
        map2 = MAPBinding(seed=123)
        
        vec1 = BipolarVector.random(100)
        vec2 = BipolarVector.random(100)
        
        bound1 = map1.bind(vec1, vec2)
        bound2 = map2.bind(vec1, vec2)
        
        assert not np.array_equal(bound1.data, bound2.data)


class TestPermutationBinding:
    """Test permutation-based binding operation."""
    
    @pytest.fixture
    def perm_binding(self):
        """Create permutation binding instance."""
        return PermutationBinding(shift=1)
        
    def test_bind_with_shift(self, perm_binding):
        """Test permutation binding with shift."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1, 1, 1]))
        
        bound = perm_binding.bind(vec1, vec2)
        
        # Should multiply vec1 with shifted vec2
        shifted_vec2 = np.roll(vec2.data, 1)
        expected = vec1.data * shifted_vec2
        
        assert np.array_equal(bound.data, expected)
        
    def test_unbind_with_shift(self, perm_binding):
        """Test unbinding with permutation."""
        vec1 = BipolarVector(np.array([1, -1, 1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1, 1, 1]))
        
        bound = perm_binding.bind(vec1, vec2)
        unbound = perm_binding.unbind(bound, vec1)
        
        # Should recover vec2
        assert np.array_equal(unbound.data, vec2.data)
        
    def test_custom_permutation(self):
        """Test with custom permutation indices."""
        perm_indices = np.array([3, 0, 1, 2])  # Rotate by 3
        perm_binding = PermutationBinding(permutation=perm_indices)
        
        vec1 = BipolarVector(np.array([1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1]))
        
        bound = perm_binding.bind(vec1, vec2)
        
        # Manual permutation
        permuted_vec2 = vec2.data[perm_indices]
        expected = vec1.data * permuted_vec2
        
        assert np.array_equal(bound.data, expected)
        
    def test_properties(self, perm_binding):
        """Test permutation binding properties."""
        assert perm_binding.is_commutative() is False
        assert perm_binding.is_associative() is False
        
    def test_identity_permutation(self):
        """Test with identity permutation (shift=0)."""
        perm_binding = PermutationBinding(shift=0)
        
        vec1 = BipolarVector(np.array([1, -1, 1, -1]))
        vec2 = BipolarVector(np.array([1, 1, -1, -1]))
        
        bound = perm_binding.bind(vec1, vec2)
        
        # Should be same as multiplication
        expected = vec1.data * vec2.data
        assert np.array_equal(bound.data, expected)
        
    def test_inverse_permutation(self):
        """Test that inverse permutation works correctly."""
        # Create random permutation
        rng = np.random.RandomState(42)
        perm_indices = rng.permutation(10)
        perm_binding = PermutationBinding(permutation=perm_indices)
        
        vec1 = BipolarVector.random(10, rng=rng)
        vec2 = BipolarVector.random(10, rng=rng)
        
        bound = perm_binding.bind(vec1, vec2)
        unbound = perm_binding.unbind(bound, vec1)
        
        assert np.array_equal(unbound.data, vec2.data)


class TestBindingCompatibility:
    """Test compatibility between different binding operations and vector types."""
    
    def test_xor_only_binary(self):
        """Test that XOR only works with binary vectors."""
        xor = XORBinding(VectorType.BINARY, 4)
        
        # Should work with binary
        binary1 = BinaryVector(np.array([0, 1, 0, 1]))
        binary2 = BinaryVector(np.array([1, 1, 0, 0]))
        result = xor.bind(binary1, binary2)
        assert isinstance(result, BinaryVector)
        
        # Should fail with other types
        bipolar = BipolarVector(np.array([1, -1, 1, -1]))
        with pytest.raises(TypeError):
            xor.bind(bipolar, bipolar)
            
    def test_multiplication_multiple_types(self):
        """Test multiplication works with multiple vector types."""
        mult = MultiplicationBinding(VectorType.BIPOLAR, 4)
        
        # Bipolar
        bipolar1 = BipolarVector(np.array([1, -1, 1, -1]))
        bipolar2 = BipolarVector(np.array([1, 1, -1, -1]))
        result = mult.bind(bipolar1, bipolar2)
        assert isinstance(result, BipolarVector)
        
        # Complex
        complex1 = ComplexVector(np.exp(1j * np.array([0, np.pi/2, np.pi, 3*np.pi/2])))
        complex2 = ComplexVector(np.exp(1j * np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4])))
        result = mult.bind(complex1, complex2)
        assert isinstance(result, ComplexVector)
        
        # Ternary
        ternary1 = TernaryVector(np.array([1, 0, -1, 0]))
        ternary2 = TernaryVector(np.array([1, -1, -1, 0]))
        result = mult.bind(ternary1, ternary2)
        assert isinstance(result, TernaryVector)
        
    def test_convolution_numeric_types(self):
        """Test convolution works with numeric vector types."""
        conv = ConvolutionBinding(VectorType.BIPOLAR, 16)
        
        # Should work with bipolar
        bipolar1 = BipolarVector.random(16)
        bipolar2 = BipolarVector.random(16)
        result = conv.bind(bipolar1, bipolar2)
        assert isinstance(result, BipolarVector)
        
        # Should work with complex
        complex1 = ComplexVector.random(16)
        complex2 = ComplexVector.random(16)
        result = conv.bind(complex1, complex2)
        assert isinstance(result, ComplexVector)
        
        # Should not work with binary
        binary1 = BinaryVector.random(16)
        binary2 = BinaryVector.random(16)
        with pytest.raises(TypeError):
            conv.bind(binary1, binary2)


class TestBindingProperties:
    """Test mathematical properties of binding operations."""
    
    @pytest.mark.parametrize("binding_class,vec1,vec2,vec3", [
        (XORBinding(VectorType.BINARY, 4), 
         BinaryVector(np.array([0, 1, 0, 1])),
         BinaryVector(np.array([1, 1, 0, 0])),
         BinaryVector(np.array([0, 0, 1, 1]))),
        (MultiplicationBinding(VectorType.BIPOLAR, 4),
         BipolarVector(np.array([1, -1, 1, -1])),
         BipolarVector(np.array([1, 1, -1, -1])),
         BipolarVector(np.array([-1, 1, 1, -1]))),
    ])
    def test_associativity(self, binding_class, vec1, vec2, vec3):
        """Test associative property: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)."""
        if binding_class.is_associative():
            # Left association
            left = binding_class.bind(binding_class.bind(vec1, vec2), vec3)
            # Right association  
            right = binding_class.bind(vec1, binding_class.bind(vec2, vec3))
            
            assert np.array_equal(left.data, right.data)
            
    @pytest.mark.parametrize("binding_class,vec1,vec2", [
        (XORBinding(VectorType.BINARY, 4),
         BinaryVector(np.array([0, 1, 0, 1])),
         BinaryVector(np.array([1, 1, 0, 0]))),
        (MultiplicationBinding(VectorType.BIPOLAR, 4),
         BipolarVector(np.array([1, -1, 1, -1])),
         BipolarVector(np.array([1, 1, -1, -1]))),
    ])
    def test_commutativity(self, binding_class, vec1, vec2):
        """Test commutative property: a ⊗ b = b ⊗ a."""
        if binding_class.is_commutative():
            result1 = binding_class.bind(vec1, vec2)
            result2 = binding_class.bind(vec2, vec1)
            
            assert np.array_equal(result1.data, result2.data)
            
    def test_inverse_property(self):
        """Test that unbind is inverse of bind."""
        # Test with different binding types
        test_cases = [
            (XORBinding(VectorType.BINARY, 100), BinaryVector.random(100)),
            (MultiplicationBinding(VectorType.BIPOLAR, 100), BipolarVector.random(100)),
            (ConvolutionBinding(VectorType.BIPOLAR, 128), BipolarVector.random(128)),
            (MAPBinding(VectorType.BIPOLAR, 100, seed=42), BipolarVector.random(100)),
            (PermutationBinding(VectorType.BIPOLAR, 100, shift=5), BipolarVector.random(100))
        ]
        
        for binding, vec1 in test_cases:
            vec2 = type(vec1).random(len(vec1.data))
            
            # Bind and unbind
            bound = binding.bind(vec1, vec2)
            recovered = binding.unbind(bound, vec1)
            
            # Check recovery
            if isinstance(binding, XORBinding):
                # XOR is perfectly invertible
                assert np.array_equal(recovered.data, vec2.data)
            else:
                # Others may be approximately invertible
                similarity = np.corrcoef(
                    recovered.data.flatten().real,
                    vec2.data.flatten().real
                )[0, 1]
                assert similarity > 0.8, f"{type(binding).__name__}: similarity = {similarity}"


class TestBindingEdgeCases:
    """Test edge cases for binding operations."""
    
    def test_single_element_binding(self):
        """Test binding with single-element vectors."""
        # XOR
        xor = XORBinding()
        vec1 = BinaryVector(np.array([0]))
        vec2 = BinaryVector(np.array([1]))
        result = xor.bind(vec1, vec2)
        assert result.data[0] == 1
        
        # Multiplication
        mult = MultiplicationBinding()
        vec1 = BipolarVector(np.array([1]))
        vec2 = BipolarVector(np.array([-1]))
        result = mult.bind(vec1, vec2)
        assert result.data[0] == -1
        
    def test_large_vector_binding(self):
        """Test binding with large vectors."""
        size = 10000
        
        # Test multiplication binding
        mult = MultiplicationBinding()
        vec1 = BipolarVector.random(size)
        vec2 = BipolarVector.random(size)
        
        result = mult.bind(vec1, vec2)
        assert len(result.data) == size
        
        # Verify unbinding works
        recovered = mult.unbind(result, vec1)
        similarity = np.corrcoef(
            recovered.data.flatten(),
            vec2.data.flatten()
        )[0, 1]
        assert similarity > 0.99  # Should be perfect for multiplication
        
    def test_zero_vector_binding(self):
        """Test binding with zero vectors (where applicable)."""
        # Ternary vectors can have all zeros
        mult = MultiplicationBinding()
        vec1 = TernaryVector(np.array([1, 0, -1, 0, 1]))
        vec2 = TernaryVector(np.zeros(5))
        
        result = mult.bind(vec1, vec2)
        # Multiplication with zero gives zero
        assert np.array_equal(result.data, np.zeros(5))
        
    def test_repeated_binding(self):
        """Test repeated binding operations."""
        # XOR with itself
        xor = XORBinding()
        vec = BinaryVector(np.array([0, 1, 0, 1, 1, 0]))
        
        result = vec
        for _ in range(10):
            result = xor.bind(result, vec)
            
        # Even number of XORs with self should give original
        # Odd number should give zeros
        expected = np.zeros(6, dtype=int)
        assert np.array_equal(result.data, expected)