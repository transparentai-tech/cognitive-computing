"""
Tests for HRR operations module.

Tests circular convolution, correlation, and vector operations.
"""

import pytest
import numpy as np
from scipy import signal

from cognitive_computing.hrr.operations import CircularConvolution, VectorOperations


class TestCircularConvolution:
    """Test circular convolution implementations."""
    
    def test_convolve_basic(self):
        """Test basic circular convolution."""
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        
        result = CircularConvolution.convolve(a, b)
        
        # Check result shape
        assert result.shape == a.shape
        
        # Verify against manual calculation
        # c[0] = 1*5 + 2*8 + 3*7 + 4*6 = 5 + 16 + 21 + 24 = 66
        assert np.abs(result[0] - 66) < 1e-10
    
    def test_convolve_direct_vs_fft(self):
        """Test that direct and FFT methods give same results."""
        np.random.seed(42)
        
        # Test various sizes
        for size in [16, 64, 128, 256]:
            a = np.random.randn(size)
            b = np.random.randn(size)
            
            result_direct = CircularConvolution.convolve(a, b, method="direct")
            result_fft = CircularConvolution.convolve(a, b, method="fft")
            
            assert np.allclose(result_direct, result_fft, rtol=1e-10)
    
    def test_convolve_auto_method_selection(self):
        """Test automatic method selection based on size."""
        # Small vectors should use direct method
        small_a = np.random.randn(64)
        small_b = np.random.randn(64)
        
        # Large vectors should use FFT method
        large_a = np.random.randn(256)
        large_b = np.random.randn(256)
        
        # Just verify they run without error
        result_small = CircularConvolution.convolve(small_a, small_b)
        result_large = CircularConvolution.convolve(large_a, large_b)
        
        assert result_small.shape == small_a.shape
        assert result_large.shape == large_a.shape
    
    def test_convolve_commutative(self):
        """Test that convolution is commutative."""
        np.random.seed(42)
        
        a = np.random.randn(128)
        b = np.random.randn(128)
        
        ab = CircularConvolution.convolve(a, b)
        ba = CircularConvolution.convolve(b, a)
        
        assert np.allclose(ab, ba, rtol=1e-10)
    
    def test_convolve_identity(self):
        """Test convolution with identity (delta function)."""
        n = 64
        a = np.random.randn(n)
        
        # Create delta function
        delta = np.zeros(n)
        delta[0] = 1.0
        
        # Convolution with delta should return original
        result = CircularConvolution.convolve(a, delta)
        assert np.allclose(result, a, rtol=1e-10)
    
    def test_convolve_complex(self):
        """Test convolution with complex vectors."""
        np.random.seed(42)
        
        a = np.random.randn(64) + 1j * np.random.randn(64)
        b = np.random.randn(64) + 1j * np.random.randn(64)
        
        result = CircularConvolution.convolve(a, b)
        
        assert result.shape == a.shape
        assert np.iscomplexobj(result)
    
    def test_correlate_basic(self):
        """Test basic circular correlation."""
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        
        result = CircularConvolution.correlate(a, b)
        
        assert result.shape == a.shape
        
        # Correlation should differ from convolution
        conv_result = CircularConvolution.convolve(a, b)
        assert not np.allclose(result, conv_result)
    
    def test_correlate_direct_vs_fft(self):
        """Test that direct and FFT correlation methods match."""
        np.random.seed(42)
        
        for size in [16, 64, 128]:
            a = np.random.randn(size)
            b = np.random.randn(size)
            
            result_direct = CircularConvolution.correlate(a, b, method="direct")
            result_fft = CircularConvolution.correlate(a, b, method="fft")
            
            assert np.allclose(result_direct, result_fft, rtol=1e-10)
    
    def test_convolve_correlate_inverse(self):
        """Test that correlation is inverse of convolution."""
        np.random.seed(42)
        n = 128
        
        # Generate unitary vector (self-inverse)
        phases = np.random.uniform(0, 2 * np.pi, n)
        u = np.cos(phases) + 1j * np.sin(phases)
        u = u / np.linalg.norm(u)
        
        # Test vector
        v = np.random.randn(n)
        
        # Convolve then correlate should recover original
        convolved = CircularConvolution.convolve(v, u)
        recovered = CircularConvolution.correlate(convolved, u)
        
        # Allow for some numerical error
        assert np.allclose(recovered, v, rtol=1e-10)
    
    def test_convolve_multiple(self):
        """Test convolution of multiple vectors."""
        np.random.seed(42)
        
        vectors = [np.random.randn(64) for _ in range(4)]
        
        result = CircularConvolution.convolve_multiple(vectors)
        
        # Verify by computing sequentially
        expected = vectors[0]
        for v in vectors[1:]:
            expected = CircularConvolution.convolve(expected, v)
        
        assert np.allclose(result, expected, rtol=1e-10)
    
    def test_convolve_multiple_empty(self):
        """Test convolution of empty list."""
        with pytest.raises(ValueError, match="Cannot convolve empty"):
            CircularConvolution.convolve_multiple([])
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        a = np.zeros(64)
        b = np.zeros(128)
        
        with pytest.raises(ValueError, match="same shape"):
            CircularConvolution.convolve(a, b)
        
        with pytest.raises(ValueError, match="same shape"):
            CircularConvolution.correlate(a, b)
    
    def test_invalid_method(self):
        """Test error on invalid method."""
        a = np.zeros(64)
        b = np.zeros(64)
        
        with pytest.raises(ValueError, match="Unknown method"):
            CircularConvolution.convolve(a, b, method="invalid")


class TestVectorOperations:
    """Test vector utility operations."""
    
    def test_normalize_real(self):
        """Test normalization of real vectors."""
        v = np.array([3, 4, 0])  # Length 5
        normalized = VectorOperations.normalize(v)
        
        assert np.abs(np.linalg.norm(normalized) - 1.0) < 1e-10
        assert np.allclose(normalized, v / 5.0)
    
    def test_normalize_complex(self):
        """Test normalization of complex vectors."""
        v = np.array([3 + 4j, 0, 0])  # Length 5
        normalized = VectorOperations.normalize(v)
        
        norm = np.sqrt(np.real(np.vdot(normalized, normalized)))
        assert np.abs(norm - 1.0) < 1e-10
    
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        v = np.zeros(10)
        normalized = VectorOperations.normalize(v)
        
        assert np.allclose(normalized, v)
    
    def test_similarity_cosine(self):
        """Test cosine similarity."""
        # Identical vectors
        v1 = np.array([1, 2, 3])
        sim = VectorOperations.similarity(v1, v1, metric="cosine")
        assert np.abs(sim - 1.0) < 1e-10
        
        # Orthogonal vectors
        v2 = np.array([1, 0, 0])
        v3 = np.array([0, 1, 0])
        sim = VectorOperations.similarity(v2, v3, metric="cosine")
        assert np.abs(sim) < 1e-10
        
        # Opposite vectors
        v4 = -v1
        sim = VectorOperations.similarity(v1, v4, metric="cosine")
        assert np.abs(sim + 1.0) < 1e-10
    
    def test_similarity_dot(self):
        """Test dot product similarity."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        
        sim = VectorOperations.similarity(v1, v2, metric="dot")
        expected = 1*4 + 2*5 + 3*6  # 32
        assert np.abs(sim - expected) < 1e-10
    
    def test_similarity_euclidean(self):
        """Test Euclidean distance similarity."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        
        sim = VectorOperations.similarity(v1, v2, metric="euclidean")
        distance = np.sqrt((4-1)**2 + (5-2)**2 + (6-3)**2)  # sqrt(27)
        assert np.abs(sim + distance) < 1e-10
    
    def test_similarity_complex(self):
        """Test similarity with complex vectors."""
        v1 = np.array([1 + 2j, 3 + 4j])
        v2 = np.array([5 + 6j, 7 + 8j])
        
        sim = VectorOperations.similarity(v1, v2, metric="cosine")
        assert isinstance(sim, float)
        assert -1 <= sim <= 1
    
    def test_similarity_invalid_metric(self):
        """Test error on invalid similarity metric."""
        v1 = np.zeros(10)
        v2 = np.zeros(10)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            VectorOperations.similarity(v1, v2, metric="invalid")
    
    def test_make_unitary_real(self):
        """Test making real vectors unitary."""
        np.random.seed(42)
        v = np.random.randn(128)
        
        u = VectorOperations.make_unitary(v)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(u) - 1.0) < 1e-6
        
        # Test self-inverse property
        identity = np.zeros(128)
        identity[0] = 1.0
        
        # u correlated with itself should give identity-like result
        self_corr = CircularConvolution.correlate(u, u)
        peak_idx = np.argmax(np.abs(self_corr))
        assert peak_idx == 0
    
    def test_make_unitary_complex(self):
        """Test making complex vectors unitary."""
        np.random.seed(42)
        v = np.random.randn(64) + 1j * np.random.randn(64)
        
        u = VectorOperations.make_unitary(v)
        
        # All magnitudes should be 1
        magnitudes = np.abs(u)
        assert np.allclose(magnitudes, 1.0, rtol=1e-10)
    
    def test_random_permutation(self):
        """Test random permutation generation."""
        perm = VectorOperations.random_permutation(10, seed=42)
        
        # Should have exactly one 1 per position
        assert perm.shape == (10,)
        assert np.sum(perm) == 1.0
        assert np.all(np.isin(perm, [0.0, 1.0]))
    
    def test_circular_shift(self):
        """Test circular shift operation."""
        v = np.array([1, 2, 3, 4, 5])
        
        # Shift right by 2
        shifted = VectorOperations.circular_shift(v, 2)
        assert np.allclose(shifted, [4, 5, 1, 2, 3])
        
        # Shift left by 2
        shifted = VectorOperations.circular_shift(v, -2)
        assert np.allclose(shifted, [3, 4, 5, 1, 2])
    
    def test_power_basic(self):
        """Test vector power operation."""
        np.random.seed(42)
        v = np.random.randn(64)
        v = v / np.linalg.norm(v)
        
        # v^0 should be identity
        v0 = VectorOperations.power(v, 0)
        assert v0[0] == 1.0
        assert np.sum(np.abs(v0[1:])) < 1e-10
        
        # v^1 should be v
        v1 = VectorOperations.power(v, 1)
        assert np.allclose(v1, v)
        
        # v^2 should be v * v
        v2 = VectorOperations.power(v, 2)
        v2_expected = CircularConvolution.convolve(v, v)
        assert np.allclose(v2, v2_expected, rtol=1e-10)
    
    def test_power_large(self):
        """Test vector power with large exponent."""
        np.random.seed(42)
        v = np.random.randn(64)
        v = v / np.linalg.norm(v)
        
        # Test v^5
        v5 = VectorOperations.power(v, 5)
        
        # Verify by sequential convolution
        expected = v
        for _ in range(4):
            expected = CircularConvolution.convolve(expected, v)
        
        assert np.allclose(v5, expected, rtol=1e-9)
    
    def test_power_negative(self):
        """Test error on negative power."""
        v = np.zeros(10)
        
        with pytest.raises(ValueError, match="non-negative"):
            VectorOperations.power(v, -1)
    
    def test_inverse_correlation(self):
        """Test vector inverse using correlation method."""
        np.random.seed(42)
        
        # Create a unitary vector
        n = 64
        phases = np.random.uniform(0, 2 * np.pi, n)
        v = np.cos(phases) + 1j * np.sin(phases)
        v = v / np.linalg.norm(v)
        
        # Get inverse
        v_inv = VectorOperations.inverse(v, method="correlation")
        
        # v * v_inv should give identity
        identity = CircularConvolution.convolve(v, v_inv)
        
        # Peak should be at 0
        assert np.argmax(np.abs(identity)) == 0
    
    def test_inverse_fft(self):
        """Test vector inverse using FFT method."""
        np.random.seed(42)
        
        # Use a vector with non-zero FFT components
        v = np.random.randn(64) + 0.1
        v = v / np.linalg.norm(v)
        
        # Get inverse
        v_inv = VectorOperations.inverse(v, method="fft")
        
        # v * v_inv should approximate identity
        identity = CircularConvolution.convolve(v, v_inv)
        
        # Check that peak is at 0
        assert np.argmax(np.abs(identity)) == 0
        
        # Check that it's close to delta function
        assert np.abs(identity[0]) > 0.5
    
    def test_inverse_invalid_method(self):
        """Test error on invalid inverse method."""
        v = np.zeros(10)
        
        with pytest.raises(ValueError, match="Unknown method"):
            VectorOperations.inverse(v, method="invalid")


class TestMathematicalProperties:
    """Test mathematical properties of operations."""
    
    def test_convolution_associative(self):
        """Test that convolution is NOT associative (important property)."""
        np.random.seed(42)
        
        a = np.random.randn(64)
        b = np.random.randn(64)
        c = np.random.randn(64)
        
        # (a * b) * c
        ab_c = CircularConvolution.convolve(
            CircularConvolution.convolve(a, b), c
        )
        
        # a * (b * c)
        a_bc = CircularConvolution.convolve(
            a, CircularConvolution.convolve(b, c)
        )
        
        # Should NOT be equal (convolution is not associative)
        assert not np.allclose(ab_c, a_bc, rtol=1e-10)
    
    def test_convolution_distributive(self):
        """Test distributive property of convolution over addition."""
        np.random.seed(42)
        
        a = np.random.randn(64)
        b = np.random.randn(64)
        c = np.random.randn(64)
        
        # a * (b + c)
        left = CircularConvolution.convolve(a, b + c)
        
        # (a * b) + (a * c)
        right = (CircularConvolution.convolve(a, b) + 
                CircularConvolution.convolve(a, c))
        
        assert np.allclose(left, right, rtol=1e-10)
    
    def test_parseval_theorem(self):
        """Test Parseval's theorem for convolution."""
        np.random.seed(42)
        
        a = np.random.randn(128)
        b = np.random.randn(128)
        
        # Energy in time domain
        conv = CircularConvolution.convolve(a, b)
        energy_time = np.sum(np.abs(conv)**2)
        
        # Energy in frequency domain
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        energy_freq = np.sum(np.abs(fft_a * fft_b)**2) / len(a)
        
        assert np.abs(energy_time - energy_freq) < 1e-10