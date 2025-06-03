"""
Tests for specific VSA architectures.

Tests Binary Spatter Codes (BSC), MAP, Fourier HRR, Sparse VSA,
and HRR-compatible architectures.
"""

import pytest
import numpy as np
from typing import List

from cognitive_computing.vsa.architectures import (
    BSC, MAP, FHRR, SparseVSA, HRRCompatible,
    create_architecture
)
from cognitive_computing.vsa.vectors import (
    BinaryVector, BipolarVector, ComplexVector, TernaryVector
)


class TestBSC:
    """Test Binary Spatter Codes architecture."""
    
    @pytest.fixture
    def bsc(self):
        """Create BSC instance."""
        return BSC(dimension=1024, seed=42)
        
    def test_initialization(self, bsc):
        """Test BSC initialization."""
        assert bsc.dimension == 1024
        assert bsc.vector_type == "binary"
        assert bsc.binding_method == "xor"
        
    def test_generate_vector(self, bsc):
        """Test vector generation."""
        vec = bsc.generate()
        
        assert isinstance(vec, BinaryVector)
        assert vec.dimension == 1024
        assert np.all(np.isin(vec.data, [0, 1]))
        
        # Should be roughly balanced
        density = np.mean(vec.data)
        assert 0.45 < density < 0.55
        
    def test_bind_unbind(self, bsc):
        """Test BSC binding and unbinding."""
        vec1 = bsc.generate()
        vec2 = bsc.generate()
        
        # Bind
        bound = bsc.bind(vec1, vec2)
        assert isinstance(bound, BinaryVector)
        
        # Unbind (XOR is self-inverse)
        recovered = bsc.unbind(bound, vec1)
        assert np.array_equal(recovered.data, vec2.data)
        
    def test_bundle(self, bsc):
        """Test BSC bundling (majority vote)."""
        vecs = [bsc.generate() for _ in range(5)]
        bundled = bsc.bundle(vecs)
        
        assert isinstance(bundled, BinaryVector)
        
        # Should have positive similarity to all inputs
        for vec in vecs:
            sim = bsc.similarity(bundled, vec)
            assert sim > 0.4  # Above chance
            
    def test_make_unitary(self, bsc):
        """Test making BSC vector unitary."""
        vec = bsc.generate()
        unitary = bsc.make_unitary(vec)
        
        # For BSC, unitary is identity (already unitary under XOR)
        assert np.array_equal(unitary.data, vec.data)
        
        # Self-binding should give zeros
        self_bound = bsc.bind(unitary, unitary)
        assert np.all(self_bound.data == 0)
        
    def test_capacity(self, bsc):
        """Test BSC capacity estimation."""
        capacity = bsc.estimate_capacity()
        
        assert capacity > 0
        # BSC capacity should be proportional to dimension
        assert capacity < bsc.dimension
        
    def test_noise_tolerance(self, bsc):
        """Test BSC noise tolerance."""
        vec1 = bsc.generate()
        vec2 = bsc.generate()
        
        # Bind
        bound = bsc.bind(vec1, vec2)
        
        # Add noise (flip some bits)
        noise_level = 0.1
        num_flips = int(noise_level * len(bound.data))
        noisy = bound.data.copy()
        flip_indices = np.random.RandomState(42).choice(
            len(noisy), num_flips, replace=False
        )
        noisy[flip_indices] = 1 - noisy[flip_indices]
        
        # Unbind noisy vector
        recovered = bsc.unbind(BinaryVector(noisy), vec1)
        
        # Should still be similar to original
        similarity = bsc.similarity(recovered, vec2)
        assert similarity > 0.7


class TestMAP:
    """Test Multiply-Add-Permute architecture."""
    
    @pytest.fixture
    def map_arch(self):
        """Create MAP instance."""
        return MAP(dimension=1024, seed=42)
        
    def test_initialization(self, map_arch):
        """Test MAP initialization."""
        assert map_arch.dimension == 1024
        assert map_arch.vector_type == "bipolar"
        assert map_arch.binding_method == "map"
        
    def test_generate_vector(self, map_arch):
        """Test MAP vector generation."""
        vec = map_arch.generate()
        
        assert isinstance(vec, BipolarVector)
        assert vec.dimension == 1024
        assert np.all(np.isin(vec.data, [-1, 1]))
        
    def test_bind_operations(self, map_arch):
        """Test MAP binding operations."""
        vec1 = map_arch.generate()
        vec2 = map_arch.generate()
        vec3 = map_arch.generate()
        
        # Bind pairs
        bound12 = map_arch.bind(vec1, vec2)
        bound23 = map_arch.bind(vec2, vec3)
        
        # MAP is not commutative
        bound21 = map_arch.bind(vec2, vec1)
        similarity = map_arch.similarity(bound12, bound21)
        assert similarity < 0.9  # Should be different
        
        # MAP is not associative
        left = map_arch.bind(bound12, vec3)
        right = map_arch.bind(vec1, bound23)
        similarity = map_arch.similarity(left, right)
        assert similarity < 0.9  # Should be different
        
    def test_unbind(self, map_arch):
        """Test MAP unbinding."""
        vec1 = map_arch.generate()
        vec2 = map_arch.generate()
        
        bound = map_arch.bind(vec1, vec2)
        recovered = map_arch.unbind(bound, vec1)
        
        # Should recover vec2 approximately
        similarity = map_arch.similarity(recovered, vec2)
        assert similarity > 0.8
        
    def test_bundle_with_permutation(self, map_arch):
        """Test MAP bundling includes permutation."""
        vecs = [map_arch.generate() for _ in range(3)]
        
        # Bundle should apply permutations
        bundled = map_arch.bundle(vecs)
        
        # Each input should contribute
        for vec in vecs:
            sim = map_arch.similarity(bundled, vec)
            assert sim > 0  # Positive contribution
            
    def test_deterministic_permutation(self):
        """Test MAP uses deterministic permutation."""
        map1 = MAP(dimension=512, seed=42)
        map2 = MAP(dimension=512, seed=42)
        
        vec = BipolarVector.random(512)
        
        # Same seed should give same permutation
        perm1 = map1._permute(vec)
        perm2 = map2._permute(vec)
        
        assert np.array_equal(perm1.data, perm2.data)


class TestFHRR:
    """Test Fourier Holographic Reduced Representations."""
    
    @pytest.fixture
    def fhrr(self):
        """Create FHRR instance."""
        return FHRR(dimension=1024, seed=42)
        
    def test_initialization(self, fhrr):
        """Test FHRR initialization."""
        assert fhrr.dimension == 1024
        assert fhrr.vector_type == "complex"
        assert fhrr.binding_method == "convolution"
        
    def test_generate_vector(self, fhrr):
        """Test FHRR vector generation."""
        vec = fhrr.generate()
        
        assert isinstance(vec, ComplexVector)
        assert vec.dimension == 1024
        
        # Should have unit magnitude
        magnitudes = np.abs(vec.data)
        assert np.allclose(magnitudes, 1.0)
        
        # Phases should be uniformly distributed
        phases = np.angle(vec.data)
        assert phases.min() >= -np.pi
        assert phases.max() <= np.pi
        
    def test_frequency_domain_binding(self, fhrr):
        """Test FHRR binding in frequency domain."""
        vec1 = fhrr.generate()
        vec2 = fhrr.generate()
        
        # Binding is element-wise multiplication in frequency domain
        bound = fhrr.bind(vec1, vec2)
        
        # Should still have unit magnitude
        assert np.allclose(np.abs(bound.data), 1.0)
        
        # Phases should add
        expected_phases = np.angle(vec1.data) + np.angle(vec2.data)
        expected_phases = np.angle(np.exp(1j * expected_phases))  # Wrap to [-π, π]
        actual_phases = np.angle(bound.data)
        
        # Allow for numerical errors
        phase_diff = np.abs(actual_phases - expected_phases)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Wrap difference
        assert np.max(phase_diff) < 0.1
        
    def test_unbind_correlation(self, fhrr):
        """Test FHRR unbinding via correlation."""
        vec1 = fhrr.generate()
        vec2 = fhrr.generate()
        
        bound = fhrr.bind(vec1, vec2)
        recovered = fhrr.unbind(bound, vec1)
        
        # Should recover vec2
        similarity = fhrr.similarity(recovered, vec2)
        assert similarity > 0.9
        
    def test_make_unitary(self, fhrr):
        """Test making FHRR vector unitary."""
        # Create non-unitary vector
        vec = ComplexVector(np.exp(1j * np.random.randn(1024)))
        
        unitary = fhrr.make_unitary(vec)
        
        # Should have unit magnitude
        assert np.allclose(np.abs(unitary.data), 1.0)
        
        # Self-correlation should give identity-like vector
        self_corr = fhrr.unbind(unitary, unitary)
        # Most energy should be at zero shift
        assert np.abs(self_corr.data[0]) > 0.9
        
    def test_real_space_conversion(self, fhrr):
        """Test conversion to/from real space."""
        vec = fhrr.generate()
        
        # Convert to real space (inverse FFT)
        real_space = np.fft.ifft(vec.data)
        
        # Should be complex with both real and imaginary parts
        assert np.any(np.abs(real_space.real) > 1e-10)
        assert np.any(np.abs(real_space.imag) > 1e-10)
        
        # Convert back to frequency space
        freq_space = np.fft.fft(real_space)
        
        # Should recover original
        assert np.allclose(freq_space, vec.data)


class TestSparseVSA:
    """Test Sparse VSA architecture."""
    
    @pytest.fixture
    def sparse_vsa(self):
        """Create Sparse VSA instance."""
        return SparseVSA(dimension=10000, sparsity=0.01, seed=42)
        
    def test_initialization(self, sparse_vsa):
        """Test Sparse VSA initialization."""
        assert sparse_vsa.dimension == 10000
        assert sparse_vsa.sparsity == 0.01
        assert sparse_vsa.vector_type == "ternary"
        
    def test_generate_sparse_vector(self, sparse_vsa):
        """Test sparse vector generation."""
        vec = sparse_vsa.generate()
        
        assert isinstance(vec, TernaryVector)
        assert vec.dimension == 10000
        
        # Check sparsity
        actual_sparsity = np.mean(vec.data == 0)
        assert 0.98 < actual_sparsity < 1.0  # Very sparse
        
        # Non-zero elements should be ±1
        nonzero = vec.data[vec.data != 0]
        assert np.all(np.isin(nonzero, [-1, 1]))
        
        # Should be roughly balanced
        if len(nonzero) > 0:
            balance = np.mean(nonzero)
            assert abs(balance) < 0.3
            
    def test_sparse_binding(self, sparse_vsa):
        """Test binding of sparse vectors."""
        vec1 = sparse_vsa.generate()
        vec2 = sparse_vsa.generate()
        
        bound = sparse_vsa.bind(vec1, vec2)
        
        # Result should also be sparse
        bound_sparsity = np.mean(bound.data == 0)
        assert bound_sparsity > 0.9  # Still sparse
        
    def test_sparse_bundle(self, sparse_vsa):
        """Test bundling sparse vectors."""
        vecs = [sparse_vsa.generate() for _ in range(10)]
        
        bundled = sparse_vsa.bundle(vecs)
        
        # Bundling many sparse vectors may reduce sparsity
        bundled_sparsity = np.mean(bundled.data == 0)
        assert bundled_sparsity > 0.5  # Still somewhat sparse
        
        # Should maintain similarity to inputs
        for vec in vecs:
            sim = sparse_vsa.similarity(bundled, vec)
            assert sim > 0
            
    def test_thinning_operation(self, sparse_vsa):
        """Test thinning to maintain sparsity."""
        # Create denser vector
        vecs = [sparse_vsa.generate() for _ in range(20)]
        dense = sparse_vsa.bundle(vecs)
        
        # Thin back to target sparsity
        thinned = sparse_vsa.thin(dense, target_sparsity=0.99)
        
        actual_sparsity = np.mean(thinned.data == 0)
        assert actual_sparsity >= 0.99
        
    def test_capacity_with_sparsity(self, sparse_vsa):
        """Test capacity of sparse VSA."""
        capacity = sparse_vsa.estimate_capacity()
        
        # Sparse VSA should have high capacity relative to active components
        active_components = int(sparse_vsa.dimension * (1 - sparse_vsa.sparsity))
        assert capacity > active_components
        
    def test_sparse_similarity(self, sparse_vsa):
        """Test similarity computation for sparse vectors."""
        vec1 = sparse_vsa.generate()
        vec2 = sparse_vsa.generate()
        
        # Random sparse vectors should have low similarity
        sim = sparse_vsa.similarity(vec1, vec2)
        assert abs(sim) < 0.1
        
        # Self-similarity should be 1
        self_sim = sparse_vsa.similarity(vec1, vec1)
        assert abs(self_sim - 1.0) < 1e-10


class TestHRRCompatible:
    """Test HRR-compatible VSA wrapper."""
    
    @pytest.fixture
    def hrr_vsa(self):
        """Create HRR-compatible VSA."""
        return HRRCompatible(dimension=1024, seed=42)
        
    def test_initialization(self, hrr_vsa):
        """Test HRR-compatible initialization."""
        assert hrr_vsa.dimension == 1024
        assert hrr_vsa.vector_type == "bipolar"
        assert hrr_vsa.binding_method == "convolution"
        
    def test_circular_convolution(self, hrr_vsa):
        """Test circular convolution binding."""
        vec1 = hrr_vsa.generate()
        vec2 = hrr_vsa.generate()
        
        bound = hrr_vsa.bind(vec1, vec2)
        
        # Verify it's circular convolution
        expected = np.real(np.fft.ifft(
            np.fft.fft(vec1.data) * np.fft.fft(vec2.data)
        ))
        
        # Normalize
        expected = expected / np.linalg.norm(expected)
        bound_normalized = bound.data / np.linalg.norm(bound.data)
        
        assert np.allclose(bound_normalized, expected, atol=1e-6)
        
    def test_circular_correlation(self, hrr_vsa):
        """Test circular correlation unbinding."""
        vec1 = hrr_vsa.generate()
        vec2 = hrr_vsa.generate()
        
        bound = hrr_vsa.bind(vec1, vec2)
        recovered = hrr_vsa.unbind(bound, vec1)
        
        # Should recover vec2
        similarity = hrr_vsa.similarity(recovered, vec2)
        assert similarity > 0.9
        
    def test_cleanup_memory(self, hrr_vsa):
        """Test cleanup memory functionality."""
        # Add items to cleanup memory
        items = {
            "cat": hrr_vsa.generate(),
            "dog": hrr_vsa.generate(),
            "bird": hrr_vsa.generate()
        }
        
        for name, vec in items.items():
            hrr_vsa.add_to_cleanup(name, vec)
            
        # Cleanup should find exact match
        cleaned = hrr_vsa.cleanup(items["cat"])
        assert cleaned == "cat"
        
        # Cleanup noisy vector
        noisy = items["dog"].data + np.random.randn(1024) * 0.1
        cleaned = hrr_vsa.cleanup(BipolarVector(np.sign(noisy)))
        assert cleaned == "dog"
        
    def test_hrr_properties(self, hrr_vsa):
        """Test HRR-specific properties."""
        # Commutativity
        vec1 = hrr_vsa.generate()
        vec2 = hrr_vsa.generate()
        
        comm1 = hrr_vsa.bind(vec1, vec2)
        comm2 = hrr_vsa.bind(vec2, vec1)
        
        similarity = hrr_vsa.similarity(comm1, comm2)
        assert similarity > 0.99  # Should be commutative
        
        # Associativity
        vec3 = hrr_vsa.generate()
        
        left = hrr_vsa.bind(hrr_vsa.bind(vec1, vec2), vec3)
        right = hrr_vsa.bind(vec1, hrr_vsa.bind(vec2, vec3))
        
        similarity = hrr_vsa.similarity(left, right)
        assert similarity > 0.99  # Should be associative


class TestArchitectureFactory:
    """Test architecture factory function."""
    
    def test_create_architectures(self):
        """Test creating all architecture types."""
        # BSC
        bsc = create_architecture("bsc", dimension=512)
        assert isinstance(bsc, BSC)
        assert bsc.dimension == 512
        
        # MAP
        map_arch = create_architecture("map", dimension=512, seed=42)
        assert isinstance(map_arch, MAP)
        
        # FHRR
        fhrr = create_architecture("fhrr", dimension=512)
        assert isinstance(fhrr, FHRR)
        
        # Sparse
        sparse = create_architecture("sparse", dimension=1000, sparsity=0.05)
        assert isinstance(sparse, SparseVSA)
        assert sparse.sparsity == 0.05
        
        # HRR-compatible
        hrr = create_architecture("hrr", dimension=512)
        assert isinstance(hrr, HRRCompatible)
        
    def test_invalid_architecture(self):
        """Test invalid architecture type."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_architecture("invalid", dimension=512)


class TestArchitectureComparison:
    """Compare different architectures."""
    
    def test_binding_comparison(self):
        """Compare binding across architectures."""
        dimension = 1024
        architectures = {
            "BSC": BSC(dimension=dimension, seed=42),
            "MAP": MAP(dimension=dimension, seed=42),
            "FHRR": FHRR(dimension=dimension, seed=42),
            "HRR": HRRCompatible(dimension=dimension, seed=42)
        }
        
        results = {}
        
        for name, arch in architectures.items():
            # Generate vectors
            vec1 = arch.generate()
            vec2 = arch.generate()
            
            # Time binding
            import time
            start = time.time()
            for _ in range(100):
                bound = arch.bind(vec1, vec2)
            bind_time = time.time() - start
            
            # Test unbinding accuracy
            recovered = arch.unbind(bound, vec1)
            similarity = arch.similarity(recovered, vec2)
            
            results[name] = {
                "bind_time": bind_time,
                "recovery_similarity": similarity
            }
            
        # All should have good recovery
        for name, result in results.items():
            assert result["recovery_similarity"] > 0.8, f"{name} recovery failed"
            
    def test_capacity_comparison(self):
        """Compare capacity across architectures."""
        dimension = 1024
        
        capacities = {
            "BSC": BSC(dimension=dimension).estimate_capacity(),
            "MAP": MAP(dimension=dimension).estimate_capacity(),
            "FHRR": FHRR(dimension=dimension).estimate_capacity(),
            "Sparse": SparseVSA(dimension=dimension*10, sparsity=0.01).estimate_capacity(),
            "HRR": HRRCompatible(dimension=dimension).estimate_capacity()
        }
        
        # All should have reasonable capacity
        for name, capacity in capacities.items():
            assert capacity > 10, f"{name} capacity too low"
            assert capacity < dimension * 10, f"{name} capacity unrealistic"
            
    def test_noise_robustness(self):
        """Test noise robustness across architectures."""
        dimension = 1024
        noise_level = 0.2
        
        architectures = {
            "BSC": BSC(dimension=dimension, seed=42),
            "MAP": MAP(dimension=dimension, seed=42),
            "HRR": HRRCompatible(dimension=dimension, seed=42)
        }
        
        for name, arch in architectures.items():
            vec1 = arch.generate()
            vec2 = arch.generate()
            
            # Bind
            bound = arch.bind(vec1, vec2)
            
            # Add noise
            if hasattr(bound, 'data'):
                noisy_data = bound.data + np.random.randn(*bound.data.shape) * noise_level
                
                # Create noisy vector of same type
                if isinstance(bound, BinaryVector):
                    noisy = BinaryVector((noisy_data > 0.5).astype(int))
                elif isinstance(bound, BipolarVector):
                    noisy = BipolarVector(np.sign(noisy_data).astype(int))
                else:
                    noisy = type(bound)(noisy_data)
                    
                # Test recovery
                recovered = arch.unbind(noisy, vec1)
                similarity = arch.similarity(recovered, vec2)
                
                assert similarity > 0.5, f"{name} not robust to noise"