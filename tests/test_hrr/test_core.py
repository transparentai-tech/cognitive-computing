"""
Tests for core HRR functionality.

Tests the main HRR class, configuration, and basic operations.
"""

import pytest
import numpy as np
from typing import List

from cognitive_computing.hrr import HRR, HRRConfig, create_hrr
from cognitive_computing.hrr.core import HRR as HRRCore


class TestHRRConfig:
    """Test HRR configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HRRConfig()
        assert config.dimension == 1024
        assert config.normalize == True
        assert config.cleanup_threshold == 0.3
        assert config.storage_method == "real"
        assert config.seed is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HRRConfig(
            dimension=2048,
            normalize=False,
            cleanup_threshold=0.5,
            storage_method="complex",
            seed=42
        )
        assert config.dimension == 2048
        assert config.normalize == False
        assert config.cleanup_threshold == 0.5
        assert config.storage_method == "complex"
        assert config.seed == 42
    
    def test_invalid_dimension(self):
        """Test invalid dimension validation."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            HRRConfig(dimension=0)
        
        with pytest.raises(ValueError, match="Dimension must be positive"):
            HRRConfig(dimension=-100)
    
    def test_invalid_storage_method(self):
        """Test invalid storage method validation."""
        with pytest.raises(ValueError, match="storage_method must be"):
            HRRConfig(storage_method="invalid")
    
    def test_complex_dimension_validation(self):
        """Test dimension validation for complex storage."""
        # Odd dimension should fail for complex storage
        with pytest.raises(ValueError, match="Dimension must be even"):
            HRRConfig(dimension=1023, storage_method="complex")
        
        # Even dimension should work
        config = HRRConfig(dimension=1024, storage_method="complex")
        assert config.dimension == 1024
    
    def test_cleanup_threshold_validation(self):
        """Test cleanup threshold validation."""
        with pytest.raises(ValueError, match="cleanup_threshold must be"):
            HRRConfig(cleanup_threshold=-0.1)
        
        with pytest.raises(ValueError, match="cleanup_threshold must be"):
            HRRConfig(cleanup_threshold=1.5)


class TestHRRCreation:
    """Test HRR creation and initialization."""
    
    def test_create_hrr_function(self):
        """Test create_hrr factory function."""
        hrr = create_hrr(dimension=512)
        assert isinstance(hrr, HRRCore)
        assert hrr.config.dimension == 512
        assert hrr.config.normalize == True
    
    def test_create_hrr_with_params(self):
        """Test create_hrr with custom parameters."""
        hrr = create_hrr(
            dimension=1024,
            normalize=False,
            cleanup_threshold=0.4,
            storage_method="complex",
            seed=123
        )
        assert hrr.config.dimension == 1024
        assert hrr.config.normalize == False
        assert hrr.config.cleanup_threshold == 0.4
        assert hrr.config.storage_method == "complex"
        assert hrr.config.seed == 123
    
    def test_hrr_initialization(self):
        """Test direct HRR initialization."""
        config = HRRConfig(dimension=256, seed=42)
        hrr = HRR(config)
        assert hrr.config == config
        assert len(hrr.memory) == 0
        assert hrr.size == 0


class TestVectorGeneration:
    """Test vector generation methods."""
    
    def test_generate_vector_real(self):
        """Test real-valued vector generation."""
        hrr = create_hrr(dimension=1000, storage_method="real", seed=42)
        
        # Generate vectors
        v1 = hrr.generate_vector()
        v2 = hrr.generate_vector()
        
        # Check properties
        assert v1.shape == (1000,)
        assert v2.shape == (1000,)
        assert v1.dtype in [np.float32, np.float64]
        
        # Should be normalized
        assert np.abs(np.linalg.norm(v1) - 1.0) < 1e-6
        assert np.abs(np.linalg.norm(v2) - 1.0) < 1e-6
        
        # Should be different
        assert not np.allclose(v1, v2)
    
    def test_generate_vector_complex(self):
        """Test complex-valued vector generation."""
        hrr = create_hrr(dimension=1024, storage_method="complex", seed=42)
        
        v = hrr.generate_vector()
        
        # Check properties
        assert v.shape == (512,)  # Half dimension for complex
        assert np.iscomplexobj(v)
        
        # Should be normalized
        assert np.abs(np.sqrt(np.real(np.vdot(v, v))) - 1.0) < 1e-6
    
    def test_generate_unitary_vector(self):
        """Test unitary vector generation."""
        hrr = create_hrr(dimension=512, seed=42)
        
        u = hrr.generate_vector(unitary=True)
        
        # Check that it's self-inverse under correlation
        identity = np.zeros(512)
        identity[0] = 1.0
        
        # u * u^(-1) should give identity-like vector
        u_inv = hrr.unbind(identity, u)
        reconstructed = hrr.bind(u, u_inv)
        
        # Peak should be at position 0
        assert np.argmax(np.abs(reconstructed)) == 0


class TestBindingOperations:
    """Test binding and unbinding operations."""
    
    def test_bind_basic(self):
        """Test basic binding operation."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        a = hrr.generate_vector()
        b = hrr.generate_vector()
        
        c = hrr.bind(a, b)
        
        # Result should be normalized
        assert np.abs(np.linalg.norm(c) - 1.0) < 1e-6
        
        # Result should be different from inputs
        assert hrr.similarity(c, a) < 0.3
        assert hrr.similarity(c, b) < 0.3
    
    def test_bind_commutative(self):
        """Test that binding is commutative."""
        hrr = create_hrr(dimension=512, seed=42)
        
        a = hrr.generate_vector()
        b = hrr.generate_vector()
        
        ab = hrr.bind(a, b)
        ba = hrr.bind(b, a)
        
        # Should be nearly identical
        assert np.allclose(ab, ba, rtol=1e-10)
    
    def test_unbind_basic(self):
        """Test basic unbinding operation."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        role = hrr.generate_vector()
        filler = hrr.generate_vector()
        
        # Bind role and filler
        binding = hrr.bind(role, filler)
        
        # Unbind to retrieve filler
        retrieved = hrr.unbind(binding, role)
        
        # Should be similar to original filler
        similarity = hrr.similarity(retrieved, filler)
        assert similarity > 0.9
    
    def test_bind_unbind_identity(self):
        """Test bind-unbind identity property."""
        hrr = create_hrr(dimension=2048, seed=42)
        
        # Multiple test cases
        for _ in range(5):
            a = hrr.generate_vector()
            b = hrr.generate_vector()
            
            # Bind then unbind
            bound = hrr.bind(a, b)
            retrieved_b = hrr.unbind(bound, a)
            retrieved_a = hrr.unbind(bound, b)
            
            # Check similarities
            assert hrr.similarity(retrieved_b, b) > 0.95
            assert hrr.similarity(retrieved_a, a) > 0.95
    
    def test_bind_with_noise(self):
        """Test binding with noisy vectors."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        a = hrr.generate_vector()
        b = hrr.generate_vector()
        
        # Add noise to a
        noise = np.random.RandomState(123).randn(1024) * 0.1
        a_noisy = a + noise
        a_noisy = a_noisy / np.linalg.norm(a_noisy)  # Renormalize
        
        # Bind and unbind
        binding = hrr.bind(a_noisy, b)
        retrieved = hrr.unbind(binding, a)
        
        # Should still recover b reasonably well
        similarity = hrr.similarity(retrieved, b)
        assert similarity > 0.7


class TestBundlingOperations:
    """Test bundling (superposition) operations."""
    
    def test_bundle_basic(self):
        """Test basic bundling operation."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        vectors = [hrr.generate_vector() for _ in range(3)]
        bundle = hrr.bundle(vectors)
        
        # Bundle should be normalized
        assert np.abs(np.linalg.norm(bundle) - 1.0) < 1e-6
        
        # Bundle should be similar to all components
        for v in vectors:
            similarity = hrr.similarity(bundle, v)
            assert similarity > 0.4  # Reasonable similarity
    
    def test_bundle_weighted(self):
        """Test weighted bundling."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        v1 = hrr.generate_vector()
        v2 = hrr.generate_vector()
        
        # Bundle with different weights
        bundle = hrr.bundle([v1, v2], weights=[2.0, 1.0])
        
        # Should be more similar to v1 due to higher weight
        sim1 = hrr.similarity(bundle, v1)
        sim2 = hrr.similarity(bundle, v2)
        assert sim1 > sim2
    
    def test_bundle_empty(self):
        """Test bundling empty list."""
        hrr = create_hrr(dimension=512)
        
        with pytest.raises(ValueError, match="Cannot bundle empty"):
            hrr.bundle([])
    
    def test_bundle_weight_mismatch(self):
        """Test bundling with mismatched weights."""
        hrr = create_hrr(dimension=512)
        vectors = [hrr.generate_vector() for _ in range(3)]
        
        with pytest.raises(ValueError, match="Number of weights"):
            hrr.bundle(vectors, weights=[1.0, 2.0])  # Only 2 weights for 3 vectors


class TestMemoryOperations:
    """Test memory storage and retrieval."""
    
    def test_store_recall_single(self):
        """Test storing and recalling a single association."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        key = hrr.generate_vector()
        value = hrr.generate_vector()
        
        # Store association
        hrr.store(key, value)
        
        # Recall value
        retrieved = hrr.recall(key)
        
        assert retrieved is not None
        similarity = hrr.similarity(retrieved, value)
        assert similarity > 0.9
    
    def test_store_recall_multiple(self):
        """Test storing and recalling multiple associations."""
        hrr = create_hrr(dimension=2048, seed=42)
        
        # Store multiple associations
        pairs = []
        for i in range(5):
            key = hrr.generate_vector()
            value = hrr.generate_vector()
            pairs.append((key, value))
            hrr.store(key, value)
        
        # Test recall for each pair
        for key, value in pairs:
            retrieved = hrr.recall(key)
            similarity = hrr.similarity(retrieved, value)
            assert similarity > 0.5  # Some interference expected
    
    def test_clear_memory(self):
        """Test clearing memory."""
        hrr = create_hrr(dimension=512, seed=42)
        
        # Store some items
        hrr.store(hrr.generate_vector(), hrr.generate_vector())
        hrr.add_item("test", hrr.generate_vector())
        
        assert "memory_trace" in hrr.memory
        assert hrr.size == 1  # One named item
        
        # Clear memory
        hrr.clear()
        
        assert len(hrr.memory) == 0
        assert hrr.size == 0
    
    def test_add_get_item(self):
        """Test adding and getting named items."""
        hrr = create_hrr(dimension=512, seed=42)
        
        # Add named items
        v1 = hrr.generate_vector()
        v2 = hrr.generate_vector()
        
        hrr.add_item("item1", v1)
        hrr.add_item("item2", v2)
        
        # Retrieve items
        retrieved1 = hrr.get_item("item1")
        retrieved2 = hrr.get_item("item2")
        
        assert np.allclose(retrieved1, v1)
        assert np.allclose(retrieved2, v2)
        
        # Non-existent item
        assert hrr.get_item("nonexistent") is None
    
    def test_size_property(self):
        """Test size property."""
        hrr = create_hrr(dimension=512)
        
        assert hrr.size == 0
        
        # Add named items
        hrr.add_item("a", hrr.generate_vector())
        assert hrr.size == 1
        
        hrr.add_item("b", hrr.generate_vector())
        assert hrr.size == 2
        
        # Store operation doesn't affect named item count
        hrr.store(hrr.generate_vector(), hrr.generate_vector())
        assert hrr.size == 2  # Still 2 named items


class TestSimilarityOperations:
    """Test similarity calculations."""
    
    def test_similarity_identical(self):
        """Test similarity of identical vectors."""
        hrr = create_hrr(dimension=512, seed=42)
        
        v = hrr.generate_vector()
        similarity = hrr.similarity(v, v)
        
        assert np.abs(similarity - 1.0) < 1e-10
    
    def test_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        hrr = create_hrr(dimension=10000, seed=42)  # Large dimension
        
        # Random vectors in high dimension are nearly orthogonal
        v1 = hrr.generate_vector()
        v2 = hrr.generate_vector()
        
        similarity = hrr.similarity(v1, v2)
        assert np.abs(similarity) < 0.1
    
    def test_similarity_negated(self):
        """Test similarity of negated vectors."""
        hrr = create_hrr(dimension=512, seed=42)
        
        v = hrr.generate_vector()
        v_neg = -v
        
        similarity = hrr.similarity(v, v_neg)
        assert np.abs(similarity + 1.0) < 1e-10  # Should be -1


class TestUnitaryOperations:
    """Test unitary vector operations."""
    
    def test_make_unitary_real(self):
        """Test making real vectors unitary."""
        hrr = create_hrr(dimension=512, storage_method="real", seed=42)
        
        v = hrr.generate_vector()
        u = hrr.make_unitary(v)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(u) - 1.0) < 1e-6
        
        # Test self-inverse property
        identity = np.zeros(512)
        identity[0] = 1.0
        
        # u * u should give peak at 0
        self_conv = hrr.bind(u, u)
        assert np.argmax(np.abs(self_conv)) == 0
    
    def test_make_unitary_complex(self):
        """Test making complex vectors unitary."""
        hrr = create_hrr(dimension=1024, storage_method="complex", seed=42)
        
        v = hrr.generate_vector()
        u = hrr.make_unitary(v)
        
        # All magnitudes should be 1
        magnitudes = np.abs(u)
        assert np.allclose(magnitudes, 1.0, rtol=1e-10)


class TestVectorValidation:
    """Test vector validation."""
    
    def test_validate_vector_shape(self):
        """Test vector shape validation."""
        hrr = create_hrr(dimension=512)
        
        # Wrong shape
        with pytest.raises(ValueError, match="Expected shape"):
            hrr.bind(np.zeros(256), np.zeros(512))
    
    def test_validate_vector_type(self):
        """Test vector type validation."""
        hrr = create_hrr(dimension=512)
        
        # Wrong type
        with pytest.raises(TypeError, match="Expected numpy array"):
            hrr.bind([1, 2, 3], np.zeros(512))


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_vector_normalization(self):
        """Test normalization of zero vector."""
        hrr = create_hrr(dimension=512)
        
        # Zero vector should remain zero after normalization
        zero = np.zeros(512)
        normalized = hrr._normalize(zero)
        assert np.allclose(normalized, zero)
    
    def test_recall_empty_memory(self):
        """Test recall from empty memory."""
        hrr = create_hrr(dimension=512)
        
        key = hrr.generate_vector()
        result = hrr.recall(key)
        
        assert result is None


class TestComplexOperations:
    """Test operations specific to complex storage."""
    
    def test_complex_binding(self):
        """Test binding with complex vectors."""
        hrr = create_hrr(dimension=1024, storage_method="complex", seed=42)
        
        a = hrr.generate_vector()
        b = hrr.generate_vector()
        
        c = hrr.bind(a, b)
        
        # Result should be complex
        assert np.iscomplexobj(c)
        
        # Test unbinding
        retrieved = hrr.unbind(c, a)
        similarity = hrr.similarity(retrieved, b)
        assert similarity > 0.9
    
    def test_complex_similarity(self):
        """Test similarity with complex vectors."""
        hrr = create_hrr(dimension=1024, storage_method="complex", seed=42)
        
        v1 = hrr.generate_vector()
        v2 = hrr.generate_vector()
        
        # Similarity should be real
        sim = hrr.similarity(v1, v2)
        assert isinstance(sim, (int, float))
        assert -1 <= sim <= 1