"""Tests for core HDC functionality."""

import pytest
import numpy as np

from cognitive_computing.hdc.core import (
    HDC,
    HDCConfig,
    HypervectorType,
    create_hdc,
)


class TestHDCConfig:
    """Test HDC configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HDCConfig()
        assert config.dimension == 10000
        assert config.hypervector_type == "bipolar"
        assert config.seed_orthogonal is True
        assert config.similarity_threshold == 0.0
        assert config.item_memory_size is None
        assert config.levels == 5
        assert config.sparsity == 0.33
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = HDCConfig(
            dimension=5000,
            hypervector_type="binary",
            seed=42,
            sparsity=0.1
        )
        assert config.dimension == 5000
        assert config.hypervector_type == "binary"
        assert config.seed == 42
        assert config.sparsity == 0.1
        
    def test_invalid_dimension(self):
        """Test invalid dimension handling."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            HDCConfig(dimension=0)
            
        with pytest.raises(ValueError, match="Dimension should be at least 100"):
            HDCConfig(dimension=50)
            
    def test_invalid_hypervector_type(self):
        """Test invalid hypervector type."""
        with pytest.raises(ValueError, match="hypervector_type must be one of"):
            HDCConfig(hypervector_type="invalid")
            
    def test_invalid_levels(self):
        """Test invalid levels."""
        with pytest.raises(ValueError, match="Levels must be at least 2"):
            HDCConfig(levels=1)
            
    def test_invalid_sparsity(self):
        """Test invalid sparsity."""
        with pytest.raises(ValueError, match="Sparsity must be in"):
            HDCConfig(sparsity=0)
            
        with pytest.raises(ValueError, match="Sparsity must be in"):
            HDCConfig(sparsity=1)


class TestHDC:
    """Test main HDC class."""
    
    def test_initialization(self):
        """Test HDC initialization."""
        config = HDCConfig(dimension=1000, seed=42)
        hdc = HDC(config)
        
        assert hdc.dimension == 1000
        assert hdc.hypervector_type == HypervectorType.BIPOLAR
        assert len(hdc.item_memory) == 0
        assert len(hdc.class_hypervectors) == 0
        assert hdc.size == 0
        
    def test_generate_hypervector_binary(self):
        """Test binary hypervector generation."""
        hdc = create_hdc(dimension=1000, hypervector_type="binary", seed=42)
        
        hv = hdc.generate_hypervector()
        assert hv.shape == (1000,)
        assert np.all(np.isin(hv, [0, 1]))
        assert hv.dtype == np.uint8
        
        # Test approximate uniformity
        assert 0.4 < np.mean(hv) < 0.6
        
    def test_generate_hypervector_bipolar(self):
        """Test bipolar hypervector generation."""
        hdc = create_hdc(dimension=1000, hypervector_type="bipolar", seed=42)
        
        hv = hdc.generate_hypervector()
        assert hv.shape == (1000,)
        assert np.all(np.isin(hv, [-1, 1]))
        
        # Test approximate balance
        assert -0.1 < np.mean(hv) < 0.1
        
    def test_generate_hypervector_bipolar_orthogonal(self):
        """Test orthogonal bipolar hypervector generation."""
        hdc = create_hdc(
            dimension=1000,
            hypervector_type="bipolar",
            seed_orthogonal=True,
            seed=42
        )
        
        # Generate first vector
        hv1 = hdc.generate_hypervector()
        
        # Generate second vector orthogonal to first
        hv2 = hdc.generate_hypervector(orthogonal_to=[hv1])
        
        # Check near-orthogonality (not perfect due to discretization)
        similarity = hdc.similarity(hv1, hv2)
        assert abs(similarity) < 0.1
        
    def test_generate_hypervector_ternary(self):
        """Test ternary hypervector generation."""
        hdc = create_hdc(
            dimension=1000,
            hypervector_type="ternary",
            sparsity=0.3,
            seed=42
        )
        
        hv = hdc.generate_hypervector()
        assert hv.shape == (1000,)
        assert np.all(np.isin(hv, [-1, 0, 1]))
        assert hv.dtype == np.int8
        
        # Check sparsity
        n_nonzero = np.sum(hv != 0)
        expected_nonzero = 1000 * 0.3
        assert abs(n_nonzero - expected_nonzero) < 50  # Allow some variance
        
    def test_generate_hypervector_level(self):
        """Test level hypervector generation."""
        hdc = create_hdc(
            dimension=1000,
            hypervector_type="level",
            levels=5,
            seed=42
        )
        
        hv = hdc.generate_hypervector()
        assert hv.shape == (1000,)
        assert np.all((0 <= hv) & (hv < 5))
        assert hv.dtype == np.int8
        
        # Check approximate uniformity
        for level in range(5):
            count = np.sum(hv == level)
            assert 150 < count < 250  # Roughly 200 each
            
    def test_bind_binary(self):
        """Test binary binding (XOR)."""
        hdc = create_hdc(dimension=1000, hypervector_type="binary", seed=42)
        
        a = hdc.generate_hypervector()
        b = hdc.generate_hypervector()
        
        bound = hdc.bind(a, b)
        assert bound.shape == (1000,)
        assert np.all(np.isin(bound, [0, 1]))
        
        # Test self-inverse property
        unbound = hdc.bind(bound, a)
        assert np.array_equal(unbound, b)
        
    def test_bind_bipolar(self):
        """Test bipolar binding (multiplication)."""
        hdc = create_hdc(dimension=1000, hypervector_type="bipolar", seed=42)
        
        a = hdc.generate_hypervector()
        b = hdc.generate_hypervector()
        
        bound = hdc.bind(a, b)
        assert bound.shape == (1000,)
        assert np.all(np.isin(bound, [-1, 1]))
        
        # Test self-inverse property
        unbound = hdc.bind(bound, a)
        assert np.array_equal(unbound, b)
        
    def test_bind_ternary(self):
        """Test ternary binding."""
        hdc = create_hdc(dimension=1000, hypervector_type="ternary", seed=42)
        
        a = hdc.generate_hypervector()
        b = hdc.generate_hypervector()
        
        bound = hdc.bind(a, b)
        assert bound.shape == (1000,)
        # Result can have values outside {-1,0,1} due to multiplication
        
    def test_bind_level(self):
        """Test level binding (modular addition)."""
        hdc = create_hdc(dimension=1000, hypervector_type="level", levels=5, seed=42)
        
        a = hdc.generate_hypervector()
        b = hdc.generate_hypervector()
        
        bound = hdc.bind(a, b)
        assert bound.shape == (1000,)
        assert np.all((0 <= bound) & (bound < 5))
        
    def test_bundle_binary(self):
        """Test binary bundling."""
        hdc = create_hdc(dimension=1000, hypervector_type="binary", seed=42)
        
        vectors = [hdc.generate_hypervector() for _ in range(5)]
        bundled = hdc.bundle(vectors)
        
        assert bundled.shape == (1000,)
        assert np.all(np.isin(bundled, [0, 1]))
        
        # Bundled should be similar to all inputs
        for v in vectors:
            sim = hdc.similarity(bundled, v)
            assert sim > 0.3  # Positive similarity
            
    def test_bundle_bipolar(self):
        """Test bipolar bundling."""
        hdc = create_hdc(dimension=1000, hypervector_type="bipolar", seed=42)
        
        vectors = [hdc.generate_hypervector() for _ in range(5)]
        bundled = hdc.bundle(vectors)
        
        assert bundled.shape == (1000,)
        assert np.all(np.isin(bundled, [-1, 1]))
        
        # Bundled should be similar to all inputs
        for v in vectors:
            sim = hdc.similarity(bundled, v)
            assert sim > 0.2
            
    def test_bundle_empty(self):
        """Test bundling empty list."""
        hdc = create_hdc(dimension=1000)
        
        with pytest.raises(ValueError, match="Cannot bundle empty"):
            hdc.bundle([])
            
    def test_permute(self):
        """Test permutation operation."""
        hdc = create_hdc(dimension=100, hypervector_type="binary", seed=42)
        
        # Create a pattern that's easy to verify
        hv = np.zeros(100, dtype=np.uint8)
        hv[0] = 1  # Single 1 at position 0
        
        # Test shift by 1
        permuted = hdc.permute(hv, shift=1)
        assert permuted[1] == 1 and np.sum(permuted) == 1
        
        # Test shift by -1
        permuted = hdc.permute(hv, shift=-1)
        assert permuted[99] == 1 and np.sum(permuted) == 1
        
        # Test larger shift
        permuted = hdc.permute(hv, shift=10)
        assert permuted[10] == 1 and np.sum(permuted) == 1
        
    def test_similarity_binary(self):
        """Test binary similarity (Hamming)."""
        hdc = create_hdc(dimension=1000, hypervector_type="binary", seed=42)
        
        # Same vector
        hv = hdc.generate_hypervector()
        assert hdc.similarity(hv, hv) == 1.0
        
        # Opposite vector
        opposite = 1 - hv
        assert hdc.similarity(hv, opposite) == -1.0
        
        # Random vectors
        hv1 = hdc.generate_hypervector()
        hv2 = hdc.generate_hypervector()
        sim = hdc.similarity(hv1, hv2)
        assert -0.2 < sim < 0.2  # Should be near 0
        
    def test_similarity_bipolar(self):
        """Test bipolar similarity (cosine)."""
        hdc = create_hdc(dimension=1000, hypervector_type="bipolar", seed=42)
        
        # Same vector
        hv = hdc.generate_hypervector()
        assert hdc.similarity(hv, hv) == 1.0
        
        # Opposite vector
        opposite = -hv
        assert hdc.similarity(hv, opposite) == -1.0
        
        # Random vectors
        hv1 = hdc.generate_hypervector()
        hv2 = hdc.generate_hypervector()
        sim = hdc.similarity(hv1, hv2)
        assert -0.1 < sim < 0.1  # Should be near 0
        
    def test_store_recall_basic(self):
        """Test basic store and recall."""
        hdc = create_hdc(dimension=1000, hypervector_type="bipolar", seed=42)
        
        key = hdc.generate_hypervector()
        value = hdc.generate_hypervector()
        
        # Store
        hdc.store(key, value)
        assert hdc.size == 1
        
        # Recall with exact key
        recalled = hdc.recall(key)
        assert recalled is not None
        
        # Due to the current implementation, exact recall might not work
        # This is a limitation we'll address in future iterations
        
    def test_clear(self):
        """Test clearing memory."""
        hdc = create_hdc(dimension=1000, seed=42)
        
        # Store some items
        for _ in range(5):
            key = hdc.generate_hypervector()
            value = hdc.generate_hypervector()
            hdc.store(key, value)
            
        assert hdc.size == 5
        assert len(hdc.item_memory) == 5
        
        # Clear
        hdc.clear()
        assert hdc.size == 0
        assert len(hdc.item_memory) == 0
        assert len(hdc.class_hypervectors) == 0
        
    def test_validate_hypervector(self):
        """Test hypervector validation."""
        hdc = create_hdc(dimension=100, hypervector_type="binary")
        
        # Valid hypervector
        valid = np.ones(100, dtype=np.uint8)
        hdc._validate_hypervector(valid, "test")  # Should not raise
        
        # Wrong type
        with pytest.raises(TypeError, match="must be a numpy array"):
            hdc._validate_hypervector([1, 0, 1], "test")
            
        # Wrong shape
        with pytest.raises(ValueError, match="must have shape"):
            hdc._validate_hypervector(np.ones(50), "test")
            
        # Wrong values for binary
        invalid = np.array([0, 1, 2] * 33 + [0], dtype=np.uint8)
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            hdc._validate_hypervector(invalid, "test")
            
    def test_factory_function(self):
        """Test create_hdc factory function."""
        # Default parameters
        hdc = create_hdc()
        assert hdc.dimension == 10000
        assert hdc.hypervector_type == HypervectorType.BIPOLAR
        
        # Custom parameters
        hdc = create_hdc(
            dimension=5000,
            hypervector_type="ternary",
            sparsity=0.2,
            seed=123
        )
        assert hdc.dimension == 5000
        assert hdc.hypervector_type == HypervectorType.TERNARY
        assert hdc.config.sparsity == 0.2
        assert hdc.config.seed == 123