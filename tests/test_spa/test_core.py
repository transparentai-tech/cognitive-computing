"""Tests for core SPA functionality."""

import pytest
import numpy as np
from cognitive_computing.spa import (
    SPAConfig, SemanticPointer, Vocabulary, SPA,
    create_spa, create_vocabulary
)


class TestSPAConfig:
    """Test SPAConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SPAConfig(dimension=512)
        assert config.dimension == 512
        assert config.subdimensions == 16
        assert config.neurons_per_dimension == 50
        assert config.max_similarity_matches == 10
        assert config.threshold == 0.3
        assert config.normalize_pointers is True
        
    def test_subdimension_validation(self):
        """Test subdimension must divide dimension evenly."""
        # Valid: 512 / 16 = 32
        config = SPAConfig(dimension=512, subdimensions=16)
        assert config.subdimensions == 16
        
        # Invalid: 512 / 15 = 34.133...
        with pytest.raises(ValueError, match="must be divisible"):
            SPAConfig(dimension=512, subdimensions=15)
            
    def test_invalid_parameters(self):
        """Test validation of configuration parameters."""
        # Negative subdimensions
        with pytest.raises(ValueError, match="subdimensions must be positive"):
            SPAConfig(dimension=512, subdimensions=-1)
            
        # Zero neurons per dimension
        with pytest.raises(ValueError, match="neurons_per_dimension must be positive"):
            SPAConfig(dimension=512, neurons_per_dimension=0)
            
        # Invalid threshold
        with pytest.raises(ValueError, match="threshold must be in"):
            SPAConfig(dimension=512, threshold=1.5)
            
        # Negative mutual inhibition
        with pytest.raises(ValueError, match="mutual_inhibition must be non-negative"):
            SPAConfig(dimension=512, mutual_inhibition=-0.5)
            
        # Zero timestep
        with pytest.raises(ValueError, match="dt must be positive"):
            SPAConfig(dimension=512, dt=0)
            
    def test_timestep_warning(self, caplog):
        """Test warning when dt > synapse."""
        config = SPAConfig(dimension=512, dt=0.02, synapse=0.01)
        assert "may lead to instability" in caplog.text


class TestSemanticPointer:
    """Test SemanticPointer operations."""
    
    def test_creation(self):
        """Test semantic pointer creation."""
        vec = np.random.randn(512)
        sp = SemanticPointer(vec, name="test")
        
        assert sp.dimension == 512
        assert sp.name == "test"
        assert np.array_equal(sp.vector, vec)
        
    def test_invalid_creation(self):
        """Test invalid pointer creation."""
        # 2D vector
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            SemanticPointer(np.random.randn(10, 10))
            
    def test_normalize(self):
        """Test pointer normalization."""
        vec = np.array([3.0, 4.0])  # Length = 5
        sp = SemanticPointer(vec)
        
        normalized = sp.normalize()
        assert np.allclose(np.linalg.norm(normalized.vector), 1.0)
        assert np.allclose(normalized.vector, [0.6, 0.8])
        
        # Original unchanged
        assert np.array_equal(sp.vector, [3.0, 4.0])
        
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        vec = np.ones(10)
        sp = SemanticPointer(vec)
        
        # Right multiplication
        result = sp * 2.5
        assert np.allclose(result.vector, 2.5)
        
        # Left multiplication
        result = 3.0 * sp
        assert np.allclose(result.vector, 3.0)
        
    def test_binding(self):
        """Test binding with circular convolution."""
        vec1 = np.random.randn(128)
        vec2 = np.random.randn(128)
        sp1 = SemanticPointer(vec1, "A")
        sp2 = SemanticPointer(vec2, "B")
        
        # Bind
        bound = sp1 * sp2
        assert bound.dimension == 128
        
        # Should not equal either original
        assert not np.allclose(bound.vector, vec1)
        assert not np.allclose(bound.vector, vec2)
        
    def test_binding_dimension_mismatch(self):
        """Test binding with mismatched dimensions."""
        sp1 = SemanticPointer(np.random.randn(128))
        sp2 = SemanticPointer(np.random.randn(256))
        
        with pytest.raises(ValueError, match="Dimensions must match"):
            sp1 * sp2
            
    def test_inverse(self):
        """Test inverse for unbinding."""
        vec = np.random.randn(128)
        sp = SemanticPointer(vec, "A")
        
        inv = ~sp
        assert inv.dimension == 128
        assert inv.name == "~A"
        
        # Test that A * ~A ≈ identity
        identity = sp * inv
        # Should have high self-similarity
        auto_sim = np.dot(identity.vector, identity.vector) / (np.linalg.norm(identity.vector)**2)
        assert auto_sim > 0.9
        
    def test_bundling(self):
        """Test bundling (addition)."""
        vec1 = np.random.randn(128)
        vec2 = np.random.randn(128)
        sp1 = SemanticPointer(vec1)
        sp2 = SemanticPointer(vec2)
        
        bundled = sp1 + sp2
        assert np.allclose(bundled.vector, vec1 + vec2)
        
    def test_subtraction(self):
        """Test pointer subtraction."""
        vec1 = np.ones(128)
        vec2 = np.ones(128) * 0.5
        sp1 = SemanticPointer(vec1)
        sp2 = SemanticPointer(vec2)
        
        diff = sp1 - sp2
        assert np.allclose(diff.vector, 0.5)
        
    def test_similarity(self):
        """Test similarity computation."""
        # Identical vectors
        vec = np.random.randn(128)
        sp1 = SemanticPointer(vec)
        sp2 = SemanticPointer(vec)
        assert np.isclose(sp1.similarity(sp2), 1.0)
        
        # Orthogonal vectors
        sp3 = SemanticPointer(np.random.randn(128))
        sp4 = SemanticPointer(np.random.randn(128))
        sim = sp3.similarity(sp4)
        assert -0.3 < sim < 0.3  # Should be near 0
        
        # Opposite vectors
        sp5 = SemanticPointer(vec)
        sp6 = SemanticPointer(-vec)
        assert np.isclose(sp5.similarity(sp6), -1.0)
        
    def test_dot_product(self):
        """Test dot product computation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sp1 = SemanticPointer(vec1)
        sp2 = SemanticPointer(vec2)
        
        # Orthogonal
        assert sp1.dot(sp2) == 0.0
        
        # Self
        assert sp1.dot(sp1) == 1.0
        
    def test_representation(self):
        """Test string representation."""
        sp1 = SemanticPointer(np.ones(512), "test")
        assert repr(sp1) == "SemanticPointer('test', dim=512)"
        
        sp2 = SemanticPointer(np.ones(256))
        assert repr(sp2) == "SemanticPointer(dim=256)"


class TestVocabulary:
    """Test Vocabulary functionality."""
    
    def test_creation(self):
        """Test vocabulary creation."""
        vocab = Vocabulary(512)
        assert vocab.dimension == 512
        assert len(vocab) == 0
        
    def test_create_pointer(self):
        """Test creating pointers in vocabulary."""
        vocab = Vocabulary(128)
        
        # Create with auto-generated vector
        sp = vocab.create_pointer("A")
        assert sp.name == "A"
        assert sp.dimension == 128
        assert len(vocab) == 1
        assert "A" in vocab
        
        # Create with specified vector
        vec = np.ones(128)
        sp2 = vocab.create_pointer("B", vec)
        assert np.array_equal(sp2.vector, vec)
        assert len(vocab) == 2
        
    def test_duplicate_pointer_strict(self):
        """Test duplicate pointer in strict mode."""
        config = SPAConfig(dimension=128, strict_vocab=True)
        vocab = Vocabulary(128, config)
        
        vocab.create_pointer("A")
        with pytest.raises(ValueError, match="already exists"):
            vocab.create_pointer("A")
            
    def test_duplicate_pointer_non_strict(self, caplog):
        """Test duplicate pointer in non-strict mode."""
        vocab = Vocabulary(128)
        
        sp1 = vocab.create_pointer("A")
        sp2 = vocab.create_pointer("A")  # Should return existing
        
        assert sp1 is sp2
        assert "already exists" in caplog.text
        
    def test_getitem(self):
        """Test vocabulary indexing."""
        vocab = Vocabulary(128)
        vocab.create_pointer("A")
        
        sp = vocab["A"]
        assert sp.name == "A"
        
        # Auto-create in non-strict mode
        sp2 = vocab["B"]
        assert sp2.name == "B"
        assert len(vocab) == 2
        
    def test_getitem_strict(self):
        """Test vocabulary indexing in strict mode."""
        config = SPAConfig(dimension=128, strict_vocab=True)
        vocab = Vocabulary(128, config)
        
        with pytest.raises(KeyError, match="not in vocabulary"):
            vocab["nonexistent"]
            
    def test_cleanup_basic(self):
        """Test basic cleanup functionality."""
        vocab = Vocabulary(128)
        
        # Create some pointers
        vec_a = np.random.randn(128)
        vec_b = np.random.randn(128)
        vec_c = np.random.randn(128)
        
        vocab.create_pointer("A", vec_a)
        vocab.create_pointer("B", vec_b)
        vocab.create_pointer("C", vec_c)
        
        # Cleanup exact vector
        results = vocab.cleanup(vec_a)
        assert len(results) > 0
        assert results[0][0] == "A"
        assert results[0][1] > 0.99
        
    def test_cleanup_noisy(self):
        """Test cleanup with noisy vector."""
        vocab = Vocabulary(128)
        
        vec = np.random.randn(128)
        vocab.create_pointer("original", vec)
        
        # Add noise
        noisy = vec + 0.2 * np.random.randn(128)
        
        results = vocab.cleanup(noisy, top_n=1)
        assert results[0][0] == "original"
        assert 0.5 < results[0][1] < 1.0
        
    def test_cleanup_empty_vocab(self):
        """Test cleanup with empty vocabulary."""
        vocab = Vocabulary(128)
        results = vocab.cleanup(np.random.randn(128))
        assert results == []
        
    def test_cleanup_threshold(self):
        """Test cleanup threshold filtering."""
        config = SPAConfig(dimension=128, threshold=0.7)
        vocab = Vocabulary(128, config)
        
        # Create orthogonal pointers
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        # Random vector unlikely to have >0.7 similarity
        results = vocab.cleanup(np.random.randn(128))
        assert len(results) == 0 or all(sim >= 0.7 for _, sim in results)
        
    def test_parse_single(self):
        """Test parsing single pointer."""
        vocab = Vocabulary(128)
        vocab.create_pointer("A")
        
        sp = vocab.parse("A")
        assert sp.name == "A"
        
    def test_parse_scalar_multiplication(self):
        """Test parsing scalar multiplication."""
        vocab = Vocabulary(128)
        vec = np.ones(128)
        vocab.create_pointer("A", vec)
        
        sp = vocab.parse("2.5*A")
        assert np.allclose(sp.vector, 2.5)
        
    def test_parse_binding(self):
        """Test parsing pointer binding."""
        vocab = Vocabulary(128)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        sp = vocab.parse("A*B")
        assert sp.dimension == 128
        
    def test_parse_complex_not_implemented(self):
        """Test complex expression parsing not implemented."""
        vocab = Vocabulary(128)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        with pytest.raises(NotImplementedError):
            vocab.parse("(A+B)*C")
            
    def test_representation(self):
        """Test vocabulary representation."""
        vocab = Vocabulary(512)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        assert repr(vocab) == "Vocabulary(dimension=512, size=2)"


class TestSPA:
    """Test main SPA system."""
    
    def test_creation(self):
        """Test SPA system creation."""
        config = SPAConfig(dimension=512)
        spa = SPA(config)
        
        assert spa.config.dimension == 512
        assert isinstance(spa.vocabulary, Vocabulary)
        assert len(spa.modules) == 0
        assert spa.time == 0.0
        
    def test_factory_function(self):
        """Test create_spa factory function."""
        spa = create_spa(dimension=256, threshold=0.5)
        
        assert spa.config.dimension == 256
        assert spa.config.threshold == 0.5
        
    def test_clear(self):
        """Test clearing the system."""
        spa = create_spa(512)
        spa.vocabulary.create_pointer("A")
        spa.time = 5.0
        
        spa.clear()
        
        assert len(spa.vocabulary) == 0
        assert spa.time == 0.0
        
    def test_size_property(self):
        """Test size property."""
        spa = create_spa(512)
        assert spa.size == 0
        
        spa.vocabulary.create_pointer("A")
        spa.vocabulary.create_pointer("B")
        assert spa.size == 2
        
    def test_store_not_implemented(self):
        """Test store raises NotImplementedError."""
        spa = create_spa(512)
        with pytest.raises(NotImplementedError):
            spa.store(np.ones(512), np.ones(512))
            
    def test_recall_not_implemented(self):
        """Test recall raises NotImplementedError."""
        spa = create_spa(512)
        with pytest.raises(NotImplementedError):
            spa.recall(np.ones(512))
            
    def test_step(self):
        """Test simulation step."""
        spa = create_spa(512, dt=0.001)
        
        initial_time = spa.time
        spa.step()
        assert spa.time == initial_time + 0.001
        
        spa.step(dt=0.01)
        assert spa.time == initial_time + 0.011
        
    def test_run(self):
        """Test running simulation."""
        spa = create_spa(512, dt=0.001)
        
        spa.run(0.1)  # Run for 100ms
        assert np.isclose(spa.time, 0.1)
        
    def test_representation(self):
        """Test string representation."""
        spa = create_spa(512)
        spa.vocabulary.create_pointer("A")
        
        repr_str = repr(spa)
        assert "SPA" in repr_str
        assert "dimension=512" in repr_str
        assert "pointers=1" in repr_str


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_vocabulary(self):
        """Test vocabulary factory function."""
        vocab = create_vocabulary(256, normalize_pointers=False)
        
        assert vocab.dimension == 256
        assert vocab.config.normalize_pointers is False
        
    def test_create_spa_with_kwargs(self):
        """Test SPA creation with various kwargs."""
        spa = create_spa(
            dimension=1024,
            subdimensions=32,
            threshold=0.4,
            mutual_inhibition=2.0
        )
        
        assert spa.config.dimension == 1024
        assert spa.config.subdimensions == 32
        assert spa.config.threshold == 0.4
        assert spa.config.mutual_inhibition == 2.0