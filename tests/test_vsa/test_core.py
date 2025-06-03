"""
Tests for VSA core functionality.

Tests the base VSA class, configuration, and factory functions.
"""

import pytest
import numpy as np
from typing import List
import tempfile
import json

from cognitive_computing.vsa.core import (
    VSA, VSAConfig, create_vsa, load_vsa_config, save_vsa_config
)
from cognitive_computing.vsa.vectors import (
    BinaryVector, BipolarVector, TernaryVector, ComplexVector, IntegerVector
)
from cognitive_computing.vsa.binding import (
    XORBinding, MultiplicationBinding, ConvolutionBinding, 
    MAPBinding, PermutationBinding
)


class TestVSAConfig:
    """Test VSA configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VSAConfig()
        assert config.dimension == 1000
        assert config.vector_type == "bipolar"
        assert config.binding_method == "multiplication"
        assert config.normalize_result is True
        assert config.seed is None
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = VSAConfig(
            dimension=2048,
            vector_type="binary",
            binding_method="xor",
            normalize_result=False,
            seed=42
        )
        assert config.dimension == 2048
        assert config.vector_type == "binary"
        assert config.binding_method == "xor"
        assert config.normalize_result is False
        assert config.seed == 42
        
    def test_invalid_dimension(self):
        """Test invalid dimension validation."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            VSAConfig(dimension=0)
            
        with pytest.raises(ValueError, match="Dimension must be positive"):
            VSAConfig(dimension=-100)
            
    def test_invalid_vector_type(self):
        """Test invalid vector type validation."""
        with pytest.raises(ValueError, match="Unknown vector type"):
            VSAConfig(vector_type="invalid")
            
    def test_invalid_binding_method(self):
        """Test invalid binding method validation."""
        with pytest.raises(ValueError, match="Unknown binding method"):
            VSAConfig(binding_method="invalid")
            
    def test_incompatible_combinations(self):
        """Test incompatible vector type and binding method combinations."""
        # XOR only works with binary vectors
        with pytest.raises(ValueError, match="XOR binding only works with binary vectors"):
            VSAConfig(vector_type="bipolar", binding_method="xor")
            
        # Convolution doesn't work with ternary vectors
        with pytest.raises(ValueError, match="Convolution binding not supported for ternary vectors"):
            VSAConfig(vector_type="ternary", binding_method="convolution")


class TestVSACreation:
    """Test VSA creation and initialization."""
    
    def test_create_default_vsa(self):
        """Test creating VSA with default settings."""
        vsa = VSA()
        assert vsa.config.dimension == 1000
        assert vsa.config.vector_type == "bipolar"
        assert vsa.config.binding_method == "multiplication"
        assert isinstance(vsa._vector_class, type)
        assert isinstance(vsa._binding_op, MultiplicationBinding)
        
    def test_create_binary_xor_vsa(self):
        """Test creating binary VSA with XOR binding."""
        config = VSAConfig(
            dimension=512,
            vector_type="binary",
            binding_method="xor"
        )
        vsa = VSA(config)
        assert vsa._vector_class == BinaryVector
        assert isinstance(vsa._binding_op, XORBinding)
        
    def test_create_complex_vsa(self):
        """Test creating complex VSA."""
        config = VSAConfig(
            dimension=1024,
            vector_type="complex",
            binding_method="multiplication"
        )
        vsa = VSA(config)
        assert vsa._vector_class == ComplexVector
        assert isinstance(vsa._binding_op, MultiplicationBinding)
        
    def test_create_vsa_with_seed(self):
        """Test VSA creation with seed for reproducibility."""
        config = VSAConfig(seed=42)
        vsa1 = VSA(config)
        vsa2 = VSA(config)
        
        # Generate vectors should be the same
        vec1_a = vsa1.generate_vector()
        vec2_a = vsa2.generate_vector()
        np.testing.assert_array_equal(vec1_a, vec2_a)
        
    def test_factory_function(self):
        """Test create_vsa factory function."""
        # Default creation
        vsa1 = create_vsa()
        assert isinstance(vsa1, VSA)
        
        # With parameters
        vsa2 = create_vsa(
            dimension=2048,
            vector_type="binary",
            binding_method="xor"
        )
        assert vsa2.config.dimension == 2048
        assert vsa2.config.vector_type == "binary"
        assert vsa2.config.binding_method == "xor"


class TestVSAOperations:
    """Test VSA basic operations."""
    
    @pytest.fixture
    def bipolar_vsa(self):
        """Create bipolar VSA for testing."""
        return create_vsa(dimension=512, vector_type="bipolar", seed=42)
        
    @pytest.fixture
    def binary_vsa(self):
        """Create binary VSA for testing."""
        return create_vsa(
            dimension=512,
            vector_type="binary",
            binding_method="xor",
            seed=42
        )
        
    def test_generate_vector(self, bipolar_vsa):
        """Test vector generation."""
        vec = bipolar_vsa.generate_vector()
        assert vec.shape == (512,)
        assert np.all(np.isin(vec, [-1, 1]))
        
    def test_bind_unbind_bipolar(self, bipolar_vsa):
        """Test binding and unbinding with bipolar vectors."""
        # Generate two vectors
        vec1 = bipolar_vsa.generate_vector()
        vec2 = bipolar_vsa.generate_vector()
        
        # Bind them
        bound = bipolar_vsa.bind(vec1, vec2)
        assert bound.shape == vec1.shape
        
        # Unbind should recover vec2
        recovered = bipolar_vsa.unbind(bound, vec1)
        
        # Check similarity (should be high but not perfect due to normalization)
        similarity = np.corrcoef(recovered.flatten(), vec2.flatten())[0, 1]
        assert similarity > 0.9
        
    def test_bind_unbind_binary(self, binary_vsa):
        """Test binding and unbinding with binary vectors."""
        # Generate two vectors
        vec1 = binary_vsa.generate_vector()
        vec2 = binary_vsa.generate_vector()
        
        # Bind them
        bound = binary_vsa.bind(vec1, vec2)
        assert bound.shape == vec1.shape
        assert np.all(np.isin(bound, [0, 1]))
        
        # XOR is self-inverse, so unbind is same as bind
        recovered = binary_vsa.unbind(bound, vec1)
        
        # Should recover exactly for XOR
        np.testing.assert_array_equal(recovered, vec2)
        
    def test_bundle(self, bipolar_vsa):
        """Test bundling operation."""
        vecs = [bipolar_vsa.generate_vector() for _ in range(5)]
        
        bundled = bipolar_vsa.bundle(vecs)
        assert bundled.shape == vecs[0].shape
        
        # Bundled vector should have some similarity to each input
        for vec in vecs:
            sim = np.corrcoef(bundled.flatten(), vec.flatten())[0, 1]
            assert sim > 0  # Should have positive similarity
            
    def test_bundle_with_weights(self, bipolar_vsa):
        """Test weighted bundling."""
        vec1 = bipolar_vsa.generate_vector()
        vec2 = bipolar_vsa.generate_vector()
        
        # Weight first vector more
        bundled = bipolar_vsa.bundle([vec1, vec2], weights=[0.8, 0.2])
        
        # Should be more similar to vec1
        sim1 = np.corrcoef(bundled.flatten(), vec1.flatten())[0, 1]
        sim2 = np.corrcoef(bundled.flatten(), vec2.flatten())[0, 1]
        assert sim1 > sim2
        
    def test_permute(self, bipolar_vsa):
        """Test permutation operation."""
        vec = bipolar_vsa.generate_vector()
        
        # Permute by 1
        perm1 = bipolar_vsa.permute(vec, shift=1)
        assert perm1.shape == vec.shape
        assert not np.array_equal(perm1, vec)
        
        # Permute back
        unperm = bipolar_vsa.permute(perm1, shift=-1)
        np.testing.assert_array_almost_equal(unperm, vec)
        
    def test_inverse_permute(self, bipolar_vsa):
        """Test inverse permutation."""
        vec = bipolar_vsa.generate_vector()
        
        # Create custom permutation
        perm_indices = np.random.RandomState(42).permutation(len(vec))
        permuted = bipolar_vsa.permute(vec, permutation=perm_indices)
        
        # Inverse should recover original
        inverse_indices = np.argsort(perm_indices)
        recovered = bipolar_vsa.permute(permuted, permutation=inverse_indices)
        np.testing.assert_array_almost_equal(recovered, vec)
        
    def test_similarity(self, bipolar_vsa):
        """Test similarity computation."""
        vec1 = bipolar_vsa.generate_vector()
        vec2 = bipolar_vsa.generate_vector()
        
        # Self-similarity should be 1
        sim_self = bipolar_vsa.similarity(vec1, vec1)
        assert abs(sim_self - 1.0) < 1e-6
        
        # Random vectors should have low similarity
        sim_random = bipolar_vsa.similarity(vec1, vec2)
        assert abs(sim_random) < 0.2
        
        # Negated vector should have -1 similarity
        sim_neg = bipolar_vsa.similarity(vec1, -vec1)
        assert abs(sim_neg + 1.0) < 1e-6


class TestVSAProperties:
    """Test mathematical properties of VSA operations."""
    
    @pytest.fixture
    def vsa_instances(self):
        """Create different VSA instances for testing."""
        return {
            "binary_xor": create_vsa(
                dimension=512, vector_type="binary", 
                binding_method="xor", seed=42
            ),
            "bipolar_mult": create_vsa(
                dimension=512, vector_type="bipolar",
                binding_method="multiplication", seed=42
            ),
            "bipolar_conv": create_vsa(
                dimension=512, vector_type="bipolar",
                binding_method="convolution", seed=42
            )
        }
        
    def test_binding_inverse_property(self, vsa_instances):
        """Test that unbind is inverse of bind."""
        for name, vsa in vsa_instances.items():
            vec1 = vsa.generate_vector()
            vec2 = vsa.generate_vector()
            
            # Bind and unbind
            bound = vsa.bind(vec1, vec2)
            recovered = vsa.unbind(bound, vec1)
            
            # Check recovery
            if name == "binary_xor":
                # XOR is perfectly invertible
                np.testing.assert_array_equal(recovered, vec2)
            else:
                # Others are approximately invertible
                sim = vsa.similarity(recovered, vec2)
                assert sim > 0.8, f"{name}: similarity = {sim}"
                
    def test_self_inverse_binding(self):
        """Test self-inverse property of XOR and permutation."""
        # XOR is self-inverse
        vsa = create_vsa(
            dimension=512, vector_type="binary",
            binding_method="xor", seed=42
        )
        vec1 = vsa.generate_vector()
        vec2 = vsa.generate_vector()
        
        # Binding twice should recover original
        bound1 = vsa.bind(vec1, vec2)
        bound2 = vsa.bind(bound1, vec2)  # XOR with vec2 again
        np.testing.assert_array_equal(bound2, vec1)
        
    def test_bundling_properties(self, vsa_instances):
        """Test properties of bundling operation."""
        for name, vsa in vsa_instances.items():
            vecs = [vsa.generate_vector() for _ in range(5)]
            
            # Bundle all
            bundled = vsa.bundle(vecs)
            
            # Bundled vector should be similar to all inputs
            similarities = [vsa.similarity(bundled, vec) for vec in vecs]
            assert all(sim > 0 for sim in similarities)
            
            # Bundling with itself should return normalized self
            vec = vecs[0]
            bundled_self = vsa.bundle([vec, vec])
            sim_self = vsa.similarity(bundled_self, vec)
            assert sim_self > 0.9
            
    def test_permutation_properties(self, vsa_instances):
        """Test properties of permutation."""
        for name, vsa in vsa_instances.items():
            vec = vsa.generate_vector()
            
            # Permuting n times by 1 should cycle back
            n = len(vec)
            permuted = vec.copy()
            for _ in range(n):
                permuted = vsa.permute(permuted, shift=1)
            np.testing.assert_array_almost_equal(permuted, vec)
            
            # Permutation should preserve similarity to self
            perm = vsa.permute(vec, shift=10)
            # Note: similarity might not be preserved for all vector types
            # but norms should be preserved
            if name != "binary_xor":  # Binary doesn't use standard norm
                assert abs(np.linalg.norm(perm) - np.linalg.norm(vec)) < 1e-6


class TestVSAConfigPersistence:
    """Test saving and loading VSA configurations."""
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        config = VSAConfig(
            dimension=2048,
            vector_type="complex",
            binding_method="convolution",
            normalize_result=False,
            seed=12345
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_vsa_config(config, f.name)
            loaded_config = load_vsa_config(f.name)
            
        assert loaded_config.dimension == config.dimension
        assert loaded_config.vector_type == config.vector_type
        assert loaded_config.binding_method == config.binding_method
        assert loaded_config.normalize_result == config.normalize_result
        assert loaded_config.seed == config.seed
        
    def test_config_to_dict_from_dict(self):
        """Test configuration serialization."""
        config = VSAConfig(
            dimension=1536,
            vector_type="ternary",
            binding_method="map"
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["dimension"] == 1536
        assert config_dict["vector_type"] == "ternary"
        assert config_dict["binding_method"] == "map"
        
        # Create from dict
        new_config = VSAConfig.from_dict(config_dict)
        assert new_config.dimension == config.dimension
        assert new_config.vector_type == config.vector_type
        assert new_config.binding_method == config.binding_method


class TestVSAEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_bundle(self):
        """Test bundling empty list."""
        vsa = create_vsa()
        with pytest.raises(ValueError, match="No vectors to bundle"):
            vsa.bundle([])
            
    def test_mismatched_dimensions(self):
        """Test operations with mismatched dimensions."""
        vsa = create_vsa(dimension=512)
        vec1 = vsa.generate_vector()
        vec2 = np.random.randn(256)  # Wrong dimension
        
        with pytest.raises(ValueError, match="dimension"):
            vsa.bind(vec1, vec2)
            
    def test_invalid_weights(self):
        """Test bundling with invalid weights."""
        vsa = create_vsa()
        vecs = [vsa.generate_vector() for _ in range(3)]
        
        # Wrong number of weights
        with pytest.raises(ValueError, match="Number of weights"):
            vsa.bundle(vecs, weights=[0.5, 0.5])
            
    def test_zero_shift_permutation(self):
        """Test permutation with zero shift."""
        vsa = create_vsa()
        vec = vsa.generate_vector()
        
        # Zero shift should return same vector
        perm0 = vsa.permute(vec, shift=0)
        np.testing.assert_array_equal(perm0, vec)
        
    def test_large_shift_permutation(self):
        """Test permutation with large shift."""
        vsa = create_vsa(dimension=100)
        vec = vsa.generate_vector()
        
        # Shift by multiple of dimension should return same
        perm = vsa.permute(vec, shift=300)  # 3 * 100
        np.testing.assert_array_almost_equal(perm, vec)


class TestVSAIntegration:
    """Integration tests for VSA functionality."""
    
    def test_symbol_binding_retrieval(self):
        """Test binding and retrieval of symbolic information."""
        vsa = create_vsa(dimension=1024, vector_type="bipolar", seed=42)
        
        # Create symbol vectors
        symbols = {
            "cat": vsa.generate_vector(),
            "dog": vsa.generate_vector(),
            "fur": vsa.generate_vector(),
            "bark": vsa.generate_vector()
        }
        
        # Bind properties to objects
        cat_fur = vsa.bind(symbols["cat"], symbols["fur"])
        dog_fur = vsa.bind(symbols["dog"], symbols["fur"])
        dog_bark = vsa.bind(symbols["dog"], symbols["bark"])
        
        # Bundle all relations
        memory = vsa.bundle([cat_fur, dog_fur, dog_bark])
        
        # Query: what does cat have?
        cat_query = vsa.unbind(memory, symbols["cat"])
        
        # Should be most similar to fur
        sims = {k: vsa.similarity(cat_query, v) for k, v in symbols.items()}
        assert max(sims, key=sims.get) == "fur"
        
    def test_sequence_encoding(self):
        """Test encoding sequences with permutation."""
        vsa = create_vsa(dimension=1024, vector_type="bipolar", seed=42)
        
        # Create sequence: A -> B -> C
        items = {
            "A": vsa.generate_vector(),
            "B": vsa.generate_vector(), 
            "C": vsa.generate_vector()
        }
        
        # Encode with position
        seq = vsa.bundle([
            items["A"],
            vsa.permute(items["B"], shift=1),
            vsa.permute(items["C"], shift=2)
        ])
        
        # Decode position 2
        pos2 = vsa.permute(seq, shift=-2)
        
        # Should be most similar to C
        sims = {k: vsa.similarity(pos2, v) for k, v in items.items()}
        assert max(sims, key=sims.get) == "C"
        
    def test_composite_structures(self):
        """Test building composite structures."""
        vsa = create_vsa(dimension=1024, vector_type="bipolar", seed=42)
        
        # Role-filler binding
        roles = {
            "agent": vsa.generate_vector(),
            "action": vsa.generate_vector(),
            "object": vsa.generate_vector()
        }
        
        fillers = {
            "john": vsa.generate_vector(),
            "eat": vsa.generate_vector(),
            "apple": vsa.generate_vector()
        }
        
        # Bind role-filler pairs
        sentence = vsa.bundle([
            vsa.bind(roles["agent"], fillers["john"]),
            vsa.bind(roles["action"], fillers["eat"]),
            vsa.bind(roles["object"], fillers["apple"])
        ])
        
        # Query: who is the agent?
        agent_query = vsa.unbind(sentence, roles["agent"])
        
        # Should be most similar to john
        sims = {k: vsa.similarity(agent_query, v) for k, v in fillers.items()}
        assert max(sims, key=sims.get) == "john"