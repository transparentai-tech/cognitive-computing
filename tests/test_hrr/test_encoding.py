"""
Tests for HRR encoding strategies.

Tests role-filler encoding, sequence encoding, and hierarchical structures.
"""

import pytest
import numpy as np

from cognitive_computing.hrr import create_hrr, HRRConfig
from cognitive_computing.hrr.encoding import (
    RoleFillerEncoder, SequenceEncoder, HierarchicalEncoder
)
from cognitive_computing.hrr.cleanup import CleanupMemory, CleanupMemoryConfig


class TestRoleFillerEncoder:
    """Test role-filler encoding."""
    
    def test_encode_pair(self):
        """Test encoding a single role-filler pair."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = RoleFillerEncoder(hrr)
        
        # Create role and filler
        role = hrr.generate_vector()
        filler = hrr.generate_vector()
        
        # Encode pair
        binding = encoder.encode_pair(role, filler)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(binding) - 1.0) < 1e-6
        
        # Should be different from inputs
        assert hrr.similarity(binding, role) < 0.3
        assert hrr.similarity(binding, filler) < 0.3
    
    def test_encode_structure_basic(self):
        """Test encoding a basic structure."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = RoleFillerEncoder(hrr)
        
        # Create structure
        red = hrr.generate_vector()
        circle = hrr.generate_vector()
        large = hrr.generate_vector()
        
        structure = encoder.encode_structure({
            "color": red,
            "shape": circle,
            "size": large
        })
        
        # Should be normalized
        assert np.abs(np.linalg.norm(structure) - 1.0) < 1e-6
        
        # Check that roles were stored
        assert hrr.get_item("role:color") is not None
        assert hrr.get_item("role:shape") is not None
        assert hrr.get_item("role:size") is not None
    
    def test_encode_structure_with_roles(self):
        """Test encoding with pre-defined role vectors."""
        hrr = create_hrr(dimension=512, seed=42)
        encoder = RoleFillerEncoder(hrr)
        
        # Define role vectors
        role_vectors = {
            "subject": hrr.generate_vector(),
            "verb": hrr.generate_vector(),
            "object": hrr.generate_vector()
        }
        
        # Create fillers
        john = hrr.generate_vector()
        loves = hrr.generate_vector()
        mary = hrr.generate_vector()
        
        # Encode sentence structure
        sentence = encoder.encode_structure(
            {
                "subject": john,
                "verb": loves,
                "object": mary
            },
            role_vectors=role_vectors
        )
        
        # Decode each component
        decoded_subject = encoder.decode_filler(sentence, role_vectors["subject"])
        decoded_verb = encoder.decode_filler(sentence, role_vectors["verb"])
        decoded_object = encoder.decode_filler(sentence, role_vectors["object"])
        
        # Check similarities
        assert hrr.similarity(decoded_subject, john) > 0.5
        assert hrr.similarity(decoded_verb, loves) > 0.5
        assert hrr.similarity(decoded_object, mary) > 0.5
    
    def test_encode_empty_structure(self):
        """Test encoding empty structure."""
        hrr = create_hrr(dimension=512)
        encoder = RoleFillerEncoder(hrr)
        
        with pytest.raises(ValueError, match="Cannot encode empty"):
            encoder.encode_structure({})
    
    def test_decode_filler(self):
        """Test decoding individual fillers."""
        hrr = create_hrr(dimension=2048, seed=42)
        encoder = RoleFillerEncoder(hrr)
        
        # Create and encode structure
        role = hrr.generate_vector()
        filler = hrr.generate_vector()
        binding = encoder.encode_pair(role, filler)
        
        # Decode filler
        decoded = encoder.decode_filler(binding, role)
        
        # Should be similar to original
        similarity = hrr.similarity(decoded, filler)
        assert similarity > 0.95
    
    def test_decode_all_fillers(self):
        """Test decoding all fillers from a structure."""
        hrr = create_hrr(dimension=2048, seed=42)
        encoder = RoleFillerEncoder(hrr)
        
        # Create roles and fillers
        roles = {
            "a": hrr.generate_vector(),
            "b": hrr.generate_vector(),
            "c": hrr.generate_vector()
        }
        
        fillers = {
            "a": hrr.generate_vector(),
            "b": hrr.generate_vector(),
            "c": hrr.generate_vector()
        }
        
        # Encode structure
        structure = encoder.encode_structure(fillers, roles)
        
        # Decode all
        decoded = encoder.decode_all_fillers(structure, roles)
        
        # Check each filler
        for key in fillers:
            similarity = hrr.similarity(decoded[key], fillers[key])
            assert similarity > 0.5  # Some interference expected
    
    def test_role_reuse(self):
        """Test that roles can be reused across structures."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = RoleFillerEncoder(hrr)
        
        # First structure
        struct1 = encoder.encode_structure({
            "type": hrr.generate_vector(),
            "value": hrr.generate_vector()
        })
        
        # Get stored roles
        type_role = hrr.get_item("role:type")
        value_role = hrr.get_item("role:value")
        
        # Second structure with same roles
        new_type = hrr.generate_vector()
        new_value = hrr.generate_vector()
        
        struct2 = encoder.encode_structure({
            "type": new_type,
            "value": new_value
        })
        
        # Roles should be the same
        assert np.allclose(hrr.get_item("role:type"), type_role)
        assert np.allclose(hrr.get_item("role:value"), value_role)
        
        # But structures should be different
        assert hrr.similarity(struct1, struct2) < 0.5


class TestSequenceEncoder:
    """Test sequence encoding."""
    
    def test_get_position_vector(self):
        """Test position vector generation."""
        hrr = create_hrr(dimension=512, seed=42)
        encoder = SequenceEncoder(hrr)
        
        # Get position vectors
        pos0 = encoder.get_position_vector(0)
        pos1 = encoder.get_position_vector(1)
        pos2 = encoder.get_position_vector(2)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(pos0) - 1.0) < 1e-6
        assert np.abs(np.linalg.norm(pos1) - 1.0) < 1e-6
        
        # Should be different
        assert hrr.similarity(pos0, pos1) < 0.1
        assert hrr.similarity(pos0, pos2) < 0.1
        assert hrr.similarity(pos1, pos2) < 0.1
        
        # Should be deterministic
        pos0_again = encoder.get_position_vector(0)
        assert np.allclose(pos0, pos0_again)
    
    def test_invalid_position(self):
        """Test invalid position index."""
        hrr = create_hrr(dimension=512)
        encoder = SequenceEncoder(hrr)
        
        with pytest.raises(ValueError, match="Position must be non-negative"):
            encoder.get_position_vector(-1)
    
    def test_encode_sequence_positional(self):
        """Test positional sequence encoding."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = SequenceEncoder(hrr)
        
        # Create sequence items
        items = [hrr.generate_vector() for _ in range(3)]
        
        # Encode sequence
        sequence = encoder.encode_sequence(items, method="positional")
        
        # Should be normalized
        assert np.abs(np.linalg.norm(sequence) - 1.0) < 1e-6
        
        # Decode each position
        for i, item in enumerate(items):
            decoded = encoder.decode_position(sequence, i, method="positional")
            similarity = hrr.similarity(decoded, item)
            assert similarity > 0.5  # Some interference expected
    
    def test_encode_sequence_chaining(self):
        """Test chaining sequence encoding."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = SequenceEncoder(hrr)
        
        # Create sequence items
        items = [hrr.generate_vector() for _ in range(4)]
        
        # Encode sequence
        sequence = encoder.encode_sequence(items, method="chaining")
        
        # Should be normalized
        assert np.abs(np.linalg.norm(sequence) - 1.0) < 1e-6
        
        # Check that permutation vector was created
        assert hrr.get_item("sequence_permutation") is not None
    
    def test_encode_empty_sequence(self):
        """Test encoding empty sequence."""
        hrr = create_hrr(dimension=512)
        encoder = SequenceEncoder(hrr)
        
        with pytest.raises(ValueError, match="Cannot encode empty"):
            encoder.encode_sequence([])
    
    def test_invalid_encoding_method(self):
        """Test invalid encoding method."""
        hrr = create_hrr(dimension=512)
        encoder = SequenceEncoder(hrr)
        
        items = [hrr.generate_vector()]
        
        with pytest.raises(ValueError, match="Unknown method"):
            encoder.encode_sequence(items, method="invalid")
    
    def test_decode_position_chaining(self):
        """Test decoding positions with chaining method."""
        hrr = create_hrr(dimension=2048, seed=42)
        encoder = SequenceEncoder(hrr)
        
        # Create sequence
        items = [hrr.generate_vector() for _ in range(3)]
        sequence = encoder.encode_sequence(items, method="chaining")
        
        # Decode each position
        for i, item in enumerate(items):
            decoded = encoder.decode_position(sequence, i, method="chaining")
            
            # First position should have highest similarity
            # Later positions accumulate more noise
            if i == 0:
                similarity = hrr.similarity(decoded, item)
                assert similarity > 0.3  # Contains noise from other positions
            else:
                # Just check it runs without error
                assert decoded.shape == item.shape
    
    def test_decode_all_positions(self):
        """Test decoding all positions in a sequence."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = SequenceEncoder(hrr)
        
        # Create sequence
        items = [hrr.generate_vector() for _ in range(3)]
        sequence = encoder.encode_sequence(items, method="positional")
        
        # Decode all
        decoded = encoder.decode_all_positions(sequence, len(items), 
                                             method="positional")
        
        assert len(decoded) == len(items)
        
        # Check similarities
        for i, (orig, dec) in enumerate(zip(items, decoded)):
            similarity = hrr.similarity(orig, dec)
            assert similarity > 0.5
    
    def test_sequence_with_cleanup(self):
        """Test sequence encoding with cleanup memory."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = SequenceEncoder(hrr)
        
        # Create cleanup memory
        cleanup = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension=1024
        )
        
        # Create and store items
        words = {}
        for word in ["the", "cat", "sat"]:
            vec = hrr.generate_vector()
            words[word] = vec
            cleanup.add_item(word, vec)
        
        # Encode sequence
        sequence = encoder.encode_sequence(
            [words["the"], words["cat"], words["sat"]],
            method="positional"
        )
        
        # Decode and cleanup
        decoded_positions = encoder.decode_all_positions(sequence, 3, 
                                                       method="positional")
        
        cleaned_words = []
        for dec in decoded_positions:
            name, _, _ = cleanup.cleanup(dec)
            cleaned_words.append(name)
        
        assert cleaned_words == ["the", "cat", "sat"]
    
    def test_pre_computed_positions(self):
        """Test using pre-computed position vectors."""
        hrr = create_hrr(dimension=512, seed=42)
        
        # Pre-compute position vectors
        position_vectors = [hrr.generate_vector() for _ in range(5)]
        
        encoder = SequenceEncoder(hrr, position_vectors)
        
        # Should use pre-computed vectors
        for i in range(5):
            pos = encoder.get_position_vector(i)
            assert np.allclose(pos, position_vectors[i])


class TestHierarchicalEncoder:
    """Test hierarchical structure encoding."""
    
    def test_encode_flat_tree(self):
        """Test encoding a flat tree structure."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = HierarchicalEncoder(hrr)
        
        # Create flat structure
        tree = {
            "name": hrr.generate_vector(),
            "age": hrr.generate_vector(),
            "city": hrr.generate_vector()
        }
        
        encoded = encoder.encode_tree(tree)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(encoded) - 1.0) < 1e-6
        
        # Should be able to decode each field
        for key in tree:
            role = hrr.get_item(f"role:{key}")
            decoded = hrr.unbind(encoded, role)
            similarity = hrr.similarity(decoded, tree[key])
            assert similarity > 0.5
    
    def test_encode_nested_tree(self):
        """Test encoding a nested tree structure."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = HierarchicalEncoder(hrr)
        
        # Create nested structure
        red = hrr.generate_vector()
        blue = hrr.generate_vector()
        circle = hrr.generate_vector()
        square = hrr.generate_vector()
        
        tree = {
            "object1": {
                "color": red,
                "shape": circle
            },
            "object2": {
                "color": blue,
                "shape": square
            }
        }
        
        encoded = encoder.encode_tree(tree)
        
        # Decode nested values
        obj1_role = hrr.get_item("role:object1")
        obj1_encoded = hrr.unbind(encoded, obj1_role)
        
        color_role = hrr.get_item("role:color")
        obj1_color = hrr.unbind(obj1_encoded, color_role)
        
        # Should be similar to red
        similarity = hrr.similarity(obj1_color, red)
        assert similarity > 0.3  # Some noise from nested encoding
    
    def test_encode_with_references(self):
        """Test encoding with string references to stored items."""
        hrr = create_hrr(dimension=512, seed=42)
        encoder = HierarchicalEncoder(hrr)
        
        # Store some items
        hrr.add_item("red", hrr.generate_vector())
        hrr.add_item("circle", hrr.generate_vector())
        
        # Create tree with references
        tree = {
            "color": "red",  # String reference
            "shape": "circle",  # String reference
            "size": hrr.generate_vector()  # Direct vector
        }
        
        encoded = encoder.encode_tree(tree)
        
        # Should work correctly
        assert encoded.shape == (512,)
    
    def test_encode_invalid_reference(self):
        """Test encoding with invalid string reference."""
        hrr = create_hrr(dimension=512)
        encoder = HierarchicalEncoder(hrr)
        
        tree = {
            "color": "nonexistent"  # Not in memory
        }
        
        with pytest.raises(ValueError, match="Item 'nonexistent' not found"):
            encoder.encode_tree(tree)
    
    def test_encode_invalid_type(self):
        """Test encoding with invalid value type."""
        hrr = create_hrr(dimension=512)
        encoder = HierarchicalEncoder(hrr)
        
        tree = {
            "value": 123  # Invalid type
        }
        
        with pytest.raises(TypeError, match="Unsupported value type"):
            encoder.encode_tree(tree)
    
    def test_decode_path_single(self):
        """Test decoding a single-level path."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = HierarchicalEncoder(hrr)
        
        # Create structure
        value = hrr.generate_vector()
        tree = {"field": value}
        encoded = encoder.encode_tree(tree)
        
        # Decode path
        decoded = encoder.decode_path(encoded, ["field"])
        
        similarity = hrr.similarity(decoded, value)
        assert similarity > 0.9
    
    def test_decode_path_nested(self):
        """Test decoding a nested path."""
        hrr = create_hrr(dimension=1024, seed=42)
        encoder = HierarchicalEncoder(hrr)
        
        # Create nested structure
        deep_value = hrr.generate_vector()
        tree = {
            "level1": {
                "level2": {
                    "level3": deep_value
                }
            }
        }
        
        encoded = encoder.encode_tree(tree)
        
        # Decode nested path
        decoded = encoder.decode_path(encoded, ["level1", "level2", "level3"])
        
        # Will have accumulated noise but should still be recognizable
        similarity = hrr.similarity(decoded, deep_value)
        assert similarity > 0.2
    
    def test_decode_empty_path(self):
        """Test decoding with empty path."""
        hrr = create_hrr(dimension=512)
        encoder = HierarchicalEncoder(hrr)
        
        tree = {"field": hrr.generate_vector()}
        encoded = encoder.encode_tree(tree)
        
        # Empty path should return the encoding itself
        decoded = encoder.decode_path(encoded, [])
        assert np.allclose(decoded, encoded)
    
    def test_decode_path_missing_role(self):
        """Test decoding with missing role vector."""
        hrr = create_hrr(dimension=512)
        encoder = HierarchicalEncoder(hrr)
        
        tree = {"field": hrr.generate_vector()}
        encoded = encoder.encode_tree(tree)
        
        # Clear memory to remove role
        hrr.clear()
        
        with pytest.raises(ValueError, match="Role vector for 'field' not found"):
            encoder.decode_path(encoded, ["field"])
    
    def test_flatten_tree(self):
        """Test flattening a hierarchical tree."""
        hrr = create_hrr(dimension=512)
        encoder = HierarchicalEncoder(hrr)
        
        # Create nested structure
        tree = {
            "person": {
                "name": "John",
                "address": {
                    "street": "Main St",
                    "city": "Boston"
                }
            },
            "age": 30
        }
        
        flat = encoder.flatten_tree(tree)
        
        expected = {
            "person.name": "John",
            "person.address.street": "Main St",
            "person.address.city": "Boston",
            "age": 30
        }
        
        assert flat == expected
    
    def test_flatten_empty_tree(self):
        """Test flattening empty tree."""
        hrr = create_hrr(dimension=512)
        encoder = HierarchicalEncoder(hrr)
        
        flat = encoder.flatten_tree({})
        assert flat == {}
    
    def test_complex_hierarchy(self):
        """Test a complex hierarchical structure."""
        hrr = create_hrr(dimension=2048, seed=42)
        encoder = HierarchicalEncoder(hrr)
        
        # Create a complex structure
        tree = {
            "company": {
                "name": hrr.generate_vector(),
                "departments": {
                    "engineering": {
                        "head": hrr.generate_vector(),
                        "size": hrr.generate_vector()
                    },
                    "sales": {
                        "head": hrr.generate_vector(),
                        "size": hrr.generate_vector()
                    }
                }
            },
            "location": hrr.generate_vector()
        }
        
        # Encode
        encoded = encoder.encode_tree(tree)
        
        # Test various paths
        paths_to_test = [
            ["company", "name"],
            ["company", "departments", "engineering", "head"],
            ["location"]
        ]
        
        for path in paths_to_test:
            decoded = encoder.decode_path(encoded, path)
            assert decoded.shape == (2048,)
            
            # Get original value
            original = tree
            for key in path:
                original = original[key]
            
            # Should have some similarity (decreases with depth)
            if len(path) <= 2:
                similarity = hrr.similarity(decoded, original)
                assert similarity > 0.3