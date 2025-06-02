"""
Tests for HRR cleanup memory.

Tests the cleanup memory functionality for retrieving clean items
from noisy vectors.
"""

import pytest
import numpy as np
import tempfile
import os

from cognitive_computing.hrr.cleanup import CleanupMemory, CleanupMemoryConfig
from cognitive_computing.hrr import create_hrr


class TestCleanupMemoryConfig:
    """Test cleanup memory configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CleanupMemoryConfig()
        assert config.threshold == 0.3
        assert config.method == "cosine"
        assert config.top_k == 1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CleanupMemoryConfig(
            threshold=0.5,
            method="dot",
            top_k=5
        )
        assert config.threshold == 0.5
        assert config.method == "dot"
        assert config.top_k == 5
    
    def test_invalid_threshold(self):
        """Test threshold validation."""
        with pytest.raises(ValueError, match="threshold must be in"):
            CleanupMemoryConfig(threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be in"):
            CleanupMemoryConfig(threshold=1.5)
    
    def test_invalid_method(self):
        """Test method validation."""
        with pytest.raises(ValueError, match="method must be"):
            CleanupMemoryConfig(method="invalid")
    
    def test_invalid_top_k(self):
        """Test top_k validation."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            CleanupMemoryConfig(top_k=0)


class TestCleanupMemoryBasics:
    """Test basic cleanup memory operations."""
    
    def test_initialization(self):
        """Test cleanup memory initialization."""
        config = CleanupMemoryConfig()
        memory = CleanupMemory(config, dimension=512)
        
        assert memory.config == config
        assert memory.dimension == 512
        assert memory.size == 0
        assert len(memory.items) == 0
    
    def test_add_item(self):
        """Test adding items to memory."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        # Add items
        v1 = np.random.randn(100)
        v2 = np.random.randn(100)
        
        memory.add_item("item1", v1)
        memory.add_item("item2", v2)
        
        assert memory.size == 2
        assert "item1" in memory.items
        assert "item2" in memory.items
        
        # Items should be normalized for cosine similarity
        assert np.abs(np.linalg.norm(memory.items["item1"]) - 1.0) < 1e-6
    
    def test_add_item_wrong_dimension(self):
        """Test adding item with wrong dimension."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        with pytest.raises(ValueError, match="Vector must have dimension"):
            memory.add_item("bad", np.zeros(50))
    
    def test_remove_item(self):
        """Test removing items from memory."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        v = np.random.randn(100)
        memory.add_item("test", v)
        assert memory.size == 1
        
        # Remove existing item
        assert memory.remove_item("test") == True
        assert memory.size == 0
        assert "test" not in memory.items
        
        # Remove non-existent item
        assert memory.remove_item("nonexistent") == False
    
    def test_get_item(self):
        """Test getting items by name."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        v = np.random.randn(100)
        memory.add_item("test", v)
        
        # Get existing item
        retrieved = memory.get_item("test")
        assert retrieved is not None
        assert retrieved.shape == (100,)
        
        # Get non-existent item
        assert memory.get_item("nonexistent") is None
    
    def test_clear(self):
        """Test clearing memory."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        # Add multiple items
        for i in range(5):
            memory.add_item(f"item{i}", np.random.randn(100))
        
        assert memory.size == 5
        
        # Clear all
        memory.clear()
        
        assert memory.size == 0
        assert len(memory.items) == 0
        assert memory._item_matrix is None


class TestCleanupOperations:
    """Test cleanup and retrieval operations."""
    
    def test_cleanup_basic(self):
        """Test basic cleanup operation."""
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.5),
            dimension=1000
        )
        
        # Add items
        v1 = np.random.randn(1000)
        v1 = v1 / np.linalg.norm(v1)
        memory.add_item("item1", v1)
        
        v2 = np.random.randn(1000)
        v2 = v2 / np.linalg.norm(v2)
        memory.add_item("item2", v2)
        
        # Query with exact vector
        name, clean, similarity = memory.cleanup(v1)
        assert name == "item1"
        assert np.allclose(clean, memory.items["item1"])
        assert similarity > 0.99
    
    def test_cleanup_noisy(self):
        """Test cleanup with noisy vector."""
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.7),
            dimension=1000
        )
        
        # Add item
        v = np.random.randn(1000)
        v = v / np.linalg.norm(v)
        memory.add_item("test", v)
        
        # Add noise
        noise = np.random.randn(1000) * 0.3
        noisy = v + noise
        noisy = noisy / np.linalg.norm(noisy)
        
        # Should still cleanup to original
        name, clean, similarity = memory.cleanup(noisy)
        assert name == "test"
        assert similarity > 0.7
    
    def test_cleanup_empty_memory(self):
        """Test cleanup with empty memory."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        with pytest.raises(ValueError, match="No items in cleanup memory"):
            memory.cleanup(np.random.randn(100))
    
    def test_cleanup_below_threshold(self):
        """Test cleanup when no item is above threshold."""
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.99),  # Very high threshold
            dimension=1000
        )
        
        # Add random items
        for i in range(3):
            memory.add_item(f"item{i}", np.random.randn(1000))
        
        # Query with unrelated vector
        query = np.random.randn(1000)
        
        with pytest.raises(ValueError, match="No item found above threshold"):
            memory.cleanup(query)
    
    def test_cleanup_without_similarity(self):
        """Test cleanup without returning similarity."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        v = np.random.randn(100)
        memory.add_item("test", v)
        
        name, clean = memory.cleanup(v, return_similarity=False)
        assert name == "test"
        assert isinstance(clean, np.ndarray)


class TestFindClosest:
    """Test finding closest items."""
    
    def test_find_closest_single(self):
        """Test finding single closest item."""
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension=1000
        )
        
        # Add items with known similarities
        base = np.random.randn(1000)
        base = base / np.linalg.norm(base)
        memory.add_item("base", base)
        
        # Add similar item
        similar = 0.8 * base + 0.2 * np.random.randn(1000)
        similar = similar / np.linalg.norm(similar)
        memory.add_item("similar", similar)
        
        # Add dissimilar item
        dissimilar = np.random.randn(1000)
        dissimilar = dissimilar / np.linalg.norm(dissimilar)
        memory.add_item("dissimilar", dissimilar)
        
        # Find closest to base
        matches = memory.find_closest(base, k=1)
        assert len(matches) == 1
        assert matches[0][0] == "base"
        assert matches[0][1] > 0.99
    
    def test_find_closest_multiple(self):
        """Test finding multiple closest items."""
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.0),  # Low threshold
            dimension=1000
        )
        
        # Add several items
        for i in range(5):
            v = np.random.randn(1000)
            memory.add_item(f"item{i}", v)
        
        # Query
        query = np.random.randn(1000)
        matches = memory.find_closest(query, k=3)
        
        assert len(matches) == 3
        # Results should be sorted by similarity
        assert matches[0][1] >= matches[1][1]
        assert matches[1][1] >= matches[2][1]
    
    def test_find_closest_with_threshold(self):
        """Test finding closest with custom threshold."""
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension=100
        )
        
        # Add items
        for i in range(5):
            memory.add_item(f"item{i}", np.random.randn(100))
        
        # Find with high threshold
        query = np.random.randn(100)
        matches = memory.find_closest(query, k=5, threshold=0.9)
        
        # Probably no matches above 0.9 for random vectors
        assert len(matches) <= 1
    
    def test_find_all_above_threshold(self):
        """Test finding all items above threshold."""
        memory = CleanupMemory(
            CleanupMemoryConfig(),
            dimension=1000
        )
        
        # Add base vector
        base = np.random.randn(1000)
        base = base / np.linalg.norm(base)
        
        # Add items with varying similarities
        for i, alpha in enumerate([0.9, 0.7, 0.5, 0.3, 0.1]):
            v = alpha * base + (1 - alpha) * np.random.randn(1000)
            v = v / np.linalg.norm(v)
            memory.add_item(f"item{i}", v)
        
        # Find all above 0.5 similarity
        matches = memory.find_all_above_threshold(base, threshold=0.5)
        
        # Should get items with alpha >= 0.5
        assert len(matches) >= 2
        for name, sim in matches:
            assert sim >= 0.5


class TestDifferentMethods:
    """Test different similarity methods."""
    
    def test_dot_product_method(self):
        """Test dot product similarity method."""
        memory = CleanupMemory(
            CleanupMemoryConfig(method="dot"),
            dimension=100
        )
        
        # Add normalized and unnormalized vectors
        v1 = np.ones(100)  # Not normalized
        v2 = np.ones(100) / 10  # Scaled down
        
        memory.add_item("big", v1)
        memory.add_item("small", v2)
        
        # Query with ones - should match "big" due to larger dot product
        query = np.ones(100)
        name, _, _ = memory.cleanup(query)
        assert name == "big"
    
    def test_euclidean_method(self):
        """Test Euclidean distance method."""
        memory = CleanupMemory(
            CleanupMemoryConfig(method="euclidean", threshold=0.7),
            dimension=100
        )
        
        # Add items
        v1 = np.zeros(100)
        v1[0] = 1.0
        memory.add_item("v1", v1)
        
        v2 = np.zeros(100)
        v2[1] = 1.0
        memory.add_item("v2", v2)
        
        # Query closer to v1
        query = np.zeros(100)
        query[0] = 0.9
        
        matches = memory.find_closest(query, k=2)
        assert matches[0][0] == "v1"
        # Similarity is negative distance
        assert matches[0][1] < 0  
    
    def test_method_consistency(self):
        """Test that methods handle normalization consistently."""
        dimension = 1000
        
        # Create memories with different methods
        configs = [
            CleanupMemoryConfig(method="cosine"),
            CleanupMemoryConfig(method="dot"),
            CleanupMemoryConfig(method="euclidean", threshold=0.5)
        ]
        
        for config in configs:
            memory = CleanupMemory(config, dimension)
            
            # Add same items
            v1 = np.random.randn(dimension)
            v2 = np.random.randn(dimension)
            
            memory.add_item("item1", v1)
            memory.add_item("item2", v2)
            
            # Should be able to retrieve both
            assert memory.get_item("item1") is not None
            assert memory.get_item("item2") is not None


class TestStatistics:
    """Test memory statistics."""
    
    def test_statistics_empty(self):
        """Test statistics on empty memory."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        stats = memory.statistics()
        assert stats["num_items"] == 0
        assert stats["avg_similarity"] == 0.0
        assert stats["min_similarity"] == 0.0
        assert stats["max_similarity"] == 0.0
    
    def test_statistics_single_item(self):
        """Test statistics with single item."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        memory.add_item("test", np.random.randn(100))
        
        stats = memory.statistics()
        assert stats["num_items"] == 1
        assert stats["avg_similarity"] == 0.0  # No pairs to compare
    
    def test_statistics_multiple_items(self):
        """Test statistics with multiple items."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=1000)
        
        # Add orthogonal vectors
        for i in range(3):
            v = np.zeros(1000)
            v[i] = 1.0
            memory.add_item(f"item{i}", v)
        
        stats = memory.statistics()
        assert stats["num_items"] == 3
        assert abs(stats["avg_similarity"]) < 0.01  # Should be near 0
        assert abs(stats["min_similarity"]) < 0.01
        assert abs(stats["max_similarity"]) < 0.01


class TestSaveLoad:
    """Test saving and loading cleanup memory."""
    
    def test_save_load_basic(self):
        """Test basic save/load functionality."""
        # Create and populate memory
        config = CleanupMemoryConfig(threshold=0.4, method="dot", top_k=3)
        memory = CleanupMemory(config, dimension=100)
        
        # Add items
        for i in range(5):
            v = np.random.randn(100)
            memory.add_item(f"item{i}", v)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
        
        try:
            memory.save(temp_path)
            
            # Load into new memory
            loaded = CleanupMemory.load(temp_path)
            
            # Check configuration
            assert loaded.config.threshold == 0.4
            assert loaded.config.method == "dot"
            assert loaded.config.top_k == 3
            assert loaded.dimension == 100
            
            # Check items
            assert loaded.size == 5
            for i in range(5):
                assert f"item{i}" in loaded.items
                assert np.allclose(loaded.items[f"item{i}"], 
                                 memory.items[f"item{i}"])
        finally:
            os.unlink(temp_path)
    
    def test_save_load_empty(self):
        """Test saving/loading empty memory."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=256)
        
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
        
        try:
            memory.save(temp_path)
            loaded = CleanupMemory.load(temp_path)
            
            assert loaded.size == 0
            assert loaded.dimension == 256
        finally:
            os.unlink(temp_path)


class TestMatrixRebuilding:
    """Test internal matrix rebuilding for efficiency."""
    
    def test_matrix_rebuild_on_add(self):
        """Test that matrix is marked for rebuild on add."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        # Initially no matrix
        assert memory._item_matrix is None
        
        # Add item
        memory.add_item("test", np.random.randn(100))
        assert memory._needs_rebuild == True
        
        # Force rebuild by finding closest
        memory.find_closest(np.random.randn(100))
        assert memory._needs_rebuild == False
        assert memory._item_matrix is not None
        
        # Add another item
        memory.add_item("test2", np.random.randn(100))
        assert memory._needs_rebuild == True
    
    def test_matrix_rebuild_on_remove(self):
        """Test that matrix is marked for rebuild on remove."""
        memory = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        # Add items and build matrix
        memory.add_item("test1", np.random.randn(100))
        memory.add_item("test2", np.random.randn(100))
        memory.find_closest(np.random.randn(100))
        
        assert memory._needs_rebuild == False
        
        # Remove item
        memory.remove_item("test1")
        assert memory._needs_rebuild == True


class TestIntegrationWithHRR:
    """Test integration with HRR system."""
    
    def test_cleanup_with_hrr(self):
        """Test using cleanup memory with HRR vectors."""
        # Create HRR system
        hrr = create_hrr(dimension=1024)
        
        # Create cleanup memory
        memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.7),
            dimension=1024
        )
        
        # Create and store some items
        items = {}
        for name in ["red", "green", "blue"]:
            vec = hrr.generate_vector()
            items[name] = vec
            memory.add_item(name, vec)
        
        # Create composite vector
        role = hrr.generate_vector()
        composite = hrr.bind(role, items["red"])
        
        # Unbind to get noisy result
        retrieved = hrr.unbind(composite, role)
        
        # Clean up
        name, clean, similarity = memory.cleanup(retrieved)
        assert name == "red"
        assert similarity > 0.7
        
        # Verify it's close to original
        assert hrr.similarity(clean, items["red"]) > 0.99