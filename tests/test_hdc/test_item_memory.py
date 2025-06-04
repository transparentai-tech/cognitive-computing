"""Tests for HDC item memory."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from cognitive_computing.hdc.item_memory import ItemMemory


class TestItemMemory:
    """Test item memory functionality."""
    
    def test_initialization(self):
        """Test item memory initialization."""
        memory = ItemMemory(dimension=100, similarity_metric="cosine")
        
        assert memory.dimension == 100
        assert memory.similarity_metric == "cosine"
        assert memory.max_items is None
        assert memory.size == 0
        assert len(memory) == 0
        
    def test_add_and_get(self):
        """Test adding and retrieving items."""
        memory = ItemMemory(dimension=10)
        
        # Add item
        hv = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
        memory.add("test", hv)
        
        assert memory.size == 1
        assert "test" in memory
        
        # Get item
        retrieved = memory.get("test")
        assert np.array_equal(retrieved, hv)
        
        # Get non-existent item
        assert memory.get("nonexistent") is None
        
    def test_add_dimension_mismatch(self):
        """Test adding with dimension mismatch."""
        memory = ItemMemory(dimension=10)
        
        hv = np.ones(5)  # Wrong dimension
        
        with pytest.raises(ValueError, match="dimension.*doesn't match"):
            memory.add("test", hv)
            
    def test_query_similarity(self):
        """Test similarity-based querying."""
        memory = ItemMemory(dimension=100, similarity_metric="cosine")
        
        # Add some items
        np.random.seed(42)
        for i in range(5):
            hv = 2 * np.random.randint(0, 2, size=100) - 1
            memory.add(f"item_{i}", hv)
            
        # Query with exact match
        query = memory.get("item_0")
        results = memory.query(query, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == "item_0"
        assert results[0][1] == 1.0  # Perfect similarity
        
        # Query with top 3
        results = memory.query(query, top_k=3)
        assert len(results) == 3
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1] >= results[2][1]
        
    def test_query_threshold(self):
        """Test querying with similarity threshold."""
        memory = ItemMemory(dimension=100, similarity_metric="cosine")
        
        # Add items
        hv1 = np.ones(100, dtype=np.int8)
        hv2 = -np.ones(100, dtype=np.int8)  # Opposite
        hv3 = np.ones(100, dtype=np.int8)
        hv3[:50] = -1  # Half similar
        
        memory.add("ones", hv1)
        memory.add("neg_ones", hv2)
        memory.add("half", hv3)
        
        # Query with threshold
        results = memory.query(hv1, top_k=10, threshold=0.5)
        
        # Should only get ones (sim=1.0) and maybe half (sim=0.0)
        assert len(results) <= 2
        assert all(sim >= 0.5 for _, sim in results)
        
    def test_cleanup(self):
        """Test cleanup operation."""
        memory = ItemMemory(dimension=100)
        
        # Add clean hypervectors
        np.random.seed(42)
        hv1 = 2 * np.random.randint(0, 2, size=100) - 1
        hv2 = 2 * np.random.randint(0, 2, size=100) - 1
        
        memory.add("clean1", hv1)
        memory.add("clean2", hv2)
        
        # Create noisy version
        noisy = hv1.copy()
        flip_indices = np.random.choice(100, 10, replace=False)
        noisy[flip_indices] = -noisy[flip_indices]
        
        # Cleanup should return closest clean vector
        cleaned, label = memory.cleanup(noisy)
        
        assert label == "clean1"
        assert np.array_equal(cleaned, hv1)
        
    def test_remove(self):
        """Test removing items."""
        memory = ItemMemory(dimension=10)
        
        hv = np.ones(10)
        memory.add("test", hv)
        
        assert memory.size == 1
        
        # Remove existing
        assert memory.remove("test") is True
        assert memory.size == 0
        assert "test" not in memory
        
        # Remove non-existent
        assert memory.remove("test") is False
        
    def test_clear(self):
        """Test clearing memory."""
        memory = ItemMemory(dimension=10)
        
        # Add multiple items
        for i in range(5):
            memory.add(f"item_{i}", np.ones(10))
            
        assert memory.size == 5
        
        # Clear
        memory.clear()
        assert memory.size == 0
        assert len(memory.labels) == 0
        
    def test_update(self):
        """Test updating items."""
        memory = ItemMemory(dimension=10)
        
        # Add initial
        hv1 = np.ones(10, dtype=np.int8)
        memory.add("test", hv1)
        
        # Update
        hv2 = -np.ones(10, dtype=np.int8)
        memory.update("test", hv2)
        
        # Should have new value
        retrieved = memory.get("test")
        assert np.array_equal(retrieved, hv2)
        
    def test_merge(self):
        """Test merging hypervectors."""
        memory = ItemMemory(dimension=4)
        
        # Add initial
        hv1 = np.array([1, 1, -1, -1], dtype=np.int8)
        memory.add("test", hv1)
        
        # Merge with another
        hv2 = np.array([-1, 1, 1, -1], dtype=np.int8)
        memory.merge("test", hv2, weight=0.5)
        
        # Result should be normalized average
        retrieved = memory.get("test")
        # Average: [0, 1, 0, -1] -> normalized: [1, 1, 1, -1]
        expected = np.array([1, 1, 1, -1], dtype=np.int8)
        assert np.array_equal(retrieved, expected)
        
    def test_merge_new_item(self):
        """Test merging with non-existent item."""
        memory = ItemMemory(dimension=10)
        
        hv = np.ones(10)
        memory.merge("new", hv, weight=0.7)
        
        # Should just add the item
        assert "new" in memory
        assert np.array_equal(memory.get("new"), hv)
        
    def test_max_items(self):
        """Test maximum items constraint."""
        memory = ItemMemory(dimension=10, max_items=3)
        
        # Add items up to limit
        for i in range(3):
            memory.add(f"item_{i}", np.ones(10))
            
        assert memory.size == 3
        
        # Add one more - should evict LRU
        memory.add("item_3", np.ones(10))
        
        assert memory.size == 3
        assert "item_0" not in memory  # LRU evicted
        assert "item_3" in memory
        
    def test_access_tracking(self):
        """Test access count and order tracking."""
        memory = ItemMemory(dimension=10, max_items=3)
        
        # Add items
        for i in range(3):
            memory.add(f"item_{i}", np.ones(10))
            
        # Access middle item
        memory.get("item_1")
        
        # Add new item - should evict item_0 (LRU)
        memory.add("item_3", np.ones(10))
        
        assert "item_0" not in memory
        assert "item_1" in memory  # Was accessed
        
    def test_save_load_pickle(self):
        """Test saving and loading with pickle."""
        memory = ItemMemory(dimension=10, similarity_metric="hamming", max_items=5)
        
        # Add items
        hv1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        hv2 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        
        memory.add("binary1", hv1)
        memory.add("binary2", hv2)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
            
        memory.save(temp_path, format="pickle")
        
        # Load
        loaded = ItemMemory.load(temp_path, format="pickle")
        
        # Verify
        assert loaded.dimension == 10
        assert loaded.similarity_metric == "hamming"
        assert loaded.max_items == 5
        assert loaded.size == 2
        
        assert np.array_equal(loaded.get("binary1"), hv1)
        assert np.array_equal(loaded.get("binary2"), hv2)
        
        # Cleanup
        temp_path.unlink()
        
    def test_save_load_json(self):
        """Test saving and loading with JSON."""
        memory = ItemMemory(dimension=5)
        
        # Add items
        hv = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        memory.add("test", hv)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
            
        memory.save(temp_path, format="json")
        
        # Load
        loaded = ItemMemory.load(temp_path, format="json")
        
        # Verify
        assert loaded.dimension == 5
        assert loaded.size == 1
        assert np.array_equal(loaded.get("test"), hv)
        
        # Cleanup
        temp_path.unlink()
        
    def test_statistics(self):
        """Test memory statistics."""
        memory = ItemMemory(dimension=10, max_items=5)
        
        # Empty stats
        stats = memory.statistics()
        assert stats["size"] == 0
        assert stats["capacity"] == 5
        assert stats["utilization"] == 0.0
        assert stats["avg_access_count"] == 0.0
        
        # Add items and access
        memory.add("item1", np.ones(10))
        memory.add("item2", np.ones(10))
        
        memory.get("item1")
        memory.get("item1")
        memory.get("item2")
        
        stats = memory.statistics()
        assert stats["size"] == 2
        assert stats["utilization"] == 0.4  # 2/5
        assert stats["avg_access_count"] == 1.5  # (2+1)/2
        assert stats["max_access_count"] == 2
        assert stats["min_access_count"] == 1
        
    def test_labels_property(self):
        """Test labels property."""
        memory = ItemMemory(dimension=10)
        
        memory.add("a", np.ones(10))
        memory.add("b", np.ones(10))
        memory.add("c", np.ones(10))
        
        labels = memory.labels
        assert len(labels) == 3
        assert set(labels) == {"a", "b", "c"}
        
    def test_repr(self):
        """Test string representation."""
        memory = ItemMemory(dimension=100, max_items=50)
        memory.add("test", np.ones(100))
        
        repr_str = repr(memory)
        assert "ItemMemory" in repr_str
        assert "dimension=100" in repr_str
        assert "size=1" in repr_str
        assert "max_items=50" in repr_str
        
    def test_query_empty_memory(self):
        """Test querying empty memory."""
        memory = ItemMemory(dimension=10)
        
        query = np.ones(10)
        results = memory.query(query)
        
        assert results == []
        
    def test_cleanup_empty_memory(self):
        """Test cleanup with empty memory."""
        memory = ItemMemory(dimension=10)
        
        query = np.ones(10)
        cleaned, label = memory.cleanup(query)
        
        assert cleaned is None
        assert label is None