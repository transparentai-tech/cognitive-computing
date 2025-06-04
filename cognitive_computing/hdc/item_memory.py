"""
Associative item memory for HDC.

This module implements associative memory storage for hypervectors,
allowing efficient storage and retrieval of labeled items.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json
import pickle
from pathlib import Path
import logging

from cognitive_computing.hdc.operations import similarity

logger = logging.getLogger(__name__)


class ItemMemory:
    """
    Associative memory for storing and retrieving hypervectors.
    
    This class implements a content-addressable memory that stores
    hypervectors with associated labels and supports similarity-based
    retrieval.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of hypervectors
    similarity_metric : str
        Metric for similarity computation: "cosine", "hamming", "euclidean"
    max_items : int, optional
        Maximum number of items to store
    
    Attributes
    ----------
    dimension : int
        Hypervector dimensionality
    similarity_metric : str
        Similarity metric used
    memory : Dict[str, np.ndarray]
        Label to hypervector mapping
    """
    
    def __init__(
        self,
        dimension: int,
        similarity_metric: str = "cosine",
        max_items: Optional[int] = None
    ):
        """Initialize item memory."""
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        self.max_items = max_items
        self.memory: Dict[str, np.ndarray] = {}
        self._access_counts: Dict[str, int] = {}
        self._access_order: List[str] = []
        
    def add(self, label: str, hypervector: np.ndarray) -> None:
        """
        Add a labeled hypervector to memory.
        
        Parameters
        ----------
        label : str
            Label for the hypervector
        hypervector : np.ndarray
            Hypervector to store
            
        Raises
        ------
        ValueError
            If dimension mismatch or memory full
        """
        if hypervector.shape != (self.dimension,):
            raise ValueError(
                f"Hypervector dimension {hypervector.shape} doesn't match "
                f"memory dimension ({self.dimension},)"
            )
            
        # Check capacity
        if self.max_items is not None and len(self.memory) >= self.max_items:
            if label not in self.memory:
                # Memory full, need to evict
                self._evict_item()
                
        # Store item
        self.memory[label] = hypervector.copy()
        self._access_counts[label] = 0
        
        # Update access order
        if label in self._access_order:
            self._access_order.remove(label)
        self._access_order.append(label)
        
        logger.debug(f"Added item '{label}' to memory")
        
    def get(self, label: str) -> Optional[np.ndarray]:
        """
        Retrieve a hypervector by exact label.
        
        Parameters
        ----------
        label : str
            Label to look up
            
        Returns
        -------
        Optional[np.ndarray]
            Hypervector if found, None otherwise
        """
        if label in self.memory:
            self._access_counts[label] += 1
            # Move to end of access order (most recently used)
            self._access_order.remove(label)
            self._access_order.append(label)
            return self.memory[label].copy()
        return None
        
    def query(
        self,
        hypervector: np.ndarray,
        top_k: int = 1,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find most similar items in memory.
        
        Parameters
        ----------
        hypervector : np.ndarray
            Query hypervector
        top_k : int
            Number of top matches to return
        threshold : float, optional
            Minimum similarity threshold
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (label, similarity) tuples, sorted by similarity
        """
        if hypervector.shape != (self.dimension,):
            raise ValueError("Query dimension mismatch")
            
        if not self.memory:
            return []
            
        # Calculate similarities
        similarities = []
        for label, stored in self.memory.items():
            sim = similarity(hypervector, stored, metric=self.similarity_metric)
            if threshold is None or sim >= threshold:
                similarities.append((label, sim))
                
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]
        
    def cleanup(self, hypervector: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Return the closest clean hypervector from memory.
        
        Parameters
        ----------
        hypervector : np.ndarray
            Noisy hypervector
            
        Returns
        -------
        cleaned : Optional[np.ndarray]
            Closest clean hypervector
        label : Optional[str]
            Label of the closest match
        """
        results = self.query(hypervector, top_k=1)
        
        if results:
            label, _ = results[0]
            return self.memory[label].copy(), label
            
        return None, None
        
    def remove(self, label: str) -> bool:
        """
        Remove an item from memory.
        
        Parameters
        ----------
        label : str
            Label to remove
            
        Returns
        -------
        bool
            True if removed, False if not found
        """
        if label in self.memory:
            del self.memory[label]
            del self._access_counts[label]
            self._access_order.remove(label)
            logger.debug(f"Removed item '{label}' from memory")
            return True
        return False
        
    def clear(self) -> None:
        """Clear all items from memory."""
        self.memory.clear()
        self._access_counts.clear()
        self._access_order.clear()
        logger.debug("Cleared item memory")
        
    def update(self, label: str, hypervector: np.ndarray) -> None:
        """
        Update an existing item or add if not present.
        
        Parameters
        ----------
        label : str
            Label to update
        hypervector : np.ndarray
            New hypervector
        """
        self.add(label, hypervector)
        
    def merge(self, label: str, hypervector: np.ndarray, weight: float = 0.5) -> None:
        """
        Merge a hypervector with an existing one.
        
        Parameters
        ----------
        label : str
            Label of item to merge with
        hypervector : np.ndarray
            Hypervector to merge
        weight : float
            Weight for new hypervector (0 to 1)
        """
        if label not in self.memory:
            self.add(label, hypervector)
            return
            
        # Weighted average
        existing = self.memory[label]
        merged = (1 - weight) * existing + weight * hypervector
        
        # Normalize based on type (assuming bipolar)
        merged = np.sign(merged)
        merged[merged == 0] = 1
        
        self.memory[label] = merged.astype(existing.dtype)
        
    def _evict_item(self) -> None:
        """Evict least recently used item."""
        if self._access_order:
            # Remove least recently used (first in order)
            lru_label = self._access_order[0]
            self.remove(lru_label)
            logger.debug(f"Evicted LRU item '{lru_label}'")
            
    @property
    def size(self) -> int:
        """Number of items in memory."""
        return len(self.memory)
        
    @property
    def labels(self) -> List[str]:
        """List of all labels in memory."""
        return list(self.memory.keys())
        
    def save(self, path: Union[str, Path], format: str = "pickle") -> None:
        """
        Save item memory to file.
        
        Parameters
        ----------
        path : str or Path
            File path to save to
        format : str
            Save format: "pickle" or "json"
        """
        path = Path(path)
        
        data = {
            "dimension": self.dimension,
            "similarity_metric": self.similarity_metric,
            "max_items": self.max_items,
            "memory": {k: v.tolist() for k, v in self.memory.items()},
            "access_counts": self._access_counts,
            "access_order": self._access_order
        }
        
        if format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        elif format == "json":
            # Convert numpy arrays to lists for JSON
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        logger.info(f"Saved item memory to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path], format: str = "pickle") -> "ItemMemory":
        """
        Load item memory from file.
        
        Parameters
        ----------
        path : str or Path
            File path to load from
        format : str
            Load format: "pickle" or "json"
            
        Returns
        -------
        ItemMemory
            Loaded item memory
        """
        path = Path(path)
        
        if format == "pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif format == "json":
            with open(path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        # Create new instance
        memory = cls(
            dimension=data["dimension"],
            similarity_metric=data["similarity_metric"],
            max_items=data["max_items"]
        )
        
        # Restore state
        memory._access_counts = data["access_counts"]
        memory._access_order = data["access_order"]
        
        # Restore memory items
        for label, vector_data in data["memory"].items():
            vector = np.array(vector_data)
            memory.memory[label] = vector
            
        logger.info(f"Loaded item memory from {path}")
        return memory
        
    def statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get memory statistics.
        
        Returns
        -------
        Dict[str, Union[int, float]]
            Statistics about memory usage
        """
        if not self.memory:
            return {
                "size": 0,
                "capacity": self.max_items,
                "utilization": 0.0,
                "avg_access_count": 0.0
            }
            
        access_counts = list(self._access_counts.values())
        
        stats = {
            "size": self.size,
            "capacity": self.max_items,
            "utilization": self.size / self.max_items if self.max_items else 0.0,
            "avg_access_count": np.mean(access_counts),
            "max_access_count": np.max(access_counts),
            "min_access_count": np.min(access_counts),
        }
        
        return stats
        
    def __len__(self) -> int:
        """Number of items in memory."""
        return self.size
        
    def __contains__(self, label: str) -> bool:
        """Check if label exists in memory."""
        return label in self.memory
        
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ItemMemory(dimension={self.dimension}, "
            f"size={self.size}, "
            f"max_items={self.max_items})"
        )