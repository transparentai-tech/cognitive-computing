"""
Cleanup memory for HRR.

This module provides cleanup memory functionality for retrieving clean
items from noisy HRR vectors.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist

from .operations import VectorOperations

logger = logging.getLogger(__name__)


@dataclass
class CleanupMemoryConfig:
    """
    Configuration for cleanup memory.
    
    Parameters
    ----------
    threshold : float
        Similarity threshold for item retrieval (0-1)
    method : str
        Similarity method: "cosine", "dot", "euclidean"
    top_k : int
        Default number of top matches to return
    """
    threshold: float = 0.3
    method: str = "cosine"
    top_k: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        
        if self.method not in ["cosine", "dot", "euclidean"]:
            raise ValueError(f"method must be 'cosine', 'dot', or 'euclidean', "
                           f"got {self.method}")
        
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


class CleanupMemory:
    """
    Item memory for cleaning up noisy HRR vectors.
    
    The cleanup memory stores a collection of clean item vectors and can
    retrieve the closest matches to a noisy query vector. This is essential
    for HRR systems to map noisy results back to known items.
    
    Parameters
    ----------
    config : CleanupMemoryConfig
        Configuration for the cleanup memory
    dimension : int
        Dimensionality of vectors
        
    Attributes
    ----------
    items : Dict[str, np.ndarray]
        Dictionary mapping item names to their vectors
    _item_matrix : Optional[np.ndarray]
        Matrix of all item vectors for efficient similarity computation
    _item_names : List[str]
        Ordered list of item names corresponding to matrix rows
    """
    
    def __init__(self, config: CleanupMemoryConfig, dimension: int):
        """Initialize cleanup memory."""
        self.config = config
        self.dimension = dimension
        self.items: Dict[str, np.ndarray] = {}
        self._item_matrix: Optional[np.ndarray] = None
        self._item_names: List[str] = []
        self._needs_rebuild = False
    
    def add_item(self, name: str, vector: np.ndarray) -> None:
        """
        Add an item to the cleanup memory.
        
        Parameters
        ----------
        name : str
            Name/label for the item
        vector : np.ndarray
            Vector representation of the item
        """
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector must have dimension {self.dimension}, "
                           f"got {vector.shape}")
        
        # Normalize vector for consistent similarity computation
        if self.config.method == "cosine":
            vector = VectorOperations.normalize(vector)
        
        self.items[name] = vector.copy()
        self._needs_rebuild = True
        
        logger.debug(f"Added item '{name}' to cleanup memory")
    
    def remove_item(self, name: str) -> bool:
        """
        Remove an item from the cleanup memory.
        
        Parameters
        ----------
        name : str
            Name of the item to remove
            
        Returns
        -------
        bool
            True if item was removed, False if not found
        """
        if name in self.items:
            del self.items[name]
            self._needs_rebuild = True
            logger.debug(f"Removed item '{name}' from cleanup memory")
            return True
        return False
    
    def cleanup(self, vector: np.ndarray, 
                return_similarity: bool = True) -> Union[Tuple[str, np.ndarray, float], 
                                                         Tuple[str, np.ndarray]]:
        """
        Clean up a noisy vector by finding the closest item.
        
        Parameters
        ----------
        vector : np.ndarray
            Noisy vector to clean up
        return_similarity : bool
            Whether to return similarity score
            
        Returns
        -------
        name : str
            Name of the closest item
        clean_vector : np.ndarray
            Clean vector of the closest item
        similarity : float (optional)
            Similarity score if return_similarity=True
            
        Raises
        ------
        ValueError
            If no items in memory or no item above threshold
        """
        if not self.items:
            raise ValueError("No items in cleanup memory")
        
        # Find best match
        matches = self.find_closest(vector, k=1)
        
        if not matches:
            raise ValueError(f"No item found above threshold {self.config.threshold}")
        
        name, similarity = matches[0]
        clean_vector = self.items[name]
        
        if return_similarity:
            return name, clean_vector, similarity
        return name, clean_vector
    
    def find_closest(self, vector: np.ndarray, 
                     k: Optional[int] = None,
                     threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find the k closest items to a query vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Query vector
        k : int, optional
            Number of matches to return (default: config.top_k)
        threshold : float, optional
            Similarity threshold (default: config.threshold)
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (item_name, similarity) tuples, sorted by similarity
        """
        if not self.items:
            return []
        
        if k is None:
            k = self.config.top_k
        if threshold is None:
            threshold = self.config.threshold
        
        # Rebuild matrix if needed
        self._rebuild_matrix()
        
        # Compute similarities
        similarities = self._compute_similarities(vector)
        
        # Get indices sorted by similarity (descending)
        if self.config.method == "euclidean":
            # For Euclidean, lower is better (negative distance)
            sorted_indices = np.argsort(-similarities)
        else:
            sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter by threshold and take top k
        results = []
        for idx in sorted_indices[:k]:
            sim = similarities[idx]
            if self._above_threshold(sim):
                results.append((self._item_names[idx], float(sim)))
        
        return results
    
    def find_all_above_threshold(self, vector: np.ndarray,
                                threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find all items above a similarity threshold.
        
        Parameters
        ----------
        vector : np.ndarray
            Query vector
        threshold : float, optional
            Similarity threshold (default: config.threshold)
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (item_name, similarity) tuples above threshold
        """
        if not self.items:
            return []
        
        if threshold is None:
            threshold = self.config.threshold
        
        # Get all matches
        all_matches = self.find_closest(vector, k=len(self.items), 
                                      threshold=threshold)
        
        # Filter by threshold
        return [(name, sim) for name, sim in all_matches 
                if self._above_threshold(sim, threshold)]
    
    def get_item(self, name: str) -> Optional[np.ndarray]:
        """
        Get an item vector by name.
        
        Parameters
        ----------
        name : str
            Name of the item
            
        Returns
        -------
        Optional[np.ndarray]
            Item vector or None if not found
        """
        return self.items.get(name)
    
    def clear(self) -> None:
        """Clear all items from memory."""
        self.items.clear()
        self._item_matrix = None
        self._item_names = []
        self._needs_rebuild = False
    
    @property
    def size(self) -> int:
        """Return number of items in memory."""
        return len(self.items)
    
    def _rebuild_matrix(self) -> None:
        """Rebuild the item matrix for efficient similarity computation."""
        if not self._needs_rebuild and self._item_matrix is not None:
            return
        
        if not self.items:
            self._item_matrix = None
            self._item_names = []
            return
        
        # Build matrix and name list
        self._item_names = list(self.items.keys())
        vectors = [self.items[name] for name in self._item_names]
        self._item_matrix = np.vstack(vectors)
        self._needs_rebuild = False
    
    def _compute_similarities(self, vector: np.ndarray) -> np.ndarray:
        """Compute similarities between query and all items."""
        if self._item_matrix is None:
            return np.array([])
        
        # Normalize query if needed
        if self.config.method == "cosine":
            vector = VectorOperations.normalize(vector)
        
        # Compute similarities based on method
        if self.config.method == "cosine":
            # Cosine similarity
            dots = np.dot(self._item_matrix, vector)
            return dots  # Already normalized
            
        elif self.config.method == "dot":
            # Dot product
            return np.dot(self._item_matrix, vector)
            
        elif self.config.method == "euclidean":
            # Negative Euclidean distance
            dists = np.linalg.norm(self._item_matrix - vector, axis=1)
            return -dists
    
    def _above_threshold(self, similarity: float, 
                        threshold: Optional[float] = None) -> bool:
        """Check if similarity is above threshold."""
        if threshold is None:
            threshold = self.config.threshold
            
        if self.config.method == "euclidean":
            # For Euclidean, we use negative distance, so check if close enough
            # Convert threshold to distance (assuming threshold is for cosine similarity)
            # For unit vectors: ||a-b||² = 2(1 - cos(a,b))
            max_dist = np.sqrt(2 * (1 - threshold))
            return -similarity < max_dist
        else:
            return similarity >= threshold
    
    def statistics(self) -> Dict[str, float]:
        """
        Compute statistics about the cleanup memory.
        
        Returns
        -------
        Dict[str, float]
            Statistics including:
            - num_items: Number of items
            - avg_similarity: Average pairwise similarity
            - min_similarity: Minimum pairwise similarity
            - max_similarity: Maximum pairwise similarity
        """
        if len(self.items) < 2:
            return {
                "num_items": len(self.items),
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
            }
        
        self._rebuild_matrix()
        
        # Compute pairwise similarities
        n = len(self._item_names)
        similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = VectorOperations.similarity(
                    self._item_matrix[i], 
                    self._item_matrix[j],
                    metric=self.config.method if self.config.method != "euclidean" else "cosine"
                )
                similarities.append(sim)
        
        similarities = np.array(similarities)
        
        return {
            "num_items": n,
            "avg_similarity": float(np.mean(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }
    
    def save(self, filename: str) -> None:
        """
        Save cleanup memory to file.
        
        Parameters
        ----------
        filename : str
            Path to save file
        """
        np.savez_compressed(
            filename,
            config_threshold=self.config.threshold,
            config_method=self.config.method,
            config_top_k=self.config.top_k,
            dimension=self.dimension,
            item_names=list(self.items.keys()),
            **self.items
        )
        logger.info(f"Saved cleanup memory with {len(self.items)} items to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> "CleanupMemory":
        """
        Load cleanup memory from file.
        
        Parameters
        ----------
        filename : str
            Path to load file
            
        Returns
        -------
        CleanupMemory
            Loaded cleanup memory
        """
        data = np.load(filename)
        
        # Reconstruct config
        config = CleanupMemoryConfig(
            threshold=float(data["config_threshold"]),
            method=str(data["config_method"]),
            top_k=int(data["config_top_k"])
        )
        
        # Create cleanup memory
        memory = cls(config, int(data["dimension"]))
        
        # Load items
        for name in data["item_names"]:
            memory.add_item(name, data[name])
        
        logger.info(f"Loaded cleanup memory with {memory.size} items from {filename}")
        return memory