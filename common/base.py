"""
Base classes and interfaces for the cognitive computing package.

This module provides abstract base classes and common interfaces that are used
across different cognitive computing implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Enumeration of supported distance metrics."""
    HAMMING = "hamming"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"


@dataclass
class MemoryConfig:
    """Base configuration for memory systems."""
    dimension: int  # Dimensionality of vectors
    capacity: Optional[int] = None  # Maximum number of items to store
    distance_metric: DistanceMetric = DistanceMetric.HAMMING
    seed: Optional[int] = None  # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {self.dimension}")
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {self.capacity}")


class CognitiveMemory(ABC):
    """
    Abstract base class for cognitive memory systems.
    
    This class defines the interface that all cognitive memory implementations
    must follow, ensuring consistency across different paradigms.
    """
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize the cognitive memory system.
        
        Parameters
        ----------
        config : MemoryConfig
            Configuration object for the memory system
        """
        self.config = config
        self._validate_config()
        self._initialize()
        
    @abstractmethod
    def _initialize(self):
        """Initialize the memory system internals."""
        pass
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Base validation is done in MemoryConfig.__post_init__
        # Subclasses can override for additional validation
        pass
    
    @abstractmethod
    def store(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Store a key-value pair in memory.
        
        Parameters
        ----------
        key : np.ndarray
            The key vector
        value : np.ndarray
            The value vector to associate with the key
        """
        pass
    
    @abstractmethod
    def recall(self, key: np.ndarray) -> Optional[np.ndarray]:
        """
        Recall a value from memory given a key.
        
        Parameters
        ----------
        key : np.ndarray
            The key vector to search for
            
        Returns
        -------
        Optional[np.ndarray]
            The recalled value or None if not found
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memories."""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Return the current number of stored items."""
        pass
    
    @property
    def is_full(self) -> bool:
        """Check if the memory is at capacity."""
        if self.config.capacity is None:
            return False
        return self.size >= self.config.capacity
    
    def __len__(self) -> int:
        """Return the current number of stored items."""
        return self.size
    
    def __repr__(self) -> str:
        """Return string representation of the memory system."""
        return (f"{self.__class__.__name__}("
                f"dimension={self.config.dimension}, "
                f"size={self.size}, "
                f"capacity={self.config.capacity})")


class VectorEncoder(ABC):
    """
    Abstract base class for encoding data into high-dimensional vectors.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the encoder.
        
        Parameters
        ----------
        dimension : int
            Target dimension for encoded vectors
        """
        self.dimension = dimension
        
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """
        Encode data into a high-dimensional vector.
        
        Parameters
        ----------
        data : Any
            Input data to encode
            
        Returns
        -------
        np.ndarray
            Encoded vector of shape (dimension,)
        """
        pass
    
    @abstractmethod
    def decode(self, vector: np.ndarray) -> Any:
        """
        Decode a high-dimensional vector back to data.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to decode
            
        Returns
        -------
        Any
            Decoded data
        """
        pass


class BinaryVector:
    """
    Utility class for binary vector operations.
    """
    
    @staticmethod
    def random(dimension: int, density: float = 0.5, 
               seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random binary vector.
        
        Parameters
        ----------
        dimension : int
            Length of the vector
        density : float, optional
            Proportion of 1s in the vector (default: 0.5)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Binary vector of shape (dimension,) with values in {0, 1}
        """
        if not 0 <= density <= 1:
            raise ValueError(f"Density must be in [0, 1], got {density}")
            
        rng = np.random.RandomState(seed)
        return (rng.rand(dimension) < density).astype(np.uint8)
    
    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        """
        Calculate Hamming distance between two binary vectors.
        
        Parameters
        ----------
        a, b : np.ndarray
            Binary vectors to compare
            
        Returns
        -------
        int
            Hamming distance (number of differing bits)
        """
        return np.sum(a != b)
    
    @staticmethod
    def to_bipolar(binary_vector: np.ndarray) -> np.ndarray:
        """
        Convert binary vector {0, 1} to bipolar {-1, +1}.
        
        Parameters
        ----------
        binary_vector : np.ndarray
            Binary vector with values in {0, 1}
            
        Returns
        -------
        np.ndarray
            Bipolar vector with values in {-1, +1}
        """
        return 2 * binary_vector - 1
    
    @staticmethod
    def from_bipolar(bipolar_vector: np.ndarray) -> np.ndarray:
        """
        Convert bipolar vector {-1, +1} to binary {0, 1}.
        
        Parameters
        ----------
        bipolar_vector : np.ndarray
            Bipolar vector with values in {-1, +1}
            
        Returns
        -------
        np.ndarray
            Binary vector with values in {0, 1}
        """
        return ((bipolar_vector + 1) / 2).astype(np.uint8)


class MemoryPerformanceMetrics:
    """
    Class for tracking and computing memory system performance metrics.
    """
    
    def __init__(self):
        """Initialize performance metrics."""
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.store_count = 0
        self.recall_count = 0
        self.successful_recalls = 0
        self.store_times = []
        self.recall_times = []
        self.noise_levels = []
        self.recall_accuracies = []
        
    def record_store(self, time_taken: float):
        """Record a store operation."""
        self.store_count += 1
        self.store_times.append(time_taken)
        
    def record_recall(self, time_taken: float, success: bool, 
                     accuracy: Optional[float] = None):
        """Record a recall operation."""
        self.recall_count += 1
        self.recall_times.append(time_taken)
        if success:
            self.successful_recalls += 1
        if accuracy is not None:
            self.recall_accuracies.append(accuracy)
            
    @property
    def recall_success_rate(self) -> float:
        """Calculate recall success rate."""
        if self.recall_count == 0:
            return 0.0
        return self.successful_recalls / self.recall_count
    
    @property
    def average_store_time(self) -> float:
        """Calculate average store time."""
        if not self.store_times:
            return 0.0
        return np.mean(self.store_times)
    
    @property
    def average_recall_time(self) -> float:
        """Calculate average recall time."""
        if not self.recall_times:
            return 0.0
        return np.mean(self.recall_times)
    
    @property
    def average_recall_accuracy(self) -> float:
        """Calculate average recall accuracy."""
        if not self.recall_accuracies:
            return 0.0
        return np.mean(self.recall_accuracies)
    
    def summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            "store_count": self.store_count,
            "recall_count": self.recall_count,
            "recall_success_rate": self.recall_success_rate,
            "average_store_time": self.average_store_time,
            "average_recall_time": self.average_recall_time,
            "average_recall_accuracy": self.average_recall_accuracy,
        }