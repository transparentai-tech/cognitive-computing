"""
Core implementation of Semantic Pointer Architecture (SPA).

This module provides the main SPA classes including semantic pointers,
vocabularies, and the base SPA system that coordinates modules and control.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import signal
from abc import ABC, abstractmethod

from ..common.base import CognitiveMemory, MemoryConfig
from ..hrr import operations as hrr_ops

logger = logging.getLogger(__name__)


@dataclass
class SPAConfig(MemoryConfig):
    """
    Configuration for Semantic Pointer Architecture models.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of semantic pointer vectors
    subdimensions : int
        Dimensions per semantic pointer component (default: 16)
    neurons_per_dimension : int
        Number of neurons per dimension for neural implementation (default: 50)
    max_similarity_matches : int
        Maximum number of matches to return from cleanup (default: 10)
    threshold : float
        Action selection threshold (default: 0.3)
    mutual_inhibition : float
        Mutual inhibition between competing actions (default: 1.0)
    bg_bias : float
        Basal ganglia bias for action selection (default: 0.0)
    routing_inhibition : float
        Thalamus routing inhibition strength (default: 3.0)
    synapse : float
        Synaptic time constant in seconds (default: 0.01)
    dt : float
        Simulation timestep in seconds (default: 0.001)
    normalize_pointers : bool
        Whether to normalize semantic pointers (default: True)
    strict_vocab : bool
        Whether to enforce strict vocabulary (no unknown pointers) (default: False)
    """
    subdimensions: int = 16
    neurons_per_dimension: int = 50
    max_similarity_matches: int = 10
    threshold: float = 0.3
    mutual_inhibition: float = 1.0
    bg_bias: float = 0.0
    routing_inhibition: float = 3.0
    synapse: float = 0.01
    dt: float = 0.001
    normalize_pointers: bool = True
    strict_vocab: bool = False
    
    def __post_init__(self):
        """Validate SPA configuration parameters."""
        super().__post_init__()
        
        if self.subdimensions <= 0:
            raise ValueError(f"subdimensions must be positive, got {self.subdimensions}")
        
        if self.dimension % self.subdimensions != 0:
            raise ValueError(f"dimension ({self.dimension}) must be divisible by "
                           f"subdimensions ({self.subdimensions})")
        
        if self.neurons_per_dimension <= 0:
            raise ValueError(f"neurons_per_dimension must be positive, "
                           f"got {self.neurons_per_dimension}")
        
        if self.max_similarity_matches <= 0:
            raise ValueError(f"max_similarity_matches must be positive, "
                           f"got {self.max_similarity_matches}")
        
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        
        if self.mutual_inhibition < 0:
            raise ValueError(f"mutual_inhibition must be non-negative, "
                           f"got {self.mutual_inhibition}")
        
        if self.routing_inhibition < 0:
            raise ValueError(f"routing_inhibition must be non-negative, "
                           f"got {self.routing_inhibition}")
        
        if self.synapse <= 0:
            raise ValueError(f"synapse must be positive, got {self.synapse}")
        
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        
        if self.dt > self.synapse:
            logger.warning(f"dt ({self.dt}) is larger than synapse ({self.synapse}), "
                         f"which may lead to instability")


class SemanticPointer:
    """
    A semantic pointer with HRR operations and neural grounding.
    
    Semantic pointers are high-dimensional vectors that can be composed
    using operations like binding and bundling to represent complex
    concepts and relationships.
    
    Parameters
    ----------
    vector : np.ndarray
        The vector representation of the semantic pointer
    name : str, optional
        Name or label for this pointer
    vocabulary : Vocabulary, optional
        Associated vocabulary for cleanup operations
    """
    
    def __init__(self, vector: np.ndarray, name: Optional[str] = None,
                 vocabulary: Optional['Vocabulary'] = None):
        """Initialize a semantic pointer."""
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1-dimensional, got shape {vector.shape}")
        
        self.vector = vector.copy()
        self.name = name
        self.vocabulary = vocabulary
        self._normalized = False
        
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the pointer."""
        return len(self.vector)
    
    def normalize(self) -> 'SemanticPointer':
        """
        Normalize the semantic pointer to unit length.
        
        Returns
        -------
        SemanticPointer
            Normalized semantic pointer
        """
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            normalized_vector = self.vector / norm
        else:
            normalized_vector = self.vector
        
        result = SemanticPointer(normalized_vector, self.name, self.vocabulary)
        result._normalized = True
        return result
    
    def __mul__(self, other: Union['SemanticPointer', float]) -> 'SemanticPointer':
        """
        Bind with another semantic pointer using circular convolution.
        
        Parameters
        ----------
        other : SemanticPointer or float
            Pointer to bind with or scalar to multiply by
            
        Returns
        -------
        SemanticPointer
            Result of binding or scaling
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return SemanticPointer(self.vector * other, vocabulary=self.vocabulary)
        elif isinstance(other, SemanticPointer):
            # Circular convolution binding
            if self.dimension != other.dimension:
                raise ValueError(f"Dimensions must match: {self.dimension} != {other.dimension}")
            
            bound = hrr_ops.CircularConvolution.convolve(self.vector, other.vector)
            return SemanticPointer(bound, vocabulary=self.vocabulary)
        else:
            raise TypeError(f"Cannot multiply SemanticPointer with {type(other)}")
    
    def __rmul__(self, other: float) -> 'SemanticPointer':
        """Right multiplication for scalar * SemanticPointer."""
        return self.__mul__(other)
    
    def __invert__(self) -> 'SemanticPointer':
        """
        Get the inverse for unbinding (~A).
        
        Returns
        -------
        SemanticPointer
            Inverse pointer for unbinding
        """
        # For circular convolution, inverse is computed by reversing the vector
        # This is equivalent to conjugating in frequency domain
        # For real vectors: inverse of a is a with indices reversed (except first)
        inverse = np.zeros_like(self.vector)
        inverse[0] = self.vector[0]
        inverse[1:] = self.vector[:0:-1]
        
        return SemanticPointer(inverse, f"~{self.name}" if self.name else None,
                             self.vocabulary)
    
    def __add__(self, other: 'SemanticPointer') -> 'SemanticPointer':
        """
        Bundle with another semantic pointer (superposition).
        
        Parameters
        ----------
        other : SemanticPointer
            Pointer to bundle with
            
        Returns
        -------
        SemanticPointer
            Bundled pointer
        """
        if not isinstance(other, SemanticPointer):
            raise TypeError(f"Cannot add SemanticPointer with {type(other)}")
        
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match: {self.dimension} != {other.dimension}")
        
        bundled = self.vector + other.vector
        return SemanticPointer(bundled, vocabulary=self.vocabulary)
    
    def __sub__(self, other: 'SemanticPointer') -> 'SemanticPointer':
        """
        Subtract another semantic pointer.
        
        Parameters
        ----------
        other : SemanticPointer
            Pointer to subtract
            
        Returns
        -------
        SemanticPointer
            Difference of pointers
        """
        if not isinstance(other, SemanticPointer):
            raise TypeError(f"Cannot subtract {type(other)} from SemanticPointer")
        
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match: {self.dimension} != {other.dimension}")
        
        diff = self.vector - other.vector
        return SemanticPointer(diff, vocabulary=self.vocabulary)
    
    def dot(self, other: 'SemanticPointer') -> float:
        """
        Compute dot product (similarity) with another pointer.
        
        Parameters
        ----------
        other : SemanticPointer
            Pointer to compare with
            
        Returns
        -------
        float
            Dot product similarity
        """
        if not isinstance(other, SemanticPointer):
            raise TypeError(f"Cannot compute dot product with {type(other)}")
        
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match: {self.dimension} != {other.dimension}")
        
        return np.dot(self.vector, other.vector)
    
    def similarity(self, other: 'SemanticPointer') -> float:
        """
        Compute cosine similarity with another pointer.
        
        Parameters
        ----------
        other : SemanticPointer
            Pointer to compare with
            
        Returns
        -------
        float
            Cosine similarity in [-1, 1]
        """
        if not isinstance(other, SemanticPointer):
            raise TypeError(f"Cannot compute similarity with {type(other)}")
        
        if self.dimension != other.dimension:
            raise ValueError(f"Dimensions must match: {self.dimension} != {other.dimension}")
        
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
        
        return np.dot(self.vector, other.vector) / (norm_self * norm_other)
    
    def __repr__(self) -> str:
        """Return string representation."""
        if self.name:
            return f"SemanticPointer('{self.name}', dim={self.dimension})"
        else:
            return f"SemanticPointer(dim={self.dimension})"


class Vocabulary:
    """
    Collection of semantic pointers with parsing and cleanup capabilities.
    
    A vocabulary stores named semantic pointers and provides methods for
    creating new pointers, parsing expressions, and cleaning up noisy
    vectors to their nearest matches.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of semantic pointers in the vocabulary
    config : SPAConfig, optional
        Configuration for the vocabulary
    rng : np.random.RandomState, optional
        Random number generator for reproducibility
    """
    
    def __init__(self, dimension: int, config: Optional[SPAConfig] = None,
                 rng: Optional[np.random.RandomState] = None):
        """Initialize vocabulary."""
        self.dimension = dimension
        self.config = config or SPAConfig(dimension=dimension)
        self.rng = rng or np.random.RandomState()
        
        # Dictionary of name -> SemanticPointer
        self.pointers: Dict[str, SemanticPointer] = {}
        
        # Precomputed matrix of all pointer vectors for efficient cleanup
        self._pointer_matrix: Optional[np.ndarray] = None
        self._pointer_names: List[str] = []
        self._needs_update = True
        
    def create_pointer(self, name: str, vector: Optional[np.ndarray] = None) -> SemanticPointer:
        """
        Create or register a semantic pointer in the vocabulary.
        
        Parameters
        ----------
        name : str
            Name for the pointer
        vector : np.ndarray, optional
            Vector representation. If None, generates random vector
            
        Returns
        -------
        SemanticPointer
            The created or registered pointer
        """
        if name in self.pointers:
            if self.config.strict_vocab:
                raise ValueError(f"Pointer '{name}' already exists in vocabulary")
            else:
                logger.warning(f"Pointer '{name}' already exists, returning existing pointer")
                return self.pointers[name]
        
        if vector is None:
            # Generate random vector
            vector = self.rng.randn(self.dimension)
            if self.config.normalize_pointers:
                vector = vector / np.linalg.norm(vector)
        else:
            if vector.shape != (self.dimension,):
                raise ValueError(f"Vector shape {vector.shape} does not match "
                               f"vocabulary dimension {self.dimension}")
            vector = vector.copy()
        
        pointer = SemanticPointer(vector, name, self)
        self.pointers[name] = pointer
        self._needs_update = True
        
        return pointer
    
    def parse(self, expression: str) -> SemanticPointer:
        """
        Parse an expression to create a semantic pointer.
        
        Supports operations:
        - Single pointer: "A"
        - Binding: "A*B" 
        - Unbinding: "A*~B"
        - Bundling: "A+B"
        - Subtraction: "A-B"
        - Parentheses: "(A+B)*C"
        - Scalars: "0.5*A"
        
        Parameters
        ----------
        expression : str
            Expression to parse
            
        Returns
        -------
        SemanticPointer
            Result of evaluating the expression
        """
        # This is a simplified parser - in production, use a proper expression parser
        # For now, we'll handle basic cases
        expression = expression.strip()
        
        # Handle single pointer
        if expression in self.pointers:
            return self.pointers[expression]
        
        # Handle scalar multiplication (simplified)
        if '*' in expression and not any(op in expression for op in ['~', '+', '-', '(', ')']):
            parts = expression.split('*')
            if len(parts) == 2:
                try:
                    scalar = float(parts[0])
                    if parts[1] in self.pointers:
                        return scalar * self.pointers[parts[1]]
                except ValueError:
                    # Not a scalar, try pointer multiplication
                    if parts[0] in self.pointers and parts[1] in self.pointers:
                        return self.pointers[parts[0]] * self.pointers[parts[1]]
        
        # For complex expressions, we need a proper parser
        # This is a placeholder that handles some basic cases
        raise NotImplementedError(f"Complex expression parsing not yet implemented: {expression}")
    
    def cleanup(self, vector: np.ndarray, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Find the closest semantic pointers to a given vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to clean up
        top_n : int, optional
            Number of top matches to return. If None, uses config.max_similarity_matches
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (name, similarity) pairs, sorted by similarity
        """
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector shape {vector.shape} does not match "
                           f"vocabulary dimension {self.dimension}")
        
        if not self.pointers:
            return []
        
        # Update pointer matrix if needed
        if self._needs_update:
            self._update_pointer_matrix()
        
        # Normalize vector for cosine similarity
        norm_vector = np.linalg.norm(vector)
        if norm_vector > 0:
            vector = vector / norm_vector
        
        # Compute similarities with all pointers
        similarities = np.dot(self._pointer_matrix, vector)
        
        # Get top matches
        if top_n is None:
            top_n = self.config.max_similarity_matches
        
        top_n = min(top_n, len(self.pointers))
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            name = self._pointer_names[idx]
            sim = similarities[idx]
            if sim > self.config.threshold:
                results.append((name, float(sim)))
        
        return results
    
    def _update_pointer_matrix(self):
        """Update the internal matrix of pointer vectors."""
        if not self.pointers:
            self._pointer_matrix = None
            self._pointer_names = []
            self._needs_update = False
            return
        
        self._pointer_names = list(self.pointers.keys())
        vectors = []
        
        for name in self._pointer_names:
            vec = self.pointers[name].vector
            # Normalize for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        
        self._pointer_matrix = np.array(vectors)
        self._needs_update = False
    
    def __len__(self) -> int:
        """Return number of pointers in vocabulary."""
        return len(self.pointers)
    
    def __contains__(self, name: str) -> bool:
        """Check if pointer name is in vocabulary."""
        return name in self.pointers
    
    def __getitem__(self, name: str) -> SemanticPointer:
        """Get pointer by name."""
        if name not in self.pointers:
            if self.config.strict_vocab:
                raise KeyError(f"Pointer '{name}' not in vocabulary")
            else:
                # Auto-create pointer
                logger.info(f"Auto-creating pointer '{name}'")
                return self.create_pointer(name)
        return self.pointers[name]
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Vocabulary(dimension={self.dimension}, size={len(self)})"


class SPA(CognitiveMemory):
    """
    Main Semantic Pointer Architecture system coordinating modules and control.
    
    The SPA class serves as the central coordinator for a cognitive model,
    managing vocabularies, modules, actions, and control flow.
    
    Parameters
    ----------
    config : SPAConfig
        Configuration for the SPA system
    """
    
    def __init__(self, config: SPAConfig):
        """Initialize SPA system."""
        super().__init__(config)
        self.config: SPAConfig = config
        
        # Main vocabulary for the system
        self.vocabulary = Vocabulary(config.dimension, config)
        
        # Modules will be added as we implement them
        self.modules: Dict[str, Any] = {}
        
        # Actions will be added as we implement action selection
        self.actions: List[Any] = []
        
        # Time tracking for simulation
        self.time = 0.0
        
    def _initialize(self):
        """Initialize the SPA system internals."""
        # Initialize will be expanded as we add modules
        pass
    
    def store(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Store a key-value pair (placeholder for CognitiveMemory interface).
        
        Note: SPA uses semantic pointers and modules rather than direct
        key-value storage. This method is here for interface compatibility.
        """
        # In SPA, storage happens through modules like AssociativeMemory
        raise NotImplementedError("Use SPA modules for storage operations")
    
    def recall(self, key: np.ndarray) -> Optional[np.ndarray]:
        """
        Recall a value (placeholder for CognitiveMemory interface).
        
        Note: SPA uses semantic pointers and modules rather than direct
        key-value recall. This method is here for interface compatibility.
        """
        # In SPA, recall happens through modules like AssociativeMemory
        raise NotImplementedError("Use SPA modules for recall operations")
    
    def clear(self) -> None:
        """Clear all stored memories and reset system."""
        self.vocabulary = Vocabulary(self.config.dimension, self.config)
        self.modules.clear()
        self.actions.clear()
        self.time = 0.0
    
    @property
    def size(self) -> int:
        """Return the current number of semantic pointers."""
        return len(self.vocabulary)
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        Run one simulation step.
        
        Parameters
        ----------
        dt : float, optional
            Timestep size. If None, uses config.dt
        """
        if dt is None:
            dt = self.config.dt
        
        # Step will be implemented when we have modules and actions
        self.time += dt
    
    def run(self, duration: float) -> None:
        """
        Run simulation for specified duration.
        
        Parameters
        ----------
        duration : float
            How long to run the simulation in seconds
        """
        steps = int(duration / self.config.dt)
        for _ in range(steps):
            self.step()
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (f"SPA(dimension={self.config.dimension}, "
                f"pointers={len(self.vocabulary)}, "
                f"modules={len(self.modules)})")


# Factory functions
def create_spa(dimension: int = 512, **kwargs) -> SPA:
    """
    Create an SPA system with the given configuration.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of semantic pointers
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    SPA
        Configured SPA system
    """
    config = SPAConfig(dimension=dimension, **kwargs)
    return SPA(config)


def create_vocabulary(dimension: int = 512, **kwargs) -> Vocabulary:
    """
    Create a vocabulary for semantic pointers.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of semantic pointers
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    Vocabulary
        New vocabulary instance
    """
    config = SPAConfig(dimension=dimension, **kwargs)
    return Vocabulary(dimension, config)