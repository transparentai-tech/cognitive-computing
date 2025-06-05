"""
Core cognitive modules for Semantic Pointer Architecture.

This module provides the fundamental building blocks for SPA models including
State, Memory, Buffer, and other cognitive components that process and
manipulate semantic pointers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

from .core import SemanticPointer, Vocabulary, SPAConfig

logger = logging.getLogger(__name__)


class Module(ABC):
    """
    Abstract base class for SPA modules.
    
    Modules are the building blocks of SPA models, processing semantic
    pointers and implementing cognitive functions.
    
    Parameters
    ----------
    name : str
        Name of the module
    dimensions : int
        Dimensionality of semantic pointers in this module
    vocab : Vocabulary, optional
        Vocabulary associated with this module
    """
    
    def __init__(self, name: str, dimensions: int, 
                 vocab: Optional[Vocabulary] = None):
        """Initialize module."""
        self.name = name
        self.dimensions = dimensions
        self.vocab = vocab if vocab is not None else Vocabulary(dimensions)
        
        # Current state of the module
        self._state = np.zeros(dimensions)
        
        # Connections to other modules
        self.inputs: Dict[str, 'Connection'] = {}
        self.outputs: Dict[str, 'Connection'] = {}
        
        # Update function for dynamic behavior
        self._update_func: Optional[Callable] = None
        
    @property
    def state(self) -> np.ndarray:
        """Get current state vector."""
        return self._state.copy()
    
    @state.setter
    def state(self, value: np.ndarray):
        """Set state vector."""
        if value.shape != (self.dimensions,):
            raise ValueError(f"State shape {value.shape} doesn't match "
                           f"module dimensions {self.dimensions}")
        self._state = value.copy()
    
    def get_semantic_pointer(self) -> SemanticPointer:
        """Get current state as a semantic pointer."""
        return SemanticPointer(self._state, vocabulary=self.vocab)
    
    def set_semantic_pointer(self, pointer: SemanticPointer):
        """Set state from a semantic pointer."""
        if pointer.dimension != self.dimensions:
            raise ValueError(f"Pointer dimension {pointer.dimension} doesn't "
                           f"match module dimensions {self.dimensions}")
        self._state = pointer.vector.copy()
    
    @abstractmethod
    def update(self, dt: float):
        """
        Update module state.
        
        Parameters
        ----------
        dt : float
            Timestep for update
        """
        pass
    
    def connect_from(self, source: 'Module', transform: Optional[np.ndarray] = None,
                     synapse: float = 0.01, label: Optional[str] = None):
        """
        Create connection from another module.
        
        Parameters
        ----------
        source : Module
            Source module to connect from
        transform : np.ndarray, optional
            Transformation matrix. If None, identity connection
        synapse : float
            Synaptic filter time constant
        label : str, optional
            Label for the connection
        """
        conn = Connection(source, self, transform, synapse)
        label = label or f"{source.name}->{self.name}"
        self.inputs[label] = conn
        source.outputs[label] = conn
        return conn
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}('{self.name}', dim={self.dimensions})"


class State(Module):
    """
    State module for representing and manipulating semantic pointer states.
    
    A State module holds a semantic pointer and can be updated through
    connections from other modules or direct assignment.
    
    Parameters
    ----------
    name : str
        Name of the state
    dimensions : int
        Dimensionality of the state
    vocab : Vocabulary, optional
        Associated vocabulary
    feedback : float
        Feedback strength (0-1) for maintaining state
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None,
                 feedback: float = 0.0):
        """Initialize state module."""
        super().__init__(name, dimensions, vocab)
        self.feedback = feedback
        
        if not 0 <= feedback <= 1:
            raise ValueError(f"Feedback must be in [0, 1], got {feedback}")
    
    def update(self, dt: float):
        """Update state from inputs."""
        # Compute input from all connections
        input_vector = np.zeros(self.dimensions)
        
        for conn in self.inputs.values():
            input_vector += conn.get_output(dt)
        
        # Apply feedback
        if self.feedback > 0:
            input_vector += self.feedback * self._state
        
        # Simple integration with time constant
        # More sophisticated dynamics can be added
        self._state += dt * (-self._state + input_vector)


class Memory(Module):
    """
    Associative memory module for storing and retrieving semantic pointers.
    
    Memory modules can store key-value pairs and retrieve values based
    on partial or noisy keys.
    
    Parameters
    ----------
    name : str
        Name of the memory
    dimensions : int
        Dimensionality of stored pointers
    vocab : Vocabulary, optional
        Associated vocabulary
    threshold : float
        Similarity threshold for retrieval
    capacity : int, optional
        Maximum number of pairs to store
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None,
                 threshold: float = 0.3,
                 capacity: Optional[int] = None):
        """Initialize memory module."""
        super().__init__(name, dimensions, vocab)
        self.threshold = threshold
        self.capacity = capacity
        
        # Stored key-value pairs
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        
    def add_pair(self, key: Union[np.ndarray, SemanticPointer],
                 value: Union[np.ndarray, SemanticPointer]):
        """
        Add a key-value pair to memory.
        
        Parameters
        ----------
        key : array or SemanticPointer
            Key for retrieval
        value : array or SemanticPointer
            Associated value
        """
        # Convert to arrays
        if isinstance(key, SemanticPointer):
            key = key.vector
        if isinstance(value, SemanticPointer):
            value = value.vector
            
        # Validate dimensions
        if key.shape != (self.dimensions,):
            raise ValueError(f"Key dimension {key.shape} doesn't match "
                           f"memory dimension {self.dimensions}")
        if value.shape != (self.dimensions,):
            raise ValueError(f"Value dimension {value.shape} doesn't match "
                           f"memory dimension {self.dimensions}")
        
        # Check capacity
        if self.capacity is not None and len(self.keys) >= self.capacity:
            # Remove oldest
            self.keys.pop(0)
            self.values.pop(0)
        
        self.keys.append(key.copy())
        self.values.append(value.copy())
    
    def recall(self, key: Union[np.ndarray, SemanticPointer]) -> Optional[np.ndarray]:
        """
        Recall value associated with key.
        
        Parameters
        ----------
        key : array or SemanticPointer
            Query key
            
        Returns
        -------
        array or None
            Retrieved value or None if no match
        """
        if isinstance(key, SemanticPointer):
            key = key.vector
            
        if not self.keys:
            return None
        
        # Compute similarities
        key_matrix = np.array(self.keys)
        similarities = np.dot(key_matrix, key) / (
            np.linalg.norm(key_matrix, axis=1) * np.linalg.norm(key)
        )
        
        # Find best match above threshold
        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= self.threshold:
            return self.values[best_idx].copy()
        
        return None
    
    def update(self, dt: float):
        """Update memory state based on input."""
        # Get input
        input_vector = np.zeros(self.dimensions)
        for conn in self.inputs.values():
            input_vector += conn.get_output(dt)
        
        # Try to recall based on input
        recalled = self.recall(input_vector)
        if recalled is not None:
            self._state = recalled
        else:
            # Decay to zero if no match
            self._state *= (1 - dt * 5.0)  # Fast decay


class AssociativeMemory(Memory):
    """
    Enhanced associative memory with multiple retrieval modes.
    
    Extends basic Memory with additional features like partial matching,
    multiple recalls, and heteroassociative storage.
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None,
                 threshold: float = 0.3,
                 capacity: Optional[int] = None,
                 input_scale: float = 1.0,
                 output_scale: float = 1.0):
        """Initialize associative memory."""
        super().__init__(name, dimensions, vocab, threshold, capacity)
        self.input_scale = input_scale
        self.output_scale = output_scale
        
    def recall_all(self, key: Union[np.ndarray, SemanticPointer],
                   top_n: int = 5) -> List[Tuple[np.ndarray, float]]:
        """
        Recall multiple values ranked by similarity.
        
        Parameters
        ----------
        key : array or SemanticPointer
            Query key
        top_n : int
            Number of top matches to return
            
        Returns
        -------
        list of (value, similarity) tuples
            Retrieved values with similarities
        """
        if isinstance(key, SemanticPointer):
            key = key.vector
            
        if not self.keys:
            return []
        
        # Compute similarities
        key_matrix = np.array(self.keys)
        similarities = np.dot(key_matrix, key) / (
            np.linalg.norm(key_matrix, axis=1) * np.linalg.norm(key)
        )
        
        # Get top matches above threshold
        matches = []
        indices = np.argsort(similarities)[::-1][:top_n]
        
        for idx in indices:
            if similarities[idx] >= self.threshold:
                matches.append((self.values[idx].copy(), float(similarities[idx])))
        
        return matches


class Buffer(Module):
    """
    Working memory buffer with gating control.
    
    Buffers can hold information temporarily and be gated to control
    information flow.
    
    Parameters
    ----------
    name : str
        Name of the buffer
    dimensions : int
        Dimensionality of the buffer
    vocab : Vocabulary, optional
        Associated vocabulary
    gate_threshold : float
        Threshold for gate to be considered open
    decay_rate : float
        Rate of decay when gate is closed
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None,
                 gate_threshold: float = 0.5,
                 decay_rate: float = 1.0):
        """Initialize buffer."""
        super().__init__(name, dimensions, vocab)
        self.gate_threshold = gate_threshold
        self.decay_rate = decay_rate
        
        # Gate control (0 = closed, 1 = open)
        self._gate = 1.0
        
    @property
    def gate(self) -> float:
        """Get gate value."""
        return self._gate
    
    @gate.setter
    def gate(self, value: float):
        """Set gate value."""
        self._gate = np.clip(value, 0.0, 1.0)
    
    def update(self, dt: float):
        """Update buffer with gating."""
        # Get input
        input_vector = np.zeros(self.dimensions)
        for conn in self.inputs.values():
            input_vector += conn.get_output(dt)
        
        if self._gate >= self.gate_threshold:
            # Gate open: update normally
            self._state += dt * (-self._state + input_vector)
        else:
            # Gate closed: decay
            self._state *= (1 - dt * self.decay_rate * (1 - self._gate))


class Gate(Module):
    """
    Gating module to control information flow between modules.
    
    Gates multiply their input by a gating signal to control flow.
    
    Parameters
    ----------
    name : str
        Name of the gate
    dimensions : int
        Dimensionality of gated signals
    vocab : Vocabulary, optional
        Associated vocabulary
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None):
        """Initialize gate."""
        super().__init__(name, dimensions, vocab)
        
        # Gating signal (scalar or vector)
        self._gate_signal = 1.0
        
    def set_gate(self, signal: Union[float, np.ndarray]):
        """
        Set gating signal.
        
        Parameters
        ----------
        signal : float or array
            Gating signal (0 = closed, 1 = open)
        """
        if isinstance(signal, (int, float)):
            self._gate_signal = float(signal)
        else:
            if signal.shape != (self.dimensions,):
                raise ValueError(f"Gate signal shape {signal.shape} doesn't "
                               f"match dimensions {self.dimensions}")
            self._gate_signal = signal.copy()
    
    def update(self, dt: float):
        """Update gated output."""
        # Get input
        input_vector = np.zeros(self.dimensions)
        for conn in self.inputs.values():
            input_vector += conn.get_output(dt)
        
        # Apply gating
        if isinstance(self._gate_signal, float):
            self._state = input_vector * self._gate_signal
        else:
            self._state = input_vector * self._gate_signal


class Compare(Module):
    """
    Module for computing similarity between semantic pointers.
    
    Compare modules output a scalar similarity value between their inputs.
    
    Parameters
    ----------
    name : str
        Name of the comparison
    dimensions : int
        Dimensionality of compared pointers
    vocab : Vocabulary, optional
        Associated vocabulary
    output_dimensions : int
        Dimensions for output (default 1 for scalar)
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None,
                 output_dimensions: int = 1):
        """Initialize compare module."""
        # For Compare, we create a minimal vocab for the output dimension
        if vocab is None and output_dimensions < 16:
            # Create a minimal config that allows small dimensions
            from .core import SPAConfig
            min_config = SPAConfig(dimension=16, subdimensions=1)
            output_vocab = Vocabulary(output_dimensions, min_config)
        else:
            output_vocab = vocab
            
        super().__init__(name, output_dimensions, output_vocab)
        self.input_dimensions = dimensions
        
        # Storage for two inputs to compare
        self._input_a = np.zeros(dimensions)
        self._input_b = np.zeros(dimensions)
        
    def set_input_a(self, pointer: Union[np.ndarray, SemanticPointer]):
        """Set first input for comparison."""
        if isinstance(pointer, SemanticPointer):
            pointer = pointer.vector
        if pointer.shape != (self.input_dimensions,):
            raise ValueError(f"Input shape {pointer.shape} doesn't match "
                           f"expected {self.input_dimensions}")
        self._input_a = pointer.copy()
    
    def set_input_b(self, pointer: Union[np.ndarray, SemanticPointer]):
        """Set second input for comparison."""
        if isinstance(pointer, SemanticPointer):
            pointer = pointer.vector
        if pointer.shape != (self.input_dimensions,):
            raise ValueError(f"Input shape {pointer.shape} doesn't match "
                           f"expected {self.input_dimensions}")
        self._input_b = pointer.copy()
    
    def update(self, dt: float):
        """Compute similarity between inputs."""
        # Normalize vectors
        norm_a = np.linalg.norm(self._input_a)
        norm_b = np.linalg.norm(self._input_b)
        
        if norm_a > 0 and norm_b > 0:
            similarity = np.dot(self._input_a, self._input_b) / (norm_a * norm_b)
        else:
            similarity = 0.0
        
        # Output is scalar similarity
        if self.dimensions == 1:
            self._state[0] = similarity
        else:
            # For higher dimensional output, broadcast similarity
            self._state[:] = similarity


class DotProduct(Module):
    """
    Module for computing dot product between semantic pointers.
    
    Similar to Compare but without normalization, useful for
    energy-based computations.
    """
    
    def __init__(self, name: str, dimensions: int,
                 vocab: Optional[Vocabulary] = None,
                 output_dimensions: int = 1):
        """Initialize dot product module."""
        # For DotProduct, we create a minimal vocab for the output dimension
        if vocab is None and output_dimensions < 16:
            # Create a minimal config that allows small dimensions
            from .core import SPAConfig
            min_config = SPAConfig(dimension=16, subdimensions=1)
            output_vocab = Vocabulary(output_dimensions, min_config)
        else:
            output_vocab = vocab
            
        super().__init__(name, output_dimensions, output_vocab)
        self.input_dimensions = dimensions
        
        # Storage for two inputs
        self._input_a = np.zeros(dimensions)
        self._input_b = np.zeros(dimensions)
        
    def set_inputs(self, a: Union[np.ndarray, SemanticPointer],
                   b: Union[np.ndarray, SemanticPointer]):
        """Set both inputs at once."""
        if isinstance(a, SemanticPointer):
            a = a.vector
        if isinstance(b, SemanticPointer):
            b = b.vector
            
        if a.shape != (self.input_dimensions,):
            raise ValueError(f"Input A shape {a.shape} doesn't match "
                           f"expected {self.input_dimensions}")
        if b.shape != (self.input_dimensions,):
            raise ValueError(f"Input B shape {b.shape} doesn't match "
                           f"expected {self.input_dimensions}")
            
        self._input_a = a.copy()
        self._input_b = b.copy()
    
    def update(self, dt: float):
        """Compute dot product."""
        dot_product = np.dot(self._input_a, self._input_b)
        
        if self.dimensions == 1:
            self._state[0] = dot_product
        else:
            self._state[:] = dot_product


@dataclass
class Connection:
    """
    Connection between modules with optional transformation.
    
    Parameters
    ----------
    source : Module
        Source module
    target : Module
        Target module
    transform : np.ndarray, optional
        Transformation matrix
    synapse : float
        Synaptic time constant
    """
    source: Module
    target: Module
    transform: Optional[np.ndarray] = None
    synapse: float = 0.01
    
    # Internal state for synaptic filtering
    _filtered_output: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Initialize connection."""
        self._filtered_output = np.zeros(self.target.dimensions)
        
        # Validate transform if provided
        if self.transform is not None:
            expected_shape = (self.target.dimensions, self.source.dimensions)
            if self.transform.shape != expected_shape:
                raise ValueError(f"Transform shape {self.transform.shape} doesn't "
                               f"match expected {expected_shape}")
    
    def get_output(self, dt: float = 0.001) -> np.ndarray:
        """
        Get filtered output from connection.
        
        Parameters
        ----------
        dt : float
            Timestep for filtering
            
        Returns
        -------
        np.ndarray
            Filtered and transformed output
        """
        # Get source state
        source_output = self.source.state
        
        # Apply transformation if present
        if self.transform is not None:
            output = np.dot(self.transform, source_output)
        else:
            output = source_output
        
        # Apply synaptic filtering (low-pass filter)
        if self.synapse > 0:
            decay = np.exp(-dt / self.synapse)
            self._filtered_output = (decay * self._filtered_output + 
                                   (1 - decay) * output)
            return self._filtered_output
        else:
            return output