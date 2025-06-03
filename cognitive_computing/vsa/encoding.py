"""
Encoding strategies for different data types in VSA.

This module provides various encoders for converting different data types
(text, numeric, spatial, temporal, graph) into high-dimensional vectors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from collections import defaultdict

from .core import VSA, VectorType

logger = logging.getLogger(__name__)


class VSAEncoder(ABC):
    """
    Abstract base class for VSA encoders.
    
    Each encoder converts specific data types into high-dimensional vectors
    suitable for VSA operations.
    """
    
    def __init__(self, vsa: VSA):
        """
        Initialize encoder with VSA system.
        
        Parameters
        ----------
        vsa : VSA
            VSA system to use for encoding
        """
        self.vsa = vsa
        self.dimension = vsa.config.dimension
        self.codebook: Dict[Any, np.ndarray] = {}
    
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Encode data into a high-dimensional vector."""
        pass
    
    @abstractmethod
    def decode(self, vector: np.ndarray) -> Any:
        """Decode a vector back to data (if possible)."""
        pass
    
    def add_to_codebook(self, key: Any, vector: np.ndarray):
        """Add an item to the codebook for later retrieval."""
        self.codebook[key] = vector.copy()
        self.vsa.store(str(key), vector)


class RandomIndexingEncoder(VSAEncoder):
    """
    Random Indexing encoder for text and sequences.
    
    Creates sparse random vectors for tokens and combines them
    for sequences. Useful for NLP applications.
    """
    
    def __init__(self, vsa: VSA, 
                 num_indices: int = 10,
                 window_size: int = 2):
        """
        Initialize Random Indexing encoder.
        
        Parameters
        ----------
        vsa : VSA
            VSA system
        num_indices : int
            Number of non-zero elements per random vector
        window_size : int
            Context window size for sequences
        """
        super().__init__(vsa)
        self.num_indices = num_indices
        self.window_size = window_size
        self.token_vectors: Dict[str, np.ndarray] = {}
    
    def _get_token_vector(self, token: str) -> np.ndarray:
        """Get or create random vector for a token."""
        if token not in self.token_vectors:
            # Create sparse random vector
            vector = np.zeros(self.dimension)
            indices = np.random.choice(
                self.dimension, self.num_indices, replace=False
            )
            values = np.random.choice([-1, 1], self.num_indices)
            vector[indices] = values
            
            self.token_vectors[token] = vector
            self.add_to_codebook(token, vector)
        
        return self.token_vectors[token]
    
    def encode(self, data: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text or sequence of tokens.
        
        Parameters
        ----------
        data : str or List[str]
            Text string or list of tokens
            
        Returns
        -------
        np.ndarray
            Encoded vector
        """
        if isinstance(data, str):
            # Split into tokens (simple whitespace tokenization)
            tokens = data.lower().split()
        else:
            tokens = data
        
        if len(tokens) == 0:
            return np.zeros(self.dimension)
        
        # Encode with context window
        result = np.zeros(self.dimension)
        
        for i, token in enumerate(tokens):
            token_vec = self._get_token_vector(token)
            
            # Add token vector
            result += token_vec
            
            # Add context vectors with position encoding
            for j in range(1, self.window_size + 1):
                if i - j >= 0:
                    context_vec = self._get_token_vector(tokens[i - j])
                    # Use permutation for position encoding
                    shifted = np.roll(context_vec, j)
                    result += shifted * (1.0 / j)  # Distance weighting
                
                if i + j < len(tokens):
                    context_vec = self._get_token_vector(tokens[i + j])
                    shifted = np.roll(context_vec, -j)
                    result += shifted * (1.0 / j)
        
        return self.vsa.vector_factory.normalize(result)
    
    def decode(self, vector: np.ndarray) -> str:
        """
        Decode vector to most similar token.
        
        This is approximate - returns single best matching token.
        """
        best_token = None
        best_similarity = -float('inf')
        
        for token, token_vec in self.token_vectors.items():
            sim = self.vsa.similarity(vector, token_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_token = token
        
        return best_token if best_token else ""


class SpatialEncoder(VSAEncoder):
    """
    Encoder for spatial data (2D/3D coordinates, images).
    
    Uses different encoding schemes for continuous coordinates
    and grid-based data.
    """
    
    def __init__(self, vsa: VSA,
                 grid_size: Tuple[int, ...] = (10, 10),
                 use_fourier: bool = True):
        """
        Initialize spatial encoder.
        
        Parameters
        ----------
        vsa : VSA
            VSA system
        grid_size : Tuple[int, ...]
            Grid dimensions for discretization
        use_fourier : bool
            Whether to use Fourier features for continuous encoding
        """
        super().__init__(vsa)
        self.grid_size = grid_size
        self.use_fourier = use_fourier
        
        # Pre-generate grid vectors
        self._generate_grid_vectors()
    
    def _generate_grid_vectors(self):
        """Generate random vectors for each grid cell."""
        self.grid_vectors = {}
        
        # Generate for all grid positions
        if len(self.grid_size) == 2:
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    pos = (i, j)
                    self.grid_vectors[pos] = self.vsa.generate_vector()
                    self.add_to_codebook(f"grid_{pos}", self.grid_vectors[pos])
        elif len(self.grid_size) == 3:
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    for k in range(self.grid_size[2]):
                        pos = (i, j, k)
                        self.grid_vectors[pos] = self.vsa.generate_vector()
                        self.add_to_codebook(f"grid_{pos}", self.grid_vectors[pos])
    
    def encode_continuous(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Encode continuous coordinates.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Coordinates array of shape (n_dims,) or (n_points, n_dims)
            
        Returns
        -------
        np.ndarray
            Encoded vector(s)
        """
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        
        n_points, n_dims = coordinates.shape
        
        if self.use_fourier:
            # Fourier feature encoding
            # Generate random frequencies
            n_frequencies = self.dimension // (2 * n_dims)
            frequencies = np.random.randn(n_frequencies, n_dims)
            
            # Compute Fourier features
            features = []
            for point in coordinates:
                angles = 2 * np.pi * frequencies @ point
                fourier_features = np.concatenate([
                    np.sin(angles),
                    np.cos(angles)
                ])
                features.append(fourier_features[:self.dimension])
            
            if n_points == 1:
                return self.vsa.vector_factory.normalize(features[0])
            else:
                return np.array([
                    self.vsa.vector_factory.normalize(f) for f in features
                ])
        else:
            # Simple linear encoding
            encoded = []
            for point in coordinates:
                # Discretize to grid
                grid_pos = tuple(
                    int(point[i] * self.grid_size[i]) % self.grid_size[i]
                    for i in range(n_dims)
                )
                encoded.append(self.grid_vectors.get(
                    grid_pos, 
                    self.vsa.generate_vector()
                ))
            
            return encoded[0] if n_points == 1 else np.array(encoded)
    
    def encode_grid(self, grid_data: np.ndarray) -> np.ndarray:
        """
        Encode grid-based data (e.g., images).
        
        Parameters
        ----------
        grid_data : np.ndarray
            Grid data of shape matching grid_size
            
        Returns
        -------
        np.ndarray
            Encoded vector
        """
        result = np.zeros(self.dimension)
        
        # Flatten grid and encode each active position
        flat_indices = np.where(grid_data.flatten() > 0)[0]
        
        for idx in flat_indices:
            # Convert flat index to grid position
            if len(self.grid_size) == 2:
                i = idx // self.grid_size[1]
                j = idx % self.grid_size[1]
                pos = (i, j)
            else:
                # 3D case
                i = idx // (self.grid_size[1] * self.grid_size[2])
                j = (idx % (self.grid_size[1] * self.grid_size[2])) // self.grid_size[2]
                k = idx % self.grid_size[2]
                pos = (i, j, k)
            
            # Weight by value
            weight = grid_data.flatten()[idx]
            result += weight * self.grid_vectors[pos]
        
        return self.vsa.vector_factory.normalize(result)
    
    def encode(self, data: Union[np.ndarray, Tuple[float, ...]]) -> np.ndarray:
        """
        Encode spatial data.
        
        Automatically detects continuous coordinates vs grid data.
        """
        data = np.array(data)
        
        if data.shape == self.grid_size:
            # Grid data
            return self.encode_grid(data)
        else:
            # Continuous coordinates
            return self.encode_continuous(data)
    
    def decode(self, vector: np.ndarray) -> Tuple[int, ...]:
        """
        Decode to nearest grid position.
        
        Returns grid coordinates of most similar position.
        """
        best_pos = None
        best_similarity = -float('inf')
        
        for pos, grid_vec in self.grid_vectors.items():
            sim = self.vsa.similarity(vector, grid_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_pos = pos
        
        return best_pos if best_pos else (0,) * len(self.grid_size)


class TemporalEncoder(VSAEncoder):
    """
    Encoder for temporal/sequential data.
    
    Handles time series, sequences with temporal dependencies.
    """
    
    def __init__(self, vsa: VSA,
                 max_sequence_length: int = 100,
                 use_position: bool = True,
                 use_decay: bool = True):
        """
        Initialize temporal encoder.
        
        Parameters
        ----------
        vsa : VSA
            VSA system
        max_sequence_length : int
            Maximum sequence length to handle
        use_position : bool
            Whether to use position encoding
        use_decay : bool
            Whether to use temporal decay
        """
        super().__init__(vsa)
        self.max_sequence_length = max_sequence_length
        self.use_position = use_position
        self.use_decay = use_decay
        
        # Generate position vectors if needed
        if self.use_position:
            self.position_vectors = [
                self.vsa.generate_vector() 
                for _ in range(max_sequence_length)
            ]
    
    def encode_sequence(self, sequence: List[np.ndarray],
                       timestamps: Optional[List[float]] = None) -> np.ndarray:
        """
        Encode a sequence of vectors with temporal information.
        
        Parameters
        ----------
        sequence : List[np.ndarray]
            Sequence of vectors
        timestamps : List[float], optional
            Timestamps for each element
            
        Returns
        -------
        np.ndarray
            Encoded sequence vector
        """
        if len(sequence) == 0:
            return np.zeros(self.dimension)
        
        result = np.zeros(self.dimension)
        
        for i, item in enumerate(sequence):
            # Position encoding
            if self.use_position and i < len(self.position_vectors):
                # Bind with position
                encoded = self.vsa.bind(item, self.position_vectors[i])
            else:
                encoded = item
            
            # Temporal decay
            if self.use_decay and timestamps is not None:
                # Exponential decay based on time difference
                if i < len(sequence) - 1:
                    time_diff = timestamps[-1] - timestamps[i]
                    decay = np.exp(-time_diff)
                    encoded = encoded * decay
            
            result += encoded
        
        return self.vsa.vector_factory.normalize(result)
    
    def encode_time_series(self, values: np.ndarray,
                          window_size: int = 10) -> List[np.ndarray]:
        """
        Encode time series data using sliding windows.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
        window_size : int
            Size of sliding window
            
        Returns
        -------
        List[np.ndarray]
            Encoded vectors for each window
        """
        encoded_windows = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            
            # Create pattern vector for window
            pattern = np.zeros(self.dimension)
            
            for j, val in enumerate(window):
                # Encode value with position
                if self.use_position:
                    base_vec = self.vsa.generate_vector()
                    # Scale by value
                    value_vec = base_vec * val
                    # Bind with position
                    pos_vec = self.position_vectors[j]
                    encoded = self.vsa.bind(value_vec, pos_vec)
                else:
                    # Simple weighted sum
                    base_vec = self.vsa.generate_vector()
                    encoded = base_vec * val * (j + 1)  # Position weighting
                
                pattern += encoded
            
            encoded_windows.append(
                self.vsa.vector_factory.normalize(pattern)
            )
        
        return encoded_windows
    
    def encode(self, data: Union[List[Any], np.ndarray]) -> np.ndarray:
        """
        Encode temporal data.
        
        Handles both sequences and time series.
        """
        if isinstance(data, np.ndarray) and data.dtype in [np.float32, np.float64]:
            # Time series data
            windows = self.encode_time_series(data)
            # Return encoding of entire series
            return self.vsa.bundle(windows) if windows else np.zeros(self.dimension)
        else:
            # Sequence data
            if all(isinstance(item, np.ndarray) for item in data):
                return self.encode_sequence(data)
            else:
                # Convert items to vectors first
                vectors = [self.vsa.generate_vector() for _ in data]
                return self.encode_sequence(vectors)
    
    def decode(self, vector: np.ndarray) -> int:
        """
        Decode to most likely position.
        
        Returns position index with highest similarity.
        """
        if not self.use_position:
            return -1
        
        best_pos = -1
        best_similarity = -float('inf')
        
        for i, pos_vec in enumerate(self.position_vectors):
            # Try unbinding position
            unbound = self.vsa.unbind(vector, pos_vec)
            # Check if result is clean (high magnitude)
            magnitude = np.linalg.norm(unbound)
            if magnitude > best_similarity:
                best_similarity = magnitude
                best_pos = i
        
        return best_pos


class LevelEncoder(VSAEncoder):
    """
    Encoder for continuous values using level quantization.
    
    Useful for encoding scalar values, sensor readings, etc.
    """
    
    def __init__(self, vsa: VSA,
                 num_levels: int = 32,
                 value_range: Tuple[float, float] = (0.0, 1.0),
                 use_thermometer: bool = True):
        """
        Initialize level encoder.
        
        Parameters
        ----------
        vsa : VSA
            VSA system
        num_levels : int
            Number of quantization levels
        value_range : Tuple[float, float]
            Range of values to encode
        use_thermometer : bool
            Whether to use thermometer encoding
        """
        super().__init__(vsa)
        self.num_levels = num_levels
        self.value_range = value_range
        self.use_thermometer = use_thermometer
        
        # Generate level vectors
        self.level_vectors = [
            self.vsa.generate_vector() for _ in range(num_levels)
        ]
    
    def encode_scalar(self, value: float) -> np.ndarray:
        """
        Encode a scalar value.
        
        Parameters
        ----------
        value : float
            Scalar value to encode
            
        Returns
        -------
        np.ndarray
            Encoded vector
        """
        # Clip to range
        value = np.clip(value, self.value_range[0], self.value_range[1])
        
        # Normalize to [0, 1]
        normalized = (value - self.value_range[0]) / (
            self.value_range[1] - self.value_range[0]
        )
        
        if self.use_thermometer:
            # Thermometer encoding: activate all levels up to value
            result = np.zeros(self.dimension)
            active_levels = int(normalized * self.num_levels)
            
            for i in range(active_levels):
                result += self.level_vectors[i]
            
            return self.vsa.vector_factory.normalize(result)
        else:
            # Single level encoding
            level = int(normalized * (self.num_levels - 1))
            return self.level_vectors[level]
    
    def encode_vector(self, values: np.ndarray) -> np.ndarray:
        """
        Encode a vector of values.
        
        Parameters
        ----------
        values : np.ndarray
            Array of values
            
        Returns
        -------
        np.ndarray
            Encoded vector
        """
        if values.ndim == 0:
            return self.encode_scalar(float(values))
        
        # Encode each dimension separately and bind
        result = self.encode_scalar(values[0])
        
        for i in range(1, len(values)):
            # Use position-specific encoding
            value_vec = self.encode_scalar(values[i])
            # Permute by dimension index
            permuted = np.roll(value_vec, i * (self.dimension // len(values)))
            result = self.vsa.bind(result, permuted)
        
        return result
    
    def encode(self, data: Union[float, np.ndarray]) -> np.ndarray:
        """Encode scalar or vector data."""
        if isinstance(data, (int, float)):
            return self.encode_scalar(float(data))
        else:
            return self.encode_vector(np.array(data))
    
    def decode(self, vector: np.ndarray) -> float:
        """
        Decode to nearest level value.
        
        Returns value corresponding to most similar level.
        """
        best_level = 0
        best_similarity = -float('inf')
        
        for i, level_vec in enumerate(self.level_vectors):
            sim = self.vsa.similarity(vector, level_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_level = i
        
        # Convert level back to value
        normalized = best_level / (self.num_levels - 1)
        value = normalized * (self.value_range[1] - self.value_range[0]) + self.value_range[0]
        
        return value


class GraphEncoder(VSAEncoder):
    """
    Encoder for graph-structured data.
    
    Encodes nodes, edges, and graph structure using VSA operations.
    """
    
    def __init__(self, vsa: VSA,
                 max_nodes: int = 100,
                 use_adjacency: bool = True):
        """
        Initialize graph encoder.
        
        Parameters
        ----------
        vsa : VSA
            VSA system
        max_nodes : int
            Maximum number of nodes to handle
        use_adjacency : bool
            Whether to encode adjacency structure
        """
        super().__init__(vsa)
        self.max_nodes = max_nodes
        self.use_adjacency = use_adjacency
        
        # Node ID vectors
        self.node_vectors: Dict[Any, np.ndarray] = {}
        
        # Edge type vectors
        self.edge_type_vectors: Dict[str, np.ndarray] = {}
    
    def _get_node_vector(self, node_id: Any) -> np.ndarray:
        """Get or create vector for node."""
        if node_id not in self.node_vectors:
            self.node_vectors[node_id] = self.vsa.generate_vector()
            self.add_to_codebook(f"node_{node_id}", self.node_vectors[node_id])
        return self.node_vectors[node_id]
    
    def _get_edge_type_vector(self, edge_type: str) -> np.ndarray:
        """Get or create vector for edge type."""
        if edge_type not in self.edge_type_vectors:
            self.edge_type_vectors[edge_type] = self.vsa.generate_vector()
            self.add_to_codebook(f"edge_{edge_type}", self.edge_type_vectors[edge_type])
        return self.edge_type_vectors[edge_type]
    
    def encode_node(self, node_id: Any, 
                   features: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Encode a node with its features.
        
        Parameters
        ----------
        node_id : Any
            Node identifier
        features : Dict[str, Any], optional
            Node features/attributes
            
        Returns
        -------
        np.ndarray
            Encoded node vector
        """
        node_vec = self._get_node_vector(node_id)
        
        if features:
            # Bind features to node
            for key, value in features.items():
                # Create feature vector
                feature_vec = self.vsa.generate_vector()
                
                # Encode value (simplified - could use other encoders)
                if isinstance(value, (int, float)):
                    # Scale feature vector by value
                    feature_vec = feature_vec * value
                
                # Bind feature to node
                node_vec = self.vsa.bind(node_vec, feature_vec)
        
        return node_vec
    
    def encode_edge(self, source: Any, target: Any,
                   edge_type: str = "default") -> np.ndarray:
        """
        Encode an edge between nodes.
        
        Parameters
        ----------
        source : Any
            Source node ID
        target : Any  
            Target node ID
        edge_type : str
            Type of edge
            
        Returns
        -------
        np.ndarray
            Encoded edge vector
        """
        source_vec = self._get_node_vector(source)
        target_vec = self._get_node_vector(target)
        edge_vec = self._get_edge_type_vector(edge_type)
        
        # Bind source and target with edge type
        # Non-commutative binding for directed edges
        edge_encoding = self.vsa.bind(
            self.vsa.bind(source_vec, edge_vec),
            target_vec
        )
        
        return edge_encoding
    
    def encode_graph(self, nodes: List[Any],
                    edges: List[Tuple[Any, Any, str]],
                    node_features: Optional[Dict[Any, Dict]] = None) -> np.ndarray:
        """
        Encode entire graph structure.
        
        Parameters
        ----------
        nodes : List[Any]
            List of node IDs
        edges : List[Tuple[Any, Any, str]]
            List of (source, target, edge_type) tuples
        node_features : Dict[Any, Dict], optional
            Features for each node
            
        Returns
        -------
        np.ndarray
            Encoded graph vector
        """
        result = np.zeros(self.dimension)
        
        # Encode nodes
        for node in nodes:
            features = node_features.get(node) if node_features else None
            node_vec = self.encode_node(node, features)
            result += node_vec
        
        # Encode edges
        for source, target, edge_type in edges:
            edge_vec = self.encode_edge(source, target, edge_type)
            result += edge_vec
        
        # Add adjacency encoding if requested
        if self.use_adjacency:
            # Build adjacency structure
            adjacency = defaultdict(list)
            for source, target, _ in edges:
                adjacency[source].append(target)
            
            # Encode neighborhoods
            for node, neighbors in adjacency.items():
                if neighbors:
                    node_vec = self._get_node_vector(node)
                    neighbor_bundle = self.vsa.bundle([
                        self._get_node_vector(n) for n in neighbors
                    ])
                    # Bind node with its neighborhood
                    neighborhood = self.vsa.bind(node_vec, neighbor_bundle)
                    result += neighborhood
        
        return self.vsa.vector_factory.normalize(result)
    
    def encode(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Encode graph data from dictionary.
        
        Expected format:
        {
            'nodes': [node_ids...],
            'edges': [(source, target, type)...],
            'features': {node_id: {feature_dict}...}  # optional
        }
        """
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        features = data.get('features', None)
        
        return self.encode_graph(nodes, edges, features)
    
    def decode(self, vector: np.ndarray) -> List[Any]:
        """
        Decode to find most similar nodes.
        
        Returns list of node IDs with high similarity.
        """
        similar_nodes = []
        threshold = 0.5
        
        for node_id, node_vec in self.node_vectors.items():
            sim = self.vsa.similarity(vector, node_vec)
            if sim > threshold:
                similar_nodes.append((node_id, sim))
        
        # Sort by similarity
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node_id for node_id, _ in similar_nodes[:5]]  # Top 5