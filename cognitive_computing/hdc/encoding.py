"""
Data encoding strategies for HDC.

This module provides various encoders to convert different data types
into hypervectors for use in hyperdimensional computing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from cognitive_computing.hdc.hypervectors import (
    generate_orthogonal_hypervectors,
    BinaryHypervector,
    BipolarHypervector,
)
from cognitive_computing.hdc.operations import (
    bind_hypervectors,
    bundle_hypervectors,
    permute_hypervector,
    PermutationMethod,
    normalize_hypervector,
)


class Encoder(ABC):
    """Abstract base class for HDC encoders."""
    
    def __init__(self, dimension: int, hypervector_type: str = "bipolar"):
        """
        Initialize encoder.
        
        Parameters
        ----------
        dimension : int
            Dimensionality of output hypervectors
        hypervector_type : str
            Type of hypervectors to generate
        """
        self.dimension = dimension
        self.hypervector_type = hypervector_type
        self._rng = np.random.RandomState()
        
    @abstractmethod
    def encode(self, data: any) -> np.ndarray:
        """
        Encode data into a hypervector.
        
        Parameters
        ----------
        data : any
            Data to encode
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        pass
        
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.RandomState(seed)


class ScalarEncoder(Encoder):
    """
    Encoder for scalar (continuous) values.
    
    This encoder uses level quantization or thermometer encoding
    to represent continuous values as hypervectors.
    """
    
    def __init__(
        self,
        dimension: int,
        min_value: float,
        max_value: float,
        n_levels: int = 100,
        method: str = "thermometer",
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize scalar encoder.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        min_value : float
            Minimum value in range
        max_value : float
            Maximum value in range
        n_levels : int
            Number of quantization levels
        method : str
            Encoding method: "thermometer" or "level"
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, hypervector_type)
        self.min_value = min_value
        self.max_value = max_value
        self.n_levels = n_levels
        self.method = method
        
        # Generate level hypervectors
        self._generate_level_hypervectors()
        
    def _generate_level_hypervectors(self) -> None:
        """Generate hypervectors for each level."""
        if self.method == "thermometer":
            # Thermometer encoding: progressive filling
            self.level_vectors = []
            
            if self.hypervector_type == "binary":
                base = np.zeros(self.dimension, dtype=np.uint8)
                step = self.dimension / self.n_levels
                
                for i in range(self.n_levels):
                    hv = base.copy()
                    fill_count = int((i + 1) * step)
                    hv[:fill_count] = 1
                    # Shuffle to distribute
                    self._rng.shuffle(hv)
                    self.level_vectors.append(hv)
                    
            elif self.hypervector_type == "bipolar":
                base = -np.ones(self.dimension, dtype=np.int8)
                step = self.dimension / self.n_levels
                
                for i in range(self.n_levels):
                    hv = base.copy()
                    fill_count = int((i + 1) * step)
                    hv[:fill_count] = 1
                    # Shuffle to distribute
                    self._rng.shuffle(hv)
                    self.level_vectors.append(hv)
                    
        elif self.method == "level":
            # Generate orthogonal vectors for each level
            self.level_vectors = generate_orthogonal_hypervectors(
                self.dimension,
                self.n_levels,
                self.hypervector_type
            )
            
    def encode(self, value: float) -> np.ndarray:
        """
        Encode a scalar value.
        
        Parameters
        ----------
        value : float
            Value to encode
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        # Clip to range
        value = np.clip(value, self.min_value, self.max_value)
        
        # Quantize to level
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        level = int(normalized * (self.n_levels - 1))
        level = np.clip(level, 0, self.n_levels - 1)
        
        return self.level_vectors[level].copy()


class CategoricalEncoder(Encoder):
    """
    Encoder for categorical (discrete) values.
    
    Each category is assigned a quasi-orthogonal hypervector.
    """
    
    def __init__(
        self,
        dimension: int,
        categories: Optional[List[str]] = None,
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize categorical encoder.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        categories : List[str], optional
            Known categories (can be extended dynamically)
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, hypervector_type)
        self.category_vectors: Dict[str, np.ndarray] = {}
        
        if categories:
            self._initialize_categories(categories)
            
    def _initialize_categories(self, categories: List[str]) -> None:
        """Initialize hypervectors for known categories."""
        vectors = generate_orthogonal_hypervectors(
            self.dimension,
            len(categories),
            self.hypervector_type
        )
        
        for cat, vec in zip(categories, vectors):
            self.category_vectors[cat] = vec
            
    def encode(self, category: str) -> np.ndarray:
        """
        Encode a category.
        
        Parameters
        ----------
        category : str
            Category to encode
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        if category not in self.category_vectors:
            # Generate new random hypervector for unknown category
            if self.hypervector_type == "binary":
                hv = self._rng.randint(0, 2, size=self.dimension, dtype=np.uint8)
            elif self.hypervector_type == "bipolar":
                hv = 2 * self._rng.randint(0, 2, size=self.dimension) - 1
                hv = hv.astype(np.int8)
            else:
                raise ValueError(f"Unsupported hypervector type: {self.hypervector_type}")
                
            self.category_vectors[category] = hv
            
        return self.category_vectors[category].copy()
        
    def get_categories(self) -> List[str]:
        """Get list of known categories."""
        return list(self.category_vectors.keys())


class SequenceEncoder(Encoder):
    """
    Encoder for sequences and time series.
    
    Supports various sequence encoding methods including
    n-grams and positional encoding.
    """
    
    def __init__(
        self,
        dimension: int,
        item_encoder: Optional[Encoder] = None,
        method: str = "ngram",
        n: int = 3,
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize sequence encoder.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        item_encoder : Encoder, optional
            Encoder for individual items
        method : str
            Encoding method: "ngram" or "position"
        n : int
            N-gram size
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, hypervector_type)
        self.item_encoder = item_encoder or CategoricalEncoder(dimension, hypervector_type=hypervector_type)
        self.method = method
        self.n = n
        
        # Generate position vectors for positional encoding
        if method == "position":
            self._generate_position_vectors()
            
    def _generate_position_vectors(self) -> None:
        """Generate position encoding vectors."""
        # Create position vectors by repeated permutation
        self.position_vectors = []
        
        if self.hypervector_type == "binary":
            base = self._rng.randint(0, 2, size=self.dimension, dtype=np.uint8)
        else:
            base = 2 * self._rng.randint(0, 2, size=self.dimension) - 1
            base = base.astype(np.int8)
            
        self.position_vectors.append(base)
        
        # Generate more position vectors by permutation
        for i in range(1, 100):  # Support up to 100 positions
            prev = self.position_vectors[-1]
            permuted = permute_hypervector(prev, shift=1)
            self.position_vectors.append(permuted)
            
    def encode(self, sequence: List[any]) -> np.ndarray:
        """
        Encode a sequence.
        
        Parameters
        ----------
        sequence : List[any]
            Sequence of items to encode
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        if not sequence:
            raise ValueError("Cannot encode empty sequence")
            
        if self.method == "ngram":
            return self._encode_ngram(sequence)
        elif self.method == "position":
            return self._encode_position(sequence)
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
            
    def _encode_ngram(self, sequence: List[any]) -> np.ndarray:
        """Encode using n-gram method."""
        # Encode individual items
        item_vectors = [self.item_encoder.encode(item) for item in sequence]
        
        if len(sequence) < self.n:
            # Sequence shorter than n-gram size
            ngram = item_vectors[0]
            for i in range(1, len(item_vectors)):
                shifted = permute_hypervector(item_vectors[i], shift=i)
                ngram = bind_hypervectors(ngram, shifted, self.hypervector_type)
            return ngram
            
        # Generate n-grams
        ngrams = []
        for i in range(len(sequence) - self.n + 1):
            # Create n-gram
            ngram = item_vectors[i]
            for j in range(1, self.n):
                shifted = permute_hypervector(item_vectors[i + j], shift=j)
                ngram = bind_hypervectors(ngram, shifted, self.hypervector_type)
            ngrams.append(ngram)
            
        # Bundle all n-grams
        return bundle_hypervectors(ngrams, hypervector_type=self.hypervector_type)
        
    def _encode_position(self, sequence: List[any]) -> np.ndarray:
        """Encode using positional encoding."""
        # Encode individual items
        item_vectors = [self.item_encoder.encode(item) for item in sequence]
        
        # Bind with position vectors
        bound_vectors = []
        for i, item_vec in enumerate(item_vectors):
            if i < len(self.position_vectors):
                pos_vec = self.position_vectors[i]
            else:
                # Generate more position vectors if needed
                pos_vec = permute_hypervector(
                    self.position_vectors[-1],
                    shift=i - len(self.position_vectors) + 1
                )
                
            bound = bind_hypervectors(item_vec, pos_vec, self.hypervector_type)
            bound_vectors.append(bound)
            
        # Bundle all position-bound items
        return bundle_hypervectors(bound_vectors, hypervector_type=self.hypervector_type)


class SpatialEncoder(Encoder):
    """
    Encoder for spatial data (2D/3D coordinates).
    
    Encodes spatial relationships and coordinates into hypervectors.
    """
    
    def __init__(
        self,
        dimension: int,
        bounds: Tuple[Tuple[float, float], ...],
        resolution: int = 10,
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize spatial encoder.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        bounds : Tuple[Tuple[float, float], ...]
            Bounds for each spatial dimension: ((x_min, x_max), (y_min, y_max), ...)
        resolution : int
            Grid resolution for each dimension
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, hypervector_type)
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.resolution = resolution
        
        # Generate basis vectors for each dimension
        self._generate_basis_vectors()
        
    def _generate_basis_vectors(self) -> None:
        """Generate basis vectors for spatial encoding."""
        self.dimension_vectors = []
        self.grid_vectors = []
        
        # Generate orthogonal vectors for each spatial dimension
        dim_vecs = generate_orthogonal_hypervectors(
            self.dimension,
            self.n_dims,
            self.hypervector_type
        )
        self.dimension_vectors = dim_vecs
        
        # Generate vectors for grid positions
        for d in range(self.n_dims):
            grid_vecs = generate_orthogonal_hypervectors(
                self.dimension,
                self.resolution,
                self.hypervector_type
            )
            self.grid_vectors.append(grid_vecs)
            
    def encode(self, coordinates: Tuple[float, ...]) -> np.ndarray:
        """
        Encode spatial coordinates.
        
        Parameters
        ----------
        coordinates : Tuple[float, ...]
            Coordinates to encode (x, y, z, ...)
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        if len(coordinates) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} coordinates, got {len(coordinates)}"
            )
            
        encoded_dims = []
        
        for d, coord in enumerate(coordinates):
            # Normalize to [0, 1]
            min_val, max_val = self.bounds[d]
            normalized = (coord - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            
            # Quantize to grid
            grid_idx = int(normalized * (self.resolution - 1))
            grid_idx = np.clip(grid_idx, 0, self.resolution - 1)
            
            # Get grid vector for this position
            grid_vec = self.grid_vectors[d][grid_idx]
            
            # Bind with dimension vector
            dim_vec = self.dimension_vectors[d]
            bound = bind_hypervectors(grid_vec, dim_vec, self.hypervector_type)
            encoded_dims.append(bound)
            
        # Bundle all dimensions
        return bundle_hypervectors(encoded_dims, hypervector_type=self.hypervector_type)


class RecordEncoder(Encoder):
    """
    Encoder for structured records with named fields.
    
    Encodes records by binding field names with field values.
    """
    
    def __init__(
        self,
        dimension: int,
        field_encoders: Optional[Dict[str, Encoder]] = None,
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize record encoder.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        field_encoders : Dict[str, Encoder], optional
            Encoders for specific fields
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, hypervector_type)
        self.field_encoders = field_encoders or {}
        self.field_vectors: Dict[str, np.ndarray] = {}
        
    def encode(self, record: Dict[str, any]) -> np.ndarray:
        """
        Encode a record.
        
        Parameters
        ----------
        record : Dict[str, any]
            Record with field names and values
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        if not record:
            raise ValueError("Cannot encode empty record")
            
        field_bindings = []
        
        for field_name, field_value in record.items():
            # Get or create field name vector
            if field_name not in self.field_vectors:
                if self.hypervector_type == "binary":
                    field_vec = self._rng.randint(0, 2, size=self.dimension, dtype=np.uint8)
                else:
                    field_vec = 2 * self._rng.randint(0, 2, size=self.dimension) - 1
                    field_vec = field_vec.astype(np.int8)
                    
                self.field_vectors[field_name] = field_vec
            else:
                field_vec = self.field_vectors[field_name]
                
            # Encode field value
            if field_name in self.field_encoders:
                value_vec = self.field_encoders[field_name].encode(field_value)
            else:
                # Use default categorical encoder
                cat_encoder = CategoricalEncoder(
                    self.dimension,
                    hypervector_type=self.hypervector_type
                )
                value_vec = cat_encoder.encode(str(field_value))
                
            # Bind field name and value
            bound = bind_hypervectors(field_vec, value_vec, self.hypervector_type)
            field_bindings.append(bound)
            
        # Bundle all field bindings
        return bundle_hypervectors(field_bindings, hypervector_type=self.hypervector_type)
        
    def add_field_encoder(self, field_name: str, encoder: Encoder) -> None:
        """Add a specific encoder for a field."""
        self.field_encoders[field_name] = encoder


class NGramEncoder(Encoder):
    """
    Specialized encoder for text using character or word n-grams.
    """
    
    def __init__(
        self,
        dimension: int,
        n: int = 3,
        level: str = "char",
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize n-gram encoder.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        n : int
            N-gram size
        level : str
            "char" for character n-grams, "word" for word n-grams
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, hypervector_type)
        self.n = n
        self.level = level
        self.token_encoder = CategoricalEncoder(dimension, hypervector_type=hypervector_type)
        
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text using n-grams.
        
        Parameters
        ----------
        text : str
            Text to encode
            
        Returns
        -------
        np.ndarray
            Encoded hypervector
        """
        if not text:
            raise ValueError("Cannot encode empty text")
            
        # Tokenize
        if self.level == "char":
            tokens = list(text)
        elif self.level == "word":
            tokens = text.split()
        else:
            raise ValueError(f"Unknown level: {self.level}")
            
        if len(tokens) < self.n:
            # Text shorter than n-gram
            # Pad with special tokens
            tokens.extend(["<PAD>"] * (self.n - len(tokens)))
            
        # Generate n-grams
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram_tokens = tokens[i:i + self.n]
            
            # Encode first token
            ngram_vec = self.token_encoder.encode(ngram_tokens[0])
            
            # Bind with shifted versions of other tokens
            for j in range(1, self.n):
                token_vec = self.token_encoder.encode(ngram_tokens[j])
                shifted = permute_hypervector(token_vec, shift=j)
                ngram_vec = bind_hypervectors(ngram_vec, shifted, self.hypervector_type)
                
            ngrams.append(ngram_vec)
            
        # Bundle all n-grams
        return bundle_hypervectors(ngrams, hypervector_type=self.hypervector_type)