# HRR API Reference

## Core Classes

### HRRConfig

Configuration dataclass for HRR systems.

```python
@dataclass
class HRRConfig(MemoryConfig):
    """Configuration for HRR system.
    
    Attributes
    ----------
    dimension : int
        Dimensionality of vectors (default: 1024)
    normalize : bool
        Whether to normalize vectors after operations (default: True)
    cleanup_threshold : float
        Similarity threshold for cleanup memory (default: 0.3)
    storage_method : str
        Method for storing vectors: "real" or "complex" (default: "real")
    seed : Optional[int]
        Random seed for reproducibility (default: None)
    """
```

**Example:**
```python
config = HRRConfig(
    dimension=2048,
    normalize=True,
    cleanup_threshold=0.25,
    storage_method="complex",
    seed=42
)
```

### HRR

Main HRR implementation class.

```python
class HRR(CognitiveMemory):
    """Holographic Reduced Representation system.
    
    Parameters
    ----------
    config : HRRConfig
        Configuration object
        
    Attributes
    ----------
    dimension : int
        Vector dimensionality
    normalize : bool
        Whether normalization is enabled
    storage_method : str
        Storage method in use
    """
```

#### Methods

##### bind(a, b)
```python
def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two vectors using circular convolution.
    
    Parameters
    ----------
    a : np.ndarray
        First vector (shape: [dimension,])
    b : np.ndarray
        Second vector (shape: [dimension,])
        
    Returns
    -------
    np.ndarray
        Bound vector (shape: [dimension,])
        
    Examples
    --------
    >>> hrr = create_hrr(dimension=1024)
    >>> role = hrr.generate_vector()
    >>> filler = hrr.generate_vector()
    >>> binding = hrr.bind(role, filler)
    """
```

##### unbind(c, a)
```python
def unbind(self, c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Unbind vectors using circular correlation.
    
    Parameters
    ----------
    c : np.ndarray
        Composite vector (shape: [dimension,])
    a : np.ndarray
        Known vector to unbind with (shape: [dimension,])
        
    Returns
    -------
    np.ndarray
        Retrieved vector (shape: [dimension,])
        
    Examples
    --------
    >>> retrieved = hrr.unbind(binding, role)
    >>> similarity = hrr.similarity(retrieved, filler)
    """
```

##### bundle(vectors, weights=None)
```python
def bundle(self, vectors: List[np.ndarray], 
          weights: Optional[List[float]] = None) -> np.ndarray:
    """Bundle multiple vectors by superposition.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of vectors to bundle
    weights : Optional[List[float]]
        Optional weights for each vector
        
    Returns
    -------
    np.ndarray
        Bundled vector (shape: [dimension,])
        
    Examples
    --------
    >>> bundle = hrr.bundle([vec1, vec2, vec3])
    >>> weighted = hrr.bundle([vec1, vec2], weights=[0.7, 0.3])
    """
```

##### similarity(a, b)
```python
def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between vectors.
    
    Parameters
    ----------
    a : np.ndarray
        First vector
    b : np.ndarray
        Second vector
        
    Returns
    -------
    float
        Cosine similarity in range [-1, 1]
    """
```

##### generate_vector(method="random")
```python
def generate_vector(self, method: str = "random") -> np.ndarray:
    """Generate a new vector.
    
    Parameters
    ----------
    method : str
        Generation method: "random" or "unitary"
        
    Returns
    -------
    np.ndarray
        Generated vector (shape: [dimension,])
    """
```

##### store(key, value)
```python
def store(self, key: np.ndarray, value: np.ndarray) -> None:
    """Store key-value pair in memory.
    
    Parameters
    ----------
    key : np.ndarray
        Key vector
    value : np.ndarray
        Value vector
    """
```

##### recall(key)
```python
def recall(self, key: np.ndarray) -> Optional[np.ndarray]:
    """Recall value associated with key.
    
    Parameters
    ----------
    key : np.ndarray
        Query key
        
    Returns
    -------
    Optional[np.ndarray]
        Retrieved value or None
    """
```

## Operations Module

### CircularConvolution

Efficient circular convolution implementation.

```python
class CircularConvolution:
    """Circular convolution operations for HRR."""
```

#### Static Methods

##### convolve(a, b, method="auto")
```python
@staticmethod
def convolve(a: np.ndarray, b: np.ndarray, 
            method: str = "auto") -> np.ndarray:
    """Perform circular convolution.
    
    Parameters
    ----------
    a : np.ndarray
        First vector
    b : np.ndarray
        Second vector
    method : str
        Method: "direct", "fft", or "auto"
        
    Returns
    -------
    np.ndarray
        Convolution result
    """
```

##### correlate(a, b, method="auto")
```python
@staticmethod
def correlate(a: np.ndarray, b: np.ndarray,
             method: str = "auto") -> np.ndarray:
    """Perform circular correlation.
    
    Parameters
    ----------
    a : np.ndarray
        First vector
    b : np.ndarray
        Second vector
    method : str
        Method: "direct", "fft", or "auto"
        
    Returns
    -------
    np.ndarray
        Correlation result
    """
```

### VectorOperations

Additional vector operations for HRR.

```python
class VectorOperations:
    """Utility operations for HRR vectors."""
```

#### Static Methods

##### normalize(vector)
```python
@staticmethod
def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector
        
    Returns
    -------
    np.ndarray
        Normalized vector
    """
```

##### make_unitary(vector)
```python
@staticmethod
def make_unitary(vector: np.ndarray) -> np.ndarray:
    """Convert vector to unitary form.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector
        
    Returns
    -------
    np.ndarray
        Unitary vector
    """
```

##### involution(vector)
```python
@staticmethod
def involution(vector: np.ndarray) -> np.ndarray:
    """Compute involution (pseudo-inverse) of vector.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector
        
    Returns
    -------
    np.ndarray
        Involution of vector
    """
```

## Cleanup Memory

### CleanupMemoryConfig

Configuration for cleanup memory.

```python
@dataclass
class CleanupMemoryConfig:
    """Configuration for cleanup memory.
    
    Attributes
    ----------
    threshold : float
        Similarity threshold for cleanup (default: 0.3)
    method : str
        Similarity method: "cosine", "dot", "euclidean" (default: "cosine")
    """
```

### CleanupMemory

Item memory for cleaning up noisy vectors.

```python
class CleanupMemory:
    """Memory for mapping noisy vectors to clean items.
    
    Parameters
    ----------
    config : CleanupMemoryConfig
        Configuration object
    dimension : int
        Vector dimensionality
    """
```

#### Methods

##### add_item(name, vector)
```python
def add_item(self, name: str, vector: np.ndarray) -> None:
    """Add an item to cleanup memory.
    
    Parameters
    ----------
    name : str
        Item identifier
    vector : np.ndarray
        Item vector
        
    Examples
    --------
    >>> cleanup = CleanupMemory(config, dimension=1024)
    >>> cleanup.add_item("cat", cat_vector)
    >>> cleanup.add_item("dog", dog_vector)
    """
```

##### cleanup(vector)
```python
def cleanup(self, vector: np.ndarray) -> Tuple[str, np.ndarray, float]:
    """Clean up noisy vector to nearest item.
    
    Parameters
    ----------
    vector : np.ndarray
        Noisy input vector
        
    Returns
    -------
    Tuple[str, np.ndarray, float]
        (item_name, clean_vector, confidence)
        
    Examples
    --------
    >>> name, clean_vec, conf = cleanup.cleanup(noisy_vector)
    >>> print(f"Cleaned to {name} with confidence {conf:.3f}")
    """
```

##### find_closest(vector, k=1)
```python
def find_closest(self, vector: np.ndarray, 
                k: int = 1) -> List[Tuple[str, float]]:
    """Find k closest items to vector.
    
    Parameters
    ----------
    vector : np.ndarray
        Query vector
    k : int
        Number of items to return
        
    Returns
    -------
    List[Tuple[str, float]]
        List of (item_name, similarity) pairs
    """
```

##### has_item(name)
```python
def has_item(self, name: str) -> bool:
    """Check if item exists in memory.
    
    Parameters
    ----------
    name : str
        Item name
        
    Returns
    -------
    bool
        True if item exists
    """
```

##### get_vector(name)
```python
def get_vector(self, name: str) -> np.ndarray:
    """Get vector for named item.
    
    Parameters
    ----------
    name : str
        Item name
        
    Returns
    -------
    np.ndarray
        Item vector
        
    Raises
    ------
    KeyError
        If item not found
    """
```

## Encoding Classes

### RoleFillerEncoder

Encode role-filler structures.

```python
class RoleFillerEncoder:
    """Encoder for role-filler structures.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to use for encoding
    """
```

#### Methods

##### encode_pair(role, filler)
```python
def encode_pair(self, role: np.ndarray, 
               filler: np.ndarray) -> np.ndarray:
    """Encode a single role-filler pair.
    
    Parameters
    ----------
    role : np.ndarray
        Role vector
    filler : np.ndarray
        Filler vector
        
    Returns
    -------
    np.ndarray
        Bound role-filler pair
    """
```

##### encode_structure(role_filler_pairs)
```python
def encode_structure(self, 
                    role_filler_pairs: Dict[str, np.ndarray]) -> np.ndarray:
    """Encode complete role-filler structure.
    
    Parameters
    ----------
    role_filler_pairs : Dict[str, np.ndarray]
        Dictionary mapping role names to filler vectors
        
    Returns
    -------
    np.ndarray
        Encoded structure
        
    Examples
    --------
    >>> structure = encoder.encode_structure({
    ...     "agent": john_vector,
    ...     "action": loves_vector,
    ...     "patient": mary_vector
    ... })
    """
```

##### decode_filler(structure, role)
```python
def decode_filler(self, structure: np.ndarray, 
                 role: np.ndarray) -> np.ndarray:
    """Extract filler for given role.
    
    Parameters
    ----------
    structure : np.ndarray
        Encoded structure
    role : np.ndarray
        Query role
        
    Returns
    -------
    np.ndarray
        Retrieved filler
    """
```

### SequenceEncoder

Encode sequences using HRR.

```python
class SequenceEncoder:
    """Encoder for sequential data.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to use
    """
```

#### Methods

##### encode_sequence(items, method="position")
```python
def encode_sequence(self, items: List[np.ndarray], 
                   method: str = "position") -> np.ndarray:
    """Encode ordered sequence of items.
    
    Parameters
    ----------
    items : List[np.ndarray]
        List of item vectors
    method : str
        Encoding method: "position" or "chaining"
        
    Returns
    -------
    np.ndarray
        Encoded sequence
        
    Examples
    --------
    >>> seq = encoder.encode_sequence([a, b, c], method="position")
    """
```

##### decode_position(sequence, position)
```python
def decode_position(self, sequence: np.ndarray, 
                   position: int) -> np.ndarray:
    """Decode item at specific position.
    
    Parameters
    ----------
    sequence : np.ndarray
        Encoded sequence
    position : int
        Position to decode (0-indexed)
        
    Returns
    -------
    np.ndarray
        Retrieved item
    """
```

##### generate_position_vectors(n_positions)
```python
def generate_position_vectors(self, 
                            n_positions: int) -> List[np.ndarray]:
    """Generate position vectors for encoding.
    
    Parameters
    ----------
    n_positions : int
        Number of positions needed
        
    Returns
    -------
    List[np.ndarray]
        List of position vectors
    """
```

### HierarchicalEncoder

Encode hierarchical structures.

```python
class HierarchicalEncoder:
    """Encoder for tree and hierarchical structures.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to use
    """
```

#### Methods

##### encode_tree(tree)
```python
def encode_tree(self, tree: Dict[str, Any]) -> np.ndarray:
    """Encode tree structure recursively.
    
    Parameters
    ----------
    tree : Dict[str, Any]
        Tree structure as nested dictionary
        
    Returns
    -------
    np.ndarray
        Encoded tree
        
    Examples
    --------
    >>> tree = {
    ...     "value": "root",
    ...     "left": {"value": "a"},
    ...     "right": {"value": "b"}
    ... }
    >>> encoded = encoder.encode_tree(tree)
    """
```

##### decode_subtree(encoding, path)
```python
def decode_subtree(self, encoding: np.ndarray, 
                  path: List[str]) -> np.ndarray:
    """Extract subtree at given path.
    
    Parameters
    ----------
    encoding : np.ndarray
        Encoded tree
    path : List[str]
        Path to subtree (e.g., ["left", "right"])
        
    Returns
    -------
    np.ndarray
        Subtree encoding
    """
```

## Utility Functions

### Vector Generation

##### generate_random_vector(dimension, method="gaussian")
```python
def generate_random_vector(dimension: int, 
                         method: str = "gaussian") -> np.ndarray:
    """Generate random vector.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    method : str
        Generation method: "gaussian", "uniform", "sparse"
        
    Returns
    -------
    np.ndarray
        Random vector
    """
```

##### generate_unitary_vector(dimension)
```python
def generate_unitary_vector(dimension: int) -> np.ndarray:
    """Generate unitary (self-inverse) vector.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
        
    Returns
    -------
    np.ndarray
        Unitary vector
    """
```

##### generate_orthogonal_set(dimension, n_vectors)
```python
def generate_orthogonal_set(dimension: int, 
                          n_vectors: int) -> np.ndarray:
    """Generate set of orthogonal vectors.
    
    Parameters
    ----------
    dimension : int
        Vector dimension
    n_vectors : int
        Number of vectors
        
    Returns
    -------
    np.ndarray
        Array of shape (n_vectors, dimension)
    """
```

### Analysis Functions

##### analyze_binding_capacity(hrr, n_pairs)
```python
def analyze_binding_capacity(hrr: HRR, 
                           n_pairs: int) -> Dict[str, float]:
    """Analyze binding capacity of HRR system.
    
    Parameters
    ----------
    hrr : HRR
        HRR system to analyze
    n_pairs : int
        Number of role-filler pairs to test
        
    Returns
    -------
    Dict[str, float]
        Analysis results including mean/min/max similarity
    """
```

##### measure_crosstalk(hrr, vectors)
```python
def measure_crosstalk(hrr: HRR, 
                     vectors: List[np.ndarray]) -> float:
    """Measure crosstalk between vectors.
    
    Parameters
    ----------
    hrr : HRR
        HRR system
    vectors : List[np.ndarray]
        Vectors to test
        
    Returns
    -------
    float
        Average crosstalk level
    """
```

##### measure_associative_capacity(hrr, n_items)
```python
def measure_associative_capacity(hrr: HRR, 
                               n_items: int) -> Dict[str, Any]:
    """Test associative memory capacity.
    
    Parameters
    ----------
    hrr : HRR
        HRR system
    n_items : int
        Number of associations to test
        
    Returns
    -------
    Dict[str, Any]
        Test results
    """
```

### Conversion Utilities

##### to_complex(vector)
```python
def to_complex(vector: np.ndarray) -> np.ndarray:
    """Convert real vector to complex representation.
    
    Parameters
    ----------
    vector : np.ndarray
        Real-valued vector
        
    Returns
    -------
    np.ndarray
        Complex-valued vector
    """
```

##### from_complex(vector)
```python
def from_complex(vector: np.ndarray) -> np.ndarray:
    """Convert complex vector to real representation.
    
    Parameters
    ----------
    vector : np.ndarray
        Complex-valued vector
        
    Returns
    -------
    np.ndarray
        Real-valued vector
    """
```

## Visualization Functions

##### plot_similarity_matrix(vectors, figsize=(10, 8))
```python
def plot_similarity_matrix(vectors: Dict[str, np.ndarray],
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Plot similarity matrix heatmap.
    
    Parameters
    ----------
    vectors : Dict[str, np.ndarray]
        Dictionary of named vectors
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
```

##### plot_binding_accuracy(hrr, test_results, figsize=(10, 6))
```python
def plot_binding_accuracy(hrr: HRR, 
                        test_results: Dict[str, Any],
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot binding accuracy results.
    
    Parameters
    ----------
    hrr : HRR
        HRR system
    test_results : Dict[str, Any]
        Results from capacity analysis
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
```

##### visualize_cleanup_space(cleanup_memory, method="pca")
```python
def visualize_cleanup_space(cleanup_memory: CleanupMemory,
                          method: str = "pca") -> plt.Figure:
    """Visualize cleanup memory space.
    
    Parameters
    ----------
    cleanup_memory : CleanupMemory
        Cleanup memory to visualize
    method : str
        Dimensionality reduction: "pca" or "tsne"
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
```

## Factory Functions

##### create_hrr(dimension=1024, **kwargs)
```python
def create_hrr(dimension: int = 1024, **kwargs) -> HRR:
    """Create HRR system with specified parameters.
    
    Parameters
    ----------
    dimension : int
        Vector dimension (default: 1024)
    **kwargs
        Additional arguments for HRRConfig
        
    Returns
    -------
    HRR
        Configured HRR system
        
    Examples
    --------
    >>> hrr = create_hrr(dimension=2048, normalize=True)
    >>> hrr = create_hrr(storage_method="complex")
    """
```

## Exception Classes

### HRRError
```python
class HRRError(Exception):
    """Base exception for HRR operations."""
```

### DimensionMismatchError
```python
class DimensionMismatchError(HRRError):
    """Raised when vector dimensions don't match."""
```

### CleanupError
```python
class CleanupError(HRRError):
    """Raised when cleanup operation fails."""
```

## Type Aliases

```python
# Type aliases for clarity
VectorArray = np.ndarray  # Shape: (dimension,)
VectorSet = np.ndarray    # Shape: (n_vectors, dimension)
SimilarityMatrix = np.ndarray  # Shape: (n, n)
```

## Constants

```python
# Default values
DEFAULT_DIMENSION = 1024
DEFAULT_THRESHOLD = 0.3
DEFAULT_SEED = None

# Method options
STORAGE_METHODS = ["real", "complex"]
SIMILARITY_METHODS = ["cosine", "dot", "euclidean"]
ENCODING_METHODS = ["position", "chaining"]
VECTOR_METHODS = ["gaussian", "uniform", "sparse"]
```