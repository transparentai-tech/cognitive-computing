# VSA API Reference

## Important API Design Notes

**Array-Based API**: VSA uses numpy arrays directly in its public API, not vector objects. This design choice ensures:
- Consistency with SDM and HRR modules
- Better performance (no object overhead)
- Easier integration with scientific computing libraries
- Direct manipulation of vector data when needed

**No Factory Functions for Architectures**: Use architecture classes directly:
```python
# Correct
from cognitive_computing.vsa import BSC, MAP, FHRR
bsc = BSC(dimension=1000)
map_arch = MAP(dimension=1000)

# Incorrect (no create_architecture function)
# arch = create_architecture('bsc', dimension=1000)  # Does not exist
```

**Vector Generation**: Use `generate_vector()` method or utility functions:
```python
# Generate vectors
vector = vsa.generate_vector()
sparse_vector = vsa.generate_vector(sparse=True)

# No encode() method for arbitrary items
# apple = vsa.encode('apple')  # Does not exist
```

## Core Classes

### VSA

The main VSA class providing core operations.

```python
class VSA(CognitiveMemory):
    """Base Vector Symbolic Architecture implementation."""
```

#### Methods

##### `__init__(config: VSAConfig)`
Initialize a VSA instance with the specified configuration.

##### `generate_vector(sparse: Optional[bool] = None) -> np.ndarray`
Generate a random vector of the configured type.

**Parameters:**
- `sparse`: Whether to generate a sparse vector (overrides config)

**Returns:**
- NumPy array representing the vector

**Example:**
```python
apple = vsa.generate_vector()
sparse_vec = vsa.generate_vector(sparse=True)
```

##### `bind(x: np.ndarray, y: np.ndarray) -> np.ndarray`
Bind two vectors using the configured binding operation.

**Parameters:**
- `x`: First vector (numpy array)
- `y`: Second vector (numpy array)

**Returns:**
- Bound vector (numpy array)

**Example:**
```python
red_apple = vsa.bind(red, apple)
```

##### `unbind(xy: np.ndarray, y: np.ndarray) -> np.ndarray`
Unbind a vector to retrieve the other component.

**Parameters:**
- `xy`: Bound vector (numpy array)
- `y`: Known component (numpy array)

**Returns:**
- Retrieved vector (numpy array)

**Example:**
```python
retrieved_apple = vsa.unbind(red_apple, red)
```

##### `bundle(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray`
Bundle multiple vectors into a single representation.

**Parameters:**
- `vectors`: List of vectors to bundle (numpy arrays)
- `weights`: Optional weights for weighted bundling

**Returns:**
- Bundled vector (numpy array)

**Example:**
```python
fruits = vsa.bundle([apple, banana, orange])
weighted = vsa.bundle([v1, v2], weights=[0.7, 0.3])
```

##### `similarity(x: np.ndarray, y: np.ndarray) -> float`
Compute similarity between two vectors.

**Parameters:**
- `x`: First vector (numpy array)
- `y`: Second vector (numpy array)

**Returns:**
- Similarity score in [-1, 1]

**Example:**
```python
sim = vsa.similarity(apple, fruits)
```

##### `permute(vector: np.ndarray, permutation: Optional[np.ndarray] = None, shift: Optional[int] = None) -> np.ndarray`
Permute elements of a vector.

**Parameters:**
- `vector`: Vector to permute (numpy array)
- `permutation`: Explicit permutation array (optional)
- `shift`: Cyclic shift amount (takes precedence over permutation)

**Returns:**
- Permuted vector (numpy array)

**Example:**
```python
shifted = vsa.permute(vector, shift=1)
rev_shifted = vsa.permute(shifted, shift=-1)
```

##### `thin(vector: np.ndarray, rate: float) -> np.ndarray`
Apply thinning to a vector by randomly zeroing elements.

**Parameters:**
- `vector`: Vector to thin (numpy array)
- `rate`: Thinning rate (0-1), proportion of elements to zero

**Returns:**
- Thinned vector (numpy array)

**Example:**
```python
sparse = vsa.thin(dense_vector, rate=0.9)  # Zero out 90%, keep 10% of elements
```

**Note**: The `rate` parameter specifies the fraction to zero out, not the sparsity level. Rate=0.9 means 90% zeros, 10% non-zero.


### VSAConfig

Configuration for VSA instances.

```python
@dataclass
class VSAConfig(MemoryConfig):
    """Configuration for Vector Symbolic Architecture."""
    
    dimension: int = 1000
    vector_type: str = 'bipolar'
    vsa_type: str = 'map'
    binding_method: Optional[str] = None
    normalize_result: bool = True
    sparsity: float = 0.0
    cleanup_threshold: float = 0.3
```

**Fields:**
- `dimension`: Vector dimensionality (default: 1000)
- `vector_type`: Type of vectors ('binary', 'bipolar', 'ternary', 'complex', 'integer')
- `vsa_type`: VSA architecture type ('bsc', 'map', 'fhrr', 'hrr', 'sparse', 'custom')
- `binding_method`: Binding operation ('xor', 'multiplication', 'convolution', 'map', 'permutation'). If None, uses architecture default
- `normalize_result`: Whether to normalize vectors after operations (default: True)
- `sparsity`: Sparsity level for sparse vectors (0-1, where 0 is dense)
- `cleanup_threshold`: Threshold for cleanup memory (default: 0.3)

## Vector Types

### Vector Types

VSA works with numpy arrays directly. Vector types are specified in the configuration and determine the internal representation and operations. The `generate_random_vector` utility function can create vectors of any type:

```python
from cognitive_computing.vsa import generate_random_vector, BinaryVector, BipolarVector

# Generate different vector types
binary_vec = generate_random_vector(1000, BinaryVector)
bipolar_vec = generate_random_vector(1000, BipolarVector)
ternary_vec = generate_random_vector(1000, TernaryVector, sparsity=0.1)
complex_vec = generate_random_vector(1000, ComplexVector)
integer_vec = generate_random_vector(1000, IntegerVector, modulus=256)
```

#### BinaryVector
- Values: {0, 1}
- Default binding: XOR
- Similarity: Hamming distance
- Use case: Hardware-efficient implementations

#### BipolarVector  
- Values: {-1, +1}
- Default binding: Multiplication
- Similarity: Cosine similarity
- Use case: General-purpose VSA

#### TernaryVector
- Values: {-1, 0, +1}
- Sparse representation
- Default binding: Multiplication
- Use case: Memory-efficient sparse coding

#### ComplexVector
- Values: Complex numbers on unit circle
- Default binding: Complex multiplication
- Similarity: Complex dot product
- Use case: Frequency domain operations

#### IntegerVector
- Values: Integers modulo N
- Default binding: Modular addition
- Similarity: Modular distance
- Use case: Discrete symbolic operations

### BipolarVector

Bipolar vectors with values in {-1, +1}.

```python
class BipolarVector(VSAVector):
    """Bipolar vector implementation."""
```

#### Class Methods

##### `random(dimension: int) -> BipolarVector`
Generate a random bipolar vector.

##### `from_binary(binary_vec: BinaryVector) -> BipolarVector`
Convert from binary vector.

#### Instance Methods

##### `bind(other: BipolarVector) -> BipolarVector`
Element-wise multiplication.

##### `normalize() -> BipolarVector`
Normalize to unit length.

### TernaryVector

Sparse ternary vectors with values in {-1, 0, +1}.

```python
class TernaryVector(VSAVector):
    """Ternary sparse vector implementation."""
```

#### Class Methods

##### `random(dimension: int, sparsity: float = 0.1) -> TernaryVector`
Generate a random sparse ternary vector.

**Parameters:**
- `dimension`: Vector dimension
- `sparsity`: Fraction of non-zero elements

### ComplexVector

Complex vectors with unit magnitude.

```python
class ComplexVector(VSAVector):
    """Complex unit vector implementation."""
```

#### Class Methods

##### `random(dimension: int) -> ComplexVector`
Generate random complex unit vector.

#### Instance Methods

##### `bind(other: ComplexVector) -> ComplexVector`
Complex multiplication (phase addition).

##### `convolve(other: ComplexVector) -> ComplexVector`
Circular convolution.

### IntegerVector

Integer vectors with modular arithmetic.

```python
class IntegerVector(VSAVector):
    """Integer modular vector implementation."""
```

#### Class Methods

##### `random(dimension: int, modulus: int = 256) -> IntegerVector`
Generate random integer vector.

**Parameters:**
- `dimension`: Vector dimension
- `modulus`: Modular arithmetic base

## Binding Operations

### XORBinding

XOR binding for binary vectors.

```python
class XORBinding(BindingOperation):
    """XOR binding operation."""
    
    def bind(self, x: BinaryVector, y: BinaryVector) -> BinaryVector
    def unbind(self, xy: BinaryVector, y: BinaryVector) -> BinaryVector
```

### MultiplicationBinding

Element-wise multiplication for bipolar/complex vectors.

```python
class MultiplicationBinding(BindingOperation):
    """Multiplication binding operation."""
    
    def bind(self, x: VSAVector, y: VSAVector) -> VSAVector
    def unbind(self, xy: VSAVector, y: VSAVector) -> VSAVector
```

### ConvolutionBinding

Circular convolution binding.

```python
class ConvolutionBinding(BindingOperation):
    """Circular convolution binding."""
    
    def bind(self, x: VSAVector, y: VSAVector) -> VSAVector
    def unbind(self, xy: VSAVector, y: VSAVector) -> VSAVector
```

### MAPBinding

Multiply-Add-Permute binding.

```python
class MAPBinding(BindingOperation):
    """MAP (Multiply-Add-Permute) binding."""
    
    def __init__(self, dimension: int, selection_ratio: float = 0.5)
    def bind(self, x: VSAVector, y: VSAVector) -> VSAVector
```

### PermutationBinding

Permutation-based binding.

```python
class PermutationBinding(BindingOperation):
    """Permutation-based binding."""
    
    def __init__(self, dimension: int)
    def bind(self, x: VSAVector, y: VSAVector) -> VSAVector
```

## Encoders

### RandomIndexingEncoder

Encode data using random indexing.

```python
class RandomIndexingEncoder:
    """Random indexing encoder for text and sequences."""
    
    def __init__(self, vsa: VSA, n_gram_size: int = 3, 
                 window_size: int = 5)
```

#### Methods

##### `encode(text: str) -> VSAVector`
Encode text using random indexing.

##### `encode_ngrams(text: str) -> List[VSAVector]`
Encode text as n-grams.

**Note**: RandomIndexingEncoder handles sequence encoding. There is no separate SequenceEncoder class.

### SpatialEncoder

Encode spatial coordinates.

```python
class SpatialEncoder:
    """Encoder for spatial data."""
    
    def __init__(self, vsa: VSA, grid_size: Tuple[int, ...])
```

#### Methods

##### `encode_2d(x: int, y: int) -> VSAVector`
Encode 2D coordinates.

##### `encode_3d(x: int, y: int, z: int) -> VSAVector`
Encode 3D coordinates.

### TemporalEncoder

Encode temporal data.

```python
class TemporalEncoder:
    """Encoder for temporal data."""
    
    def __init__(self, vsa: VSA, max_lag: int = 10)
```

#### Methods

##### `encode_time_point(t: int) -> VSAVector`
Encode a discrete time point.

##### `encode_time_series(values: List[float], timestamps: List[int]) -> VSAVector`
Encode a time series.

### LevelEncoder

Encode continuous values as discrete levels.

```python
class LevelEncoder:
    """Encoder for continuous values using levels."""
    
    def __init__(self, vsa: VSA, num_levels: int = 10,
                 min_value: float = 0.0, max_value: float = 1.0)
```

#### Methods

##### `encode(value: float) -> VSAVector`
Encode a continuous value.

##### `decode(vector: VSAVector) -> int`
Decode to nearest level.

##### `level_to_value(level: int) -> float`
Convert level back to continuous value.

### GraphEncoder

Encode graph structures.

```python
class GraphEncoder:
    """Encoder for graph structures."""
    
    def __init__(self, vsa: VSA)
```

#### Methods

##### `encode_edge(source: VSAVector, target: VSAVector) -> VSAVector`
Encode a directed edge.

##### `encode_path(nodes: List[VSAVector]) -> VSAVector`
Encode a path through nodes.

##### `encode_graph(edges: List[Tuple[VSAVector, VSAVector]]) -> VSAVector`
Encode entire graph structure.

## Architectures

### BSC (Binary Spatter Codes)

```python
class BSC(VSA):
    """Binary Spatter Codes architecture."""
    
    def __init__(self, dimension: int = 10000)
```

Optimized for:
- Hardware implementation
- Minimal memory usage
- XOR binding operations

### MAP Architecture

```python
class MAP(VSA):
    """Multiply-Add-Permute architecture."""
    
    def __init__(self, dimension: int = 10000,
                 selection_ratio: float = 0.5)
```

Optimized for:
- Noise robustness
- Multiple bindings
- Cognitive modeling

**Note**: MAP unbinding is approximate. Expect similarity ~0.3 after unbinding due to the permutation additions.

### FHRR (Fourier HRR)

```python
class FHRR(VSA):
    """Fourier Holographic Reduced Representation."""
    
    def __init__(self, dimension: int = 10000)
```

Optimized for:
- Frequency domain operations
- HRR compatibility
- Complex vectors

**Note**: FHRR uses unit norm vectors (not unit magnitude per element). Expect similarity ~0.35-0.4 after unbinding due to FFT-based convolution.

### SparseVSA

```python
class SparseVSA(VSA):
    """Sparse Vector Symbolic Architecture."""
    
    def __init__(self, dimension: int = 10000,
                 sparsity: float = 0.05)
```

Optimized for:
- Memory efficiency
- Large-scale systems
- Biological plausibility

**Note**: The `sparsity` parameter represents the fraction of zeros (e.g., 0.95 = 95% zeros, 5% non-zero).

## Factory Functions

### create_vsa

Create a VSA instance with the specified parameters.

```python
def create_vsa(dimension: int, 
               vector_type: Union[str, VectorType],
               vsa_type: Union[str, VSAType],
               **kwargs) -> VSA:
    """Create VSA instance with configuration."""
```

**Parameters:**
- `dimension`: Vector dimension
- `vector_type`: Type of vectors ('binary', 'bipolar', 'ternary', 'complex', 'integer')
- `vsa_type`: VSA architecture ('bsc', 'map', 'fhrr', 'hrr', 'sparse', 'custom')
- `**kwargs`: Additional configuration parameters

**Example:**
```python
# Create Binary Spatter Codes
vsa = create_vsa(
    dimension=1000,
    vector_type='binary',
    vsa_type='bsc'
)

# Create MAP architecture
vsa = create_vsa(
    dimension=1000,
    vector_type='bipolar',
    vsa_type='map'
)

# Create custom VSA with specific binding
vsa = create_vsa(
    dimension=1000,
    vector_type='bipolar',
    vsa_type='custom',
    binding_method='multiplication'
)
```

### Architecture Classes

Direct instantiation of architecture classes:

```python
# Binary Spatter Codes
bsc = BSC(dimension=1000)

# MAP Architecture  
map_arch = MAP(dimension=1000)

# Fourier HRR
fhrr = FHRR(dimension=1000)

# Sparse VSA
sparse = SparseVSA(dimension=1000, sparsity=0.95)
```

## Utility Functions

### analyze_capacity

Analyze the capacity of a VSA system.

```python
def analyze_capacity(vsa: VSA, num_items: int = 100,
                    num_trials: int = 10) -> Dict[str, float]:
    """Analyze storage and retrieval capacity."""
```

**Returns:**
- Dictionary with capacity metrics

### compare_architectures

Compare different VSA architectures.

```python
def compare_architectures(architectures: List[VSA],
                         benchmark: str = 'all') -> pd.DataFrame:
    """Compare performance of different architectures."""
```

**Parameters:**
- `architectures`: List of VSA instances
- `benchmark`: Type of benchmark ('binding', 'capacity', 'noise', 'all')

### optimize_dimension

Find optimal dimension for given requirements.

```python
def optimize_dimension(num_items: int, 
                      error_rate: float = 0.01) -> int:
    """Calculate optimal dimension for storage requirements."""
```

## Visualization Functions

### plot_similarity_matrix

Plot similarity matrix between vectors.

```python
def plot_similarity_matrix(vectors: List[VSAVector],
                          labels: Optional[List[str]] = None) -> None:
    """Plot similarity matrix heatmap."""
```

### plot_vector_space

Visualize vectors in 2D/3D space.

```python
def plot_vector_space(vectors: List[VSAVector],
                     method: str = 'pca',
                     labels: Optional[List[str]] = None) -> None:
    """Plot vectors in reduced dimensional space."""
```

**Parameters:**
- `vectors`: List of vectors to plot
- `method`: Dimensionality reduction ('pca', 'tsne', 'umap')
- `labels`: Optional labels for vectors

### plot_binding_operation

Visualize binding operation effects.

```python
def plot_binding_operation(x: VSAVector, y: VSAVector,
                          operation: str = 'multiplication') -> None:
    """Visualize the effect of binding operations."""
```

## Exception Classes

### VSAError

Base exception for VSA-related errors.

```python
class VSAError(Exception):
    """Base exception for VSA errors."""
```

### DimensionMismatchError

Raised when vector dimensions don't match.

```python
class DimensionMismatchError(VSAError):
    """Raised when vector dimensions don't match."""
```

### UnsupportedOperationError

Raised when operation not supported for vector type.

```python
class UnsupportedOperationError(VSAError):
    """Raised when operation not supported."""
```

## Constants

```python
# Default dimensions
DEFAULT_DIMENSION = 1000
MIN_DIMENSION = 100
MAX_DIMENSION = 100000

# Vector type constants
VECTOR_TYPES = ['binary', 'bipolar', 'ternary', 'complex', 'integer']

# Binding method constants
BINDING_METHODS = ['xor', 'multiplication', 'convolution', 'map', 'permutation']

# Architecture names
ARCHITECTURES = ['bsc', 'map', 'fhrr', 'sparse', 'hrr']
```

## Type Definitions

```python
from typing import TypeVar, Union

# VSA vector type
VSAVectorType = TypeVar('VSAVectorType', bound='VSAVector')

# Numeric types for encoding
Numeric = Union[int, float, complex]

# Item types that can be encoded
Encodable = Union[str, Numeric, tuple, Any]
```