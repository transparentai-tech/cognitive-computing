# VSA API Reference

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

##### `encode(item: Any) -> VSAVector`
Encode an item into a VSA vector.

**Parameters:**
- `item`: Item to encode (string, number, or any hashable)

**Returns:**
- VSA vector representation

**Example:**
```python
apple = vsa.encode('apple')
number = vsa.encode(42)
```

##### `bind(x: VSAVector, y: VSAVector) -> VSAVector`
Bind two vectors using the configured binding operation.

**Parameters:**
- `x`: First vector
- `y`: Second vector

**Returns:**
- Bound vector

**Example:**
```python
red_apple = vsa.bind(red, apple)
```

##### `unbind(xy: VSAVector, y: VSAVector) -> VSAVector`
Unbind a vector to retrieve the other component.

**Parameters:**
- `xy`: Bound vector
- `y`: Known component

**Returns:**
- Retrieved vector

**Example:**
```python
retrieved_apple = vsa.unbind(red_apple, red)
```

##### `bundle(vectors: List[VSAVector], weights: Optional[List[float]] = None) -> VSAVector`
Bundle multiple vectors into a single representation.

**Parameters:**
- `vectors`: List of vectors to bundle
- `weights`: Optional weights for weighted bundling

**Returns:**
- Bundled vector

**Example:**
```python
fruits = vsa.bundle([apple, banana, orange])
weighted = vsa.bundle([v1, v2], weights=[0.7, 0.3])
```

##### `similarity(x: VSAVector, y: VSAVector) -> float`
Compute similarity between two vectors.

**Parameters:**
- `x`: First vector
- `y`: Second vector

**Returns:**
- Similarity score in [-1, 1]

**Example:**
```python
sim = vsa.similarity(apple, fruits)
```

##### `zero() -> VSAVector`
Get the zero vector for the current vector type.

##### `identity() -> VSAVector`
Get the identity vector for the current binding operation.

##### `inverse(x: VSAVector) -> VSAVector`
Get the inverse of a vector for unbinding.

##### `permute(x: VSAVector, n: int) -> VSAVector`
Permute vector elements by n positions.

##### `thin(x: VSAVector, sparsity: float) -> VSAVector`
Create a sparse version of the vector.

### VSAConfig

Configuration for VSA instances.

```python
@dataclass
class VSAConfig(MemoryConfig):
    """Configuration for Vector Symbolic Architecture."""
    
    dimension: int = 10000
    vector_type: str = 'bipolar'
    binding_method: str = 'multiplication'
    similarity_threshold: float = 0.3
    cleanup_threshold: float = 0.7
```

**Fields:**
- `dimension`: Vector dimensionality (default: 10000)
- `vector_type`: Type of vectors ('binary', 'bipolar', 'ternary', 'complex', 'integer')
- `binding_method`: Binding operation ('xor', 'multiplication', 'convolution', 'map', 'permutation')
- `similarity_threshold`: Threshold for similarity detection
- `cleanup_threshold`: Threshold for cleanup memory

## Vector Types

### BinaryVector

Binary vectors with values in {0, 1}.

```python
class BinaryVector(VSAVector):
    """Binary vector implementation."""
```

#### Class Methods

##### `random(dimension: int) -> BinaryVector`
Generate a random binary vector.

##### `zeros(dimension: int) -> BinaryVector`
Create an all-zeros vector.

##### `ones(dimension: int) -> BinaryVector`
Create an all-ones vector.

#### Instance Methods

##### `bind(other: BinaryVector) -> BinaryVector`
XOR binding operation.

##### `similarity(other: BinaryVector) -> float`
Hamming similarity.

##### `to_bipolar() -> BipolarVector`
Convert to bipolar representation.

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

### SequenceEncoder

Encode sequential data.

```python
class SequenceEncoder:
    """Encoder for sequential data."""
    
    def __init__(self, vsa: VSA)
```

#### Methods

##### `encode_sequence(items: List[VSAVector], method: str = 'positional') -> VSAVector`
Encode a sequence of vectors.

**Parameters:**
- `items`: List of vectors
- `method`: Encoding method ('positional', 'chaining', 'temporal')

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

## Factory Functions

### create_vsa

Create a VSA instance with configuration.

```python
def create_vsa(config: VSAConfig) -> VSA:
    """Create VSA instance from configuration."""
```

**Example:**
```python
vsa = create_vsa(VSAConfig(
    dimension=10000,
    vector_type='bipolar',
    binding_method='multiplication'
))
```

### create_architecture

Create a specific VSA architecture.

```python
def create_architecture(name: str, **kwargs) -> VSA:
    """Create a specific VSA architecture."""
```

**Parameters:**
- `name`: Architecture name ('bsc', 'map', 'fhrr', 'sparse', 'hrr')
- `**kwargs`: Architecture-specific parameters

**Example:**
```python
bsc = create_architecture('bsc', dimension=10000)
sparse = create_architecture('sparse', dimension=10000, sparsity=0.05)
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
DEFAULT_DIMENSION = 10000
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