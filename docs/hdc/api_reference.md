# HDC API Reference

## Core Classes

### HDCConfig

Configuration class for HDC systems.

```python
@dataclass
class HDCConfig(MemoryConfig):
    """Configuration for HDC system."""
    dimension: int = 10000
    hypervector_type: str = "bipolar"
    sparsity: float = 0.5
    levels: int = 3
    seed: Optional[int] = None
```

**Parameters:**
- `dimension`: Dimensionality of hypervectors (default: 10000)
- `hypervector_type`: Type of hypervectors ("binary", "bipolar", "ternary", "level")
- `sparsity`: Sparsity level for sparse vectors (0.0 to 1.0)
- `levels`: Number of levels for level hypervectors
- `seed`: Random seed for reproducibility

### HDC

Main HDC class implementing core operations.

```python
class HDC(CognitiveMemory):
    def __init__(self, config: HDCConfig)
```

**Methods:**

#### generate_hypervector
```python
def generate_hypervector(self, orthogonal_to: Optional[List[np.ndarray]] = None) -> np.ndarray
```
Generate a random hypervector, optionally orthogonal to given vectors.

#### bind
```python
def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray
```
Bind two hypervectors using appropriate operation for the vector type.

#### unbind
```python
def unbind(self, composite: np.ndarray, known: np.ndarray) -> np.ndarray
```
Unbind a known hypervector from a composite (inverse of bind).

#### bundle
```python
def bundle(self, hypervectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray
```
Bundle multiple hypervectors into a single hypervector.

#### permute
```python
def permute(self, hypervector: np.ndarray, shift: int = 1, inverse: bool = False) -> np.ndarray
```
Permute hypervector elements by cyclic shift.

#### similarity
```python
def similarity(self, a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float
```
Calculate similarity between two hypervectors.

## Hypervector Types

### BinaryHypervector

Binary hypervectors with values in {0, 1}.

```python
class BinaryHypervector(Hypervector):
    @staticmethod
    def random(dimension: int, sparsity: float = 0.5, seed: Optional[int] = None) -> np.ndarray
    
    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray
    
    @staticmethod
    def bundle(hypervectors: List[np.ndarray], method: str = "majority") -> np.ndarray
```

### BipolarHypervector

Bipolar hypervectors with values in {-1, +1}.

```python
class BipolarHypervector(Hypervector):
    @staticmethod
    def random(dimension: int, seed: Optional[int] = None) -> np.ndarray
    
    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray
    
    @staticmethod
    def bundle(hypervectors: List[np.ndarray], method: str = "average") -> np.ndarray
```

### TernaryHypervector

Ternary hypervectors with values in {-1, 0, +1}.

```python
class TernaryHypervector(Hypervector):
    @staticmethod
    def random(dimension: int, sparsity: float = 0.33, seed: Optional[int] = None) -> np.ndarray
```

### LevelHypervector

Multi-level discrete hypervectors.

```python
class LevelHypervector(Hypervector):
    @staticmethod
    def random(dimension: int, levels: int = 3, seed: Optional[int] = None) -> np.ndarray
```

## Operations

### bind_hypervectors
```python
def bind_hypervectors(a: np.ndarray, b: np.ndarray, hypervector_type: str = "bipolar") -> np.ndarray
```
Bind two hypervectors based on their type.

### bundle_hypervectors
```python
def bundle_hypervectors(
    hypervectors: List[np.ndarray],
    method: BundlingMethod = BundlingMethod.MAJORITY,
    weights: Optional[List[float]] = None,
    hypervector_type: str = "bipolar"
) -> np.ndarray
```
Bundle multiple hypervectors using specified method.

**Bundling Methods:**
- `MAJORITY`: Majority voting (binary) or sign (bipolar)
- `AVERAGE`: Average and threshold
- `SAMPLE`: Random sampling from inputs
- `WEIGHTED`: Weighted sum

### permute_hypervector
```python
def permute_hypervector(
    hypervector: np.ndarray,
    method: PermutationMethod = PermutationMethod.CYCLIC,
    shift: int = 1,
    permutation: Optional[np.ndarray] = None,
    block_size: Optional[int] = None,
    inverse: bool = False
) -> np.ndarray
```
Permute hypervector elements.

**Permutation Methods:**
- `CYCLIC`: Circular shift
- `RANDOM`: Random permutation
- `BLOCK`: Block-wise permutation
- `INVERSE`: Inverse permutation

### similarity
```python
def similarity(
    a: np.ndarray,
    b: np.ndarray,
    metric: str = "cosine"
) -> float
```
Calculate similarity between hypervectors.

**Metrics:**
- `cosine`: Cosine similarity
- `hamming`: Hamming similarity (1 - normalized Hamming distance)
- `euclidean`: Negative Euclidean distance
- `jaccard`: Jaccard similarity (for binary vectors)

## Item Memory

### ItemMemory

Associative memory for storing and retrieving hypervectors.

```python
class ItemMemory:
    def __init__(
        self,
        dimension: int,
        similarity_metric: str = "cosine",
        max_items: Optional[int] = None
    )
```

**Methods:**

#### add
```python
def add(self, label: str, hypervector: np.ndarray) -> None
```
Add a labeled hypervector to memory.

#### query
```python
def query(
    self,
    hypervector: np.ndarray,
    top_k: int = 5,
    threshold: Optional[float] = None
) -> List[Tuple[str, float]]
```
Query memory with a hypervector, returning top-k similar items.

#### cleanup
```python
def cleanup(self, hypervector: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]
```
Find the closest stored hypervector (cleanup memory).

#### update
```python
def update(self, label: str, hypervector: np.ndarray) -> None
```
Update an existing item's hypervector.

#### merge
```python
def merge(self, label: str, hypervector: np.ndarray, weight: float = 0.5) -> None
```
Merge a new hypervector with an existing one.

## Encoders

### ScalarEncoder

Encode continuous scalar values.

```python
class ScalarEncoder(Encoder):
    def __init__(
        self,
        dimension: int,
        min_value: float,
        max_value: float,
        n_levels: int = 100,
        method: str = "thermometer",
        hypervector_type: str = "bipolar"
    )
```

**Methods:**
- `encode(value: float) -> np.ndarray`: Encode a scalar value

### CategoricalEncoder

Encode discrete categories.

```python
class CategoricalEncoder(Encoder):
    def __init__(
        self,
        dimension: int,
        categories: Optional[List[str]] = None,
        hypervector_type: str = "bipolar"
    )
```

**Methods:**
- `encode(category: str) -> np.ndarray`: Encode a category
- `get_categories() -> List[str]`: Get list of known categories

### SequenceEncoder

Encode sequences using n-grams or positional encoding.

```python
class SequenceEncoder(Encoder):
    def __init__(
        self,
        dimension: int,
        item_encoder: Optional[Encoder] = None,
        method: str = "ngram",
        n: int = 3,
        hypervector_type: str = "bipolar"
    )
```

**Methods:**
- `encode(sequence: List[any]) -> np.ndarray`: Encode a sequence

### SpatialEncoder

Encode spatial coordinates.

```python
class SpatialEncoder(Encoder):
    def __init__(
        self,
        dimension: int,
        bounds: Tuple[Tuple[float, float], ...],
        resolution: int = 10,
        hypervector_type: str = "bipolar"
    )
```

**Methods:**
- `encode(coordinates: Tuple[float, ...]) -> np.ndarray`: Encode spatial coordinates

### RecordEncoder

Encode structured records with named fields.

```python
class RecordEncoder(Encoder):
    def __init__(
        self,
        dimension: int,
        field_encoders: Optional[Dict[str, Encoder]] = None,
        hypervector_type: str = "bipolar"
    )
```

**Methods:**
- `encode(record: Dict[str, any]) -> np.ndarray`: Encode a record
- `add_field_encoder(field_name: str, encoder: Encoder) -> None`: Add encoder for specific field

### NGramEncoder

Encode text using character or word n-grams.

```python
class NGramEncoder(Encoder):
    def __init__(
        self,
        dimension: int,
        n: int = 3,
        level: str = "char",
        hypervector_type: str = "bipolar"
    )
```

**Methods:**
- `encode(text: str) -> np.ndarray`: Encode text

## Classifiers

### OneShotClassifier

Learn from single examples per class.

```python
class OneShotClassifier(HDClassifier):
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        similarity_metric: str = "cosine",
        hypervector_type: str = "bipolar",
        similarity_threshold: float = 0.0
    )
```

**Methods:**
- `train(X: List[any], y: List[str]) -> None`: Train on examples
- `add_example(x: any, label: str) -> None`: Add single example
- `predict(X: List[any]) -> List[str]`: Predict labels
- `remove_class(label: str) -> bool`: Remove a class

### AdaptiveClassifier

Online learning classifier with momentum.

```python
class AdaptiveClassifier(HDClassifier):
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        learning_rate: float = 0.1,
        momentum: float = 0.9
    )
```

**Methods:**
- `update(x: any, true_label: str, predicted_label: Optional[str] = None) -> None`: Update with feedback

### EnsembleClassifier

Combine multiple HDC classifiers.

```python
class EnsembleClassifier(HDClassifier):
    def __init__(
        self,
        classifiers: List[HDClassifier],
        voting: str = "hard"
    )
```

**Voting methods:**
- `hard`: Majority voting
- `soft`: Average probabilities

### HierarchicalClassifier

Multi-level hierarchical classification.

```python
class HierarchicalClassifier(HDClassifier):
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        hierarchy: Dict[str, List[str]],
        similarity_metric: str = "cosine",
        hypervector_type: str = "bipolar"
    )
```

## Utility Functions

### measure_capacity
```python
def measure_capacity(
    hdc: HDC,
    num_items: int = 1000,
    noise_levels: List[float] = None,
    similarity_threshold: float = 0.1
) -> HDCPerformanceMetrics
```
Measure capacity and noise tolerance of HDC system.

### benchmark_operations
```python
def benchmark_operations(hdc: HDC, num_trials: int = 100) -> Dict[str, float]
```
Benchmark HDC operations and return timing in milliseconds.

### estimate_required_dimension
```python
def estimate_required_dimension(
    num_items: int,
    similarity_threshold: float = 0.1,
    confidence: float = 0.99
) -> int
```
Estimate required dimension for storing items.

### create_codebook
```python
def create_codebook(
    num_symbols: int,
    dimension: int,
    hypervector_type: str = "bipolar"
) -> Dict[str, np.ndarray]
```
Create a codebook of quasi-orthogonal hypervectors.

## Factory Functions

### create_hdc
```python
def create_hdc(
    dimension: int = 10000,
    hypervector_type: str = "bipolar",
    **kwargs
) -> HDC
```
Create an HDC instance with specified configuration.

### generate_orthogonal_hypervectors
```python
def generate_orthogonal_hypervectors(
    dimension: int,
    num_vectors: int,
    hypervector_type: str = "bipolar"
) -> List[np.ndarray]
```
Generate a set of quasi-orthogonal hypervectors.