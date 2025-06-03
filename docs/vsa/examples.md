# VSA Examples Guide

This guide provides detailed walkthroughs of the VSA example scripts, explaining key concepts and demonstrating practical applications.

## Example Scripts Overview

1. **basic_vsa_demo.py** - Introduction to VSA concepts
2. **binding_comparison.py** - Comparing binding operations
3. **vector_types_demo.py** - Understanding vector types
4. **symbolic_reasoning.py** - Advanced reasoning capabilities
5. **data_encoding.py** - Encoding various data types

## Basic VSA Demo

**Location**: `examples/vsa/basic_vsa_demo.py`

This example introduces fundamental VSA concepts through hands-on demonstrations.

### Key Concepts Demonstrated

#### 1. Vector Types
```python
# Binary vectors for hardware efficiency
binary_vec = BinaryVector.random(1000)

# Bipolar vectors for general use
bipolar_vec = BipolarVector.random(1000)

# Sparse ternary vectors
ternary_vec = TernaryVector.random(1000, sparsity=0.1)
```

**Learning Points:**
- Different vector types have different properties
- Binary is memory-efficient (1 bit per element)
- Bipolar is most versatile (-1/+1 values)
- Ternary enables sparse representations

#### 2. Binding Operations
```python
# XOR binding (binary vectors)
red_apple = vsa.bind(red, apple)

# Self-inverse property
original = vsa.bind(red_apple, red)  # Recovers apple
```

**Key Insights:**
- Binding creates associations between concepts
- Different binding operations have different properties
- XOR is self-inverse: `A ⊕ B ⊕ B = A`

#### 3. Bundling (Superposition)
```python
# Create a set representation
fruits = vsa.bundle([apple, banana, orange])

# Check membership
similarity = vsa.similarity(fruits, apple)  # High
similarity = vsa.similarity(fruits, car)    # Low
```

**Important Concepts:**
- Bundling creates set-like structures
- Items remain recoverable from bundles
- Capacity limited to ~√dimension items

### Running the Example

```bash
python examples/vsa/basic_vsa_demo.py
```

Expected output shows:
- Vector type properties
- Binding operation results
- Bundling demonstrations
- Different VSA architectures

## Binding Comparison

**Location**: `examples/vsa/binding_comparison.py`

This example provides an in-depth comparison of different binding operations.

### Performance Analysis

The script measures binding performance:

```python
# Time different binding operations
results = measure_binding_performance(
    dimension=1000,
    num_operations=1000
)
```

**Results Interpretation:**
- XOR: Fastest (bitwise operations)
- Multiplication: Fast (element-wise)
- Convolution: Slower (FFT required)
- MAP: Moderate (multiple steps)

### Mathematical Properties

```python
# Test commutativity: A ⊗ B = B ⊗ A
ab = vsa.bind(a, b)
ba = vsa.bind(b, a)
commutative = vsa.similarity(ab, ba) > 0.95
```

**Property Summary:**
| Operation | Commutative | Associative | Self-Inverse | Distributive |
|-----------|-------------|-------------|--------------|--------------|
| XOR | ✓ | ✓ | ✓ | ✗ |
| Multiplication | ✓ | ✓ | ✗ | ✗ |
| Convolution | ✓ | ✓ | ✗ | ✓ |
| MAP | ✓ | ≈ | ✗ | ≈ |
| Permutation | ✗ | ✗ | ✗ | ✗ |

### Noise Tolerance

```python
# Add noise and test recovery
noisy_bound = add_noise(bound, noise_level=0.2)
recovered = vsa.unbind(noisy_bound, key)
```

**Findings:**
- MAP most noise-tolerant
- Convolution good for moderate noise
- XOR sensitive to bit flips

### Capacity Testing

```python
# Test multiple bindings
pairs = [(key_i, value_i) for i in range(n)]
bundle = vsa.bundle([vsa.bind(k, v) for k, v in pairs])
```

**Capacity Guidelines:**
- Single binding: High fidelity
- 5-10 bindings: Good recovery
- 20+ bindings: Degraded performance

### Use Case Recommendations

1. **XOR Binding**
   - Binary data
   - Hardware implementations
   - Cryptographic applications

2. **Multiplication Binding**
   - General-purpose VSA
   - Neural network integration
   - Real-valued data

3. **Convolution Binding**
   - Sequential data
   - Signal processing
   - HRR compatibility

4. **MAP Binding**
   - Noisy environments
   - Multiple simultaneous bindings
   - Cognitive modeling

## Vector Types Demo

**Location**: `examples/vsa/vector_types_demo.py`

Comprehensive exploration of different vector types and their properties.

### Binary Vectors

```python
# Properties
density = np.mean(binary_vec.data)  # ~0.5
hamming_dist = np.sum(vec1.data != vec2.data)
```

**Use Cases:**
- Bloom filters
- Hardware accelerators
- Memory-constrained systems

### Bipolar Vectors

```python
# Statistical properties
mean = np.mean(bipolar_vec.data)  # ~0
correlation = np.corrcoef(vec1.data, vec2.data)[0,1]  # ~0
```

**Applications:**
- Neural networks
- Continuous embeddings
- General VSA operations

### Ternary Vectors

```python
# Sparse representation
sparsity = 0.1  # 90% zeros
active_elements = np.sum(ternary_vec.data != 0)
```

**Benefits:**
- Memory efficiency
- Faster operations on sparse data
- Biological plausibility

### Complex Vectors

```python
# Phase-based encoding
phases = np.angle(complex_vec.data)
magnitudes = np.abs(complex_vec.data)  # All 1.0
```

**Advantages:**
- Frequency domain operations
- Holographic properties
- Rich mathematical structure

### Integer Vectors

```python
# Modular arithmetic
modulus = 256
result = (vec1.data + vec2.data) % modulus
```

**Applications:**
- Finite field operations
- Cryptography
- Error correction

### Memory Comparison

| Vector Type | Bits per Element | 10K Dimension Size |
|-------------|------------------|-------------------|
| Binary | 1 | 1.25 KB |
| Bipolar | 8 | 10 KB |
| Ternary (10% sparse) | ~3.2 | 4 KB |
| Complex | 128 | 160 KB |
| Integer (mod 256) | 8 | 10 KB |

## Symbolic Reasoning

**Location**: `examples/vsa/symbolic_reasoning.py`

Advanced reasoning capabilities using VSA.

### Analogical Reasoning

```python
# A:B :: C:?
# King:Queen :: Man:Woman

# Learn transformation
transform = vsa.bind(queen, vsa.inverse(king))

# Apply to new input
result = vsa.bind(transform, man)
# Result similar to woman
```

**Key Concept**: VSA can learn and apply relational transformations.

### Knowledge Representation

```python
class KnowledgeBase:
    def add_fact(self, subject, predicate, object):
        # Store as role-filler structure
        fact = vsa.bundle([
            vsa.bind(SUBJECT_ROLE, subject),
            vsa.bind(PREDICATE_ROLE, predicate),
            vsa.bind(OBJECT_ROLE, object)
        ])
```

**Applications:**
- Semantic networks
- Question answering
- Inference systems

### Compositional Structures

```python
# Nested structures
red_car = vsa.bundle([
    vsa.bind(COLOR, red),
    vsa.bind(TYPE, car)
])

scene = vsa.bundle([
    vsa.bind(OBJECT1, red_car),
    vsa.bind(OBJECT2, blue_house)
])
```

**Capabilities:**
- Arbitrary nesting depth
- Role-based access
- Graceful degradation

### Logic Operations

```python
# Fuzzy logic with VSA
true = vsa.encode('TRUE')
false = vsa.encode('FALSE')
maybe = vsa.bundle([true, false], weights=[0.5, 0.5])
```

**Features:**
- Continuous truth values
- Probabilistic reasoning
- Soft logic operations

### Question Answering

```python
# Query knowledge base
def query(self, role, fact_vector):
    result = vsa.unbind(fact_vector, role)
    # Find closest match in vocabulary
    return find_closest_concept(result)
```

**Example Queries:**
- "What is John's occupation?"
- "Who lives in New York?"
- "Which doctors live in Boston?"

### Cognitive Operations

```python
# Attention mechanism
attention_query = vsa.bind(COLOR, red)
for object in scene:
    relevance = vsa.similarity(object, attention_query)
    # Higher relevance = more attention
```

**Cognitive Models:**
- Working memory (limited capacity)
- Attention (similarity-based selection)
- Priming (spreading activation)

## Data Encoding

**Location**: `examples/vsa/data_encoding.py`

Comprehensive guide to encoding different data types.

### Text Encoding

```python
# Character-level encoding
word = "hello"
char_vectors = []
for i, char in enumerate(word):
    pos_vec = vsa.encode(f'pos_{i}')
    char_vec = vsa.encode(f'char_{char}')
    char_vectors.append(vsa.bind(pos_vec, char_vec))
word_vec = vsa.bundle(char_vectors)
```

**Techniques:**
- Character-level: Fine-grained control
- Word-level: Semantic units
- N-grams: Local context
- Random indexing: Dimensionality reduction

### Numerical Encoding

```python
# Continuous values with levels
level_encoder = LevelEncoder(
    vsa, 
    num_levels=10,
    min_value=0.0,
    max_value=100.0
)
temp_vec = level_encoder.encode(25.5)
```

**Strategies:**
- Discretization: Map to levels
- Magnitude-phase: Complex representation
- Thermometer coding: Cumulative activation

### Spatial Encoding

```python
# 2D coordinates
spatial_encoder = SpatialEncoder(vsa, grid_size=(100, 100))
location = spatial_encoder.encode_2d(x=45, y=67)

# Scene representation
scene = vsa.bundle([
    vsa.bind(car, location1),
    vsa.bind(tree, location2)
])
```

**Applications:**
- Image representation
- Spatial reasoning
- Navigation tasks

### Temporal Encoding

```python
# Time series
temporal_encoder = TemporalEncoder(vsa, max_lag=5)
series_vec = vsa.zero()

for t, value in enumerate(time_series):
    time_vec = temporal_encoder.encode_time_point(t)
    value_vec = encode_value(value)
    series_vec = vsa.bundle([
        series_vec,
        vsa.bind(time_vec, value_vec)
    ])
```

**Use Cases:**
- Sensor data
- Event sequences
- Periodic patterns

### Graph Encoding

```python
# Graph structure
graph_encoder = GraphEncoder(vsa)
edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]
graph_vec = vsa.bundle([
    graph_encoder.encode_edge(
        vsa.encode(src), 
        vsa.encode(dst)
    )
    for src, dst in edges
])
```

**Capabilities:**
- Node embeddings
- Edge relationships
- Path encoding
- Attributed graphs

### Mixed Data Types

```python
# IoT sensor reading
sensor_data = {
    'device_id': 'sensor_42',
    'timestamp': 1234567890,
    'location': (37.7749, -122.4194),
    'temperature': 22.5,
    'status': 'active'
}

# Encode each component appropriately
sensor_vec = encode_mixed_data(sensor_data)
```

**Best Practices:**
- Choose appropriate encoder for each field
- Maintain consistent dimensionality
- Use role-filler for structure
- Consider sparsity for efficiency

## Running All Examples

To run all examples:

```bash
# Run individual examples
python examples/vsa/basic_vsa_demo.py
python examples/vsa/binding_comparison.py
python examples/vsa/vector_types_demo.py
python examples/vsa/symbolic_reasoning.py
python examples/vsa/data_encoding.py

# Run with custom parameters
python examples/vsa/basic_vsa_demo.py --dimension 5000
python examples/vsa/binding_comparison.py --num-trials 100
```

## Key Takeaways

1. **Vector Types**: Choose based on hardware, sparsity, and mathematical needs
2. **Binding Operations**: Select based on properties and noise requirements
3. **Encoding Strategies**: Match encoder to data type and structure
4. **Capacity Limits**: Respect √dimension rule for reliable storage
5. **Compositional Power**: Build complex structures from simple operations

## Next Steps

- Experiment with different dimensions and parameters
- Combine techniques for your specific application
- Explore the [API Reference](api_reference.md) for advanced features
- Read the [Performance Guide](performance.md) for optimization tips

## Additional Resources

- [VSA Overview](overview.md) - Conceptual introduction
- [Theory Guide](theory.md) - Mathematical foundations
- [API Reference](api_reference.md) - Complete API documentation
- [Performance Guide](performance.md) - Optimization strategies