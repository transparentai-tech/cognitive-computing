# HDC Examples

This guide provides practical examples of using the Hyperdimensional Computing (HDC) module for various tasks including classification, associative memory, and data encoding.

## Table of Contents

1. [Basic HDC Operations](#basic-hdc-operations)
2. [Classification Tasks](#classification-tasks)
3. [Associative Memory](#associative-memory)
4. [Data Encoding](#data-encoding)
5. [Capacity Analysis](#capacity-analysis)
6. [Advanced Patterns](#advanced-patterns)

## Basic HDC Operations

### Creating and Manipulating Hypervectors

```python
from cognitive_computing.hdc import create_hdc, BinaryHypervector, BipolarHypervector
from cognitive_computing.hdc.operations import bundle, bind, permute
import numpy as np

# Create HDC instance
hdc = create_hdc(dimension=10000, vector_type="binary")

# Generate random hypervectors
hv1 = hdc.generate_random()
hv2 = hdc.generate_random()

# Basic operations
bundled = bundle([hv1, hv2])  # Superposition
bound = bind(hv1, hv2)  # Binding
shifted = permute(hv1, shift=5)  # Circular shift

# Similarity computation
similarity = hdc.similarity(hv1, hv2)
print(f"Random vector similarity: {similarity:.4f}")
```

### Different Vector Types

```python
# Binary vectors (0, 1)
binary_hdc = create_hdc(dimension=10000, vector_type="binary")
binary_hv = binary_hdc.generate_random()

# Bipolar vectors (-1, +1)
bipolar_hdc = create_hdc(dimension=10000, vector_type="bipolar")
bipolar_hv = bipolar_hdc.generate_random()

# Ternary vectors (-1, 0, +1)
ternary_hdc = create_hdc(dimension=10000, vector_type="ternary", sparsity=0.9)
ternary_hv = ternary_hdc.generate_random()

# Level vectors (multi-level quantized)
level_hdc = create_hdc(dimension=10000, vector_type="level", levels=5)
level_hv = level_hdc.generate_random()
```

## Classification Tasks

### Basic Classification Example

```python
from cognitive_computing.hdc import HDCClassifier
import numpy as np

# Create synthetic data
np.random.seed(42)
n_samples = 100
n_features = 20

# Two classes: class 0 has low values, class 1 has high values
X_class0 = np.random.randn(n_samples // 2, n_features) - 1
X_class1 = np.random.randn(n_samples // 2, n_features) + 1
X_train = np.vstack([X_class0, X_class1])
y_train = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Create and train classifier
classifier = HDCClassifier(dimension=10000, encoding_method="thermometer")
classifier.fit(X_train, y_train)

# Test on new data
X_test = np.random.randn(20, n_features)
predictions = classifier.predict(X_test)
print(f"Predictions: {predictions}")
```

### Multi-class Classification

```python
# Create multi-class dataset
n_classes = 5
samples_per_class = 50
X_train = []
y_train = []

for class_id in range(n_classes):
    # Each class has different mean
    class_data = np.random.randn(samples_per_class, n_features) + class_id * 2
    X_train.append(class_data)
    y_train.extend([class_id] * samples_per_class)

X_train = np.vstack(X_train)
y_train = np.array(y_train)

# Train classifier with different encoding methods
encodings = ["thermometer", "circular", "random_projection"]
for encoding in encodings:
    clf = HDCClassifier(dimension=10000, encoding_method=encoding)
    clf.fit(X_train, y_train)
    
    # Cross-validation accuracy
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{encoding} encoding - Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Online Learning

```python
# Create classifier with online learning
online_clf = HDCClassifier(
    dimension=10000,
    encoding_method="circular",
    learning_rate=0.1
)

# Initial training
online_clf.fit(X_train[:100], y_train[:100])

# Online updates
for i in range(100, len(X_train)):
    # Predict on new sample
    pred = online_clf.predict(X_train[i:i+1])
    
    # Update if wrong
    if pred[0] != y_train[i]:
        online_clf.partial_fit(X_train[i:i+1], y_train[i:i+1])
```

## Associative Memory

### Basic Item Memory

```python
from cognitive_computing.hdc import ItemMemory

# Create item memory
memory = ItemMemory(dimension=10000)

# Store items
memory.add("apple", memory.generate_random())
memory.add("banana", memory.generate_random())
memory.add("orange", memory.generate_random())

# Create composite concepts
fruit_salad = memory.bundle(["apple", "banana", "orange"])
memory.add("fruit_salad", fruit_salad)

# Query memory
query = memory.get("apple")
similar_items = memory.query_similar(query, k=3)
print(f"Items similar to apple: {similar_items}")
```

### Semantic Relationships

```python
# Create semantic vectors
memory = ItemMemory(dimension=10000)

# Base concepts
memory.add("king", memory.generate_random())
memory.add("queen", memory.generate_random())
memory.add("man", memory.generate_random())
memory.add("woman", memory.generate_random())

# Analogy: king - man + woman ≈ queen
king = memory.get("king")
man = memory.get("man")
woman = memory.get("woman")

# Compute analogy
result = memory.unbind(memory.bind(king, memory.invert(man)), memory.invert(woman))

# Find nearest item
nearest = memory.query_similar(result, k=1)
print(f"king - man + woman ≈ {nearest[0]}")  # Should be "queen"
```

### Hierarchical Memory

```python
# Build hierarchical structure
memory = ItemMemory(dimension=10000)

# Animals hierarchy
memory.add("animal", memory.generate_random())
memory.add("mammal", memory.bind(memory.get("animal"), memory.generate_random()))
memory.add("dog", memory.bind(memory.get("mammal"), memory.generate_random()))
memory.add("cat", memory.bind(memory.get("mammal"), memory.generate_random()))

# Check relationships
dog = memory.get("dog")
mammal = memory.get("mammal")
similarity = memory.similarity(dog, mammal)
print(f"Dog-Mammal similarity: {similarity:.3f}")
```

## Data Encoding

### Numeric Data Encoding

```python
from cognitive_computing.hdc import (
    ThermometerEncoder, CircularEncoder, 
    RandomProjectionEncoder, ScalarEncoder
)

# Different encoding methods for numeric data
dimension = 10000
value = 0.7

# Thermometer encoding (good for ordered data)
therm_enc = ThermometerEncoder(dimension=dimension, min_value=0, max_value=1)
therm_hv = therm_enc.encode(value)

# Circular encoding (good for periodic data)
circ_enc = CircularEncoder(dimension=dimension, min_value=0, max_value=1)
circ_hv = circ_enc.encode(value)

# Random projection (good for high-dimensional data)
rp_enc = RandomProjectionEncoder(dimension=dimension, input_dim=10)
rp_hv = rp_enc.encode(np.random.randn(10))

# Scalar encoding (simple threshold-based)
scalar_enc = ScalarEncoder(dimension=dimension, min_value=0, max_value=1)
scalar_hv = scalar_enc.encode(value)
```

### Categorical Data Encoding

```python
from cognitive_computing.hdc import CategoricalEncoder, SymbolEncoder

# Categorical encoding
cat_enc = CategoricalEncoder(dimension=10000)
categories = ["red", "green", "blue", "yellow"]
for cat in categories:
    cat_enc.add_category(cat)

# Encode categorical values
color_hv = cat_enc.encode("red")

# Symbol encoding (for text)
sym_enc = SymbolEncoder(dimension=10000)
text = "hello world"
text_hv = sym_enc.encode(text)
```

### Spatial Data Encoding

```python
from cognitive_computing.hdc import SpatialEncoder

# Encode 2D coordinates
spatial_enc = SpatialEncoder(dimension=10000, grid_size=(10, 10))
position = (3, 7)
pos_hv = spatial_enc.encode(position)

# Encode with proximity
nearby_pos = (4, 7)
nearby_hv = spatial_enc.encode(nearby_pos)
similarity = np.dot(pos_hv, nearby_hv) / dimension
print(f"Nearby position similarity: {similarity:.3f}")
```

### Temporal Sequence Encoding

```python
from cognitive_computing.hdc import SequenceEncoder

# Encode sequences
seq_enc = SequenceEncoder(dimension=10000, method="position")

# Encode a sequence of symbols
sequence = ["start", "middle", "end"]
seq_hv = seq_enc.encode(sequence)

# Encode with n-gram binding
ngram_enc = SequenceEncoder(dimension=10000, method="ngram", n=2)
ngram_hv = ngram_enc.encode(sequence)
```

## Capacity Analysis

### Memory Capacity Testing

```python
from cognitive_computing.hdc import ItemMemory
from cognitive_computing.hdc.utils import estimate_capacity
import matplotlib.pyplot as plt

# Test capacity at different dimensions
dimensions = [1000, 5000, 10000, 20000]
capacities = []

for dim in dimensions:
    memory = ItemMemory(dimension=dim)
    capacity = estimate_capacity(memory, target_accuracy=0.9)
    capacities.append(capacity)
    print(f"Dimension {dim}: capacity ≈ {capacity} items")

# Plot results
plt.plot(dimensions, capacities, 'o-')
plt.xlabel("Dimension")
plt.ylabel("Capacity (90% accuracy)")
plt.title("HDC Memory Capacity vs Dimension")
plt.show()
```

### Noise Robustness Analysis

```python
from cognitive_computing.hdc.utils import test_noise_robustness

# Create memory with items
memory = ItemMemory(dimension=10000)
for i in range(100):
    memory.add(f"item_{i}", memory.generate_random())

# Test noise robustness
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []

for noise in noise_levels:
    acc = test_noise_robustness(memory, noise_level=noise, n_trials=100)
    accuracies.append(acc)
    print(f"Noise {noise:.1f}: accuracy = {acc:.3f}")
```

## Advanced Patterns

### Compositional Structures

```python
# Build complex compositional structures
memory = ItemMemory(dimension=10000)

# Basic attributes
memory.add("red", memory.generate_random())
memory.add("round", memory.generate_random())
memory.add("sweet", memory.generate_random())

# Compose apple from attributes
apple_attributes = memory.bundle(["red", "round", "sweet"])
memory.add("apple", apple_attributes)

# Query by partial attributes
query = memory.bundle(["red", "round"])
matches = memory.query_similar(query, k=5)
print(f"Red and round objects: {matches}")
```

### Reasoning with Analogies

```python
# Implement analogical reasoning
def solve_analogy(memory, a, b, c):
    """Solve analogy: a:b :: c:?"""
    # Get vectors
    va = memory.get(a)
    vb = memory.get(b)
    vc = memory.get(c)
    
    # Compute transformation and apply
    transform = memory.bind(vb, memory.invert(va))
    result = memory.bind(vc, transform)
    
    # Find nearest item
    nearest = memory.query_similar(result, k=1)
    return nearest[0]

# Example: capital city analogies
memory.add("france", memory.generate_random())
memory.add("paris", memory.generate_random())
memory.add("germany", memory.generate_random())
memory.add("berlin", memory.generate_random())

# Bind relationships
france_paris = memory.bind(memory.get("france"), memory.get("paris"))
germany_berlin = memory.bind(memory.get("germany"), memory.get("berlin"))

answer = solve_analogy(memory, "france", "paris", "germany")
print(f"france:paris :: germany:{answer}")
```

### Dynamic Binding for Variables

```python
# Variable binding for symbolic computation
memory = ItemMemory(dimension=10000)

# Create variable placeholders
memory.add("X", memory.generate_random())
memory.add("Y", memory.generate_random())

# Create operations
memory.add("plus", memory.generate_random())
memory.add("equals", memory.generate_random())

# Encode equation: X + Y = 5
equation = memory.bundle([
    memory.bind(memory.get("X"), memory.get("plus")),
    memory.bind(memory.get("Y"), memory.get("plus")),
    memory.bind(memory.generate_random(), memory.get("equals"))
])

# Substitute values
x_value = memory.generate_random()  # Represents 2
y_value = memory.generate_random()  # Represents 3

substituted = memory.substitute(equation, {"X": x_value, "Y": y_value})
```

### Multi-Modal Encoding

```python
# Combine different modalities
from cognitive_computing.hdc import MultiModalEncoder

# Create multi-modal encoder
mm_encoder = MultiModalEncoder(dimension=10000)

# Add encoders for different modalities
mm_encoder.add_modality("visual", ThermometerEncoder(dimension=10000))
mm_encoder.add_modality("text", SymbolEncoder(dimension=10000))
mm_encoder.add_modality("audio", RandomProjectionEncoder(dimension=10000, input_dim=128))

# Encode multi-modal data
data = {
    "visual": np.array([0.5, 0.7, 0.3]),  # RGB values
    "text": "red apple",
    "audio": np.random.randn(128)  # Audio features
}

combined_hv = mm_encoder.encode(data)
```

## Best Practices

1. **Dimension Selection**: Use 10,000+ dimensions for robust performance
2. **Vector Type**: Binary for memory efficiency, bipolar for better properties
3. **Encoding Method**: Match to data characteristics (thermometer for ordered, circular for periodic)
4. **Bundling**: Use weighted bundling for unequal importance
5. **Memory Management**: Clean up unused items to maintain performance
6. **Similarity Threshold**: Typically 0.3-0.4 for random vectors
7. **Noise Handling**: Add controlled noise during training for robustness

## Common Pitfalls

1. **Too Small Dimensions**: Less than 1000D loses the quasi-orthogonality property
2. **Wrong Encoding**: Using circular encoding for non-periodic data
3. **Overloading Memory**: Storing too many items reduces recall accuracy
4. **Normalization**: Forgetting to normalize after operations
5. **Direct Comparison**: Comparing vectors from different vector types

## Performance Tips

1. **Batch Operations**: Process multiple vectors at once
2. **Sparse Vectors**: Use ternary/sparse for memory-constrained applications
3. **Caching**: Cache frequently used base vectors
4. **Parallel Processing**: Many HDC operations are embarrassingly parallel
5. **GPU Acceleration**: Use CuPy backend for large-scale operations