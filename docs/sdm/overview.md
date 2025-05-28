# Sparse Distributed Memory (SDM) Overview

## Introduction

Sparse Distributed Memory (SDM) is a mathematical model of human long-term memory developed by Pentti Kanerva in 1988. It provides a content-addressable memory system that exhibits remarkable properties similar to biological memory, including:

- **Content-based retrieval**: Access data by content rather than location
- **Noise tolerance**: Retrieve correct data even with corrupted input
- **Graceful degradation**: Performance degrades gradually with damage
- **Automatic generalization**: Similar inputs retrieve similar outputs
- **Massive capacity**: Store vast amounts of information efficiently

## Key Concepts

### Address Space and Hard Locations

SDM operates in a high-dimensional binary space (typically 256-10,000 dimensions). The key insight is that while the potential address space is enormous (2^n possible addresses), only a sparse subset of "hard locations" are actually implemented in physical memory.

```python
from cognitive_computing.sdm import SDM, SDMConfig

# Create SDM with 1000-dimensional space and 1000 hard locations
config = SDMConfig(
    dimension=1000,
    num_hard_locations=1000,
    activation_radius=451
)
sdm = SDM(config)
```

### Distributed Storage

When storing data, SDM doesn't write to a single location. Instead, it activates multiple hard locations within a certain Hamming distance (activation radius) of the input address and distributes the data across all activated locations.

### Critical Distance

The critical distance is approximately 0.451 × n for n-dimensional binary vectors. This represents the optimal activation radius where:
- Storage capacity is maximized
- Interference between patterns is minimized
- Noise tolerance is optimal

## How SDM Works

### Storage Process

1. **Input**: Receive an address-data pair (both binary vectors)
2. **Activation**: Find all hard locations within the activation radius
3. **Distribution**: Write data to all activated locations
   - Counter method: Increment/decrement counters based on data bits
   - Binary method: OR the data with existing contents

```python
import numpy as np

# Store a pattern
address = np.random.randint(0, 2, 1000)  # Random binary address
data = np.random.randint(0, 2, 1000)     # Random binary data
sdm.store(address, data)
```

### Recall Process

1. **Input**: Receive a query address (possibly noisy)
2. **Activation**: Find hard locations within activation radius
3. **Aggregation**: Sum the contents of activated locations
4. **Thresholding**: Convert sums back to binary data

```python
# Recall with perfect address
recalled_data = sdm.recall(address)

# Recall with noisy address
from cognitive_computing.sdm.utils import add_noise
noisy_address = add_noise(address, noise_level=0.1)
recalled_data = sdm.recall(noisy_address)
```

## Mathematical Foundation

### Capacity Analysis

The theoretical capacity of SDM is approximately:

**C ≈ 0.15 × M**

Where:
- C = number of patterns that can be stored
- M = number of hard locations

This assumes optimal parameters (activation radius ≈ critical distance).

### Probability of Activation

For random addresses, the probability that a hard location is activated:

**P(activation) = Σ(i=0 to r) C(n,i) × 2^(-n)**

Where:
- n = dimension
- r = activation radius
- C(n,i) = binomial coefficient

### Signal-to-Noise Ratio

The signal-to-noise ratio for recalled patterns:

**SNR = S × √(k) / √(M × p × (1-p))**

Where:
- S = number of storage operations
- k = number of activated locations
- M = total hard locations
- p = bit probability in data

## Storage Methods

### Counter-Based Storage

The default method uses integer counters at each hard location:

```python
config = SDMConfig(
    dimension=1000,
    num_hard_locations=1000,
    activation_radius=451,
    storage_method="counters",
    counter_bits=8,
    saturation_value=127
)
```

**Advantages:**
- Multiple patterns can be superimposed
- Confidence information available
- Better noise tolerance

**Disadvantages:**
- Higher memory usage
- Counter saturation possible

### Binary Storage

Alternative method using only binary values:

```python
config = SDMConfig(
    dimension=1000,
    num_hard_locations=1000,
    activation_radius=451,
    storage_method="binary"
)
```

**Advantages:**
- Lower memory usage
- No saturation issues
- Simpler implementation

**Disadvantages:**
- Less information per location
- Reduced capacity

## Address Decoders

SDM supports various address decoding strategies that determine which hard locations are activated:

### Hamming Decoder (Default)
- Classic approach using Hamming distance
- Uniform activation probability
- Predictable performance

### Jaccard Decoder
- Better for sparse binary data
- Activation based on set similarity
- Adapts to data density

### Random Decoder
- Hash-based activation
- O(1) complexity
- No distance computation

### Adaptive Decoder
- Dynamically adjusts activation patterns
- Optimizes for uniform usage
- Self-balancing

### Hierarchical Decoder
- Multi-level activation
- Natural clustering
- Efficient for structured data

### LSH Decoder
- Locality-Sensitive Hashing
- Sub-linear query time
- Scalable to millions of locations

## Practical Considerations

### Choosing Parameters

1. **Dimension**: Higher dimensions provide more capacity but require more hard locations
   - Typical range: 256-10,000 bits
   - Sweet spot: 1,000-2,000 bits

2. **Number of Hard Locations**: Determines memory usage and capacity
   - Rule of thumb: √(2^dimension) for balanced performance
   - Minimum: 0.001% of address space
   - Maximum: Limited by available RAM

3. **Activation Radius**: Controls the distribution spread
   - Optimal: ~0.451 × dimension (critical distance)
   - Smaller: Less interference, lower capacity
   - Larger: More interference, higher capacity

### Performance Optimization

```python
# Enable parallel processing for large SDMs
config = SDMConfig(
    dimension=5000,
    num_hard_locations=10000,
    activation_radius=2250,
    parallel=True,
    num_workers=4
)

# Use appropriate decoder for your data
from cognitive_computing.sdm.address_decoder import create_decoder
decoder = create_decoder('lsh', config, hard_locations)
```

### Memory Requirements

Memory usage can be estimated as:

**Memory (bytes) = M × n × (b/8) × k**

Where:
- M = number of hard locations
- n = dimension
- b = bits per counter (counter method) or 1 (binary method)
- k = overhead factor (~1.5 for metadata)

## Applications

### Pattern Recognition
```python
# Store multiple variants of a pattern
base_pattern = generate_pattern()
for i in range(10):
    variant = add_noise(base_pattern, 0.05)
    sdm.store(variant, class_label)

# Recognition with noisy input
noisy_input = add_noise(base_pattern, 0.15)
recognized_class = sdm.recall(noisy_input)
```

### Sequence Memory
```python
# Store sequences by using previous item as address
sequence = generate_sequence()
for i in range(len(sequence)-1):
    sdm.store(sequence[i], sequence[i+1])

# Recall sequence
current = sequence[0]
recalled_sequence = [current]
for _ in range(len(sequence)-1):
    current = sdm.recall(current)
    recalled_sequence.append(current)
```

### Associative Memory
```python
# Store associations between concepts
concept_pairs = load_associations()
for concept_a, concept_b in concept_pairs:
    sdm.store(encode(concept_a), encode(concept_b))

# Retrieve associations
query_concept = encode("example")
associated = sdm.recall(query_concept)
result = decode(associated)
```

## Advantages and Limitations

### Advantages

1. **Biological Plausibility**: Models properties of human memory
2. **Fault Tolerance**: Continues functioning with damaged locations
3. **No Training Required**: Immediate storage and recall
4. **Scalability**: Performance scales with resources
5. **Flexibility**: Works with any binary data

### Limitations

1. **Fixed Capacity**: Limited by number of hard locations
2. **Interference**: Patterns can interfere with each other
3. **Binary Constraint**: Data must be encoded as binary
4. **Memory Overhead**: Requires significant RAM for large systems
5. **No Forgetting**: Patterns accumulate unless manually cleared

## Best Practices

### 1. Data Encoding
```python
from cognitive_computing.sdm.utils import PatternEncoder

encoder = PatternEncoder(dimension=1000)

# Encode different data types
binary_int = encoder.encode_integer(42)
binary_float = encoder.encode_float(3.14159)
binary_text = encoder.encode_string("Hello SDM")
binary_vector = encoder.encode_vector(continuous_data)
```

### 2. Capacity Management
```python
# Monitor capacity utilization
stats = sdm.get_memory_stats()
if stats['locations_used'] / stats['num_hard_locations'] > 0.8:
    print("Warning: Approaching capacity limit")

# Analyze interference
crosstalk = sdm.analyze_crosstalk()
if crosstalk['avg_recall_error'] > 0.1:
    print("High interference detected")
```

### 3. Performance Monitoring
```python
from cognitive_computing.sdm.utils import test_sdm_performance

# Run performance tests
results = test_sdm_performance(sdm, test_patterns=100)
print(f"Average recall accuracy: {results.recall_accuracy_mean:.2%}")
print(f"Noise tolerance at 20%: {results.noise_tolerance[0.2]:.2%}")
```

### 4. Visualization
```python
from cognitive_computing.sdm.visualizations import (
    plot_memory_distribution,
    plot_recall_accuracy
)

# Visualize memory state
fig = plot_memory_distribution(sdm)

# Analyze performance
test_results = evaluate_sdm(sdm)
fig = plot_recall_accuracy(test_results)
```

## Integration with Other Systems

SDM can be integrated with other cognitive computing paradigms:

### With Neural Networks
- Use SDM as external memory for neural networks
- Store and retrieve neural network states
- Implement differentiable SDM layers

### With Hyperdimensional Computing
- Use HDC vectors as SDM addresses
- Combine HDC operations with SDM storage
- Implement symbolic reasoning

### With Vector Symbolic Architectures
- Store VSA representations in SDM
- Use SDM for cleanup memory
- Implement compositional structures

## Future Directions

1. **Quantum SDM**: Exploiting quantum superposition for exponential capacity
2. **Neuromorphic Implementation**: Hardware acceleration using memristors
3. **Adaptive Architectures**: Self-organizing SDM structures
4. **Hybrid Systems**: Combining SDM with deep learning
5. **Distributed SDM**: Scaling across multiple machines

## References

1. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
2. Kanerva, P. (1993). "Sparse Distributed Memory and Related Models." *Associative Neural Memories*.
3. Jaeckel, L. A. (1989). "An Alternative Design for a Sparse Distributed Memory." RIACS Technical Report.
4. Flynn, M. J., Kanerva, P., & Bhadkamkar, N. (1989). "Sparse Distributed Memory: Principles and Operation." Stanford University Technical Report.

## Quick Reference

### Basic Usage
```python
from cognitive_computing.sdm import create_sdm

# Quick creation with defaults
sdm = create_sdm(dimension=1000)

# Store and recall
sdm.store(address, data)
recalled = sdm.recall(address)
```

### Advanced Configuration
```python
from cognitive_computing.sdm import SDM, SDMConfig
from cognitive_computing.sdm.address_decoder import create_decoder

# Custom configuration
config = SDMConfig(
    dimension=2000,
    num_hard_locations=5000,
    activation_radius=900,
    storage_method="counters",
    parallel=True
)

# Custom decoder
decoder = create_decoder('adaptive', config, hard_locations)
sdm = SDM(config)
```

### Performance Analysis
```python
from cognitive_computing.sdm.utils import analyze_activation_patterns
from cognitive_computing.sdm.memory import MemoryStatistics

# Analyze patterns
analysis = analyze_activation_patterns(sdm, sample_size=1000)

# Detailed statistics
stats = MemoryStatistics(sdm)
report = stats.generate_report()
```

## Conclusion

Sparse Distributed Memory provides a powerful and biologically-inspired approach to content-addressable storage. Its unique properties make it valuable for applications requiring noise tolerance, pattern completion, and associative recall. By understanding its principles and parameters, you can effectively apply SDM to various cognitive computing tasks.

For more detailed information, see:
- [Theory Guide](theory.md) - Mathematical foundations
- [API Reference](api_reference.md) - Complete API documentation
- [Examples](examples.md) - Practical code examples
- [Performance Guide](performance.md) - Optimization strategies