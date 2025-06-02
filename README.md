# Cognitive Computing

A comprehensive Python package for cognitive computing, implementing various brain-inspired computing paradigms for robust, efficient, and adaptive information processing.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

The `cognitive-computing` package provides implementations of several cognitive computing paradigms:

- **Sparse Distributed Memory (SDM)** ✅ - *Fully implemented*
- **Holographic Reduced Representations (HRR)** ✅ - *Core implementation complete*
- **Vector Symbolic Architectures (VSA)** 🚧 - *Coming soon*
- **Hyperdimensional Computing (HDC)** 🚧 - *Coming soon*

These technologies enable:
- 🧠 **Brain-inspired computing** - Models based on human memory and cognition
- 🔍 **Content-addressable storage** - Retrieve data by content, not location
- 🌊 **Noise tolerance** - Graceful degradation with noisy inputs
- ⚡ **Fast approximate computing** - Trade precision for speed and robustness
- 🔗 **Symbolic reasoning** - Combine neural and symbolic approaches

## Installation

### From Source (Recommended during development)

```bash
git clone https://github.com/transparentai-tech/cognitive-computing.git
cd cognitive-computing
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

### Install with Visualization Support

```bash
pip install -e ".[viz]"
```

### Install with GPU Acceleration

```bash
pip install -e ".[gpu]"
```

## Quick Start

### Sparse Distributed Memory (SDM)

```python
import numpy as np
from cognitive_computing.sdm import create_sdm

# Create a 1000-dimensional SDM
sdm = create_sdm(dimension=1000)

# Generate random binary patterns
address = np.random.randint(0, 2, 1000)
data = np.random.randint(0, 2, 1000)

# Store the pattern
sdm.store(address, data)

# Recall with perfect address
recalled_data = sdm.recall(address)
print(f"Perfect recall accuracy: {np.mean(recalled_data == data):.2%}")

# Recall with noisy address (10% noise)
from cognitive_computing.sdm.utils import add_noise
noisy_address = add_noise(address, noise_level=0.1)
recalled_noisy = sdm.recall(noisy_address)
print(f"Noisy recall accuracy: {np.mean(recalled_noisy == data):.2%}")
```

### Advanced Configuration

```python
from cognitive_computing.sdm import SDM, SDMConfig

# Custom configuration
config = SDMConfig(
    dimension=2000,
    num_hard_locations=5000,
    activation_radius=900,
    storage_method="counters",  # or "binary"
    parallel=True,
    num_workers=4
)

# Create SDM with custom parameters
sdm = SDM(config)

# Use different address decoders
from cognitive_computing.sdm.address_decoder import create_decoder

# Options: 'hamming', 'jaccard', 'random', 'adaptive', 'hierarchical', 'lsh'
decoder = create_decoder('adaptive', config, sdm.hard_locations)
```

### Holographic Reduced Representations (HRR)

```python
from cognitive_computing.hrr import create_hrr
from cognitive_computing.hrr.encoding import RoleFillerEncoder

# Create HRR system
hrr = create_hrr(dimension=1024)

# Basic binding and unbinding
role = hrr.generate_vector()
filler = hrr.generate_vector()
binding = hrr.bind(role, filler)
retrieved = hrr.unbind(binding, role)
print(f"Similarity: {hrr.similarity(retrieved, filler):.3f}")

# Encode structured information
encoder = RoleFillerEncoder(hrr)
person = encoder.encode_structure({
    "name": hrr.generate_vector(),  # Vector for "John"
    "age": hrr.generate_vector(),    # Vector for "25"
    "city": hrr.generate_vector()    # Vector for "Boston"
})

# Cleanup memory for robust retrieval
from cognitive_computing.hrr.cleanup import CleanupMemory, CleanupMemoryConfig

cleanup = CleanupMemory(CleanupMemoryConfig(threshold=0.3), dimension=1024)
cleanup.add_item("john", hrr.generate_vector())
cleanup.add_item("mary", hrr.generate_vector())

# Clean up noisy vectors
noisy_vector = retrieved + np.random.randn(1024) * 0.2
name, clean_vector, similarity = cleanup.cleanup(noisy_vector)
```

## Features

### Sparse Distributed Memory (SDM)

- **Multiple Storage Methods**
  - Counter-based (default) - Better noise tolerance
  - Binary - Lower memory usage

- **Six Address Decoders**
  - Hamming - Classic distance-based
  - Jaccard - For sparse data
  - Random - O(1) hashing
  - Adaptive - Self-adjusting
  - Hierarchical - Multi-level
  - LSH - Locality-sensitive hashing

- **Comprehensive Analysis Tools**
  - Memory capacity estimation
  - Activation pattern analysis
  - Performance benchmarking
  - Crosstalk measurement

- **Data Encoding Utilities**
  - Integer encoding
  - Float encoding
  - String encoding
  - Vector encoding

- **Visualization Support**
  - Memory distribution plots
  - Activation patterns
  - Recall accuracy curves
  - Interactive 3D visualizations

### Holographic Reduced Representations (HRR)

- **Core Operations**
  - Circular convolution binding
  - Circular correlation unbinding
  - Vector bundling (superposition)
  - Real and complex storage modes

- **Encoding Strategies**
  - Role-filler binding
  - Sequence encoding (positional/chaining)
  - Hierarchical structures
  - Tree encoding

- **Cleanup Memory**
  - Item storage and retrieval
  - Similarity-based cleanup
  - Multiple similarity metrics
  - Persistence support

- **Analysis Tools**
  - Binding capacity analysis
  - Crosstalk measurement
  - Performance benchmarking
  - Vector generation utilities

- **Visualization Support**
  - Similarity matrices
  - Convolution spectra
  - Cleanup space visualization
  - Performance dashboards

## Examples

### Pattern Recognition

```python
from cognitive_computing.sdm import create_sdm
from cognitive_computing.sdm.utils import generate_random_patterns, add_noise

# Create SDM
sdm = create_sdm(dimension=1000)

# Generate base pattern and variations
base_pattern = np.random.randint(0, 2, 1000)
variations = [add_noise(base_pattern, 0.05) for _ in range(10)]

# Store all variations with same label
label = np.array([1, 0, 0, 0] + [0] * 996)  # One-hot encoded label
for variant in variations:
    sdm.store(variant, label)

# Recognize noisy input
test_input = add_noise(base_pattern, 0.15)
recalled_label = sdm.recall(test_input)
print(f"Pattern recognized: {np.argmax(recalled_label)}")
```

### Sequence Memory

```python
# Store sequential patterns
sequence = generate_random_patterns(10, 1000)[0]  # 10 patterns

for i in range(len(sequence) - 1):
    sdm.store(sequence[i], sequence[i + 1])

# Recall sequence
current = sequence[0]
recalled_sequence = [current]

for _ in range(len(sequence) - 1):
    current = sdm.recall(current)
    recalled_sequence.append(current)
```

### Real-World Data Encoding

```python
from cognitive_computing.sdm.utils import PatternEncoder

encoder = PatternEncoder(dimension=1000)
sdm = create_sdm(dimension=1000)

# Encode and store different data types
# Integer
age = 25
age_encoded = encoder.encode_integer(age)
sdm.store(age_encoded, age_encoded)

# String
name = "Alice"
name_encoded = encoder.encode_string(name)
sdm.store(name_encoded, name_encoded)

# Float array
features = np.array([0.1, 0.5, 0.9, 0.2])
features_encoded = encoder.encode_vector(features)
sdm.store(features_encoded, features_encoded)
```

## Performance

SDM operations are highly efficient:

```python
from cognitive_computing.sdm.utils import evaluate_sdm_performance

# Run performance benchmark
results = evaluate_sdm_performance(sdm, test_patterns=100)

print(f"Write time: {results.write_time_mean*1000:.2f} ms")
print(f"Read time: {results.read_time_mean*1000:.2f} ms")
print(f"Recall accuracy: {results.recall_accuracy_mean:.2%}")
```

### Parallel Processing

For large-scale applications, enable parallel processing:

```python
config = SDMConfig(
    dimension=5000,
    num_hard_locations=10000,
    parallel=True,
    num_workers=8
)
sdm = SDM(config)
```

## Visualization

```python
from cognitive_computing.sdm.visualizations import (
    plot_memory_distribution,
    plot_recall_accuracy,
    visualize_memory_contents
)

# Analyze memory distribution
fig = plot_memory_distribution(sdm)

# Test and plot recall accuracy
test_results = evaluate_sdm_performance(sdm)
fig = plot_recall_accuracy(test_results)

# Interactive 3D visualization
fig = visualize_memory_contents(sdm, interactive=True)
```

## Documentation

- [Installation Guide](docs/installation.md)
- [SDM Overview](docs/sdm/overview.md)
- [API Reference](docs/sdm/api_reference.md)
- [Theory and Mathematics](docs/sdm/theory.md)
- [Examples](docs/sdm/examples.md)
- [Performance Guide](docs/sdm/performance.md)

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_sdm/

# Run with coverage
pytest --cov=cognitive_computing

# Run only fast tests
pytest -m "not slow"
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

### Development Setup

```bash
# Install in development mode with all dependencies
pip install -e ".[dev,viz]"

# Run code formatting
black cognitive_computing tests

# Run linting
flake8 cognitive_computing tests

# Run type checking
mypy cognitive_computing
```

## Roadmap

### Current Status
- ✅ Sparse Distributed Memory (SDM) - **Complete**
  - Core implementation with counter/binary storage
  - Six address decoder strategies
  - Comprehensive utilities and visualizations
  - Full test coverage

- ✅ Holographic Reduced Representations (HRR) - **Core Complete**
  - Circular convolution/correlation operations
  - Role-filler and sequence encoding
  - Cleanup memory implementation
  - Comprehensive test suite (12 files)
  - **Still needed**: Examples and documentation

### Upcoming Features

- 🚧 Vector Symbolic Architectures (VSA)
  - Bundling and binding operations
  - Cleanup memory
  - Compositional structures

- 🚧 Hyperdimensional Computing (HDC)
  - Random indexing
  - Item memory
  - Classification capabilities

- 🚧 Integration Features
  - Cross-paradigm operations
  - Neural network interfaces
  - Distributed computing support

## Citation

If you use this package in your research, please cite:

```bibtex
@software{cognitive_computing,
  title = {Cognitive Computing: A Python Package for Brain-Inspired Computing},
  author = {Ian Hamilton},
  year = {2025},
  url = {https://github.com/transparentai-tech/cognitive-computing}
}
```

## References

### Sparse Distributed Memory
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Kanerva, P. (1993). "Sparse Distributed Memory and Related Models." *Associative Neural Memories*.

### Coming Soon
- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks*.
- Gayler, R. W. (2003). "Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience."
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation."

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Pentti Kanerva's groundbreaking work on Sparse Distributed Memory
- Thanks to all contributors and the cognitive computing research community

---

**Note**: This package is under active development. APIs may change in future versions. We recommend pinning your dependencies to specific versions for production use.