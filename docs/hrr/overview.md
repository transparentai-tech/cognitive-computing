# Holographic Reduced Representations (HRR) Overview

## Introduction

Holographic Reduced Representations (HRR) is a powerful method for encoding compositional structures in fixed-size distributed representations. Developed by Tony Plate in the 1990s, HRR provides a mathematically principled way to represent and manipulate symbolic structures using high-dimensional vectors.

## What is HRR?

HRR is a **Vector Symbolic Architecture (VSA)** that enables:

- **Compositional representation**: Complex structures encoded in fixed-size vectors
- **Symbolic manipulation**: Operations on symbols using vector algebra
- **Distributed storage**: Information spread across all dimensions
- **Graceful degradation**: Robust to noise and partial information

### Key Properties

1. **Fixed dimensionality**: All representations use the same vector size
2. **Compositionality**: Complex structures built from simpler ones
3. **Invertibility**: Can extract components from composite representations
4. **Similarity preservation**: Similar structures have similar representations

## Core Concepts

### 1. Binding Operation (⊛)

Binding combines two vectors into a composite representation using **circular convolution**:

```python
from cognitive_computing.hrr import create_hrr

hrr = create_hrr(dimension=1024)

# Bind role and filler
role = hrr.generate_vector()
filler = hrr.generate_vector()
binding = hrr.bind(role, filler)
```

Properties:
- Distributes information across all dimensions
- Approximately preserves vector magnitude
- Creates dissimilar output from inputs

### 2. Unbinding Operation (⊘)

Unbinding extracts components using **circular correlation**:

```python
# Extract filler given role
retrieved_filler = hrr.unbind(binding, role)

# Check similarity
similarity = hrr.similarity(filler, retrieved_filler)
print(f"Retrieval similarity: {similarity:.3f}")
```

Properties:
- Approximate inverse of binding
- Works best with unitary vectors
- Quality depends on vector properties

### 3. Bundling Operation (+)

Bundling superposes multiple vectors:

```python
# Bundle multiple items
items = [item1, item2, item3]
bundle = hrr.bundle(items)

# Each item is partially present
for item in items:
    sim = hrr.similarity(bundle, item)
    print(f"Similarity: {sim:.3f}")
```

Properties:
- Creates set-like representations
- Allows weighted combinations
- Limited capacity (typically 3-7 items)

### 4. Cleanup Memory

Cleanup memory maps noisy vectors to clean items:

```python
from cognitive_computing.hrr import CleanupMemory, CleanupMemoryConfig

# Create cleanup memory
cleanup = CleanupMemory(CleanupMemoryConfig(threshold=0.3), dimension=1024)

# Add known items
cleanup.add_item("cat", cat_vector)
cleanup.add_item("dog", dog_vector)

# Clean up noisy vector
name, clean_vector, confidence = cleanup.cleanup(noisy_vector)
```

## Mathematical Foundations

### Circular Convolution

The binding operation uses circular convolution:

```
c[k] = Σ(i=0 to n-1) a[i] * b[(k-i) mod n]
```

Efficient implementation via FFT:
```
C = F⁻¹(F(A) ⊙ F(B))
```

Where:
- F = Fourier transform
- ⊙ = element-wise multiplication

### Vector Properties

**Random Vectors**:
- Drawn from normal distribution N(0, 1/n)
- Nearly orthogonal in high dimensions
- Good for general representations

**Unitary Vectors**:
- Magnitude = 1
- Self-inverse: A ⊛ A = I
- Ideal for roles and relations

## Use Cases and Applications

### 1. Symbolic Reasoning
```python
# Encode "John loves Mary"
john_loves_mary = hrr.bundle([
    hrr.bind(agent_role, john),
    hrr.bind(action_role, loves),
    hrr.bind(patient_role, mary)
])
```

### 2. Analogical Reasoning
```python
# A:B :: C:?
# If A transforms to B, apply same to C
transform = hrr.unbind(B, A)
D = hrr.bind(transform, C)
```

### 3. Sequence Processing
```python
# Encode sequence with positions
sequence = hrr.bundle([
    hrr.bind(pos1, item1),
    hrr.bind(pos2, item2),
    hrr.bind(pos3, item3)
])
```

### 4. Hierarchical Structures
```python
# Encode tree structures recursively
tree = hrr.bundle([
    hrr.bind(root_role, root_value),
    hrr.bind(left_child, left_subtree),
    hrr.bind(right_child, right_subtree)
])
```

## Comparison with Other VSA Methods

| Feature | HRR | Binary Spatter Code | MAP | HDC/Hyperdimensional |
|---------|-----|-------------------|-----|---------------------|
| Vector Type | Real/Complex | Binary | Real | Binary/Integer |
| Binding Op | Convolution | XOR | Multiplication | Various |
| Capacity | Moderate | Low | High | High |
| Noise Tolerance | Good | Moderate | Good | Excellent |
| Implementation | Moderate | Simple | Complex | Simple |

## Advantages and Limitations

### Advantages
- **Flexible**: Handles various data structures
- **Compositional**: Natural hierarchy support
- **Robust**: Graceful degradation with noise
- **Theoretically grounded**: Solid mathematical foundation

### Limitations
- **Capacity constraints**: Limited bundling capacity
- **Computational cost**: FFT operations for large dimensions
- **Approximate retrieval**: Not exact for complex structures
- **Dimension selection**: Performance sensitive to dimension choice

## Getting Started

### Basic Example
```python
from cognitive_computing.hrr import create_hrr

# Create HRR system
hrr = create_hrr(dimension=1024)

# Generate vectors
A = hrr.generate_vector()
B = hrr.generate_vector()

# Basic operations
C = hrr.bind(A, B)          # Binding
B_retrieved = hrr.unbind(C, A)  # Unbinding
bundle = hrr.bundle([A, B, C])  # Bundling

# Check similarity
sim = hrr.similarity(B, B_retrieved)
print(f"Similarity: {sim:.3f}")
```

### Role-Filler Example
```python
from cognitive_computing.hrr import RoleFillerEncoder

# Create encoder
encoder = RoleFillerEncoder(hrr)

# Encode structure
structure = {
    "color": red_vector,
    "shape": circle_vector,
    "size": large_vector
}
encoded = encoder.encode_structure(structure)

# Query structure
red_retrieved = encoder.decode_filler(encoded, color_role)
```

## Best Practices

### 1. Dimension Selection
- Typical range: 512-4096 dimensions
- Higher dimensions = better capacity but more computation
- Consider your application's requirements

### 2. Vector Generation
- Use unitary vectors for roles/relations
- Use random vectors for content/fillers
- Normalize when needed for stability

### 3. Cleanup Memory
- Essential for symbolic applications
- Set appropriate threshold (typically 0.2-0.4)
- Pre-populate with expected items

### 4. Noise Handling
- Add noise during training for robustness
- Use cleanup memory for discrete outputs
- Consider redundant encoding for critical data

## Integration with Cognitive Computing

HRR integrates seamlessly with other components:

```python
# Combine with SDM for robust storage
from cognitive_computing.sdm import create_sdm

sdm = create_sdm(dimension=1024)
hrr = create_hrr(dimension=1024)

# Store HRR structures in SDM
structure = encoder.encode_structure({"subj": "cat", "verb": "sleeps"})
sdm.store(structure, structure)
```

## Next Steps

1. Explore the [Theory Guide](theory.md) for mathematical details
2. Check the [API Reference](api_reference.md) for complete documentation
3. Try the [Examples](examples.md) for practical applications
4. Review [Performance Guide](performance.md) for optimization tips

## References

1. Plate, T. A. (2003). *Holographic Reduced Representation: Distributed Representation for Cognitive Structures*. CSLI Publications.

2. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139-159.

3. Gayler, R. W. (2003). Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience. *ICCS/ASCS International Conference on Cognitive Science*.

4. Eliasmith, C. (2013). *How to build a brain: A neural architecture for biological cognition*. Oxford University Press.