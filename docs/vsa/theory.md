# Vector Symbolic Architectures - Mathematical Theory

## Introduction

This document provides the mathematical foundations of Vector Symbolic Architectures (VSA). VSA represents a class of computational models that use high-dimensional vectors to encode and manipulate symbolic information. The theory combines insights from linear algebra, probability theory, and information theory.

## Mathematical Foundations

### Vector Spaces

VSA operates in high-dimensional vector spaces, typically ℝ^d or {0,1}^d where d ∈ [1000, 10000].

**Definition**: A VSA vector space V is a d-dimensional space equipped with:
- A similarity measure: sim: V × V → [-1, 1]
- A binding operation: ⊗: V × V → V
- A bundling operation: ⊕: V^n → V
- An identity element: I ∈ V

### Random Vector Generation

Vectors in VSA are typically generated randomly to ensure approximately orthogonal representations.

**Binary Vectors**: For v ∈ {0,1}^d:
```
P(v_i = 1) = 0.5, independently for each i
```

**Bipolar Vectors**: For v ∈ {-1,+1}^d:
```
P(v_i = 1) = P(v_i = -1) = 0.5
```

**Complex Vectors**: For v ∈ ℂ^d with |v_i| = 1:
```
v_i = e^(iθ_i), where θ_i ~ Uniform(0, 2π)
```

### Similarity Measures

Different vector types use different similarity measures:

**Hamming Similarity** (Binary):
```
sim_H(x, y) = 1 - (1/d)∑|x_i - y_i|
```

**Cosine Similarity** (Bipolar/Real):
```
sim_C(x, y) = (x · y)/(||x|| ||y||)
```

**Complex Dot Product** (Complex):
```
sim_Z(x, y) = Re(∑x_i * conj(y_i))/d
```

## Binding Operations

### XOR Binding (Binary Vectors)

For x, y ∈ {0,1}^d:
```
(x ⊕ y)_i = x_i ⊕ y_i (mod 2)
```

**Properties**:
- Self-inverse: x ⊕ x = 0
- Commutative: x ⊕ y = y ⊕ x
- Associative: (x ⊕ y) ⊕ z = x ⊕ (y ⊕ z)
- Identity: x ⊕ 0 = x

### Multiplication Binding (Bipolar Vectors)

For x, y ∈ {-1,+1}^d:
```
(x ⊗ y)_i = x_i × y_i
```

**Properties**:
- Commutative: x ⊗ y = y ⊗ x
- Associative: (x ⊗ y) ⊗ z = x ⊗ (y ⊗ z)
- Identity: x ⊗ 1 = x
- Inverse: x ⊗ x = 1 (element-wise)

### Circular Convolution (Real/Complex Vectors)

For x, y ∈ ℝ^d or ℂ^d:
```
(x ⊛ y)_j = ∑_{k=0}^{d-1} x_k × y_{(j-k) mod d}
```

Or in frequency domain:
```
x ⊛ y = IFFT(FFT(x) ⊙ FFT(y))
```

**Properties**:
- Commutative: x ⊛ y = y ⊛ x
- Associative: (x ⊛ y) ⊛ z = x ⊛ (y ⊛ z)
- Distributive over addition: x ⊛ (y + z) = (x ⊛ y) + (x ⊛ z)
- Approximate inverse via correlation

### MAP Binding (Multiply-Add-Permute)

For x, y ∈ V:
```
MAP(x, y) = s ⊙ (x ⊗ y) + (1-s) ⊙ ρ(x + y)
```

Where:
- s ∈ [0,1]^d is a random selection vector
- ρ is a permutation operation
- ⊙ is element-wise multiplication

## Bundling Operations

### Majority Rule Bundling (Binary/Bipolar)

For vectors x₁, x₂, ..., x_n:
```
Bundle(x₁, ..., x_n)_i = sign(∑_{j=1}^n x_{ji})
```

### Normalized Addition (Real/Complex)

For vectors x₁, x₂, ..., x_n:
```
Bundle(x₁, ..., x_n) = normalize(∑_{j=1}^n w_j × x_j)
```

Where w_j are optional weights with ∑w_j = 1.

### Properties of Bundling

1. **Similarity Preservation**: Bundle(X) is similar to each x_i ∈ X
2. **Capacity**: Can reliably store ~√d items
3. **Noise Tolerance**: Robust to missing or corrupted items

## Permutation Operations

### Cyclic Shift

For vector x and shift amount k:
```
ρ_k(x)_i = x_{(i-k) mod d}
```

**Properties**:
- Invertible: ρ_{-k}(ρ_k(x)) = x
- Order preserving: ρ_k preserves sequential relationships
- Orthogonality: ρ_k(x) ⊥ x for random x and k ≠ 0

### Random Permutation

A fixed random permutation π: {1,...,d} → {1,...,d}:
```
π(x)_i = x_{π(i)}
```

## Information Capacity

### Storage Capacity

The number of items that can be reliably stored in a bundle:

**Binary Vectors**:
```
Capacity ≈ 0.5 × √d
```

**Bipolar Vectors**:
```
Capacity ≈ √(d / (2 ln d))
```

### Channel Capacity

Information transmission through binding:
```
I(X; Y) ≤ d × log₂(1 + SNR)
```

Where SNR is the signal-to-noise ratio.

### Binding Capacity

Number of role-filler pairs in a single vector:
```
Pairs ≈ d / (k × log d)
```

Where k depends on the required fidelity.

## Probability Theory in VSA

### Random Vector Properties

For random vectors x, y ∈ V:

**Expected Similarity**:
```
E[sim(x, y)] = 0 (for x ≠ y)
Var[sim(x, y)] = 1/d
```

**Concentration Bounds** (Hoeffding):
```
P(|sim(x, y) - E[sim(x, y)]| > ε) ≤ 2 exp(-2dε²)
```

### Noise Model

For noisy retrieval with noise rate p:
```
P(correct retrieval) ≈ Φ((1-2p)√d)
```

Where Φ is the standard normal CDF.

## VSA Algebra

### Algebraic Structure

VSA forms an approximate algebra with:
- **Group**: (V, ⊗) forms an abelian group
- **Ring**: (V, ⊕, ⊗) approximates a ring structure
- **Module**: V is a module over scalars

### Key Identities

1. **Binding-Unbinding**: x ⊗ (x ⊗ y) ≈ y
2. **Bundle Decomposition**: Bundle(x, y) · x > 0
3. **Permutation Commutation**: ρ(x ⊗ y) = ρ(x) ⊗ ρ(y)

## Geometric Interpretation

### Hypersphere Model

Normalized vectors lie on the unit hypersphere S^{d-1}.

**Geodesic Distance**:
```
d_geo(x, y) = arccos(x · y)
```

**Volume of ε-ball**:
```
Vol(B_ε) ∝ sin^{d-2}(ε)
```

### Projection Properties

For random subspaces U ⊂ V:
```
||P_U(x)||² ≈ (dim U / d) × ||x||²
```

## Encoding Theory

### Random Indexing

For item i with context words c_j:
```
v_i = ∑_{j} ε_j × r_j
```

Where:
- r_j are random index vectors
- ε_j ∈ {-1, 0, +1} are sparse random signs

### Spatial Encoding

For 2D position (x, y) in grid [0, N]²:
```
v_{x,y} = e^{2πi(ux/N + vy/N)}
```

Where u, v are frequency components.

### Temporal Encoding

For time series with lag k:
```
v_t = ∑_{k=1}^K w_k × ρ^k(x_{t-k})
```

## Optimization Theory

### Cleanup Memory

Finding nearest stored vector:
```
x* = argmin_{x_i ∈ Memory} ||q - x_i||²
```

Efficient via:
```
x* = argmax_{x_i ∈ Memory} q · x_i
```

### Sparse Approximation

For sparse VSA with sparsity s:
```
x_sparse = TopK(x, s×d)
```

Preserves similarity while reducing computation.

## Theoretical Guarantees

### Binding Preservation Theorem

For random vectors x, y, z:
```
P(|(x ⊗ y) · z| > ε) ≤ 2 exp(-dε²/2)
```

### Bundle Recovery Theorem

For bundle B = Bundle(x₁, ..., x_n) with n < √d:
```
P(argmax_i(B · x_i) = j) ≥ 1 - n × exp(-d/8n²)
```

### Robustness Theorem

With noise rate p < 0.5:
```
P(successful decoding) → 1 as d → ∞
```

## Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Encoding | O(1) | O(d) |
| Binding | O(d) or O(d log d) | O(d) |
| Bundling | O(nd) | O(d) |
| Similarity | O(d) | O(1) |
| Cleanup | O(Md) | O(Md) |

Where M is the number of stored items.

## Applications to Cognitive Modeling

### Working Memory Model

Capacity limit emerges from:
```
Items ≈ √d ≈ 7±2 for d ≈ 50
```

### Semantic Memory Model

Similarity structure preservation:
```
sim_semantic(A, B) ≈ sim_VSA(v_A, v_B)
```

## Further Reading

1. **Plate (2003)**: Holographic Reduced Representation
2. **Kanerva (2009)**: Hyperdimensional Computing
3. **Gayler (2003)**: Vector Symbolic Architectures
4. **Eliasmith (2013)**: How to Build a Brain
5. **Schlegel et al. (2021)**: A Comparison of VSA Architectures

## Summary

VSA provides a mathematically rigorous framework for cognitive computing with:
- Strong theoretical foundations
- Provable properties and guarantees
- Scalable computational complexity
- Direct applications to cognitive modeling

The high-dimensional nature provides robustness while the algebraic structure enables symbolic reasoning, making VSA a powerful tool for neural-symbolic integration.