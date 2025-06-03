# HRR Theory: Mathematical Foundations

## Introduction

This document provides a comprehensive mathematical treatment of Holographic Reduced Representations (HRR). We explore the theoretical foundations, mathematical properties, and analytical framework that make HRR a powerful tool for cognitive computing.

## Mathematical Framework

### Vector Space

HRR operates in an n-dimensional real vector space ℝⁿ (or complex space ℂⁿ). The key insight is that high-dimensional random vectors are nearly orthogonal, providing a basis for distributed representation.

**Definition**: An HRR system consists of:
- Vector space V ⊆ ℝⁿ (or ℂⁿ)
- Binding operation: ⊛ : V × V → V
- Unbinding operation: ⊘ : V × V → V
- Bundling operation: ⊕ : V × V → V
- Similarity measure: sim : V × V → ℝ

### Circular Convolution

The fundamental operation in HRR is circular convolution, which serves as the binding operator.

**Definition**: For vectors **a**, **b** ∈ ℝⁿ, circular convolution is:

```
(a ⊛ b)ₖ = Σᵢ₌₀ⁿ⁻¹ aᵢ · b₍ₖ₋ᵢ₎ mod n
```

**Matrix Form**:
```
a ⊛ b = Cᵃ · b
```

Where Cᵃ is the circulant matrix of **a**:

```
Cᵃ = [
    a₀   a₁   a₂   ... aₙ₋₁
    aₙ₋₁ a₀   a₁   ... aₙ₋₂
    aₙ₋₂ aₙ₋₁ a₀   ... aₙ₋₃
    ...  ...  ...  ... ...
    a₁   a₂   a₃   ... a₀
]
```

### Fourier Transform Optimization

Circular convolution can be efficiently computed using the Discrete Fourier Transform (DFT):

**Convolution Theorem**:
```
a ⊛ b = F⁻¹(F(a) ⊙ F(b))
```

Where:
- F denotes the DFT
- F⁻¹ denotes the inverse DFT
- ⊙ denotes element-wise multiplication

**Complexity**:
- Direct convolution: O(n²)
- FFT-based convolution: O(n log n)

### Circular Correlation

The unbinding operation uses circular correlation, defined as:

```
(a ⊘ b)ₖ = Σᵢ₌₀ⁿ⁻¹ aᵢ · b₍ᵢ₊ₖ₎ mod n
```

**Relationship to Convolution**:
```
a ⊘ b = a ⊛ b†
```

Where b† is the involution of b:
- Real vectors: b† = [b₀, bₙ₋₁, bₙ₋₂, ..., b₁]
- Complex vectors: b† = conj([b₀, bₙ₋₁, bₙ₋₂, ..., b₁])

## Vector Properties

### Random Vectors

**Distribution**: Elements drawn from N(0, σ²) where σ² = 1/n

**Properties**:
1. **Expected dot product**: E[⟨a, b⟩] = 0 for independent a, b
2. **Variance**: Var[⟨a, b⟩] = 1/n
3. **Near orthogonality**: P(|⟨a, b⟩| > ε) ≈ 2exp(-nε²/2)

### Unitary Vectors

**Definition**: A vector **u** is unitary if:
```
u ⊛ u† = e₀ = [1, 0, 0, ..., 0]
```

**Properties**:
1. **Self-inverse**: u ⊛ u† = u† ⊛ u = e₀
2. **Magnitude preservation**: |u| = 1
3. **Perfect unbinding**: (u ⊛ v) ⊘ u = v

**Construction**:
```python
# In frequency domain
U = F(u)
|Uₖ| = 1 for all k  # Unit magnitude
arg(Uₖ) = arbitrary  # Random phase
```

## Binding and Unbinding Analysis

### Binding Properties

**Theorem 1**: Circular convolution approximately preserves vector magnitude.

*Proof sketch*: For random vectors a, b with E[aᵢ] = 0, Var[aᵢ] = 1/n:
```
E[|(a ⊛ b)|²] = E[|a|²] · E[|b|²]
```

**Theorem 2**: Binding produces dissimilar output.

For random unit vectors a, b, c:
```
E[⟨a ⊛ b, a⟩] = E[⟨a ⊛ b, b⟩] = 0
Var[⟨a ⊛ b, a⟩] = 1/n
```

### Unbinding Accuracy

**Theorem 3**: Unbinding with unitary vectors is exact.

If u is unitary:
```
(u ⊛ v) ⊘ u = v
```

**Theorem 4**: Unbinding with random vectors is approximate.

For random vectors a, b, c:
```
(a ⊛ b) ⊘ a = b + noise
E[noise] = 0
Var[noise] ∝ 1/n
```

### Signal-to-Noise Ratio

For retrieval of b from a ⊛ b using a:

```
SNR = signal_power / noise_power
    = n / (1 + |a|² · |b|² - n)
    ≈ n  (for unit vectors)
```

## Bundling Capacity

### Superposition Model

Bundling creates a sum of vectors:
```
s = Σᵢ₌₁ᵐ wᵢ · vᵢ
```

Where wᵢ are optional weights.

### Capacity Analysis

**Theorem 5**: Bundling capacity is limited by interference.

For m items bundled with equal weight:
```
⟨s, vᵢ⟩ = 1/m + Σⱼ≠ᵢ (1/m)⟨vⱼ, vᵢ⟩
```

Signal strength: 1/m
Noise variance: (m-1)/mn

**Critical capacity**: m* ≈ 0.14n (for 50% retrieval accuracy)

### Weighted Bundling

Optimal weights for unequal importance:
```
wᵢ = pᵢ / √(Σⱼ pⱼ²)
```

Where pᵢ is the relative importance of item i.

## Compositional Structures

### Role-Filler Binding

For role-filler pairs:
```
S = Σᵢ rᵢ ⊛ fᵢ
```

**Retrieval**:
```
fⱼ ≈ S ⊘ rⱼ = fⱼ + Σᵢ≠ⱼ (rᵢ ⊛ fᵢ) ⊘ rⱼ
```

**Crosstalk**: The noise term Σᵢ≠ⱼ (rᵢ ⊛ fᵢ) ⊘ rⱼ

### Recursive Structures

HRR supports recursive binding:
```
Tree = value ⊕ (left_role ⊛ left_tree) ⊕ (right_role ⊛ right_tree)
```

**Depth limitation**: Signal degrades exponentially with depth
- After d levels: SNR ∝ (1/m)^d

## Similarity Measures

### Cosine Similarity

Standard similarity measure:
```
sim(a, b) = ⟨a, b⟩ / (|a| · |b|)
```

**Properties**:
- Range: [-1, 1]
- Invariant to scaling
- Expected value for random vectors: 0

### Dot Product

For normalized vectors:
```
sim(a, b) = ⟨a, b⟩ = Σᵢ aᵢ · bᵢ
```

### Complex Vectors

For complex HRR:
```
sim(a, b) = Re(⟨a, b*⟩) / (|a| · |b|)
```

Where b* is the complex conjugate of b.

## Noise Analysis

### Gaussian Noise Model

Adding Gaussian noise N(0, σ²) to vectors:

**Effect on binding**:
```
(a + nₐ) ⊛ (b + nᵦ) = a ⊛ b + a ⊛ nᵦ + nₐ ⊛ b + nₐ ⊛ nᵦ
```

**SNR degradation**:
```
SNR_noisy = SNR_clean / (1 + σ²(2 + σ²))
```

### Cleanup Threshold

Optimal cleanup threshold:
```
θ* = 1 / (1 + exp(-2SNR))
```

This maximizes correct retrieval probability.

## Capacity and Scaling

### Storage Capacity

**Theorem 6**: Maximum reliable storage

For m role-filler pairs with error probability p:
```
m_max ≈ n / (2 log(1/p))
```

### Dimensional Scaling

Required dimensions for given capacity:
```
n ≥ 7m  (for 95% accuracy)
n ≥ 14m (for 99% accuracy)
```

### Computational Complexity

| Operation | Direct | FFT-based |
|-----------|--------|-----------|
| Binding | O(n²) | O(n log n) |
| Unbinding | O(n²) | O(n log n) |
| Bundling | O(n) | O(n) |
| Similarity | O(n) | O(n) |

## Theoretical Limitations

### 1. Binding Depth

Maximum recursive depth before signal loss:
```
d_max ≈ log_m(n/7)
```

Where m is branching factor.

### 2. Bundling Limit

Maximum superposition items:
```
m_max ≈ √(n/7)
```

### 3. Precision Bounds

Retrieval precision limited by:
```
ε ≥ 1/√n
```

## Extensions and Variants

### 1. Complex HRR

Using complex vectors doubles capacity:
- Real HRR: n dimensions
- Complex HRR: n complex = 2n real parameters

### 2. Binary HRR

Quantized version using {-1, +1}:
- Reduced precision
- Faster computation
- Hardware-friendly

### 3. Sparse HRR

Using sparse vectors:
- Reduced storage
- Faster operations
- Lower capacity

## Connections to Neuroscience

### Neural Implementation

HRR operations map to neural processes:
- **Binding**: Multiplicative synapses
- **Unbinding**: Pattern completion
- **Bundling**: Dendritic summation
- **Cleanup**: Attractor dynamics

### Biological Plausibility

1. **Distributed representation**: Like neural population codes
2. **Graceful degradation**: Matches neural robustness
3. **Fixed resources**: Analogous to fixed neuron count

## Conclusions

HRR provides a mathematically rigorous framework for:
- Encoding structured information in fixed-size vectors
- Manipulating symbolic structures with vector operations
- Bridging symbolic and connectionist approaches

Key theoretical insights:
1. High-dimensional random vectors provide quasi-orthogonal basis
2. Circular convolution enables reversible binding
3. Capacity scales with square root of dimensions
4. Noise tolerance emerges from distributed representation

## References

1. Plate, T. A. (1995). Holographic reduced representations. *IEEE Transactions on Neural Networks*, 6(3), 623-641.

2. Plate, T. A. (2003). *Holographic Reduced Representation: Distributed Representation for Cognitive Structures*. CSLI Publications.

3. Gayler, R. W. (1998). Multiplicative binding, representation operators & analogy. *Advances in Analogy Research*.

4. Kanerva, P. (1996). Binary spatter-coding of ordered k-tuples. *International Conference on Artificial Neural Networks*.

5. Eliasmith, C., & Anderson, C. H. (2003). *Neural engineering: Computation, representation, and dynamics in neurobiological systems*. MIT Press.