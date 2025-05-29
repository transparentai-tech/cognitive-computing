# Sparse Distributed Memory: Mathematical Theory

This document provides a rigorous mathematical treatment of Sparse Distributed Memory (SDM), including theoretical foundations, proofs, and analysis.

## Table of Contents
- [Mathematical Foundations](#mathematical-foundations)
- [The SDM Model](#the-sdm-model)
- [Storage and Retrieval Operations](#storage-and-retrieval-operations)
- [Capacity Analysis](#capacity-analysis)
- [Signal-to-Noise Ratio](#signal-to-noise-ratio)
- [Convergence Properties](#convergence-properties)
- [Distance Metrics and Activation](#distance-metrics-and-activation)
- [Theoretical Optimality](#theoretical-optimality)
- [Extensions and Variations](#extensions-and-variations)
- [Proofs](#proofs)

## Mathematical Foundations

### Binary Vector Space

SDM operates in an n-dimensional binary vector space:

**Definition 1.1**: The address space is defined as A = {0,1}ⁿ, where n is the dimension.

**Definition 1.2**: The Hamming distance between two vectors x, y ∈ A is:
```
d_H(x, y) = Σᵢ₌₁ⁿ |xᵢ - yᵢ| = ||x ⊕ y||₁
```

where ⊕ denotes XOR operation.

### Probability Distributions

**Definition 1.3**: For random binary vectors with P(xᵢ = 1) = p:
- Mean number of 1s: μ = np
- Variance: σ² = np(1-p)
- For p = 0.5 (uniform distribution): μ = n/2, σ² = n/4

**Theorem 1.1**: The probability that two random vectors have Hamming distance k is:
```
P(d_H(x, y) = k) = (n choose k) × 2⁻ⁿ
```

This follows a binomial distribution with parameters (n, 0.5).

## The SDM Model

### Hard Locations

**Definition 2.1**: SDM consists of M hard locations {h₁, h₂, ..., h_M}, where each hᵢ ∈ A.

**Definition 2.2**: The activation function for location i given address x is:
```
a_i(x) = {
    1, if d_H(x, hᵢ) ≤ r
    0, otherwise
}
```
where r is the activation radius.

### Storage Arrays

**Definition 2.3**: Each hard location i has an associated storage array:
- **Counter method**: Cᵢ ∈ ℤⁿ (n-dimensional integer vector)
- **Binary method**: Bᵢ ∈ {0,1}ⁿ (n-dimensional binary vector)

## Storage and Retrieval Operations

### Storage Operation

**Counter Method**:
For storing data vector d at address x:
```
Cᵢ(t+1) = Cᵢ(t) + aᵢ(x) × (2d - 1)
```
where (2d - 1) converts binary {0,1} to bipolar {-1,+1}.

**Binary Method**:
```
Bᵢ(t+1) = Bᵢ(t) ∨ (aᵢ(x) × d)
```
where ∨ denotes bitwise OR.

### Retrieval Operation

**Counter Method**:
```
S(x) = Σᵢ₌₁ᴹ aᵢ(x) × Cᵢ
d̂ⱼ = {
    1, if Sⱼ(x) > θ
    0, otherwise
}
```
where θ is the threshold (typically 0).

**Binary Method**:
```
d̂ⱼ = {
    1, if Σᵢ₌₁ᴹ aᵢ(x) × Bᵢⱼ > Σᵢ₌₁ᴹ aᵢ(x) / 2
    0, otherwise
}
```

## Capacity Analysis

### Activation Probability

**Theorem 3.1**: The probability that a random hard location is activated by a random address is:
```
P_a = Σₖ₌₀ʳ (n choose k) × 2⁻ⁿ
```

For large n and r ≈ n/2:
```
P_a ≈ 2⁻ⁿᴴ⁽ʳ/ⁿ⁾
```
where H(p) = -p log₂(p) - (1-p) log₂(1-p) is the binary entropy function.

### Expected Number of Active Locations

**Theorem 3.2**: The expected number of active locations is:
```
E[K] = M × P_a
```

**Corollary 3.1**: For optimal performance, E[K] should be approximately √M.

### Storage Capacity

**Theorem 3.3** (Kanerva's Capacity Formula): The capacity of SDM is approximately:
```
C ≈ M × (S²/2) × (1/E[K])
```
where S is the signal strength (number of storage operations per pattern).

**Corollary 3.2**: For S = 1 and optimal activation:
```
C ≈ M / (2√M) = √M / 2
```

### Critical Distance

**Definition 4.1**: The critical distance r_c is the activation radius that maximizes capacity:
```
r_c ≈ n × (0.5 - 1/(2√(2πn)))
```

For large n:
```
r_c ≈ 0.451n
```

## Signal-to-Noise Ratio

### Noise Model

**Definition 5.1**: The noise in retrieval consists of:
1. **Crosstalk noise**: Interference from other stored patterns
2. **Activation noise**: Variance in activation patterns

### SNR Analysis

**Theorem 5.1**: The signal-to-noise ratio for retrieving pattern p is:
```
SNR = S_p × √K / √(N × σ²)
```
where:
- S_p = number of times pattern p was stored
- K = number of active locations
- N = total number of stored patterns
- σ² = variance of the noise

**Corollary 5.1**: The bit error rate is approximately:
```
BER ≈ Q(√SNR)
```
where Q is the complementary error function.

### Information Theoretic Bounds

**Theorem 5.2**: The information capacity per hard location is bounded by:
```
I ≤ log₂(1 + SNR) bits
```

**Corollary 5.2**: Total information capacity:
```
I_total ≤ M × log₂(1 + SNR) bits
```

## Convergence Properties

### Iterative Retrieval

**Definition 6.1**: Iterative retrieval uses the output as the new input:
```
x^(t+1) = f(x^(t))
```
where f is the retrieval function.

**Theorem 6.1**: Under certain conditions, iterative retrieval converges to a fixed point:
```
||x^(t+1) - x^(t)|| → 0 as t → ∞
```

### Basin of Attraction

**Definition 6.2**: The basin of attraction for pattern p is:
```
B(p) = {x ∈ A : lim_{t→∞} f^t(x) = p}
```

**Theorem 6.2**: The radius of the basin of attraction is approximately:
```
r_B ≈ r_c × √(S_p / N_avg)
```
where N_avg is the average noise level.

## Distance Metrics and Activation

### Hamming Distance Properties

**Lemma 7.1**: Hamming distance satisfies the triangle inequality:
```
d_H(x, z) ≤ d_H(x, y) + d_H(y, z)
```

**Lemma 7.2**: Expected Hamming distance between random vectors:
```
E[d_H(x, y)] = n/2
Var[d_H(x, y)] = n/4
```

### Alternative Distance Metrics

**Definition 7.1**: Jaccard distance for binary vectors:
```
d_J(x, y) = 1 - |x ∩ y| / |x ∪ y|
```

**Theorem 7.3**: Relationship between Hamming and Jaccard:
```
d_J(x, y) = d_H(x, y) / (n - n_00)
```
where n_00 is the number of positions where both x and y are 0.

### Activation Function Analysis

**Theorem 7.4**: The activation volume (number of addresses activating location h) is:
```
V(h) = Σₖ₌₀ʳ (n choose k)
```

**Corollary 7.3**: The activation density is:
```
ρ = V(h) / 2ⁿ = P_a
```

## Theoretical Optimality

### Optimal Parameters

**Theorem 8.1**: For maximum capacity, the optimal parameters satisfy:
```
r_opt = arg max_r [C(r)] ≈ 0.451n
M_opt ≈ 2^(αn) where α ∈ [0.1, 0.2]
```

### Trade-offs

**Theorem 8.2**: The fundamental trade-off in SDM:
```
Capacity × Reliability × Speed = Constant
```

More precisely:
```
C × (1 - BER) × (1/T) ≤ K
```
where T is retrieval time and K is a system-dependent constant.

### Efficiency Metrics

**Definition 8.1**: Storage efficiency:
```
η_s = C × n / (M × n × b)
```
where b is bits per counter.

**Definition 8.2**: Energy efficiency:
```
η_e = C / (M × E[K] × E_op)
```
where E_op is energy per operation.

## Extensions and Variations

### Weighted SDM

**Definition 9.1**: Weighted activation function:
```
w_i(x) = exp(-λ × d_H(x, h_i))
```

**Theorem 9.1**: Weighted SDM has smoother interpolation properties.

### Hierarchical SDM

**Definition 9.2**: Multi-level SDM with L levels:
```
M_total = Σₗ₌₁ᴸ M_l
```

**Theorem 9.2**: Hierarchical SDM can achieve:
```
C_hierarchical ≈ L × C_single
```

### Continuous SDM

**Definition 9.3**: Extension to continuous vectors:
```
x ∈ [0,1]ⁿ or x ∈ ℝⁿ
```

**Theorem 9.3**: With appropriate activation functions, continuous SDM maintains similar properties.

## Proofs

### Proof of Theorem 3.1 (Activation Probability)

For a random hard location h and random address x:
```
P(d_H(x, h) = k) = (n choose k) × 2⁻ⁿ
```

Therefore:
```
P_a = P(d_H(x, h) ≤ r) = Σₖ₌₀ʳ (n choose k) × 2⁻ⁿ
```

Using Stirling's approximation and the central limit theorem for large n:
```
P_a ≈ Φ((r - n/2) / √(n/4))
```
where Φ is the cumulative normal distribution.

### Proof of Critical Distance (Sketch)

The capacity function is:
```
C(r) = M × f(r) × g(r)
```
where:
- f(r) = storage efficiency (decreases with r)
- g(r) = noise tolerance (increases with r)

Taking the derivative:
```
dC/dr = M × (f'(r)g(r) + f(r)g'(r)) = 0
```

Solving yields r_c ≈ 0.451n.

### Proof of SNR Formula (Sketch)

Signal strength from K active locations:
```
Signal = S_p × K
```

Noise from crosstalk (N patterns, each activating ~K locations):
```
Noise² = N × K × σ²
```

Therefore:
```
SNR = Signal² / Noise² = (S_p × K)² / (N × K × σ²) = S_p² × K / (N × σ²)
```

## References

1. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.

2. Kanerva, P. (1993). "Sparse Distributed Memory and Related Models." In *Associative Neural Memories: Theory and Implementation*, pp. 50-76.

3. Jaeckel, L. A. (1989). "An Alternative Design for a Sparse Distributed Memory." RIACS Technical Report 89.28.

4. Chou, P. A. (1989). "The Capacity of the Kanerva Associative Memory." *IEEE Transactions on Information Theory*, 35(2), 281-298.

5. Keeler, J. D. (1988). "Comparison Between Kanerva's SDM and Hopfield-Type Neural Networks." *Cognitive Science*, 12(3), 299-329.

6. Rogers, D. (1989). "Statistical Prediction with Kanerva's Sparse Distributed Memory." In *Advances in Neural Information Processing Systems*.

7. Anwar, A., & Franklin, S. (2003). "Sparse Distributed Memory for 'Conscious' Software Agents." *Cognitive Systems Research*, 4(4), 339-354.

## Appendix: Notation Summary

- n: Dimension of binary space
- M: Number of hard locations
- r: Activation radius
- d_H: Hamming distance
- P_a: Activation probability
- C: Capacity (number of patterns)
- K: Number of active locations
- SNR: Signal-to-noise ratio
- BER: Bit error rate
- r_c: Critical distance