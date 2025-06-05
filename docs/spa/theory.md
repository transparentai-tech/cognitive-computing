# SPA Theory and Mathematical Foundations

## Introduction

The Semantic Pointer Architecture (SPA) is grounded in mathematical principles from vector symbolic architectures, neuroscience, and cognitive science. This document provides a comprehensive overview of the theoretical foundations underlying SPA.

## Semantic Pointers

### Definition
A semantic pointer is a high-dimensional vector that represents information in a distributed manner. Formally:

```
S ∈ ℝᵈ, where d is the dimensionality (typically 256-1024)
```

### Properties
1. **Unit Length**: Semantic pointers are typically normalized: ||S|| = 1
2. **Random Distribution**: Initially drawn from a Gaussian distribution
3. **Orthogonality**: Random vectors are nearly orthogonal in high dimensions
4. **Compositionality**: Support vector operations for combining information

### Mathematical Operations

#### Binding (Circular Convolution)
Binding combines two semantic pointers into a composite representation:

```
C = A ⊛ B
```

Where ⊛ denotes circular convolution:
```
C[k] = Σᵢ A[i] × B[(k-i) mod d]
```

#### Unbinding (Circular Correlation)
Unbinding extracts a component from a composite:

```
A' = C ⊛ B*
```

Where B* is the approximate inverse of B.

#### Bundling (Superposition)
Bundling creates a representation containing multiple items:

```
S = (A + B + C) / √3
```

Normalization maintains unit length.

#### Similarity (Dot Product)
Similarity between pointers is measured by their dot product:

```
sim(A, B) = A · B = Σᵢ A[i] × B[i]
```

## Vocabulary and Cleanup

### Vocabulary Structure
A vocabulary V is a set of labeled semantic pointers:

```
V = {(label₁, S₁), (label₂, S₂), ..., (labelₙ, Sₙ)}
```

### Cleanup Memory
Cleanup finds the closest vocabulary item to a noisy vector:

```
cleanup(x) = argmax_{Si ∈ V} (x · Si)
```

This implements a form of associative memory with error correction.

### Parsing Expressions
SPA supports parsing complex expressions:

```
parse("A*B + C") = (SA ⊛ SB) + SC
```

## Action Selection Mechanism

### Basal Ganglia Model
The basal ganglia implements action selection through competitive dynamics:

```
utility_i = condition_i(state)
output_i = activation(utility_i - Σⱼ≠ᵢ wᵢⱼ × utility_j)
```

Where:
- utility_i is the raw utility of action i
- wᵢⱼ is the mutual inhibition weight
- activation is a threshold function

### Thalamus Routing
The thalamus routes information based on selected actions:

```
gate_i = thalamus_output_i
module_input = Σᵢ gate_i × routing_matrix[i] × source_output
```

### Action Rules
Actions are defined by condition-effect pairs:

```
Action = {
    condition: state → ℝ (utility function)
    effect: state → state' (state transformation)
}
```

## Cognitive Modules

### State Module
Maintains information with optional feedback:

```
state(t+1) = feedback × state(t) + (1-feedback) × input(t)
```

### Memory Module
Implements associative memory using HRR:

```
memory = Σᵢ key_i ⊛ value_i
recall(key) = cleanup(memory ⊛ key*)
```

### Gate Module
Controls information flow:

```
output = gate_signal × input
```

### Compare Module
Computes similarity between inputs:

```
output = input_a · input_b
```

## Production System

### Production Rules
A production is a condition-effect pair:

```
Production = {
    condition: state → bool
    effect: state → state'
    priority: ℝ
}
```

### Conflict Resolution
When multiple productions match:

```
selected = argmax_i (priority_i × match_strength_i)
```

### Pattern Matching
Conditions match against semantic pointer patterns:

```
match(pattern, state) = pattern · state > threshold
```

## Cognitive Control

### Working Memory Management
Working memory capacity is limited by interference:

```
interference = Σᵢ≠ⱼ |item_i · item_j|
capacity ≈ d / (k × average_interference)
```

### Attention Mechanism
Attention biases processing:

```
attended_state = state + α × attention_vector
```

### Task Switching
Task representations guide processing:

```
effective_rule = base_rule + task_context ⊛ task_modifier
```

## Neural Implementation

### Neural Engineering Framework (NEF)
SPA can be implemented using spiking neurons:

```
a_i(x) = G_i[α_i · x + bias_i]
```

Where:
- a_i is the activity of neuron i
- G_i is the neuron's activation function
- α_i is the encoding vector
- x is the represented value

### Decoding
The represented value is decoded from neural activity:

```
x̂ = Σᵢ a_i × d_i
```

Where d_i are decoding weights.

### Transformations
Linear transformations between populations:

```
y = Wx → connection_weights = W × decoders × encoders
```

## Information Capacity

### Single Pointer Capacity
The information capacity of a semantic pointer:

```
I ≈ d × log₂(1/ε)
```

Where ε is the desired error rate.

### Vocabulary Capacity
Number of items that can be reliably stored:

```
N ≈ 0.1 × d (for 95% accuracy)
```

### Binding Depth
Maximum binding depth before degradation:

```
depth_max ≈ log(d) / log(k)
```

Where k is the average binding noise factor.

## Biological Inspiration

### Basal Ganglia Correspondence
- **Striatum**: Evaluates action utilities
- **Globus Pallidus**: Implements mutual inhibition
- **Substantia Nigra**: Provides learning signals
- **Thalamus**: Routes selected actions

### Cortical Implementation
- **Semantic Pointers**: Distributed cortical representations
- **Binding**: Implemented via synchronous firing
- **Cleanup**: Pattern completion in associative cortex

### Working Memory
- **Prefrontal Cortex**: Maintains state representations
- **Feedback Loops**: Sustain activity over time
- **Gating**: Controlled by basal ganglia outputs

## Comparison with Other Approaches

### vs. Classical HRR
SPA extends HRR with:
- Integrated control mechanisms
- Modular architecture
- Production system capabilities
- Biological grounding

### vs. Tensor Product Representations
Advantages:
- Fixed dimensionality
- Efficient operations
- Graceful degradation
- Neural implementation

### vs. Symbolic Systems
Benefits:
- Continuous representations
- Similarity-based reasoning
- Noise tolerance
- Learning capability

## Mathematical Properties

### Algebraic Structure
Semantic pointers form an approximate algebra:
- **Identity**: Exists (approximately)
- **Inverse**: Approximate inverse via correlation
- **Associativity**: Approximate for binding
- **Commutativity**: Bundling is commutative

### Convergence Properties
- **Cleanup**: Converges to nearest vocabulary item
- **Action Selection**: Converges to stable selection
- **Production System**: Reaches fixed point or cycle

### Complexity Analysis
- **Binding/Unbinding**: O(d log d) with FFT
- **Similarity**: O(d)
- **Cleanup**: O(N × d) for N vocabulary items
- **Action Selection**: O(A²) for A actions

## Summary

SPA's theoretical foundations combine:
1. **Vector Symbolic Architectures**: For compositional representations
2. **Neural Engineering**: For biological implementation
3. **Cognitive Science**: For architectural principles
4. **Control Theory**: For action selection and routing

This mathematical framework enables SPA to bridge symbolic and neural approaches while maintaining biological plausibility and computational efficiency.

## References

1. Eliasmith, C. (2013). How to build a brain: A neural architecture for biological cognition.
2. Stewart, T. C., & Eliasmith, C. (2012). Compositionality and biologically plausible models.
3. Plate, T. A. (2003). Holographic reduced representations.
4. Gayler, R. W. (2003). Vector symbolic architectures answer Jackendoff's challenges.
5. Gurney, K., Prescott, T. J., & Redgrave, P. (2001). A computational model of action selection in the basal ganglia.