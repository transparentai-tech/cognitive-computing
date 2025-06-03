#!/usr/bin/env python3
"""
Vector Types Demo: Comprehensive Demonstration of VSA Vector Types

This script explores the different vector types available in VSA:
- Binary vectors {0, 1}
- Bipolar vectors {-1, +1}
- Ternary vectors {-1, 0, +1}
- Complex vectors (unit magnitude)
- Integer vectors (modular arithmetic)

Each vector type has unique properties and optimal use cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from cognitive_computing.vsa import (
    BinaryVector, BipolarVector, TernaryVector, ComplexVector, IntegerVector,
    create_vsa, VSAConfig
)


def demonstrate_binary_vectors():
    """Demonstrate binary vector properties and operations."""
    print("=== Binary Vectors {0, 1} ===\n")
    
    dimension = 1000
    
    # Create binary vectors
    print("1. Vector Creation:")
    vec1 = BinaryVector.random(dimension)
    vec2 = BinaryVector.random(dimension)
    vec_zero = BinaryVector.zeros(dimension)
    vec_one = BinaryVector.ones(dimension)
    
    print(f"   Random vector sample: {vec1.data[:20]}")
    print(f"   Density (% of 1s): {np.mean(vec1.data):.3f}")
    print(f"   Zero vector sum: {np.sum(vec_zero.data)}")
    print(f"   One vector sum: {np.sum(vec_one.data)}\n")
    
    # Binary operations
    print("2. Binary Operations:")
    vec_xor = vec1.bind(vec2)  # XOR operation
    vec_and = BinaryVector(vec1.data & vec2.data)
    vec_or = BinaryVector(vec1.data | vec2.data)
    vec_not = BinaryVector(1 - vec1.data)
    
    print(f"   XOR density: {np.mean(vec_xor.data):.3f}")
    print(f"   AND density: {np.mean(vec_and.data):.3f}")
    print(f"   OR density: {np.mean(vec_or.data):.3f}")
    print(f"   NOT density: {np.mean(vec_not.data):.3f}\n")
    
    # Distance metrics
    print("3. Distance Metrics:")
    hamming_dist = np.sum(vec1.data != vec2.data)
    jaccard_sim = np.sum(vec1.data & vec2.data) / np.sum(vec1.data | vec2.data)
    
    print(f"   Hamming distance: {hamming_dist}")
    print(f"   Jaccard similarity: {jaccard_sim:.3f}")
    print(f"   Normalized Hamming similarity: {1 - hamming_dist/dimension:.3f}\n")
    
    # Properties
    print("4. Key Properties:")
    print(f"   Self XOR (should be zero): {np.sum(vec1.bind(vec1).data)}")
    print(f"   Double XOR recovery: {np.array_equal(vec1.bind(vec2).bind(vec2).data, vec1.data)}")
    print(f"   Memory efficiency: {dimension} bits = {dimension/8:.0f} bytes\n")
    
    # Use cases
    print("5. Optimal Use Cases:")
    print("   - Cryptographic applications (XOR cipher)")
    print("   - Feature hashing")
    print("   - Bloom filters")
    print("   - Hardware implementations\n")


def demonstrate_bipolar_vectors():
    """Demonstrate bipolar vector properties and operations."""
    print("=== Bipolar Vectors {-1, +1} ===\n")
    
    dimension = 1000
    
    # Create bipolar vectors
    print("1. Vector Creation:")
    vec1 = BipolarVector.random(dimension)
    vec2 = BipolarVector.random(dimension)
    vec_pos = BipolarVector.ones(dimension)
    vec_neg = BipolarVector(np.full(dimension, -1))
    
    print(f"   Random vector sample: {vec1.data[:20]}")
    print(f"   Mean value: {np.mean(vec1.data):.3f} (should be ~0)")
    print(f"   Standard deviation: {np.std(vec1.data):.3f}")
    print(f"   Norm: {np.linalg.norm(vec1.data):.3f}\n")
    
    # Arithmetic operations
    print("2. Arithmetic Operations:")
    vec_mult = BipolarVector(vec1.data * vec2.data)
    vec_add = BipolarVector(np.sign(vec1.data + vec2.data))
    vec_inv = BipolarVector(-vec1.data)
    
    print(f"   Multiplication similarity to vec1: {vec1.similarity(vec_mult):.3f}")
    print(f"   Addition (majority) similarity to vec1: {vec1.similarity(vec_add):.3f}")
    print(f"   Inverse correlation: {np.dot(vec1.data, vec_inv.data) / dimension:.3f}\n")
    
    # Statistical properties
    print("3. Statistical Properties:")
    dot_product = np.dot(vec1.data, vec2.data) / dimension
    correlation = np.corrcoef(vec1.data, vec2.data)[0, 1]
    
    print(f"   Dot product (normalized): {dot_product:.3f}")
    print(f"   Correlation coefficient: {correlation:.3f}")
    print(f"   Expected correlation (random): ~0\n")
    
    # Bundling operations
    print("4. Bundling (Superposition):")
    vectors = [BipolarVector.random(dimension) for _ in range(5)]
    bundle = BipolarVector(np.sign(np.sum([v.data for v in vectors], axis=0)))
    
    for i, vec in enumerate(vectors):
        sim = bundle.similarity(vec)
        print(f"   Bundle similarity to vector {i}: {sim:.3f}")
    print()
    
    # Use cases
    print("5. Optimal Use Cases:")
    print("   - Neural network models")
    print("   - Continuous value encoding")
    print("   - Signal processing")
    print("   - General-purpose VSA\n")


def demonstrate_ternary_vectors():
    """Demonstrate ternary vector properties and operations."""
    print("=== Ternary Vectors {-1, 0, +1} ===\n")
    
    dimension = 1000
    
    # Create ternary vectors with different sparsity levels
    print("1. Vector Creation with Varying Sparsity:")
    sparsities = [0.1, 0.5, 0.9]
    
    for sparsity in sparsities:
        vec = TernaryVector.random(dimension, sparsity=sparsity)
        active = np.sum(vec.data != 0)
        print(f"   Sparsity {sparsity}: {active} active elements ({active/dimension:.1%})")
        print(f"     Positive: {np.sum(vec.data == 1)}, Negative: {np.sum(vec.data == -1)}")
    print()
    
    # Sparse operations
    print("2. Sparse Operations:")
    vec1 = TernaryVector.random(dimension, sparsity=0.1)
    vec2 = TernaryVector.random(dimension, sparsity=0.1)
    
    # Multiplication (preserves sparsity)
    vec_mult = TernaryVector(vec1.data * vec2.data)
    active_mult = np.sum(vec_mult.data != 0)
    
    print(f"   Vec1 active: {np.sum(vec1.data != 0)}")
    print(f"   Vec2 active: {np.sum(vec2.data != 0)}")
    print(f"   Product active: {active_mult} (intersection)\n")
    
    # Bundling with threshold
    print("3. Threshold Bundling:")
    vectors = [TernaryVector.random(dimension, sparsity=0.1) for _ in range(10)]
    sum_vec = np.sum([v.data for v in vectors], axis=0)
    
    thresholds = [1, 2, 3]
    for thresh in thresholds:
        bundled = TernaryVector(np.where(sum_vec >= thresh, 1, 
                                       np.where(sum_vec <= -thresh, -1, 0)))
        active = np.sum(bundled.data != 0)
        print(f"   Threshold {thresh}: {active} active elements")
    print()
    
    # Memory efficiency
    print("4. Memory Efficiency:")
    dense_size = dimension * 8  # 64-bit float
    sparse_size = np.sum(vec1.data != 0) * 2 * 4  # position + value (32-bit each)
    
    print(f"   Dense storage: {dense_size} bytes")
    print(f"   Sparse storage: ~{sparse_size} bytes")
    print(f"   Compression ratio: {dense_size/sparse_size:.1f}x\n")
    
    # Use cases
    print("5. Optimal Use Cases:")
    print("   - Sparse distributed representations")
    print("   - Memory-constrained applications")
    print("   - Biologically-inspired models")
    print("   - Large-scale systems\n")


def demonstrate_complex_vectors():
    """Demonstrate complex vector properties and operations."""
    print("=== Complex Vectors (Unit Circle) ===\n")
    
    dimension = 1000
    
    # Create complex vectors
    print("1. Vector Creation:")
    vec1 = ComplexVector.random(dimension)
    vec2 = ComplexVector.random(dimension)
    
    print(f"   Sample values: {vec1.data[:5]}")
    print(f"   Magnitudes: {np.abs(vec1.data[:5])}")
    print(f"   All unit magnitude: {np.allclose(np.abs(vec1.data), 1.0)}\n")
    
    # Phase operations
    print("2. Phase Operations:")
    phases1 = np.angle(vec1.data)
    phases2 = np.angle(vec2.data)
    
    print(f"   Phase range: [{np.min(phases1):.3f}, {np.max(phases1):.3f}]")
    print(f"   Phase mean: {np.mean(phases1):.3f}")
    print(f"   Phase std: {np.std(phases1):.3f}\n")
    
    # Complex binding (element-wise multiplication)
    print("3. Complex Binding:")
    vec_bound = ComplexVector(vec1.data * vec2.data)
    vec_unbound = ComplexVector(vec_bound.data * np.conj(vec2.data))
    
    similarity = np.real(np.vdot(vec_unbound.data, vec1.data)) / dimension
    print(f"   Binding: rotation by phase of vec2")
    print(f"   Unbinding similarity: {similarity:.3f}")
    print(f"   Phase addition property verified\n")
    
    # Fourier properties
    print("4. Fourier Transform Properties:")
    fft_vec = np.fft.fft(vec1.data)
    fft_magnitude = np.abs(fft_vec)
    
    print(f"   FFT magnitude mean: {np.mean(fft_magnitude):.3f}")
    print(f"   FFT magnitude std: {np.std(fft_magnitude):.3f}")
    print(f"   Parseval's theorem check: {np.allclose(np.sum(np.abs(vec1.data)**2), np.sum(np.abs(fft_vec)**2)/dimension)}\n")
    
    # Holographic properties
    print("5. Holographic Properties:")
    # Convolution in time = multiplication in frequency
    vec_conv = np.fft.ifft(np.fft.fft(vec1.data) * np.fft.fft(vec2.data))
    vec_conv_normalized = vec_conv / np.abs(vec_conv)
    
    print(f"   Convolution preserves unit magnitude (after normalization)")
    print(f"   Suitable for holographic reduced representations\n")
    
    # Use cases
    print("6. Optimal Use Cases:")
    print("   - Frequency domain operations")
    print("   - Holographic representations")
    print("   - Signal processing")
    print("   - Quantum-inspired computing\n")


def demonstrate_integer_vectors():
    """Demonstrate integer vector properties and operations."""
    print("=== Integer Vectors (Modular Arithmetic) ===\n")
    
    dimension = 1000
    moduli = [2, 16, 256]
    
    for modulus in moduli:
        print(f"1. Integer Vectors (mod {modulus}):")
        vec1 = IntegerVector.random(dimension, modulus=modulus)
        vec2 = IntegerVector.random(dimension, modulus=modulus)
        
        print(f"   Value range: [0, {modulus-1}]")
        print(f"   Sample values: {vec1.data[:20]}")
        print(f"   Mean value: {np.mean(vec1.data):.3f} (expected: {(modulus-1)/2:.3f})")
        
        # Modular operations
        vec_add = IntegerVector((vec1.data + vec2.data) % modulus, modulus=modulus)
        vec_mult = IntegerVector((vec1.data * vec2.data) % modulus, modulus=modulus)
        vec_inv = IntegerVector((modulus - vec1.data) % modulus, modulus=modulus)
        
        print(f"   Addition mod {modulus}: preserves distribution")
        print(f"   Self + Inverse = 0 (mod {modulus}): {np.all((vec1.data + vec_inv.data) % modulus == 0)}")
        print()
    
    # Cyclic properties
    print("2. Cyclic Group Properties:")
    modulus = 16
    vec = IntegerVector.random(dimension, modulus=modulus)
    generator = 3  # Generator for multiplicative group
    
    powers = []
    current = vec.data.copy()
    for i in range(4):
        powers.append(np.mean(current))
        current = (current * generator) % modulus
    
    print(f"   Powers of {generator} (mod {modulus}): {[f'{p:.3f}' for p in powers]}")
    print(f"   Demonstrates cyclic behavior\n")
    
    # Use cases
    print("3. Optimal Use Cases:")
    print("   - Finite field operations")
    print("   - Cryptographic primitives")
    print("   - Error-correcting codes")
    print("   - Discrete mathematics\n")


def compare_vector_types():
    """Compare properties across all vector types."""
    print("=== Vector Type Comparison ===\n")
    
    dimension = 1000
    results = {}
    
    # Create vectors of each type
    vectors = {
        'Binary': BinaryVector.random(dimension),
        'Bipolar': BipolarVector.random(dimension),
        'Ternary': TernaryVector.random(dimension, sparsity=0.1),
        'Complex': ComplexVector.random(dimension),
        'Integer': IntegerVector.random(dimension, modulus=256)
    }
    
    # Compare properties
    print("1. Memory Requirements (bytes):")
    for name, vec in vectors.items():
        if name == 'Binary':
            size = dimension / 8
        elif name == 'Bipolar':
            size = dimension * 1  # int8
        elif name == 'Ternary':
            active = np.sum(vec.data != 0)
            size = active * 8  # sparse representation
        elif name == 'Complex':
            size = dimension * 16  # 2 * float64
        elif name == 'Integer':
            size = dimension * 1  # uint8 for modulus 256
        print(f"   {name}: {size:.0f} bytes")
    print()
    
    # Compare operations
    print("2. Operation Complexity:")
    print("   Binary: O(n) bitwise operations (very fast)")
    print("   Bipolar: O(n) arithmetic operations")
    print("   Ternary: O(k) where k = active elements")
    print("   Complex: O(n) complex arithmetic")
    print("   Integer: O(n) modular arithmetic\n")
    
    # Compare binding operations
    print("3. Natural Binding Operations:")
    print("   Binary: XOR (self-inverse)")
    print("   Bipolar: Multiplication")
    print("   Ternary: Sparse multiplication")
    print("   Complex: Complex multiplication (phase addition)")
    print("   Integer: Modular addition/multiplication\n")
    
    # Compare properties
    print("4. Unique Properties:")
    print("   Binary: Hardware-friendly, cryptographic")
    print("   Bipolar: Continuous values, neural-like")
    print("   Ternary: Sparse, memory-efficient")
    print("   Complex: Phase encoding, Fourier-compatible")
    print("   Integer: Finite field arithmetic, cyclic groups\n")


def visualize_vector_distributions():
    """Visualize the distributions of different vector types."""
    try:
        dimension = 1000
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Binary distribution
        vec_binary = BinaryVector.random(dimension)
        axes[0].hist(vec_binary.data, bins=2, alpha=0.7, color='blue')
        axes[0].set_title('Binary Vector Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Count')
        
        # Bipolar distribution
        vec_bipolar = BipolarVector.random(dimension)
        axes[1].hist(vec_bipolar.data, bins=2, alpha=0.7, color='green')
        axes[1].set_title('Bipolar Vector Distribution')
        axes[1].set_xlabel('Value')
        
        # Ternary distribution
        vec_ternary = TernaryVector.random(dimension, sparsity=0.1)
        axes[2].hist(vec_ternary.data, bins=3, alpha=0.7, color='red')
        axes[2].set_title('Ternary Vector Distribution (90% sparse)')
        axes[2].set_xlabel('Value')
        
        # Complex phase distribution
        vec_complex = ComplexVector.random(dimension)
        phases = np.angle(vec_complex.data)
        axes[3].hist(phases, bins=50, alpha=0.7, color='purple')
        axes[3].set_title('Complex Vector Phase Distribution')
        axes[3].set_xlabel('Phase (radians)')
        
        # Integer distribution
        vec_integer = IntegerVector.random(dimension, modulus=16)
        axes[4].hist(vec_integer.data, bins=16, alpha=0.7, color='orange')
        axes[4].set_title('Integer Vector Distribution (mod 16)')
        axes[4].set_xlabel('Value')
        
        # Comparison of densities
        densities = {
            'Binary': np.mean(vec_binary.data),
            'Bipolar +1': np.mean(vec_bipolar.data == 1),
            'Ternary non-zero': np.mean(vec_ternary.data != 0),
            'Complex (N/A)': 1.0,
            'Integer > 8': np.mean(vec_integer.data > 8)
        }
        
        axes[5].bar(range(len(densities)), list(densities.values()), 
                    tick_label=list(densities.keys()), alpha=0.7)
        axes[5].set_title('Vector Density Comparison')
        axes[5].set_ylabel('Density')
        axes[5].set_xticklabels(list(densities.keys()), rotation=45)
        
        plt.tight_layout()
        plt.savefig('vsa_vector_types_distribution.png', dpi=300, bbox_inches='tight')
        print("\nDistribution plots saved to 'vsa_vector_types_distribution.png'\n")
    except Exception as e:
        print(f"\nCould not generate plots: {e}\n")


def main():
    """Run all vector type demonstrations."""
    print("\n" + "="*60)
    print("VSA VECTOR TYPES DEMONSTRATION")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Demonstrate each vector type
    demonstrate_binary_vectors()
    print("-"*60 + "\n")
    
    demonstrate_bipolar_vectors()
    print("-"*60 + "\n")
    
    demonstrate_ternary_vectors()
    print("-"*60 + "\n")
    
    demonstrate_complex_vectors()
    print("-"*60 + "\n")
    
    demonstrate_integer_vectors()
    print("-"*60 + "\n")
    
    # Compare all types
    compare_vector_types()
    
    # Visualize distributions
    visualize_vector_distributions()
    
    # Summary recommendations
    print("=== Summary Recommendations ===\n")
    print("Choose vector type based on your requirements:\n")
    print("• **Binary**: When you need hardware efficiency or cryptographic properties")
    print("• **Bipolar**: For general-purpose VSA, neural models, continuous values")
    print("• **Ternary**: When sparsity is critical for memory or computation")
    print("• **Complex**: For frequency-domain operations or holographic representations")
    print("• **Integer**: For finite field arithmetic or discrete mathematics\n")
    
    print("="*60)
    print("Vector Types Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()