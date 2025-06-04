#!/usr/bin/env python3
"""
Binding Comparison Demo: Compare Different VSA Binding Methods

This script provides a comprehensive comparison of different binding operations
in Vector Symbolic Architectures, including:
- Performance characteristics
- Properties (commutativity, associativity, self-inverse)
- Noise tolerance
- Capacity for multiple bindings
- Use case recommendations
"""

import numpy as np
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from cognitive_computing.vsa import create_vsa, BSC, MAP, FHRR


def measure_binding_performance(dimension: int = 1000, num_operations: int = 1000) -> Dict[str, Dict[str, float]]:
    """Measure performance of different binding operations."""
    print("=== Binding Performance Comparison ===\n")
    
    results = {}
    
    # Test configurations
    configs = [
        ('XOR (Binary)', 'binary', 'bsc', None),
        ('Multiplication (Bipolar)', 'bipolar', 'custom', 'multiplication'),
        ('Convolution (Complex)', 'complex', 'fhrr', None),
        ('MAP (Bipolar)', 'bipolar', 'map', None),
        ('Permutation (Bipolar)', 'bipolar', 'custom', 'permutation')
    ]
    
    for name, vector_type, vsa_type, binding_method in configs:
        print(f"Testing {name}...")
        
        # Create VSA instance
        kwargs = {'dimension': dimension, 'vector_type': vector_type, 'vsa_type': vsa_type}
        if binding_method:
            kwargs['binding_method'] = binding_method
        vsa = create_vsa(**kwargs)
        
        # Generate test vectors
        vectors_a = [vsa.generate_vector() for i in range(num_operations)]
        vectors_b = [vsa.generate_vector() for i in range(num_operations)]
        
        # Measure binding time
        start_time = time.time()
        for a, b in zip(vectors_a, vectors_b):
            _ = vsa.bind(a, b)
        bind_time = (time.time() - start_time) / num_operations * 1000  # ms per operation
        
        # Measure unbinding time
        bound_vectors = [vsa.bind(a, b) for a, b in zip(vectors_a[:100], vectors_b[:100])]
        start_time = time.time()
        for bound, key in zip(bound_vectors, vectors_a[:100]):
            _ = vsa.unbind(bound, key)
        unbind_time = (time.time() - start_time) / 100 * 1000  # ms per operation
        
        results[name] = {
            'bind_time': bind_time,
            'unbind_time': unbind_time,
            'total_time': bind_time + unbind_time
        }
        
        print(f"  Bind time: {bind_time:.3f} ms/op")
        print(f"  Unbind time: {unbind_time:.3f} ms/op\n")
    
    return results


def test_binding_properties(dimension: int = 1000) -> Dict[str, Dict[str, bool]]:
    """Test mathematical properties of binding operations."""
    print("=== Binding Properties Test ===\n")
    
    properties = {}
    
    configs = [
        ('XOR', 'binary', 'bsc', None),
        ('Multiplication', 'bipolar', 'custom', 'multiplication'),
        ('Convolution', 'complex', 'fhrr', None),
        ('MAP', 'bipolar', 'map', None),
        ('Permutation', 'bipolar', 'custom', 'permutation')
    ]
    
    for name, vector_type, vsa_type, binding_method in configs:
        print(f"Testing {name} properties...")
        
        kwargs = {'dimension': dimension, 'vector_type': vector_type, 'vsa_type': vsa_type}
        if binding_method:
            kwargs['binding_method'] = binding_method
        vsa = create_vsa(**kwargs)
        
        # Generate test vectors
        a = vsa.generate_vector()
        b = vsa.generate_vector()
        c = vsa.generate_vector()
        # Note: VSA doesn't have identity() method - use appropriate identity for binding type
        if name == 'XOR':
            identity = np.zeros(dimension, dtype=np.uint8)  # All zeros for XOR
        elif name in ['Multiplication', 'MAP']:
            identity = np.ones(dimension)  # All ones for multiplication
        else:
            identity = vsa.generate_vector()  # No clear identity for other operations
        
        # Test commutativity: a ⊗ b = b ⊗ a
        ab = vsa.bind(a, b)
        ba = vsa.bind(b, a)
        commutative = vsa.similarity(ab, ba) > 0.95
        
        # Test associativity: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
        ab_c = vsa.bind(vsa.bind(a, b), c)
        a_bc = vsa.bind(a, vsa.bind(b, c))
        associative = vsa.similarity(ab_c, a_bc) > 0.95
        
        # Test identity: a ⊗ I = a
        a_identity = vsa.bind(a, identity)
        has_identity = vsa.similarity(a_identity, a) > 0.95
        
        # Test self-inverse: a ⊗ a = I (for some operations)
        aa = vsa.bind(a, a)
        self_inverse = vsa.similarity(aa, identity) > 0.95
        
        # Test distributivity over bundling (approximate)
        bundle = vsa.bundle([b, c])
        a_bundle = vsa.bind(a, bundle)
        ab_ac = vsa.bundle([vsa.bind(a, b), vsa.bind(a, c)])
        distributive = vsa.similarity(a_bundle, ab_ac) > 0.7
        
        properties[name] = {
            'commutative': commutative,
            'associative': associative,
            'has_identity': has_identity,
            'self_inverse': self_inverse,
            'distributive': distributive
        }
        
        print(f"  Commutative: {commutative}")
        print(f"  Associative: {associative}")
        print(f"  Has identity: {has_identity}")
        print(f"  Self-inverse: {self_inverse}")
        print(f"  Distributive: {distributive}\n")
    
    return properties


def test_noise_tolerance(dimension: int = 1000, noise_levels: List[float] = None) -> Dict[str, List[float]]:
    """Test noise tolerance of different binding methods."""
    print("=== Noise Tolerance Test ===\n")
    
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {}
    
    configs = [
        ('XOR', 'binary', 'bsc', None),
        ('Multiplication', 'bipolar', 'custom', 'multiplication'),
        ('Convolution', 'complex', 'fhrr', None),
        ('MAP', 'bipolar', 'map', None)
    ]
    
    for name, vector_type, vsa_type, binding_method in configs:
        print(f"Testing {name} noise tolerance...")
        
        kwargs = {'dimension': dimension, 'vector_type': vector_type, 'vsa_type': vsa_type}
        if binding_method:
            kwargs['binding_method'] = binding_method
        vsa = create_vsa(**kwargs)
        
        # Create test vectors
        key = vsa.generate_vector()
        value = vsa.generate_vector()
        bound = vsa.bind(key, value)
        
        similarities = []
        
        for noise_level in noise_levels:
            # Add noise to bound vector
            if vector_type == 'binary':
                # Flip bits with probability noise_level
                noise_mask = np.random.random(dimension) < noise_level
                noisy_bound = bound.copy()
                noisy_bound[noise_mask] = 1 - noisy_bound[noise_mask]
            elif vector_type == 'bipolar':
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, dimension)
                noisy_bound = bound + noise
            elif vector_type == 'complex':
                # Add complex noise
                noise = np.random.normal(0, noise_level, dimension) + 1j * np.random.normal(0, noise_level, dimension)
                noisy_bound = bound + noise
                # Renormalize
                noisy_bound = noisy_bound / np.abs(noisy_bound)
            else:
                noisy_bound = bound  # Skip noise for other types
            
            # Try to recover value
            recovered = vsa.unbind(noisy_bound, key)
            similarity = vsa.similarity(recovered, value)
            similarities.append(similarity)
        
        results[name] = similarities
        print(f"  Similarities at different noise levels: {[f'{s:.3f}' for s in similarities]}\n")
    
    return results, noise_levels


def test_binding_capacity(dimension: int = 1000, max_pairs: int = 20) -> Dict[str, List[float]]:
    """Test capacity for multiple bindings."""
    print("=== Binding Capacity Test ===\n")
    
    results = {}
    
    configs = [
        ('XOR', 'binary', 'bsc', None),
        ('Multiplication', 'bipolar', 'custom', 'multiplication'),
        ('MAP', 'bipolar', 'map', None)
    ]
    
    for name, vector_type, vsa_type, binding_method in configs:
        print(f"Testing {name} capacity...")
        
        kwargs = {'dimension': dimension, 'vector_type': vector_type, 'vsa_type': vsa_type}
        if binding_method:
            kwargs['binding_method'] = binding_method
        vsa = create_vsa(**kwargs)
        
        similarities = []
        
        for num_pairs in range(1, max_pairs + 1):
            # Create multiple key-value pairs
            keys = [vsa.generate_vector() for i in range(num_pairs)]
            values = [vsa.generate_vector() for i in range(num_pairs)]
            
            # Bind all pairs and bundle
            bound_pairs = [vsa.bind(k, v) for k, v in zip(keys, values)]
            bundle = vsa.bundle(bound_pairs)
            
            # Try to recover each value
            recovered_sims = []
            for key, value in zip(keys, values):
                recovered = vsa.unbind(bundle, key)
                sim = vsa.similarity(recovered, value)
                recovered_sims.append(sim)
            
            # Average similarity
            avg_similarity = np.mean(recovered_sims)
            similarities.append(avg_similarity)
        
        results[name] = similarities
        print(f"  Average similarities for 1-{max_pairs} pairs: {[f'{s:.3f}' for s in similarities[::5]]}\n")
    
    return results


def demonstrate_use_cases():
    """Demonstrate specific use cases for each binding method."""
    print("=== Binding Method Use Cases ===\n")
    
    dimension = 1000
    
    # 1. XOR for Binary Classification
    print("1. XOR Binding - Binary Feature Binding:")
    vsa_xor = create_vsa(
        dimension=dimension,
        vector_type='binary',
        vsa_type='bsc'
    )
    
    # Bind binary features
    has_fur = vsa_xor.generate_vector()
    mammal = vsa_xor.generate_vector()
    furry_mammal = vsa_xor.bind(has_fur, mammal)
    
    # XOR is self-inverse
    recovered_mammal = vsa_xor.bind(furry_mammal, has_fur)
    print(f"   Self-inverse property: {vsa_xor.similarity(recovered_mammal, mammal):.3f}")
    print(f"   Use case: Binary feature combinations, cryptographic applications\n")
    
    # 2. Multiplication for Continuous Values
    print("2. Multiplication Binding - Weighted Combinations:")
    vsa_mult = create_vsa(
        dimension=dimension,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )
    
    # Combine features with weights
    feature1 = vsa_mult.generate_vector()
    feature2 = vsa_mult.generate_vector()
    weight1 = 0.7
    weight2 = 0.3
    
    # Create weight vectors
    weight_vec1 = vsa_mult.generate_vector()
    weight_vec2 = vsa_mult.generate_vector()
    
    weighted_combo = vsa_mult.bundle([
        vsa_mult.bind(feature1, weight_vec1),
        vsa_mult.bind(feature2, weight_vec2)
    ], weights=[weight1, weight2])
    print(f"   Use case: Weighted feature combinations, neural network emulation\n")
    
    # 3. Convolution for Sequential Data
    print("3. Convolution Binding - Sequential Processing:")
    vsa_conv = create_vsa(
        dimension=dimension,
        vector_type='complex',
        vsa_type='fhrr'
    )
    
    # Encode sequence
    words = ['the', 'quick', 'brown', 'fox']
    positions = [vsa_conv.generate_vector() for i in range(len(words))]
    word_vecs = [vsa_conv.generate_vector() for w in words]
    
    sequence = vsa_conv.bundle([
        vsa_conv.bind(pos, word) for pos, word in zip(positions, word_vecs)
    ])
    
    # Recover word at position 2
    recovered = vsa_conv.unbind(sequence, positions[2])
    print(f"   Recovered 'brown': {vsa_conv.similarity(recovered, word_vecs[2]):.3f}")
    print(f"   Use case: Natural language processing, sequence modeling\n")
    
    # 4. MAP for Robust Binding
    print("4. MAP Binding - Noise-Robust Combinations:")
    vsa_map = create_vsa(
        dimension=dimension,
        vector_type='bipolar',
        vsa_type='map'
    )
    
    # Multiple bindings with MAP
    role1 = vsa_map.generate_vector()
    role2 = vsa_map.generate_vector()
    role3 = vsa_map.generate_vector()
    
    filler1 = vsa_map.generate_vector()
    filler2 = vsa_map.generate_vector()
    filler3 = vsa_map.generate_vector()
    
    sentence = vsa_map.bundle([
        vsa_map.bind(role1, filler1),
        vsa_map.bind(role2, filler2),
        vsa_map.bind(role3, filler3)
    ])
    
    print(f"   MAP combines multiplication, addition, and permutation")
    print(f"   Use case: Robust symbolic reasoning, cognitive modeling\n")
    
    # 5. Permutation for Order-Sensitive Data
    print("5. Permutation Binding - Order-Preserving Operations:")
    vsa_perm = create_vsa(
        dimension=dimension,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='permutation'
    )
    
    # Encode ordered list
    items = ['first', 'second', 'third']
    item_vecs = [vsa_perm.generate_vector() for item in items]
    
    # Use permutation to encode order
    ordered_list = vsa_perm.bundle([
        vsa_perm.permute(vec, shift=i) for i, vec in enumerate(item_vecs)
    ])
    
    print(f"   Permutation preserves order information")
    print(f"   Use case: List processing, stack operations\n")


def plot_comparison_results(performance: Dict, noise_results: Tuple, capacity_results: Dict):
    """Plot comparison results."""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Performance comparison
        methods = list(performance.keys())
        bind_times = [performance[m]['bind_time'] for m in methods]
        unbind_times = [performance[m]['unbind_time'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax1.bar(x - width/2, bind_times, width, label='Bind')
        ax1.bar(x + width/2, unbind_times, width, label='Unbind')
        ax1.set_xlabel('Binding Method')
        ax1.set_ylabel('Time (ms/operation)')
        ax1.set_title('Binding Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split()[0] for m in methods], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Noise tolerance
        noise_data, noise_levels = noise_results
        for method, similarities in noise_data.items():
            ax2.plot(noise_levels, similarities, 'o-', label=method)
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Similarity')
        ax2.set_title('Noise Tolerance Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Capacity comparison
        max_pairs = len(next(iter(capacity_results.values())))
        x_capacity = range(1, max_pairs + 1)
        for method, similarities in capacity_results.items():
            ax3.plot(x_capacity, similarities, 'o-', label=method)
        ax3.set_xlabel('Number of Bound Pairs')
        ax3.set_ylabel('Average Similarity')
        ax3.set_title('Binding Capacity Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('vsa_binding_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plots saved to 'vsa_binding_comparison.png'\n")
    except Exception as e:
        print(f"\nCould not generate plots: {e}\n")


def print_summary_table(properties: Dict[str, Dict[str, bool]]):
    """Print a summary table of binding properties."""
    print("\n=== Binding Methods Summary Table ===\n")
    
    # Header
    print(f"{'Method':<15} {'Commut.':<10} {'Assoc.':<10} {'Identity':<10} {'Self-Inv':<10} {'Distrib.':<10}")
    print("-" * 65)
    
    # Rows
    for method, props in properties.items():
        print(f"{method:<15} "
              f"{'✓' if props['commutative'] else '✗':<10} "
              f"{'✓' if props['associative'] else '✗':<10} "
              f"{'✓' if props['has_identity'] else '✗':<10} "
              f"{'✓' if props['self_inverse'] else '✗':<10} "
              f"{'✓' if props['distributive'] else '✗':<10}")
    
    print("\n" + "="*65 + "\n")


def main():
    """Run all binding comparison demonstrations."""
    print("\n" + "="*60)
    print("VSA BINDING METHODS COMPARISON")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run comparisons
    performance = measure_binding_performance(dimension=1000, num_operations=1000)
    print("-"*60 + "\n")
    
    properties = test_binding_properties(dimension=1000)
    print("-"*60 + "\n")
    
    noise_results = test_noise_tolerance(dimension=1000)
    print("-"*60 + "\n")
    
    capacity_results = test_binding_capacity(dimension=1000, max_pairs=20)
    print("-"*60 + "\n")
    
    demonstrate_use_cases()
    
    # Print summary
    print_summary_table(properties)
    
    # Plot results
    plot_comparison_results(performance, noise_results, capacity_results)
    
    # Recommendations
    print("=== Recommendations ===\n")
    print("1. **XOR Binding**: Best for binary data, cryptographic applications")
    print("   - Pros: Self-inverse, very fast, hardware-friendly")
    print("   - Cons: Limited to binary vectors\n")
    
    print("2. **Multiplication**: Good general-purpose binding")
    print("   - Pros: Simple, works with continuous values")
    print("   - Cons: Not self-inverse, less noise-robust\n")
    
    print("3. **Convolution**: Best for sequential and structured data")
    print("   - Pros: Preserves structure, compatible with HRR")
    print("   - Cons: Computationally expensive\n")
    
    print("4. **MAP**: Most robust for multiple bindings")
    print("   - Pros: Noise-tolerant, good capacity")
    print("   - Cons: More complex, slower\n")
    
    print("5. **Permutation**: Best for order-sensitive applications")
    print("   - Pros: Preserves order, efficient")
    print("   - Cons: Not commutative\n")
    
    print("="*60)
    print("Binding Comparison Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()