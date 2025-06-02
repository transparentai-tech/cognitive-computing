#!/usr/bin/env python3
"""
Basic HRR demonstration showing core operations and concepts.

This example demonstrates:
1. Basic vector binding and unbinding
2. Bundling multiple items
3. Simple associative memory
4. Performance benchmarks
5. Noise tolerance testing
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from cognitive_computing.hrr import (
    create_hrr,
    HRR,
    CleanupMemory,
    CleanupMemoryConfig,
    generate_random_vector,
    generate_unitary_vector,
    plot_similarity_matrix,
    analyze_binding_capacity,
)


def demonstrate_basic_binding(hrr: HRR, dimension: int) -> None:
    """Demonstrate basic binding and unbinding operations."""
    print("\n" + "="*60)
    print("1. BASIC BINDING AND UNBINDING")
    print("="*60)
    
    # Generate random vectors
    A = generate_random_vector(dimension)
    B = generate_random_vector(dimension)
    
    # Bind vectors
    C = hrr.bind(A, B)
    
    # Unbind to retrieve original vectors
    B_retrieved = hrr.unbind(C, A)
    A_retrieved = hrr.unbind(C, B)
    
    # Calculate similarities
    sim_B = hrr.similarity(B, B_retrieved)
    sim_A = hrr.similarity(A, A_retrieved)
    
    print(f"Dimension: {dimension}")
    print(f"Similarity after unbinding:")
    print(f"  B similarity: {sim_B:.4f}")
    print(f"  A similarity: {sim_A:.4f}")
    
    # Test with unitary vectors (better for unbinding)
    print("\nUsing unitary vectors:")
    A_unitary = generate_unitary_vector(dimension)
    B_unitary = generate_unitary_vector(dimension)
    
    C_unitary = hrr.bind(A_unitary, B_unitary)
    B_unitary_retrieved = hrr.unbind(C_unitary, A_unitary)
    
    sim_unitary = hrr.similarity(B_unitary, B_unitary_retrieved)
    print(f"  Unitary vector similarity: {sim_unitary:.4f}")


def demonstrate_bundling(hrr: HRR, dimension: int) -> None:
    """Demonstrate bundling (superposition) of multiple items."""
    print("\n" + "="*60)
    print("2. BUNDLING MULTIPLE ITEMS")
    print("="*60)
    
    # Create multiple items
    n_items = 5
    items = [generate_random_vector(dimension) for _ in range(n_items)]
    labels = [f"Item_{i}" for i in range(n_items)]
    
    # Bundle items together
    bundle = hrr.bundle(items)
    
    # Test retrieval of each item
    print(f"Bundled {n_items} items together")
    print("Similarity of bundle to each item:")
    
    for i, (item, label) in enumerate(zip(items, labels)):
        sim = hrr.similarity(bundle, item)
        print(f"  {label}: {sim:.4f}")
    
    # Test with weighted bundling
    print("\nWeighted bundling (different importance):")
    weights = np.array([3.0, 2.0, 1.0, 1.0, 1.0])
    weighted_items = [w * item for w, item in zip(weights, items)]
    weighted_bundle = hrr.bundle(weighted_items)
    
    for i, (item, label, weight) in enumerate(zip(items, labels, weights)):
        sim = hrr.similarity(weighted_bundle, item)
        print(f"  {label} (weight={weight}): {sim:.4f}")


def demonstrate_associative_memory(hrr: HRR, dimension: int) -> None:
    """Demonstrate simple associative memory using HRR."""
    print("\n" + "="*60)
    print("3. ASSOCIATIVE MEMORY")
    print("="*60)
    
    # Create key-value pairs
    pairs = {
        "color": "red",
        "shape": "circle",
        "size": "large",
        "texture": "smooth"
    }
    
    # Generate vectors for keys and values
    key_vectors = {k: generate_unitary_vector(dimension) for k in pairs.keys()}
    value_vectors = {v: generate_random_vector(dimension) for v in pairs.values()}
    
    # Create associative memory by binding and bundling
    memory_items = []
    for key, value in pairs.items():
        binding = hrr.bind(key_vectors[key], value_vectors[value])
        memory_items.append(binding)
    
    memory = hrr.bundle(memory_items)
    
    # Create cleanup memory for values
    cleanup_config = CleanupMemoryConfig(threshold=0.3)
    cleanup_memory = CleanupMemory(cleanup_config, dimension)
    
    for value, vector in value_vectors.items():
        cleanup_memory.add_item(value, vector)
    
    # Query the memory
    print("Stored associations:")
    for key, value in pairs.items():
        print(f"  {key} -> {value}")
    
    print("\nQuerying memory:")
    for key in pairs.keys():
        # Unbind with key to get value
        retrieved = hrr.unbind(memory, key_vectors[key])
        
        # Clean up the retrieved vector
        cleaned_name, cleaned_vector, confidence = cleanup_memory.cleanup(retrieved)
        
        print(f"  Query '{key}': Retrieved '{cleaned_name}' (confidence: {confidence:.3f})")


def benchmark_operations(hrr: HRR, dimension: int) -> Dict[str, float]:
    """Benchmark HRR operations."""
    print("\n" + "="*60)
    print("4. PERFORMANCE BENCHMARKS")
    print("="*60)
    
    n_trials = 1000
    results = {}
    
    # Generate test vectors
    A = generate_random_vector(dimension)
    B = generate_random_vector(dimension)
    vectors = [generate_random_vector(dimension) for _ in range(10)]
    
    # Benchmark binding
    start = time.time()
    for _ in range(n_trials):
        _ = hrr.bind(A, B)
    results['binding'] = (time.time() - start) / n_trials * 1000
    
    # Benchmark unbinding
    C = hrr.bind(A, B)
    start = time.time()
    for _ in range(n_trials):
        _ = hrr.unbind(C, A)
    results['unbinding'] = (time.time() - start) / n_trials * 1000
    
    # Benchmark bundling
    start = time.time()
    for _ in range(n_trials):
        _ = hrr.bundle(vectors)
    results['bundling'] = (time.time() - start) / n_trials * 1000
    
    # Benchmark similarity
    start = time.time()
    for _ in range(n_trials):
        _ = hrr.similarity(A, B)
    results['similarity'] = (time.time() - start) / n_trials * 1000
    
    print(f"Average operation times (dimension={dimension}):")
    for op, time_ms in results.items():
        print(f"  {op}: {time_ms:.3f} ms")
    
    return results


def test_noise_tolerance(hrr: HRR, dimension: int) -> None:
    """Test HRR's tolerance to noise."""
    print("\n" + "="*60)
    print("5. NOISE TOLERANCE")
    print("="*60)
    
    # Create test vectors
    A = generate_unitary_vector(dimension)
    B = generate_random_vector(dimension)
    C = hrr.bind(A, B)
    
    # Test with different noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    similarities = []
    
    print("Testing unbinding with noisy composite vector:")
    
    for noise_level in noise_levels:
        # Add noise to composite
        noise = np.random.normal(0, noise_level, dimension)
        C_noisy = C + noise
        
        # Unbind and measure similarity
        B_retrieved = hrr.unbind(C_noisy, A)
        sim = hrr.similarity(B, B_retrieved)
        similarities.append(sim)
        
        print(f"  Noise level {noise_level:.1f}: similarity = {sim:.4f}")
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, similarities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Retrieval Similarity')
    plt.title('HRR Noise Tolerance')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def demonstrate_capacity_analysis(dimension: int) -> None:
    """Analyze binding capacity of HRR."""
    print("\n" + "="*60)
    print("6. CAPACITY ANALYSIS")
    print("="*60)
    
    # Test different dimensions
    dimensions = [256, 512, 1024, 2048]
    n_pairs_list = [5, 10, 20, 30, 40, 50]
    
    results = {}
    
    for dim in dimensions:
        hrr = create_hrr(dimension=dim)
        capacities = []
        
        for n_pairs in n_pairs_list:
            analysis = analyze_binding_capacity(hrr, n_pairs)
            capacities.append(analysis['mean_similarity'])
        
        results[dim] = capacities
        
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for dim, capacities in results.items():
        plt.plot(n_pairs_list, capacities, 'o-', label=f'D={dim}', linewidth=2)
    
    plt.xlabel('Number of Bound Pairs')
    plt.ylabel('Mean Retrieval Similarity')
    plt.title('HRR Binding Capacity vs Dimensionality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Capacity analysis complete - see plots")


def compare_storage_methods() -> None:
    """Compare real vs complex storage methods."""
    print("\n" + "="*60)
    print("7. STORAGE METHOD COMPARISON")
    print("="*60)
    
    dimension = 1024
    n_pairs = 20
    
    # Test real storage
    hrr_real = create_hrr(dimension=dimension, storage_method="real")
    real_analysis = analyze_binding_capacity(hrr_real, n_pairs)
    
    # Test complex storage
    hrr_complex = create_hrr(dimension=dimension, storage_method="complex")
    complex_analysis = analyze_binding_capacity(hrr_complex, n_pairs)
    
    print(f"Binding {n_pairs} pairs (dimension={dimension}):")
    print(f"  Real storage:")
    print(f"    Mean similarity: {real_analysis['mean_similarity']:.4f}")
    print(f"    Min similarity: {real_analysis['min_similarity']:.4f}")
    print(f"  Complex storage:")
    print(f"    Mean similarity: {complex_analysis['mean_similarity']:.4f}")
    print(f"    Min similarity: {complex_analysis['min_similarity']:.4f}")


def visualize_similarity_structure(hrr: HRR, dimension: int) -> None:
    """Visualize similarity relationships between vectors."""
    print("\n" + "="*60)
    print("8. SIMILARITY STRUCTURE VISUALIZATION")
    print("="*60)
    
    # Create a set of related vectors
    vectors = {}
    
    # Base concepts
    vectors['DOG'] = generate_random_vector(dimension)
    vectors['CAT'] = generate_random_vector(dimension)
    vectors['ANIMAL'] = generate_random_vector(dimension)
    vectors['PET'] = generate_random_vector(dimension)
    
    # Create relationships
    vectors['DOG_IS_ANIMAL'] = hrr.bind(vectors['DOG'], vectors['ANIMAL'])
    vectors['CAT_IS_ANIMAL'] = hrr.bind(vectors['CAT'], vectors['ANIMAL'])
    vectors['DOG_IS_PET'] = hrr.bind(vectors['DOG'], vectors['PET'])
    vectors['CAT_IS_PET'] = hrr.bind(vectors['CAT'], vectors['PET'])
    
    # Create similarity matrix plot
    fig = plot_similarity_matrix(vectors)
    plt.show()
    
    print("Similarity matrix visualization complete")


def main():
    """Run all HRR demonstrations."""
    print("="*60)
    print("HOLOGRAPHIC REDUCED REPRESENTATIONS (HRR) DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create HRR system
    dimension = 1024
    hrr = create_hrr(dimension=dimension, normalize=True)
    
    # Run demonstrations
    demonstrate_basic_binding(hrr, dimension)
    demonstrate_bundling(hrr, dimension)
    demonstrate_associative_memory(hrr, dimension)
    benchmark_operations(hrr, dimension)
    test_noise_tolerance(hrr, dimension)
    demonstrate_capacity_analysis(dimension)
    compare_storage_methods()
    visualize_similarity_structure(hrr, dimension)
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()