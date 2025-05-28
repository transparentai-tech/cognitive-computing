#!/usr/bin/env python3
"""
Basic Sparse Distributed Memory (SDM) demonstration.

This example demonstrates the fundamental capabilities of SDM including:
- Creating and configuring an SDM instance
- Storing and recalling patterns
- Testing noise tolerance
- Comparing storage methods
- Visualizing memory behavior
- Analyzing performance characteristics

To run this example:
    python basic_sdm_demo.py

Optional arguments:
    --dimension: Address/data space dimension (default: 1000)
    --locations: Number of hard locations (default: 1000)
    --patterns: Number of patterns to store (default: 100)
    --visualize: Create visualization plots (default: True)
    --save-plots: Save plots to files (default: False)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import List, Tuple, Dict
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cognitive_computing.sdm import SDM, SDMConfig, create_sdm
from cognitive_computing.sdm.utils import (
    add_noise,
    generate_random_patterns,
    test_sdm_performance,
    analyze_activation_patterns,
    PatternEncoder,
    compute_memory_capacity
)
from cognitive_computing.sdm.visualizations import (
    plot_memory_distribution,
    plot_activation_pattern,
    plot_recall_accuracy
)
from cognitive_computing.sdm.memory import MemoryStatistics


def demonstrate_basic_operations(dimension: int = 1000) -> None:
    """
    Demonstrate basic SDM operations: store and recall.
    
    Parameters
    ----------
    dimension : int
        Dimension of address/data space
    """
    print("\n" + "="*60)
    print("1. BASIC SDM OPERATIONS")
    print("="*60)
    
    # Create SDM with default parameters
    sdm = create_sdm(dimension=dimension)
    print(f"\nCreated SDM with:")
    print(f"  - Dimension: {sdm.config.dimension}")
    print(f"  - Hard locations: {sdm.config.num_hard_locations}")
    print(f"  - Activation radius: {sdm.config.activation_radius}")
    print(f"  - Estimated capacity: {sdm.config.capacity} patterns")
    
    # Generate a random pattern
    address = np.random.randint(0, 2, dimension)
    data = np.random.randint(0, 2, dimension)
    
    print("\nStoring a pattern...")
    sdm.store(address, data)
    
    # Recall the pattern
    print("Recalling with exact address...")
    recalled = sdm.recall(address)
    
    if recalled is not None:
        accuracy = np.mean(recalled == data)
        print(f"  - Recall successful!")
        print(f"  - Accuracy: {accuracy:.2%}")
        print(f"  - Bit errors: {np.sum(recalled != data)}/{dimension}")
    else:
        print("  - Recall failed (no data retrieved)")
    
    # Test with noisy address
    print("\nRecalling with noisy address (10% noise)...")
    noisy_address = add_noise(address, noise_level=0.1)
    hamming_distance = np.sum(address != noisy_address)
    print(f"  - Hamming distance from original: {hamming_distance}")
    
    recalled_noisy = sdm.recall(noisy_address)
    if recalled_noisy is not None:
        accuracy_noisy = np.mean(recalled_noisy == data)
        print(f"  - Recall successful!")
        print(f"  - Accuracy: {accuracy_noisy:.2%}")
        print(f"  - Bit errors: {np.sum(recalled_noisy != data)}/{dimension}")
    else:
        print("  - Recall failed")
    
    # Show memory statistics
    stats = sdm.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"  - Locations used: {stats['locations_used']}/{stats['num_hard_locations']}")
    print(f"  - Average location usage: {stats['avg_location_usage']:.2f}")


def demonstrate_noise_tolerance(sdm: SDM, num_patterns: int = 50) -> Dict[float, float]:
    """
    Demonstrate SDM's noise tolerance capabilities.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to test
    num_patterns : int
        Number of test patterns
        
    Returns
    -------
    dict
        Noise levels mapped to recall accuracies
    """
    print("\n" + "="*60)
    print("2. NOISE TOLERANCE DEMONSTRATION")
    print("="*60)
    
    # Generate and store test patterns
    print(f"\nStoring {num_patterns} random patterns...")
    addresses, data_patterns = generate_random_patterns(num_patterns, sdm.config.dimension)
    
    for addr, data in zip(addresses, data_patterns):
        sdm.store(addr, data)
    
    # Test recall with different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    results = {}
    
    print("\nTesting recall with noise:")
    print("Noise Level | Avg Accuracy | Successful Recalls")
    print("-" * 50)
    
    for noise in noise_levels:
        accuracies = []
        successes = 0
        
        # Test subset of patterns
        test_size = min(20, num_patterns)
        test_indices = np.random.choice(num_patterns, test_size, replace=False)
        
        for idx in test_indices:
            # Add noise to address
            noisy_addr = add_noise(addresses[idx], noise_level=noise)
            
            # Attempt recall
            recalled = sdm.recall(noisy_addr)
            
            if recalled is not None:
                successes += 1
                accuracy = np.mean(recalled == data_patterns[idx])
                accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        success_rate = successes / test_size
        
        results[noise] = avg_accuracy
        print(f"   {noise:4.2f}    |    {avg_accuracy:5.2%}    |      {success_rate:5.2%}")
    
    return results


def demonstrate_capacity_limits(dimension: int = 1000) -> None:
    """
    Demonstrate SDM capacity and interference effects.
    
    Parameters
    ----------
    dimension : int
        Dimension of address/data space
    """
    print("\n" + "="*60)
    print("3. CAPACITY AND INTERFERENCE ANALYSIS")
    print("="*60)
    
    # Create fresh SDM
    sdm = create_sdm(dimension=dimension)
    theoretical_capacity = sdm.config.capacity
    
    print(f"\nTheoretical capacity: {theoretical_capacity} patterns")
    print("Testing capacity by gradually adding patterns...\n")
    
    # Test different load levels
    load_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    
    print("Load Level | Patterns | Avg Accuracy | Crosstalk")
    print("-" * 50)
    
    for load in load_levels:
        # Clear and reload SDM
        sdm.clear()
        num_patterns = int(theoretical_capacity * load)
        
        # Generate and store patterns
        addresses, data_patterns = generate_random_patterns(num_patterns, dimension)
        for addr, data in zip(addresses, data_patterns):
            sdm.store(addr, data)
        
        # Test recall accuracy on sample
        test_size = min(50, num_patterns)
        test_indices = np.random.choice(num_patterns, test_size, replace=False)
        
        accuracies = []
        for idx in test_indices:
            recalled = sdm.recall(addresses[idx])
            if recalled is not None:
                accuracy = np.mean(recalled == data_patterns[idx])
                accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        # Analyze crosstalk
        crosstalk = sdm.analyze_crosstalk(num_samples=20)
        crosstalk_error = crosstalk.get('avg_recall_error', 0.0)
        
        results.append({
            'load': load,
            'patterns': num_patterns,
            'accuracy': avg_accuracy,
            'crosstalk': crosstalk_error
        })
        
        print(f"   {load:4.2f}    |   {num_patterns:4d}   |    {avg_accuracy:5.2%}    |  {crosstalk_error:5.2%}")
    
    print(f"\nMemory utilization at {load_levels[-1]:.1f}x capacity:")
    stats = sdm.get_memory_stats()
    print(f"  - Locations used: {stats['locations_used']}/{stats['num_hard_locations']}")
    print(f"  - Max location usage: {stats['max_location_usage']}")
    
    return results


def demonstrate_storage_methods(dimension: int = 1000) -> None:
    """
    Compare counter-based and binary storage methods.
    
    Parameters
    ----------
    dimension : int
        Dimension of address/data space
    """
    print("\n" + "="*60)
    print("4. STORAGE METHOD COMPARISON")
    print("="*60)
    
    # Create SDMs with different storage methods
    config_counter = SDMConfig(
        dimension=dimension,
        num_hard_locations=1000,
        activation_radius=int(0.451 * dimension),
        storage_method="counters"
    )
    sdm_counter = SDM(config_counter)
    
    config_binary = SDMConfig(
        dimension=dimension,
        num_hard_locations=1000,
        activation_radius=int(0.451 * dimension),
        storage_method="binary"
    )
    sdm_binary = SDM(config_binary)
    
    # Generate test patterns
    num_patterns = 50
    addresses, data_patterns = generate_random_patterns(num_patterns, dimension)
    
    print(f"\nStoring {num_patterns} patterns in both SDMs...")
    
    # Store in both
    for addr, data in zip(addresses, data_patterns):
        sdm_counter.store(addr, data)
        sdm_binary.store(addr, data)
    
    # Compare performance
    print("\nComparing recall performance with noise:")
    print("Method    | Perfect Recall | 10% Noise | 20% Noise | Memory Usage")
    print("-" * 65)
    
    for method, sdm in [("Counters", sdm_counter), ("Binary  ", sdm_binary)]:
        accuracies = []
        
        # Test at different noise levels
        for noise in [0.0, 0.1, 0.2]:
            test_accuracies = []
            
            for i in range(min(20, num_patterns)):
                noisy_addr = add_noise(addresses[i], noise_level=noise)
                recalled = sdm.recall(noisy_addr)
                
                if recalled is not None:
                    accuracy = np.mean(recalled == data_patterns[i])
                    test_accuracies.append(accuracy)
            
            accuracies.append(np.mean(test_accuracies) if test_accuracies else 0.0)
        
        # Calculate memory usage
        if method.strip() == "Counters":
            memory_bytes = sdm.counters.nbytes
        else:
            memory_bytes = sdm.binary_storage.nbytes
        
        memory_mb = memory_bytes / (1024 * 1024)
        
        print(f"{method} |    {accuracies[0]:5.2%}     |   {accuracies[1]:5.2%}   |   {accuracies[2]:5.2%}   | {memory_mb:6.1f} MB")
    
    # Compare capacity characteristics
    print("\nCapacity characteristics:")
    
    # Get confidence for counter method
    test_addr = addresses[0]
    recalled_counter, confidence = sdm_counter.recall_with_confidence(test_addr)
    
    print(f"  - Counter method provides confidence scores")
    print(f"    Average confidence: {np.mean(confidence):.2f}")
    print(f"  - Binary method uses less memory but saturates faster")


def demonstrate_data_encoding(sdm: SDM) -> None:
    """
    Demonstrate encoding different data types for SDM storage.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to use
    """
    print("\n" + "="*60)
    print("5. DATA ENCODING EXAMPLES")
    print("="*60)
    
    encoder = PatternEncoder(dimension=sdm.config.dimension)
    
    # Example 1: Store integers
    print("\nStoring integers:")
    numbers = [42, 123, 456, 789]
    for num in numbers:
        # Use the number's encoding as both address and data
        encoded = encoder.encode_integer(num)
        sdm.store(encoded, encoded)
        print(f"  - Stored: {num}")
    
    # Recall integers
    print("\nRecalling integers:")
    for num in numbers:
        encoded = encoder.encode_integer(num)
        recalled = sdm.recall(encoded)
        if recalled is not None:
            accuracy = np.mean(recalled == encoded)
            print(f"  - Recalled {num} with {accuracy:.2%} accuracy")
    
    # Example 2: Store strings
    print("\nStoring strings:")
    words = ["hello", "world", "sparse", "memory"]
    word_encodings = {}
    
    for word in words:
        encoded = encoder.encode_string(word)
        word_encodings[word] = encoded
        sdm.store(encoded, encoded)
        print(f"  - Stored: '{word}'")
    
    # Test associative recall
    print("\nTesting associative recall with partial/noisy input:")
    test_word = "hello"
    original = word_encodings[test_word]
    noisy = add_noise(original, noise_level=0.15)
    
    recalled = sdm.recall(noisy)
    if recalled is not None:
        # Find closest stored word
        best_match = None
        best_similarity = 0
        
        for word, encoding in word_encodings.items():
            similarity = np.mean(recalled == encoding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = word
        
        print(f"  - Query: '{test_word}' with 15% noise")
        print(f"  - Best match: '{best_match}' (similarity: {best_similarity:.2%})")
    
    # Example 3: Store continuous vectors
    print("\nStoring continuous vectors:")
    vectors = [
        np.array([0.1, 0.5, 0.9, 0.2, 0.7]),
        np.array([0.9, 0.1, 0.4, 0.8, 0.3]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ]
    
    for i, vec in enumerate(vectors):
        encoded = encoder.encode_vector(vec, method='threshold')
        sdm.store(encoded, encoded)
        print(f"  - Stored vector {i+1}: {vec}")


def demonstrate_activation_analysis(sdm: SDM, visualize: bool = True) -> None:
    """
    Demonstrate activation pattern analysis.
    
    Parameters
    ----------
    sdm : SDM
        SDM instance to analyze
    visualize : bool
        Whether to create visualizations
    """
    print("\n" + "="*60)
    print("6. ACTIVATION PATTERN ANALYSIS")
    print("="*60)
    
    # Analyze activation patterns
    print("\nAnalyzing activation patterns...")
    analysis = analyze_activation_patterns(sdm, sample_size=500, visualize=False)
    
    print(f"\nActivation Statistics:")
    print(f"  - Mean activations per address: {analysis['mean_activation_count']:.1f}")
    print(f"  - Standard deviation: {analysis['std_activation_count']:.1f}")
    print(f"  - Range: [{analysis['min_activations']}, {analysis['max_activations']}]")
    print(f"  - Mean overlap between patterns: {analysis['mean_overlap']:.1f}")
    print(f"  - Location usage uniformity: {analysis['usage_uniformity']:.2%}")
    
    if 'similarity_correlation' in analysis:
        print(f"  - Address similarity vs activation overlap correlation: {analysis['similarity_correlation']:.3f}")
    
    # Theoretical comparison
    capacity_info = compute_memory_capacity(
        sdm.config.dimension,
        sdm.config.num_hard_locations,
        sdm.config.activation_radius
    )
    
    print(f"\nTheoretical vs Actual:")
    print(f"  - Expected activations: {capacity_info['expected_activated']:.1f}")
    print(f"  - Actual mean: {analysis['mean_activation_count']:.1f}")
    print(f"  - Deviation: {abs(analysis['mean_activation_count'] - capacity_info['expected_activated']) / capacity_info['expected_activated']:.1%}")
    
    if visualize:
        # Show a specific activation pattern
        test_address = np.random.randint(0, 2, sdm.config.dimension)
        fig = plot_activation_pattern(sdm, test_address)
        plt.show()


def performance_benchmark(dimension: int = 1000) -> None:
    """
    Benchmark SDM performance.
    
    Parameters
    ----------
    dimension : int
        Dimension of address/data space
    """
    print("\n" + "="*60)
    print("7. PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Create SDM
    sdm = create_sdm(dimension=dimension)
    
    # Run performance tests
    print("\nRunning performance benchmark...")
    results = test_sdm_performance(sdm, test_patterns=100, progress=True)
    
    print(f"\nPerformance Results:")
    print(f"  - Patterns tested: {results.pattern_count}")
    print(f"  - Dimension: {results.dimension}")
    print(f"\nTiming:")
    print(f"  - Average write time: {results.write_time_mean*1000:.2f} ± {results.write_time_std*1000:.2f} ms")
    print(f"  - Average read time: {results.read_time_mean*1000:.2f} ± {results.read_time_std*1000:.2f} ms")
    print(f"\nAccuracy:")
    print(f"  - Mean recall accuracy: {results.recall_accuracy_mean:.2%} ± {results.recall_accuracy_std:.2%}")
    print(f"\nNoise Tolerance:")
    for noise, accuracy in sorted(results.noise_tolerance.items()):
        print(f"  - {noise*100:3.0f}% noise: {accuracy:.2%} accuracy")
    print(f"\nCapacity:")
    print(f"  - Location utilization: {results.capacity_utilization:.1%}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Basic SDM demonstration")
    parser.add_argument("--dimension", type=int, default=1000,
                       help="Address/data space dimension (default: 1000)")
    parser.add_argument("--locations", type=int, default=1000,
                       help="Number of hard locations (default: 1000)")
    parser.add_argument("--patterns", type=int, default=100,
                       help="Number of patterns to store (default: 100)")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Create visualization plots (default: True)")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots to files (default: False)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SPARSE DISTRIBUTED MEMORY (SDM) DEMONSTRATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Dimension: {args.dimension}")
    print(f"  - Hard locations: {args.locations}")
    print(f"  - Test patterns: {args.patterns}")
    
    # Run demonstrations
    try:
        # 1. Basic operations
        demonstrate_basic_operations(dimension=args.dimension)
        
        # 2. Noise tolerance
        config = SDMConfig(
            dimension=args.dimension,
            num_hard_locations=args.locations,
            activation_radius=int(0.451 * args.dimension)
        )
        sdm = SDM(config)
        noise_results = demonstrate_noise_tolerance(sdm, num_patterns=args.patterns)
        
        # 3. Capacity limits
        demonstrate_capacity_limits(dimension=args.dimension)
        
        # 4. Storage methods
        demonstrate_storage_methods(dimension=args.dimension)
        
        # 5. Data encoding
        demonstrate_data_encoding(sdm)
        
        # 6. Activation analysis
        demonstrate_activation_analysis(sdm, visualize=args.visualize)
        
        # 7. Performance benchmark
        performance_benchmark(dimension=args.dimension)
        
        # Create visualizations if requested
        if args.visualize:
            print("\n" + "="*60)
            print("8. VISUALIZATIONS")
            print("="*60)
            
            # Memory distribution
            print("\nCreating memory distribution plot...")
            fig1 = plot_memory_distribution(sdm)
            if args.save_plots:
                fig1.savefig("sdm_memory_distribution.png", dpi=300, bbox_inches='tight')
                print("  - Saved: sdm_memory_distribution.png")
            
            # Recall accuracy
            print("Creating recall accuracy plot...")
            test_results = {
                'noise_tolerance': noise_results,
                'label': 'SDM Performance'
            }
            fig2 = plot_recall_accuracy(test_results)
            if args.save_plots:
                fig2.savefig("sdm_recall_accuracy.png", dpi=300, bbox_inches='tight')
                print("  - Saved: sdm_recall_accuracy.png")
            
            plt.show()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nKey Takeaways:")
        print("  1. SDM provides content-addressable memory with noise tolerance")
        print("  2. Capacity scales with number of hard locations (~15% for optimal parameters)")
        print("  3. Performance degrades gracefully with noise and overloading")
        print("  4. Different storage methods offer different trade-offs")
        print("  5. SDM can store various data types with appropriate encoding")
        print("\nFor more examples, see the other scripts in this directory.")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()