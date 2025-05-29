# SDM Performance Optimization Guide

This document provides comprehensive guidance on optimizing Sparse Distributed Memory performance for various use cases and scales.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Benchmarking Methodology](#benchmarking-methodology)
3. [Parameter Optimization](#parameter-optimization)
4. [Memory Optimization](#memory-optimization)
5. [Computational Optimization](#computational-optimization)
6. [Scaling Strategies](#scaling-strategies)
7. [Hardware Considerations](#hardware-considerations)
8. [Performance Monitoring](#performance-monitoring)
9. [Common Bottlenecks](#common-bottlenecks)
10. [Best Practices](#best-practices)
11. [Performance Benchmarks](#performance-benchmarks)

---

## Performance Overview

### Key Performance Metrics

SDM performance is characterized by several key metrics:

1. **Storage Speed** - Time to store a pattern (write operation)
2. **Recall Speed** - Time to recall a pattern (read operation)
3. **Memory Capacity** - Number of patterns that can be stored
4. **Recall Accuracy** - Correctness of recalled patterns
5. **Noise Tolerance** - Ability to recall with noisy inputs
6. **Memory Efficiency** - RAM usage per stored pattern

### Performance Characteristics

```python
import numpy as np
import time
import matplotlib.pyplot as plt
from cognitive_computing.sdm import create_sdm, SDMConfig
from cognitive_computing.sdm.utils import generate_random_patterns, add_noise

def measure_sdm_characteristics(dimension, num_locations, test_patterns=100):
    """Measure fundamental SDM performance characteristics."""
    
    # Create SDM
    sdm = create_sdm(dimension=dimension, num_locations=num_locations)
    
    # Generate test data
    addresses, data = generate_random_patterns(test_patterns, dimension)
    
    # Measure write performance
    write_times = []
    for addr, dat in zip(addresses, data):
        start = time.time()
        sdm.store(addr, dat)
        write_times.append(time.time() - start)
    
    # Measure read performance
    read_times = []
    accuracies = []
    for addr, dat in zip(addresses, data):
        start = time.time()
        recalled = sdm.recall(addr)
        read_times.append(time.time() - start)
        
        if recalled is not None:
            accuracies.append(np.mean(recalled == dat))
    
    # Measure noise tolerance
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
    noise_accuracies = []
    
    for noise in noise_levels:
        noise_accs = []
        for i in range(min(20, test_patterns)):
            noisy_addr = add_noise(addresses[i], noise)
            recalled = sdm.recall(noisy_addr)
            if recalled is not None:
                noise_accs.append(np.mean(recalled == data[i]))
        noise_accuracies.append(np.mean(noise_accs) if noise_accs else 0)
    
    results = {
        'dimension': dimension,
        'num_locations': num_locations,
        'patterns_stored': test_patterns,
        'avg_write_time': np.mean(write_times),
        'avg_read_time': np.mean(read_times),
        'write_throughput': 1.0 / np.mean(write_times),
        'read_throughput': 1.0 / np.mean(read_times),
        'recall_accuracy': np.mean(accuracies),
        'noise_tolerance': dict(zip(noise_levels, noise_accuracies)),
        'memory_usage_mb': (num_locations * dimension * 9) / 8 / 1024 / 1024  # 9 bits per cell
    }
    
    return results

# Example measurement
results = measure_sdm_characteristics(dimension=1000, num_locations=1000, test_patterns=100)
print("SDM Performance Characteristics:")
for key, value in results.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v:.3f}")
    elif isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")
```

### Theoretical Performance Limits

```python
def calculate_theoretical_limits(dimension, num_locations):
    """Calculate theoretical performance limits for SDM."""
    
    # Critical distance
    critical_distance = 0.451 * dimension
    
    # Activation probability at critical distance
    from scipy.stats import binom
    activation_prob = sum(binom.pmf(k, dimension, 0.5) 
                         for k in range(int(critical_distance) + 1))
    
    # Expected activations
    expected_activations = num_locations * activation_prob
    
    # Theoretical capacity (Kanerva's formula)
    capacity = 0.15 * num_locations
    
    # Information capacity (bits)
    info_capacity = capacity * dimension
    
    # Minimum operations per recall
    min_operations = expected_activations * dimension
    
    print(f"Theoretical Limits for SDM (D={dimension}, M={num_locations}):")
    print(f"  Critical Distance: {critical_distance:.0f} bits")
    print(f"  Activation Probability: {activation_prob:.6f}")
    print(f"  Expected Activations: {expected_activations:.0f}")
    print(f"  Storage Capacity: {capacity:.0f} patterns")
    print(f"  Information Capacity: {info_capacity/1e6:.2f} Mbits")
    print(f"  Min Operations/Recall: {min_operations/1e6:.2f}M")
    
    return {
        'critical_distance': critical_distance,
        'activation_prob': activation_prob,
        'expected_activations': expected_activations,
        'capacity': capacity,
        'info_capacity': info_capacity,
        'min_operations': min_operations
    }

# Calculate limits for different scales
for dim, locs in [(1000, 1000), (2000, 5000), (5000, 10000)]:
    print(f"\n{'='*60}")
    calculate_theoretical_limits(dim, locs)
```

---

## Benchmarking Methodology

### Comprehensive Benchmarking Suite

```python
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import psutil
import gc

@dataclass
class BenchmarkConfig:
    """Configuration for SDM benchmark."""
    dimension: int
    num_locations: int
    test_patterns: int
    noise_levels: List[float]
    pattern_sparsity: float = 0.5
    num_iterations: int = 3

class SDMBenchmark:
    """Comprehensive SDM benchmarking suite."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        
    def run_benchmark(self, sdm_config: SDMConfig):
        """Run complete benchmark suite."""
        print(f"Running benchmark: D={sdm_config.dimension}, "
              f"M={sdm_config.num_hard_locations}, "
              f"Method={sdm_config.storage_method}")
        
        results = {
            'config': sdm_config.__dict__.copy(),
            'metrics': {}
        }
        
        # Memory before
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create SDM
        from cognitive_computing.sdm import SDM
        sdm = SDM(sdm_config)
        
        # Memory after creation
        memory_after_creation = psutil.Process().memory_info().rss / 1024 / 1024
        results['metrics']['memory_creation_mb'] = memory_after_creation - memory_before
        
        # Generate test data
        addresses, data = generate_random_patterns(
            self.config.test_patterns,
            self.config.dimension,
            sparsity=self.config.pattern_sparsity
        )
        
        # Write performance
        write_times = []
        for _ in range(self.config.num_iterations):
            sdm.clear()
            start = time.time()
            for addr, dat in zip(addresses, data):
                sdm.store(addr, dat)
            write_times.append(time.time() - start)
        
        results['metrics']['total_write_time'] = np.mean(write_times)
        results['metrics']['write_time_per_pattern'] = np.mean(write_times) / self.config.test_patterns
        results['metrics']['write_throughput'] = self.config.test_patterns / np.mean(write_times)
        
        # Memory after storage
        memory_after_storage = psutil.Process().memory_info().rss / 1024 / 1024
        results['metrics']['memory_storage_mb'] = memory_after_storage - memory_after_creation
        results['metrics']['memory_per_pattern_kb'] = (
            (memory_after_storage - memory_after_creation) * 1024 / self.config.test_patterns
        )
        
        # Read performance (perfect recall)
        read_times = []
        accuracies = []
        
        for _ in range(self.config.num_iterations):
            iter_times = []
            iter_accs = []
            
            for addr, dat in zip(addresses, data):
                start = time.time()
                recalled = sdm.recall(addr)
                iter_times.append(time.time() - start)
                
                if recalled is not None:
                    iter_accs.append(np.mean(recalled == dat))
            
            read_times.extend(iter_times)
            accuracies.extend(iter_accs)
        
        results['metrics']['avg_read_time'] = np.mean(read_times)
        results['metrics']['read_throughput'] = 1.0 / np.mean(read_times)
        results['metrics']['perfect_recall_accuracy'] = np.mean(accuracies)
        
        # Noise tolerance
        noise_results = {}
        for noise_level in self.config.noise_levels:
            noise_accs = []
            noise_times = []
            
            # Test subset for efficiency
            test_size = min(50, self.config.test_patterns)
            for i in range(test_size):
                noisy_addr = add_noise(addresses[i], noise_level)
                
                start = time.time()
                recalled = sdm.recall(noisy_addr)
                noise_times.append(time.time() - start)
                
                if recalled is not None:
                    noise_accs.append(np.mean(recalled == data[i]))
            
            noise_results[f'noise_{noise_level}'] = {
                'accuracy': np.mean(noise_accs) if noise_accs else 0,
                'recall_time': np.mean(noise_times)
            }
        
        results['metrics']['noise_tolerance'] = noise_results
        
        # Location utilization
        stats = sdm.get_memory_stats()
        results['metrics']['locations_used'] = stats['locations_used']
        results['metrics']['location_utilization'] = stats['locations_used'] / sdm_config.num_hard_locations
        results['metrics']['avg_location_usage'] = stats['avg_location_usage']
        
        # CPU usage estimate (operations per second)
        total_ops = self.config.test_patterns * stats['locations_used'] * self.config.dimension
        ops_per_second = total_ops / results['metrics']['total_write_time']
        results['metrics']['ops_per_second'] = ops_per_second
        
        return results
    
    def run_comparison(self, configurations: List[Dict]):
        """Run benchmarks comparing multiple configurations."""
        all_results = []
        
        for config_dict in configurations:
            sdm_config = SDMConfig(**config_dict)
            result = self.run_benchmark(sdm_config)
            all_results.append(result)
            
            # Clean up
            gc.collect()
        
        return all_results
    
    def generate_report(self, results: List[Dict]):
        """Generate performance comparison report."""
        # Create DataFrame for easy comparison
        rows = []
        for result in results:
            row = {'name': result['config'].get('name', 'unnamed')}
            row.update(result['config'])
            row.update(result['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Print summary
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        summary_cols = ['name', 'dimension', 'num_hard_locations', 'storage_method',
                       'write_throughput', 'read_throughput', 'perfect_recall_accuracy',
                       'memory_per_pattern_kb', 'location_utilization']
        
        print(df[summary_cols].to_string(index=False))
        
        return df

# Run comprehensive benchmark
benchmark_config = BenchmarkConfig(
    dimension=1000,
    num_locations=1000,
    test_patterns=200,
    noise_levels=[0.05, 0.1, 0.15, 0.2],
    num_iterations=3
)

benchmark = SDMBenchmark(benchmark_config)

# Compare different configurations
configurations = [
    {
        'name': 'Counter-8bit',
        'dimension': 1000,
        'num_hard_locations': 1000,
        'activation_radius': 451,
        'storage_method': 'counters',
        'counter_bits': 8
    },
    {
        'name': 'Counter-16bit',
        'dimension': 1000,
        'num_hard_locations': 1000,
        'activation_radius': 451,
        'storage_method': 'counters',
        'counter_bits': 16
    },
    {
        'name': 'Binary',
        'dimension': 1000,
        'num_hard_locations': 1000,
        'activation_radius': 451,
        'storage_method': 'binary'
    },
    {
        'name': 'Parallel-Counter',
        'dimension': 1000,
        'num_hard_locations': 1000,
        'activation_radius': 451,
        'storage_method': 'counters',
        'parallel': True,
        'num_workers': 4
    }
]

results = benchmark.run_comparison(configurations)
report_df = benchmark.generate_report(results)
```

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_sdm_operations(sdm, num_operations=100):
    """Profile SDM operations to identify bottlenecks."""
    
    # Generate test data
    addresses, data = generate_random_patterns(num_operations, sdm.config.dimension)
    
    # Profile storage
    print("Profiling STORE operations...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    for addr, dat in zip(addresses, data):
        sdm.store(addr, dat)
    
    profiler.disable()
    
    # Print storage profile
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Top 10 functions
    print(s.getvalue())
    
    # Profile recall
    print("\nProfiling RECALL operations...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    for addr in addresses:
        _ = sdm.recall(addr)
    
    profiler.disable()
    
    # Print recall profile
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())

# Example profiling
sdm = create_sdm(dimension=1000)
profile_sdm_operations(sdm, num_operations=50)
```

---

## Parameter Optimization

### Optimal Activation Radius

```python
def find_optimal_activation_radius(dimension, num_locations, test_patterns=100):
    """Find optimal activation radius through empirical testing."""
    
    # Test range around critical distance
    critical_dist = int(0.451 * dimension)
    test_radii = range(
        max(1, critical_dist - 50),
        min(dimension, critical_dist + 50),
        5
    )
    
    results = []
    
    for radius in test_radii:
        # Create SDM with test radius
        config = SDMConfig(
            dimension=dimension,
            num_hard_locations=num_locations,
            activation_radius=radius
        )
        sdm = SDM(config)
        
        # Generate test data
        addresses, data = generate_random_patterns(test_patterns, dimension)
        
        # Store patterns
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Test recall accuracy
        accuracies = []
        noise_accuracies = []
        
        for i in range(min(50, test_patterns)):
            # Perfect recall
            recalled = sdm.recall(addresses[i])
            if recalled is not None:
                accuracies.append(np.mean(recalled == data[i]))
            
            # Noisy recall (10% noise)
            noisy = add_noise(addresses[i], 0.1)
            recalled_noisy = sdm.recall(noisy)
            if recalled_noisy is not None:
                noise_accuracies.append(np.mean(recalled_noisy == data[i]))
        
        # Get statistics
        stats = sdm.get_memory_stats()
        
        results.append({
            'radius': radius,
            'accuracy': np.mean(accuracies) if accuracies else 0,
            'noise_accuracy': np.mean(noise_accuracies) if noise_accuracies else 0,
            'locations_used': stats['locations_used'],
            'avg_activations': stats['locations_used'] * test_patterns / num_locations
        })
    
    # Find optimal radius
    # Balance between accuracy and efficiency
    scores = []
    for r in results:
        # Weighted score: accuracy + noise tolerance - overactivation penalty
        score = (r['accuracy'] + r['noise_accuracy']) / 2
        
        # Penalty for too many activations
        if r['avg_activations'] > 150:
            score *= 0.9
        elif r['avg_activations'] > 200:
            score *= 0.7
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_result = results[optimal_idx]
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    radii = [r['radius'] for r in results]
    plt.plot(radii, [r['accuracy'] for r in results], 'o-', label='Perfect Recall')
    plt.plot(radii, [r['noise_accuracy'] for r in results], 's-', label='10% Noise')
    plt.axvline(critical_dist, color='red', linestyle='--', label='Critical Distance')
    plt.axvline(optimal_result['radius'], color='green', linestyle='--', label='Optimal')
    plt.xlabel('Activation Radius')
    plt.ylabel('Recall Accuracy')
    plt.title('Accuracy vs Activation Radius')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(radii, [r['avg_activations'] for r in results], 'o-')
    plt.axvline(optimal_result['radius'], color='green', linestyle='--', label='Optimal')
    plt.xlabel('Activation Radius')
    plt.ylabel('Average Activations')
    plt.title('Activation Count vs Radius')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nOptimal Activation Radius: {optimal_result['radius']}")
    print(f"  Accuracy: {optimal_result['accuracy']:.3f}")
    print(f"  Noise Accuracy: {optimal_result['noise_accuracy']:.3f}")
    print(f"  Avg Activations: {optimal_result['avg_activations']:.1f}")
    
    return optimal_result

# Find optimal radius
optimal = find_optimal_activation_radius(dimension=1000, num_locations=1000)
```

### Capacity vs Locations Trade-off

```python
def analyze_capacity_scaling(dimension, location_counts, test_patterns=200):
    """Analyze how capacity scales with number of locations."""
    
    results = []
    
    for num_locs in location_counts:
        print(f"\nTesting with {num_locs} locations...")
        
        # Use optimal radius for each configuration
        from cognitive_computing.sdm.memory import MemoryOptimizer
        optimal_radius = MemoryOptimizer.find_optimal_radius(dimension, num_locs)
        
        config = SDMConfig(
            dimension=dimension,
            num_hard_locations=num_locs,
            activation_radius=optimal_radius
        )
        sdm = SDM(config)
        
        # Progressive capacity test
        capacities = []
        pattern_counts = range(50, test_patterns + 1, 50)
        
        for count in pattern_counts:
            # Clear and store patterns
            sdm.clear()
            addresses, data = generate_random_patterns(count, dimension)
            
            for addr, dat in zip(addresses, data):
                sdm.store(addr, dat)
            
            # Test recall accuracy
            test_size = min(50, count)
            accuracies = []
            
            for i in range(test_size):
                recalled = sdm.recall(addresses[i])
                if recalled is not None:
                    accuracies.append(np.mean(recalled == data[i]))
            
            capacities.append({
                'patterns': count,
                'accuracy': np.mean(accuracies) if accuracies else 0
            })
        
        # Find effective capacity (where accuracy drops below 95%)
        effective_capacity = 0
        for cap in capacities:
            if cap['accuracy'] >= 0.95:
                effective_capacity = cap['patterns']
            else:
                break
        
        # Memory usage
        memory_mb = (num_locs * dimension * 9) / 8 / 1024 / 1024
        
        results.append({
            'num_locations': num_locs,
            'optimal_radius': optimal_radius,
            'effective_capacity': effective_capacity,
            'theoretical_capacity': int(0.15 * num_locs),
            'memory_mb': memory_mb,
            'patterns_per_mb': effective_capacity / memory_mb if memory_mb > 0 else 0,
            'capacity_data': capacities
        })
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Capacity scaling
    ax = axes[0, 0]
    locs = [r['num_locations'] for r in results]
    eff_cap = [r['effective_capacity'] for r in results]
    theo_cap = [r['theoretical_capacity'] for r in results]
    
    ax.plot(locs, eff_cap, 'o-', label='Effective (95% accuracy)', linewidth=2)
    ax.plot(locs, theo_cap, 's--', label='Theoretical (0.15M)', linewidth=2)
    ax.set_xlabel('Number of Locations')
    ax.set_ylabel('Pattern Capacity')
    ax.set_title('Capacity Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory efficiency
    ax = axes[0, 1]
    memory = [r['memory_mb'] for r in results]
    efficiency = [r['patterns_per_mb'] for r in results]
    
    ax.plot(locs, efficiency, 'o-', linewidth=2)
    ax.set_xlabel('Number of Locations')
    ax.set_ylabel('Patterns per MB')
    ax.set_title('Memory Efficiency')
    ax.grid(True, alpha=0.3)
    
    # Optimal radius
    ax = axes[1, 0]
    radii = [r['optimal_radius'] for r in results]
    ax.plot(locs, radii, 'o-', linewidth=2)
    ax.set_xlabel('Number of Locations')
    ax.set_ylabel('Optimal Radius')
    ax.set_title('Optimal Activation Radius')
    ax.grid(True, alpha=0.3)
    
    # Capacity curves
    ax = axes[1, 1]
    for i, result in enumerate(results[::2]):  # Show every other for clarity
        cap_data = result['capacity_data']
        patterns = [c['patterns'] for c in cap_data]
        accuracies = [c['accuracy'] for c in cap_data]
        ax.plot(patterns, accuracies, 'o-', 
                label=f"{result['num_locations']} locs", alpha=0.7)
    
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Patterns Stored')
    ax.set_ylabel('Recall Accuracy')
    ax.set_title('Capacity Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.02)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Analyze capacity scaling
location_counts = [500, 1000, 2000, 5000, 10000]
scaling_results = analyze_capacity_scaling(dimension=1000, location_counts=location_counts)

# Print summary
print("\n" + "="*70)
print("CAPACITY SCALING SUMMARY")
print("="*70)
print(f"{'Locations':<12} {'Effective':<12} {'Theoretical':<12} {'Memory(MB)':<12} {'Pat/MB':<12}")
print("-"*70)
for r in scaling_results:
    print(f"{r['num_locations']:<12} {r['effective_capacity']:<12} "
          f"{r['theoretical_capacity']:<12} {r['memory_mb']:<12.1f} "
          f"{r['patterns_per_mb']:<12.1f}")
```

### Dimension Selection

```python
def analyze_dimension_impact(dimensions, num_locations=1000, test_patterns=100):
    """Analyze impact of dimension on SDM performance."""
    
    results = []
    
    for dim in dimensions:
        print(f"\nTesting dimension {dim}...")
        
        # Create SDM with proportional radius
        radius = int(0.451 * dim)
        config = SDMConfig(
            dimension=dim,
            num_hard_locations=num_locations,
            activation_radius=radius
        )
        sdm = SDM(config)
        
        # Generate and store patterns
        addresses, data = generate_random_patterns(test_patterns, dim)
        
        store_times = []
        for addr, dat in zip(addresses, data):
            start = time.time()
            sdm.store(addr, dat)
            store_times.append(time.time() - start)
        
        # Test recall
        recall_times = []
        accuracies = []
        noise_accuracies = []
        
        for i in range(min(50, test_patterns)):
            # Perfect recall
            start = time.time()
            recalled = sdm.recall(addresses[i])
            recall_times.append(time.time() - start)
            
            if recalled is not None:
                accuracies.append(np.mean(recalled == data[i]))
            
            # Noisy recall
            noisy = add_noise(addresses[i], 0.15)
            recalled_noisy = sdm.recall(noisy)
            if recalled_noisy is not None:
                noise_accuracies.append(np.mean(recalled_noisy == data[i]))
        
        # Calculate metrics
        stats = sdm.get_memory_stats()
        
        results.append({
            'dimension': dim,
            'store_time': np.mean(store_times),
            'recall_time': np.mean(recall_times),
            'accuracy': np.mean(accuracies),
            'noise_accuracy': np.mean(noise_accuracies),
            'locations_used': stats['locations_used'],
            'theoretical_capacity': int(0.15 * num_locations),
            'bits_per_pattern': dim,
            'activation_probability': stats['locations_used'] / (test_patterns * num_locations)
        })
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dims = [r['dimension'] for r in results]
    
    # Performance times
    ax = axes[0, 0]
    ax.plot(dims, [r['store_time']*1000 for r in results], 'o-', label='Store')
    ax.plot(dims, [r['recall_time']*1000 for r in results], 's-', label='Recall')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Operation Times vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(dims, [r['accuracy'] for r in results], 'o-', label='Perfect')
    ax.plot(dims, [r['noise_accuracy'] for r in results], 's-', label='15% Noise')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Recall Accuracy')
    ax.set_title('Accuracy vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Activation probability
    ax = axes[1, 0]
    ax.semilogy(dims, [r['activation_probability'] for r in results], 'o-')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Activation Probability (log)')
    ax.set_title('Sparsity vs Dimension')
    ax.grid(True, alpha=0.3)
    
    # Information density
    ax = axes[1, 1]
    # Bits per pattern / theoretical capacity
    info_density = [r['bits_per_pattern'] / r['theoretical_capacity'] 
                   for r in results]
    ax.plot(dims, info_density, 'o-')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Bits per Pattern Slot')
    ax.set_title('Information Density')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Analyze different dimensions
dimensions = [256, 512, 1000, 2000, 4000, 8000]
dim_results = analyze_dimension_impact(dimensions)
```

---

## Memory Optimization

### Memory-Efficient Storage Methods

```python
class CompressedSDM:
    """Memory-efficient SDM implementation using compression techniques."""
    
    def __init__(self, dimension, num_locations, activation_radius):
        self.dimension = dimension
        self.num_locations = num_locations
        self.activation_radius = activation_radius
        
        # Use bit-packed storage for addresses
        self.hard_locations = self._generate_compressed_locations()
        
        # Use sparse storage for counters
        self.sparse_counters = {}  # Only store non-zero counters
        
        # Track active locations
        self.active_locations = set()
        
    def _generate_compressed_locations(self):
        """Generate bit-packed hard location addresses."""
        # Each location stored as packed bytes
        locations = []
        bytes_per_location = (self.dimension + 7) // 8
        
        for _ in range(self.num_locations):
            # Generate random bytes
            loc_bytes = np.random.randint(0, 256, bytes_per_location, dtype=np.uint8)
            locations.append(loc_bytes)
        
        return locations
    
    def _unpack_location(self, loc_bytes):
        """Unpack bytes to binary vector."""
        return np.unpackbits(loc_bytes)[:self.dimension]
    
    def _hamming_distance(self, addr, loc_bytes):
        """Compute Hamming distance with packed location."""
        loc_binary = self._unpack_location(loc_bytes)
        return np.sum(addr != loc_binary)
    
    def store(self, address, data):
        """Store pattern with sparse counter updates."""
        # Find activated locations
        activated = []
        for i, loc_bytes in enumerate(self.hard_locations):
            if self._hamming_distance(address, loc_bytes) <= self.activation_radius:
                activated.append(i)
                self.active_locations.add(i)
        
        # Update sparse counters
        for loc_idx in activated:
            if loc_idx not in self.sparse_counters:
                self.sparse_counters[loc_idx] = np.zeros(self.dimension, dtype=np.int8)
            
            # Convert to bipolar and update
            bipolar_data = 2 * data - 1
            self.sparse_counters[loc_idx] += bipolar_data
    
    def get_memory_usage(self):
        """Calculate actual memory usage."""
        # Packed addresses
        address_memory = self.num_locations * ((self.dimension + 7) // 8)
        
        # Sparse counters
        counter_memory = len(self.sparse_counters) * self.dimension
        
        # Python object overhead (approximate)
        overhead = len(self.sparse_counters) * 100  # bytes per dict entry
        
        total_bytes = address_memory + counter_memory + overhead
        
        return {
            'address_mb': address_memory / 1024 / 1024,
            'counter_mb': counter_memory / 1024 / 1024,
            'overhead_mb': overhead / 1024 / 1024,
            'total_mb': total_bytes / 1024 / 1024,
            'active_locations': len(self.active_locations),
            'sparsity': 1 - len(self.active_locations) / self.num_locations
        }

# Compare memory usage
def compare_memory_implementations(dimension=2000, num_locations=5000, test_patterns=200):
    """Compare memory usage of different SDM implementations."""
    
    print("Comparing memory implementations...")
    
    # Standard SDM
    standard_config = SDMConfig(
        dimension=dimension,
        num_hard_locations=num_locations,
        activation_radius=int(0.451 * dimension)
    )
    standard_sdm = SDM(standard_config)
    
    # Compressed SDM
    compressed_sdm = CompressedSDM(
        dimension=dimension,
        num_locations=num_locations,
        activation_radius=int(0.451 * dimension)
    )
    
    # Generate test patterns
    addresses, data = generate_random_patterns(test_patterns, dimension)
    
    # Store in both
    for addr, dat in zip(addresses, data):
        standard_sdm.store(addr, dat)
        compressed_sdm.store(addr, dat)
    
    # Memory usage
    standard_memory = (num_locations * dimension * 9) / 8 / 1024 / 1024  # 9 bits per cell
    compressed_memory = compressed_sdm.get_memory_usage()
    
    print(f"\nStandard SDM:")
    print(f"  Memory: {standard_memory:.2f} MB")
    print(f"  Locations used: {standard_sdm.get_memory_stats()['locations_used']}")
    
    print(f"\nCompressed SDM:")
    print(f"  Address memory: {compressed_memory['address_mb']:.2f} MB")
    print(f"  Counter memory: {compressed_memory['counter_mb']:.2f} MB")
    print(f"  Total memory: {compressed_memory['total_mb']:.2f} MB")
    print(f"  Active locations: {compressed_memory['active_locations']}")
    print(f"  Sparsity: {compressed_memory['sparsity']:.2%}")
    print(f"  Savings: {(1 - compressed_memory['total_mb']/standard_memory)*100:.1f}%")
    
    return standard_memory, compressed_memory

# Run comparison
compare_memory_implementations()
```

### Dynamic Memory Allocation

```python
class DynamicSDM:
    """SDM with dynamic memory allocation for hard locations."""
    
    def __init__(self, dimension, initial_locations=100, growth_factor=2.0):
        self.dimension = dimension
        self.growth_factor = growth_factor
        
        # Start with small number of locations
        self.hard_locations = []
        self.counters = []
        self.location_usage = []
        
        # Add initial locations
        self._add_locations(initial_locations)
        
        # Track performance
        self.resize_history = []
        
    def _add_locations(self, count):
        """Add new hard locations."""
        for _ in range(count):
            # Random location
            loc = np.random.randint(0, 2, self.dimension, dtype=np.uint8)
            self.hard_locations.append(loc)
            self.counters.append(np.zeros(self.dimension, dtype=np.int16))
            self.location_usage.append(0)
        
        print(f"Added {count} locations. Total: {len(self.hard_locations)}")
    
    def _check_capacity(self):
        """Check if we need more locations."""
        # Calculate usage ratio
        used_locations = sum(1 for u in self.location_usage if u > 0)
        usage_ratio = used_locations / len(self.hard_locations)
        
        # Grow if usage is high
        if usage_ratio > 0.8:
            old_count = len(self.hard_locations)
            new_count = int(old_count * (self.growth_factor - 1))
            self._add_locations(new_count)
            
            self.resize_history.append({
                'operation': len(self.resize_history) + 1,
                'old_size': old_count,
                'new_size': len(self.hard_locations),
                'usage_ratio': usage_ratio
            })
    
    def store(self, address, data):
        """Store with dynamic allocation."""
        # Check if we need to grow
        self._check_capacity()
        
        # Find activated locations
        activated = []
        for i, loc in enumerate(self.hard_locations):
            dist = np.sum(loc != address)
            if dist <= int(0.451 * self.dimension):
                activated.append(i)
                self.location_usage[i] += 1
        
        # Update counters
        bipolar_data = 2 * data - 1
        for idx in activated:
            self.counters[idx] += bipolar_data
    
    def get_growth_history(self):
        """Return memory growth history."""
        return self.resize_history

# Test dynamic allocation
def test_dynamic_allocation(dimension=1000, initial_patterns=50, total_patterns=500):
    """Test SDM with dynamic memory allocation."""
    
    dynamic_sdm = DynamicSDM(dimension=dimension, initial_locations=100)
    
    # Generate patterns
    addresses, data = generate_random_patterns(total_patterns, dimension)
    
    # Store patterns and track growth
    location_counts = []
    pattern_counts = []
    
    for i, (addr, dat) in enumerate(zip(addresses, data)):
        dynamic_sdm.store(addr, dat)
        
        if i % 10 == 0:
            location_counts.append(len(dynamic_sdm.hard_locations))
            pattern_counts.append(i + 1)
    
    # Visualize growth
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(pattern_counts, location_counts, 'o-', linewidth=2)
    plt.xlabel('Patterns Stored')
    plt.ylabel('Hard Locations')
    plt.title('Dynamic Memory Growth')
    plt.grid(True, alpha=0.3)
    
    # Show resize events
    plt.subplot(2, 1, 2)
    history = dynamic_sdm.get_growth_history()
    if history:
        resize_points = [h['operation'] * 10 for h in history]  # Approximate
        resize_sizes = [h['new_size'] for h in history]
        plt.scatter(resize_points, resize_sizes, color='red', s=100, zorder=5)
        
        for h in history:
            plt.annotate(f"{h['usage_ratio']:.0%}", 
                        (h['operation'] * 10, h['new_size']),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.plot(pattern_counts, location_counts, 'o-', linewidth=2, alpha=0.5)
    plt.xlabel('Patterns Stored')
    plt.ylabel('Hard Locations')
    plt.title('Memory Resize Events')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal statistics:")
    print(f"  Patterns stored: {total_patterns}")
    print(f"  Final locations: {len(dynamic_sdm.hard_locations)}")
    print(f"  Resize events: {len(history)}")
    print(f"  Memory efficiency: {total_patterns / len(dynamic_sdm.hard_locations):.2f} patterns/location")

# Test dynamic allocation
test_dynamic_allocation()
```

---

## Computational Optimization

### Vectorized Operations

```python
class VectorizedSDM:
    """SDM with optimized vectorized operations."""
    
    def __init__(self, config: SDMConfig):
        self.config = config
        
        # Pre-allocate arrays
        self.hard_locations = np.random.randint(
            0, 2, (config.num_hard_locations, config.dimension), dtype=np.uint8
        )
        
        if config.storage_method == 'counters':
            self.counters = np.zeros(
                (config.num_hard_locations, config.dimension), 
                dtype=np.int16
            )
        else:
            self.binary_storage = np.zeros(
                (config.num_hard_locations, config.dimension),
                dtype=np.uint8
            )
        
        # Pre-compute for fast operations
        self._precompute_optimizations()
    
    def _precompute_optimizations(self):
        """Pre-compute values for optimization."""
        # Pre-compute location sums for fast Hamming distance
        self.location_sums = np.sum(self.hard_locations, axis=1)
        
        # Pre-allocate work arrays
        self.work_distances = np.empty(self.config.num_hard_locations, dtype=np.int32)
        self.work_activated = np.empty(self.config.num_hard_locations, dtype=bool)
    
    def store_batch(self, addresses, data):
        """Optimized batch storage."""
        n_patterns = len(addresses)
        
        # Vectorized distance computation for all patterns
        for i in range(n_patterns):
            # Compute distances using pre-computed sums
            addr_sum = np.sum(addresses[i])
            
            # Vectorized XOR and sum
            self.work_distances = addr_sum + self.location_sums - 2 * (self.hard_locations @ addresses[i])
            
            # Find activated locations
            self.work_activated = self.work_distances <= self.config.activation_radius
            activated_indices = np.where(self.work_activated)[0]
            
            if len(activated_indices) > 0:
                if self.config.storage_method == 'counters':
                    # Vectorized counter update
                    bipolar_data = 2 * data[i] - 1
                    self.counters[activated_indices] += bipolar_data
                else:
                    # Vectorized OR operation
                    self.binary_storage[activated_indices] = np.logical_or(
                        self.binary_storage[activated_indices],
                        data[i]
                    )
    
    def recall_batch(self, addresses):
        """Optimized batch recall."""
        n_queries = len(addresses)
        results = []
        
        for i in range(n_queries):
            # Fast distance computation
            addr_sum = np.sum(addresses[i])
            self.work_distances = addr_sum + self.location_sums - 2 * (self.hard_locations @ addresses[i])
            
            # Find activated locations
            self.work_activated = self.work_distances <= self.config.activation_radius
            activated_indices = np.where(self.work_activated)[0]
            
            if len(activated_indices) > 0:
                if self.config.storage_method == 'counters':
                    # Vectorized sum
                    summed = np.sum(self.counters[activated_indices], axis=0)
                    recalled = (summed > 0).astype(np.uint8)
                else:
                    # Vectorized voting
                    summed = np.sum(self.binary_storage[activated_indices], axis=0)
                    recalled = (summed > len(activated_indices) // 2).astype(np.uint8)
                
                results.append(recalled)
            else:
                results.append(None)
        
        return results

# Benchmark vectorized operations
def benchmark_vectorized_ops(dimension=1000, num_locations=1000, batch_size=100):
    """Compare standard vs vectorized operations."""
    
    config = SDMConfig(
        dimension=dimension,
        num_hard_locations=num_locations,
        activation_radius=int(0.451 * dimension)
    )
    
    # Standard SDM
    standard_sdm = SDM(config)
    
    # Vectorized SDM
    vectorized_sdm = VectorizedSDM(config)
    
    # Generate test data
    addresses, data = generate_random_patterns(batch_size, dimension)
    
    # Benchmark standard (sequential)
    start = time.time()
    for addr, dat in zip(addresses, data):
        standard_sdm.store(addr, dat)
    standard_store_time = time.time() - start
    
    start = time.time()
    for addr in addresses:
        _ = standard_sdm.recall(addr)
    standard_recall_time = time.time() - start
    
    # Benchmark vectorized (batch)
    start = time.time()
    vectorized_sdm.store_batch(addresses, data)
    vectorized_store_time = time.time() - start
    
    start = time.time()
    _ = vectorized_sdm.recall_batch(addresses)
    vectorized_recall_time = time.time() - start
    
    print(f"Performance Comparison (batch size={batch_size}):")
    print(f"\nStandard SDM:")
    print(f"  Store time: {standard_store_time:.3f}s")
    print(f"  Recall time: {standard_recall_time:.3f}s")
    print(f"  Total: {standard_store_time + standard_recall_time:.3f}s")
    
    print(f"\nVectorized SDM:")
    print(f"  Store time: {vectorized_store_time:.3f}s")
    print(f"  Recall time: {vectorized_recall_time:.3f}s")
    print(f"  Total: {vectorized_store_time + vectorized_recall_time:.3f}s")
    
    print(f"\nSpeedup:")
    print(f"  Store: {standard_store_time / vectorized_store_time:.2f}x")
    print(f"  Recall: {standard_recall_time / vectorized_recall_time:.2f}x")
    print(f"  Overall: {(standard_store_time + standard_recall_time) / (vectorized_store_time + vectorized_recall_time):.2f}x")

# Run benchmark
benchmark_vectorized_ops()
```

### GPU Acceleration (if available)

```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available. GPU acceleration disabled.")

if GPU_AVAILABLE:
    class GPUSDM:
        """GPU-accelerated SDM using CuPy."""
        
        def __init__(self, config: SDMConfig):
            self.config = config
            
            # Allocate on GPU
            self.hard_locations = cp.random.randint(
                0, 2, (config.num_hard_locations, config.dimension), dtype=cp.uint8
            )
            
            if config.storage_method == 'counters':
                self.counters = cp.zeros(
                    (config.num_hard_locations, config.dimension),
                    dtype=cp.int16
                )
            else:
                self.binary_storage = cp.zeros(
                    (config.num_hard_locations, config.dimension),
                    dtype=cp.uint8
                )
            
            # Pre-compute for fast operations
            self.location_sums = cp.sum(self.hard_locations, axis=1)
        
        def store_batch_gpu(self, addresses, data):
            """GPU-accelerated batch storage."""
            # Transfer to GPU
            addresses_gpu = cp.asarray(addresses)
            data_gpu = cp.asarray(data)
            
            n_patterns = len(addresses)
            
            for i in range(n_patterns):
                # GPU distance computation
                addr_sum = cp.sum(addresses_gpu[i])
                distances = addr_sum + self.location_sums - 2 * cp.dot(self.hard_locations, addresses_gpu[i])
                
                # Find activated locations
                activated = cp.where(distances <= self.config.activation_radius)[0]
                
                if len(activated) > 0:
                    if self.config.storage_method == 'counters':
                        bipolar_data = 2 * data_gpu[i] - 1
                        self.counters[activated] += bipolar_data
                    else:
                        self.binary_storage[activated] = cp.logical_or(
                            self.binary_storage[activated],
                            data_gpu[i]
                        )
        
        def recall_batch_gpu(self, addresses):
            """GPU-accelerated batch recall."""
            addresses_gpu = cp.asarray(addresses)
            results = []
            
            for i in range(len(addresses)):
                addr_sum = cp.sum(addresses_gpu[i])
                distances = addr_sum + self.location_sums - 2 * cp.dot(self.hard_locations, addresses_gpu[i])
                
                activated = cp.where(distances <= self.config.activation_radius)[0]
                
                if len(activated) > 0:
                    if self.config.storage_method == 'counters':
                        summed = cp.sum(self.counters[activated], axis=0)
                        recalled = (summed > 0).astype(cp.uint8)
                    else:
                        summed = cp.sum(self.binary_storage[activated], axis=0)
                        recalled = (summed > len(activated) // 2).astype(cp.uint8)
                    
                    # Transfer back to CPU
                    results.append(cp.asnumpy(recalled))
                else:
                    results.append(None)
            
            return results
    
    # Benchmark GPU acceleration
    def benchmark_gpu_acceleration(dimension=2000, num_locations=5000, batch_size=200):
        """Compare CPU vs GPU performance."""
        
        config = SDMConfig(
            dimension=dimension,
            num_hard_locations=num_locations,
            activation_radius=int(0.451 * dimension)
        )
        
        # CPU SDM (vectorized)
        cpu_sdm = VectorizedSDM(config)
        
        # GPU SDM
        gpu_sdm = GPUSDM(config)
        
        # Generate test data
        addresses, data = generate_random_patterns(batch_size, dimension)
        
        # Warm up GPU
        _ = gpu_sdm.recall_batch_gpu(addresses[:10])
        
        # Benchmark CPU
        start = time.time()
        cpu_sdm.store_batch(addresses, data)
        cpu_store_time = time.time() - start
        
        start = time.time()
        _ = cpu_sdm.recall_batch(addresses)
        cpu_recall_time = time.time() - start
        
        # Benchmark GPU
        start = time.time()
        gpu_sdm.store_batch_gpu(addresses, data)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        gpu_store_time = time.time() - start
        
        start = time.time()
        _ = gpu_sdm.recall_batch_gpu(addresses)
        cp.cuda.Stream.null.synchronize()
        gpu_recall_time = time.time() - start
        
        print(f"GPU Acceleration Results (batch size={batch_size}):")
        print(f"\nCPU (Vectorized):")
        print(f"  Store time: {cpu_store_time:.3f}s")
        print(f"  Recall time: {cpu_recall_time:.3f}s")
        
        print(f"\nGPU:")
        print(f"  Store time: {gpu_store_time:.3f}s")
        print(f"  Recall time: {gpu_recall_time:.3f}s")
        
        print(f"\nGPU Speedup:")
        print(f"  Store: {cpu_store_time / gpu_store_time:.2f}x")
        print(f"  Recall: {cpu_recall_time / gpu_recall_time:.2f}x")
    
    # Run GPU benchmark if available
    benchmark_gpu_acceleration()
```

---

## Scaling Strategies

### Distributed SDM

```python
class DistributedSDM:
    """SDM distributed across multiple nodes/processes."""
    
    def __init__(self, dimension, total_locations, num_nodes):
        self.dimension = dimension
        self.total_locations = total_locations
        self.num_nodes = num_nodes
        
        # Divide locations among nodes
        self.locations_per_node = total_locations // num_nodes
        
        # Simulate nodes (in practice, these would be separate processes)
        self.nodes = []
        for i in range(num_nodes):
            node_config = SDMConfig(
                dimension=dimension,
                num_hard_locations=self.locations_per_node,
                activation_radius=int(0.451 * dimension)
            )
            self.nodes.append(SDM(node_config))
        
        # Load balancing statistics
        self.node_loads = [0] * num_nodes
    
    def _hash_to_nodes(self, address, k=2):
        """Hash address to k nodes for redundancy."""
        # Simple hash function
        hash_value = hash(address.tobytes())
        
        # Select k nodes
        selected = []
        for i in range(k):
            node_id = (hash_value + i) % self.num_nodes
            selected.append(node_id)
        
        return selected
    
    def store_distributed(self, address, data, redundancy=2):
        """Store pattern with redundancy across nodes."""
        # Select nodes
        target_nodes = self._hash_to_nodes(address, k=redundancy)
        
        # Store on selected nodes
        for node_id in target_nodes:
            self.nodes[node_id].store(address, data)
            self.node_loads[node_id] += 1
    
    def recall_distributed(self, address, redundancy=2):
        """Recall from distributed nodes with voting."""
        # Query same nodes used for storage
        target_nodes = self._hash_to_nodes(address, k=redundancy)
        
        # Collect recalls from all nodes
        recalls = []
        for node_id in target_nodes:
            recalled = self.nodes[node_id].recall(address)
            if recalled is not None:
                recalls.append(recalled)
        
        if not recalls:
            return None
        
        # Vote on final result
        if len(recalls) == 1:
            return recalls[0]
        else:
            # Bit-wise voting
            stacked = np.stack(recalls)
            votes = np.sum(stacked, axis=0)
            return (votes > len(recalls) // 2).astype(np.uint8)
    
    def get_load_balance(self):
        """Analyze load distribution across nodes."""
        loads = np.array(self.node_loads)
        return {
            'mean_load': np.mean(loads),
            'std_load': np.std(loads),
            'min_load': np.min(loads),
            'max_load': np.max(loads),
            'balance_ratio': np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0
        }

# Test distributed SDM
def test_distributed_scaling(dimension=1000, num_nodes_list=[1, 2, 4, 8], 
                           patterns_per_test=200):
    """Test scaling with distributed SDM."""
    
    results = []
    
    for num_nodes in num_nodes_list:
        print(f"\nTesting with {num_nodes} nodes...")
        
        # Total locations scaled with nodes
        total_locations = 1000 * num_nodes
        
        # Create distributed SDM
        dist_sdm = DistributedSDM(dimension, total_locations, num_nodes)
        
        # Generate test patterns
        addresses, data = generate_random_patterns(patterns_per_test, dimension)
        
        # Measure performance
        start = time.time()
        for addr, dat in zip(addresses, data):
            dist_sdm.store_distributed(addr, dat)
        store_time = time.time() - start
        
        start = time.time()
        accuracies = []
        for addr, dat in zip(addresses, data):
            recalled = dist_sdm.recall_distributed(addr)
            if recalled is not None:
                accuracies.append(np.mean(recalled == dat))
        recall_time = time.time() - start
        
        # Load balance
        balance = dist_sdm.get_load_balance()
        
        results.append({
            'nodes': num_nodes,
            'total_locations': total_locations,
            'store_time': store_time,
            'recall_time': recall_time,
            'throughput': patterns_per_test / (store_time + recall_time),
            'accuracy': np.mean(accuracies),
            'load_balance': balance['balance_ratio']
        })
    
    # Visualize scaling
    plt.figure(figsize=(12, 5))
    
    nodes = [r['nodes'] for r in results]
    
    plt.subplot(1, 2, 1)
    throughput = [r['throughput'] for r in results]
    plt.plot(nodes, throughput, 'o-', linewidth=2)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Throughput (patterns/second)')
    plt.title('Throughput Scaling')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Efficiency = throughput / nodes (normalized)
    efficiency = [t / (n * throughput[0]) for t, n in zip(throughput, nodes)]
    plt.plot(nodes, efficiency, 'o-', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Scaling Efficiency')
    plt.title('Scaling Efficiency')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.2)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Test distributed scaling
dist_results = test_distributed_scaling()
```

### Hierarchical SDM

```python
class HierarchicalSDM:
    """Multi-level hierarchical SDM for large-scale applications."""
    
    def __init__(self, dimension, levels):
        """
        Parameters
        ----------
        dimension : int
            Base dimension
        levels : list of dict
            Configuration for each level
            [{'locations': 100, 'dimension': 500, 'radius': 225}, ...]
        """
        self.dimension = dimension
        self.levels = []
        
        for i, level_config in enumerate(levels):
            config = SDMConfig(
                dimension=level_config['dimension'],
                num_hard_locations=level_config['locations'],
                activation_radius=level_config['radius']
            )
            sdm = SDM(config)
            
            self.levels.append({
                'sdm': sdm,
                'config': level_config,
                'level': i
            })
        
        # Projections between levels
        self.projections = self._create_projections()
    
    def _create_projections(self):
        """Create random projections between levels."""
        projections = []
        
        for i in range(len(self.levels) - 1):
            dim_from = self.levels[i]['config']['dimension']
            dim_to = self.levels[i + 1]['config']['dimension']
            
            # Random projection matrix
            proj = np.random.randn(dim_from, dim_to)
            proj = proj / np.linalg.norm(proj, axis=0)
            
            projections.append(proj)
        
        return projections
    
    def store_hierarchical(self, address, data):
        """Store pattern through hierarchy."""
        current_addr = address
        current_data = data
        
        for i, level in enumerate(self.levels):
            # Store at current level
            level['sdm'].store(current_addr, current_data)
            
            # Project to next level if not last
            if i < len(self.levels) - 1:
                # Project address
                proj_addr = current_addr @ self.projections[i]
                current_addr = (proj_addr > 0).astype(np.uint8)
                
                # Project data
                proj_data = current_data @ self.projections[i]
                current_data = (proj_data > 0).astype(np.uint8)
    
    def recall_hierarchical(self, address, start_level=0):
        """Recall through hierarchy with refinement."""
        results = []
        
        current_addr = address
        
        for i in range(start_level, len(self.levels)):
            level = self.levels[i]
            
            # Recall at current level
            recalled = level['sdm'].recall(current_addr)
            
            if recalled is not None:
                results.append({
                    'level': i,
                    'data': recalled,
                    'confidence': 1.0  # Could add confidence calculation
                })
                
                # Project to next level for refinement
                if i < len(self.levels) - 1:
                    proj_addr = recalled @ self.projections[i]
                    current_addr = (proj_addr > 0).astype(np.uint8)
            else:
                # Try next level with projected address
                if i < len(self.levels) - 1:
                    proj_addr = current_addr @ self.projections[i]
                    current_addr = (proj_addr > 0).astype(np.uint8)
        
        return results

# Test hierarchical SDM
def test_hierarchical_sdm():
    """Test multi-level hierarchical SDM."""
    
    # Define hierarchy
    levels = [
        {'locations': 500, 'dimension': 2000, 'radius': 900},   # Fine level
        {'locations': 200, 'dimension': 1000, 'radius': 450},   # Medium level
        {'locations': 100, 'dimension': 500, 'radius': 225}     # Coarse level
    ]
    
    hier_sdm = HierarchicalSDM(dimension=2000, levels=levels)
    
    # Generate test patterns
    num_patterns = 100
    addresses, data = generate_random_patterns(num_patterns, 2000)
    
    # Store patterns
    print("Storing patterns in hierarchy...")
    for addr, dat in zip(addresses, data):
        hier_sdm.store_hierarchical(addr, dat)
    
    # Test recall at different levels
    print("\nTesting recall from different levels...")
    test_size = 20
    
    for start_level in range(len(levels)):
        accuracies = []
        
        for i in range(test_size):
            # Add noise based on level
            noise_level = 0.05 * (start_level + 1)
            noisy_addr = add_noise(addresses[i], noise_level)
            
            # Recall from hierarchy
            results = hier_sdm.recall_hierarchical(noisy_addr, start_level=start_level)
            
            if results:
                # Use finest level result
                recalled = results[0]['data']
                
                # Need to project back to original dimension for comparison
                if len(recalled) < len(data[i]):
                    # Pad or project back
                    recalled_full = np.zeros(len(data[i]), dtype=np.uint8)
                    recalled_full[:len(recalled)] = recalled
                    recalled = recalled_full
                
                accuracy = np.mean(recalled[:len(data[i])] == data[i])
                accuracies.append(accuracy)
        
        print(f"  Level {start_level}: {np.mean(accuracies):.2%} accuracy "
              f"(noise: {noise_level:.0%})")
    
    return hier_sdm

# Test hierarchical system
hier_sdm = test_hierarchical_sdm()
```

---

## Hardware Considerations

### CPU Optimization

```python
def analyze_cpu_optimization():
    """Analyze impact of CPU optimization techniques."""
    
    import platform
    import multiprocessing
    
    # System information
    print("System Information:")
    print(f"  CPU: {platform.processor()}")
    print(f"  Cores: {multiprocessing.cpu_count()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  NumPy: {np.__version__}")
    
    # Test different optimization levels
    dimension = 2000
    num_locations = 5000
    test_patterns = 100
    
    # Standard implementation
    config_standard = SDMConfig(
        dimension=dimension,
        num_hard_locations=num_locations,
        activation_radius=int(0.451 * dimension),
        parallel=False
    )
    
    # Parallel implementation
    config_parallel = SDMConfig(
        dimension=dimension,
        num_hard_locations=num_locations,
        activation_radius=int(0.451 * dimension),
        parallel=True,
        num_workers=multiprocessing.cpu_count()
    )
    
    # Test data
    addresses, data = generate_random_patterns(test_patterns, dimension)
    
    # Benchmark standard
    sdm_standard = SDM(config_standard)
    start = time.time()
    for addr, dat in zip(addresses, data):
        sdm_standard.store(addr, dat)
    standard_time = time.time() - start
    
    # Benchmark parallel
    sdm_parallel = SDM(config_parallel)
    start = time.time()
    for addr, dat in zip(addresses, data):
        sdm_parallel.store(addr, dat)
    parallel_time = time.time() - start
    
    # NumPy threading
    print(f"\nNumPy Threading:")
    print(f"  MKL threads: {os.environ.get('MKL_NUM_THREADS', 'default')}")
    print(f"  OpenBLAS threads: {os.environ.get('OPENBLAS_NUM_THREADS', 'default')}")
    
    print(f"\nPerformance Results:")
    print(f"  Standard: {standard_time:.3f}s")
    print(f"  Parallel: {parallel_time:.3f}s")
    print(f"  Speedup: {standard_time / parallel_time:.2f}x")
    
    # Memory bandwidth test
    print(f"\nMemory Bandwidth Test:")
    size = 100_000_000  # 100M elements
    
    # Write bandwidth
    arr = np.zeros(size)
    start = time.time()
    arr[:] = 1
    write_time = time.time() - start
    write_bandwidth = size * 8 / write_time / 1e9  # GB/s
    
    # Read bandwidth
    start = time.time()
    _ = np.sum(arr)
    read_time = time.time() - start
    read_bandwidth = size * 8 / read_time / 1e9  # GB/s
    
    print(f"  Write: {write_bandwidth:.1f} GB/s")
    print(f"  Read: {read_bandwidth:.1f} GB/s")

# Analyze CPU optimization
analyze_cpu_optimization()
```

### Memory Bandwidth Optimization

```python
def optimize_memory_access(dimension=2000, num_locations=5000):
    """Demonstrate memory access optimization techniques."""
    
    print("Memory Access Optimization Analysis")
    
    # Create test data
    hard_locations = np.random.randint(0, 2, (num_locations, dimension), dtype=np.uint8)
    test_address = np.random.randint(0, 2, dimension, dtype=np.uint8)
    
    # Method 1: Row-wise access (cache-friendly)
    start = time.time()
    distances_row = []
    for i in range(num_locations):
        dist = np.sum(hard_locations[i] != test_address)
        distances_row.append(dist)
    row_time = time.time() - start
    
    # Method 2: Vectorized (optimal)
    start = time.time()
    distances_vec = np.sum(hard_locations != test_address, axis=1)
    vec_time = time.time() - start
    
    # Method 3: Column-wise access (cache-unfriendly)
    hard_locations_T = hard_locations.T
    start = time.time()
    distances_col = []
    for i in range(num_locations):
        dist = 0
        for j in range(dimension):
            if hard_locations_T[j, i] != test_address[j]:
                dist += 1
        distances_col.append(dist)
    col_time = time.time() - start
    
    print(f"\nTiming Results:")
    print(f"  Row-wise: {row_time:.4f}s")
    print(f"  Vectorized: {vec_time:.4f}s ({row_time/vec_time:.1f}x faster)")
    print(f"  Column-wise: {col_time:.4f}s ({col_time/vec_time:.1f}x slower)")
    
    # Memory layout impact
    print(f"\nMemory Layout:")
    print(f"  Array shape: {hard_locations.shape}")
    print(f"  Array strides: {hard_locations.strides}")
    print(f"  C-contiguous: {hard_locations.flags['C_CONTIGUOUS']}")
    print(f"  Cache line efficiency: {64 / hard_locations.itemsize} elements per line")

# Run memory optimization analysis
optimize_memory_access()
```

---

## Performance Monitoring

### Real-time Performance Monitor

```python
class SDMPerformanceMonitor:
    """Real-time performance monitoring for SDM operations."""
    
    def __init__(self, sdm, window_size=100):
        self.sdm = sdm
        self.window_size = window_size
        
        # Circular buffers for metrics
        self.store_times = collections.deque(maxlen=window_size)
        self.recall_times = collections.deque(maxlen=window_size)
        self.accuracies = collections.deque(maxlen=window_size)
        self.activation_counts = collections.deque(maxlen=window_size)
        
        # Cumulative statistics
        self.total_stores = 0
        self.total_recalls = 0
        self.total_time = 0
        
        # Wrap SDM methods
        self._wrap_methods()
    
    def _wrap_methods(self):
        """Wrap SDM methods with monitoring."""
        # Save original methods
        self._original_store = self.sdm.store
        self._original_recall = self.sdm.recall
        
        # Replace with monitored versions
        self.sdm.store = self._monitored_store
        self.sdm.recall = self._monitored_recall
    
    def _monitored_store(self, address, data):
        """Monitored store operation."""
        start = time.time()
        
        # Get activation count before store
        activated = self.sdm._get_activated_locations(address)
        self.activation_counts.append(len(activated))
        
        # Perform store
        self._original_store(address, data)
        
        # Record time
        elapsed = time.time() - start
        self.store_times.append(elapsed)
        self.total_stores += 1
        self.total_time += elapsed
    
    def _monitored_recall(self, address):
        """Monitored recall operation."""
        start = time.time()
        
        # Perform recall
        result = self._original_recall(address)
        
        # Record time
        elapsed = time.time() - start
        self.recall_times.append(elapsed)
        self.total_recalls += 1
        self.total_time += elapsed
        
        return result
    
    def get_current_stats(self):
        """Get current performance statistics."""
        return {
            'avg_store_time': np.mean(self.store_times) if self.store_times else 0,
            'avg_recall_time': np.mean(self.recall_times) if self.recall_times else 0,
            'avg_activations': np.mean(self.activation_counts) if self.activation_counts else 0,
            'store_throughput': len(self.store_times) / sum(self.store_times) if self.store_times else 0,
            'recall_throughput': len(self.recall_times) / sum(self.recall_times) if self.recall_times else 0,
            'total_operations': self.total_stores + self.total_recalls,
            'uptime': self.total_time
        }
    
    def plot_realtime(self, duration=30):
        """Plot real-time performance metrics."""
        import matplotlib.animation as animation
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Initialize data
        times = []
        store_throughputs = []
        recall_throughputs = []
        activations = []
        
        def animate(frame):
            # Get current stats
            stats = self.get_current_stats()
            
            # Update data
            times.append(frame)
            store_throughputs.append(stats['store_throughput'])
            recall_throughputs.append(stats['recall_throughput'])
            activations.append(stats['avg_activations'])
            
            # Keep last N points
            if len(times) > 50:
                times.pop(0)
                store_throughputs.pop(0)
                recall_throughputs.pop(0)
                activations.pop(0)
            
            # Clear and plot
            for ax in axes.flat:
                ax.clear()
            
            # Throughput
            axes[0, 0].plot(times, store_throughputs, 'b-', label='Store')
            axes[0, 0].plot(times, recall_throughputs, 'r-', label='Recall')
            axes[0, 0].set_ylabel('Operations/second')
            axes[0, 0].set_title('Throughput')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Latency histogram
            if self.store_times:
                axes[0, 1].hist(list(self.store_times), bins=20, alpha=0.5, label='Store')
            if self.recall_times:
                axes[0, 1].hist(list(self.recall_times), bins=20, alpha=0.5, label='Recall')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_title('Latency Distribution')
            axes[0, 1].legend()
            
            # Activations
            axes[1, 0].plot(times, activations, 'g-')
            axes[1, 0].set_ylabel('Activated Locations')
            axes[1, 0].set_title('Average Activations')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Stats text
            axes[1, 1].axis('off')
            stats_text = f"""Current Statistics:
            
Total Operations: {stats['total_operations']}
Uptime: {stats['uptime']:.1f}s

Store:
  Avg Time: {stats['avg_store_time']*1000:.2f}ms
  Throughput: {stats['store_throughput']:.1f}/s
  
Recall:
  Avg Time: {stats['avg_recall_time']*1000:.2f}ms
  Throughput: {stats['recall_throughput']:.1f}/s"""
            
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                          verticalalignment='top', fontfamily='monospace')
            
            return axes.flat
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, interval=1000, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim

# Example usage
sdm = create_sdm(dimension=1000)
monitor = SDMPerformanceMonitor(sdm)

# Simulate operations
print("Running performance monitor demo...")
addresses, data = generate_random_patterns(200, 1000)

# Store patterns with monitoring
for i, (addr, dat) in enumerate(zip(addresses, data)):
    sdm.store(addr, dat)
    
    # Occasional recalls
    if i % 5 == 0:
        _ = sdm.recall(addresses[i // 5])

# Display stats
stats = monitor.get_current_stats()
print("\nPerformance Summary:")
for key, value in stats.items():
    if 'time' in key:
        print(f"  {key}: {value*1000:.2f}ms")
    elif 'throughput' in key:
        print(f"  {key}: {value:.1f}/s")
    else:
        print(f"  {key}: {value:.2f}")
```

---

## Common Bottlenecks

### Bottleneck Analysis

```python
def analyze_bottlenecks(sdm, test_patterns=100):
    """Identify performance bottlenecks in SDM operations."""
    
    import cProfile
    import pstats
    from line_profiler import LineProfiler
    
    # Generate test data
    addresses, data = generate_random_patterns(test_patterns, sdm.config.dimension)
    
    print("Analyzing SDM Performance Bottlenecks...")
    
    # 1. Overall profiling
    print("\n1. Function-level Profiling:")
    profiler = cProfile.Profile()
    
    profiler.enable()
    for addr, dat in zip(addresses[:50], data[:50]):
        sdm.store(addr, dat)
        _ = sdm.recall(addr)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    # 2. Memory allocation analysis
    print("\n2. Memory Allocation Analysis:")
    import tracemalloc
    
    tracemalloc.start()
    
    # Perform operations
    for i in range(20):
        sdm.store(addresses[i], data[i])
        _ = sdm.recall(addresses[i])
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"  Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    # Get top memory allocations
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\n  Top memory allocations:")
    for stat in top_stats[:5]:
        print(f"    {stat}")
    
    tracemalloc.stop()
    
    # 3. Cache efficiency
    print("\n3. Cache Efficiency Analysis:")
    
    # Sequential access pattern
    start = time.time()
    for i in range(100):
        _ = np.sum(sdm.hard_locations[i % sdm.config.num_hard_locations])
    sequential_time = time.time() - start
    
    # Random access pattern
    indices = np.random.randint(0, sdm.config.num_hard_locations, 100)
    start = time.time()
    for i in indices:
        _ = np.sum(sdm.hard_locations[i])
    random_time = time.time() - start
    
    print(f"  Sequential access: {sequential_time*1000:.2f}ms")
    print(f"  Random access: {random_time*1000:.2f}ms")
    print(f"  Cache penalty: {(random_time/sequential_time - 1)*100:.1f}%")
    
    # 4. Bottleneck identification
    print("\n4. Identified Bottlenecks:")
    
    bottlenecks = []
    
    # Check activation computation
    test_addr = addresses[0]
    start = time.time()
    for _ in range(100):
        _ = sdm._get_activated_locations(test_addr)
    activation_time = (time.time() - start) / 100
    
    if activation_time > 0.001:  # > 1ms
        bottlenecks.append(f"Activation computation: {activation_time*1000:.2f}ms")
    
    # Check memory bandwidth
    data_size = sdm.config.num_hard_locations * sdm.config.dimension / 8 / 1024 / 1024  # MB
    if data_size > 100:
        bottlenecks.append(f"Large memory footprint: {data_size:.1f}MB")
    
    # Check parallelization potential
    if not sdm.config.parallel and sdm.config.num_hard_locations > 1000:
        bottlenecks.append("Parallelization not enabled for large SDM")
    
    if bottlenecks:
        for b in bottlenecks:
            print(f"  - {b}")
    else:
        print("  No major bottlenecks identified")
    
    return bottlenecks

# Analyze bottlenecks
sdm = create_sdm(dimension=2000, num_locations=5000)
bottlenecks = analyze_bottlenecks(sdm)
```

---

## Best Practices

### Performance Best Practices Summary

```python
def print_best_practices():
    """Print SDM performance best practices."""
    
    best_practices = """
SDM PERFORMANCE BEST PRACTICES
==============================

1. DIMENSION SELECTION
   - Use 1000-2000 dimensions for balanced performance
   - Higher dimensions: better separation, more memory
   - Lower dimensions: faster operations, less capacity
   
2. ACTIVATION RADIUS
   - Start with critical distance (0.451 × dimension)
   - Decrease for less interference, lower capacity
   - Increase for more capacity, higher interference
   
3. NUMBER OF HARD LOCATIONS
   - Rule of thumb: sqrt(2^dimension) or 0.001% of space
   - More locations = more capacity but slower operations
   - Scale with available memory
   
4. STORAGE METHOD
   - Counters: Better accuracy, higher memory (default)
   - Binary: Lower memory, reduced capacity
   - Choose based on memory constraints
   
5. PARALLELIZATION
   - Enable for > 1000 locations
   - Set workers = CPU cores
   - Batch operations when possible
   
6. MEMORY OPTIMIZATION
   - Use sparse storage for < 50% utilization
   - Consider compression for large deployments
   - Monitor memory growth
   
7. DECODER SELECTION
   - Hamming: General purpose (default)
   - Jaccard: Sparse data
   - Random: Fast, no generalization
   - Adaptive: Self-tuning
   - LSH: Very large scale
   
8. BATCH OPERATIONS
   - Process multiple patterns together
   - Amortize overhead costs
   - Better cache utilization
   
9. NOISE TOLERANCE
   - Design for expected noise level
   - Test with realistic noise patterns
   - Consider iterative recall for denoising
   
10. MONITORING
    - Track operation times
    - Monitor memory usage
    - Analyze activation patterns
    - Profile bottlenecks regularly
"""
    
    print(best_practices)

# Print best practices
print_best_practices()
```

### Performance Tuning Checklist

```python
def performance_tuning_checklist(sdm):
    """Run performance tuning checklist for SDM."""
    
    print("SDM PERFORMANCE TUNING CHECKLIST")
    print("="*40)
    
    checks = []
    
    # 1. Dimension check
    if 500 <= sdm.config.dimension <= 5000:
        checks.append(("✓", "Dimension in optimal range"))
    else:
        checks.append(("✗", f"Dimension {sdm.config.dimension} outside optimal range (500-5000)"))
    
    # 2. Activation radius check
    critical = int(0.451 * sdm.config.dimension)
    if abs(sdm.config.activation_radius - critical) < 50:
        checks.append(("✓", "Activation radius near critical distance"))
    else:
        checks.append(("⚠", f"Activation radius {sdm.config.activation_radius} far from critical {critical}"))
    
    # 3. Location count check
    if sdm.config.num_hard_locations >= 100:
        checks.append(("✓", "Sufficient hard locations"))
    else:
        checks.append(("✗", "Too few hard locations (< 100)"))
    
    # 4. Parallelization check
    if sdm.config.num_hard_locations > 1000:
        if sdm.config.parallel:
            checks.append(("✓", "Parallelization enabled for large SDM"))
        else:
            checks.append(("⚠", "Consider enabling parallelization"))
    else:
        checks.append(("✓", "SDM size appropriate for sequential processing"))
    
    # 5. Memory usage check
    memory_mb = (sdm.config.num_hard_locations * sdm.config.dimension * 9) / 8 / 1024 / 1024
    if memory_mb < 1000:
        checks.append(("✓", f"Reasonable memory usage ({memory_mb:.1f} MB)"))
    else:
        checks.append(("⚠", f"High memory usage ({memory_mb:.1f} MB)"))
    
    # 6. Storage method check
    if memory_mb > 500 and sdm.config.storage_method == "counters":
        checks.append(("⚠", "Consider binary storage for large memory usage"))
    else:
        checks.append(("✓", "Storage method appropriate"))
    
    # Print results
    for status, message in checks:
        print(f"{status} {message}")
    
    # Overall score
    score = sum(1 for status, _ in checks if status == "✓") / len(checks)
    print(f"\nOverall Score: {score:.0%}")
    
    if score < 0.8:
        print("\nRecommendations:")
        for status, message in checks:
            if status != "✓":
                print(f"  - {message}")

# Run checklist
sdm = create_sdm(dimension=1000, num_locations=1000)
performance_tuning_checklist(sdm)
```

---

## Performance Benchmarks

### Standard Benchmark Results

```python
def run_standard_benchmarks():
    """Run and display standard SDM benchmarks."""
    
    print("STANDARD SDM PERFORMANCE BENCHMARKS")
    print("="*60)
    print("Hardware: CPU-based, single machine")
    print("Test: 100 patterns store/recall")
    print("="*60)
    
    # Benchmark configurations
    configs = [
        {"name": "Small", "dim": 256, "locs": 500},
        {"name": "Medium", "dim": 1000, "locs": 1000},
        {"name": "Large", "dim": 2000, "locs": 5000},
        {"name": "XLarge", "dim": 5000, "locs": 10000},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\nBenchmarking {cfg['name']} configuration...")
        
        # Create SDM
        sdm = create_sdm(dimension=cfg['dim'], num_locations=cfg['locs'])
        
        # Generate test data
        addresses, data = generate_random_patterns(100, cfg['dim'])
        
        # Benchmark
        start = time.time()
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        store_time = time.time() - start
        
        start = time.time()
        for addr in addresses:
            _ = sdm.recall(addr)
        recall_time = time.time() - start
        
        # Memory
        memory_mb = (cfg['locs'] * cfg['dim'] * 9) / 8 / 1024 / 1024
        
        results.append({
            'config': cfg['name'],
            'dimension': cfg['dim'],
            'locations': cfg['locs'],
            'store_ms': (store_time / 100) * 1000,
            'recall_ms': (recall_time / 100) * 1000,
            'memory_mb': memory_mb
        })
    
    # Display results table
    print("\n" + "="*80)
    print(f"{'Config':<10} {'Dimension':<12} {'Locations':<12} {'Store(ms)':<12} {'Recall(ms)':<12} {'Memory(MB)':<12}")
    print("="*80)
    
    for r in results:
        print(f"{r['config']:<10} {r['dimension']:<12} {r['locations']:<12} "
              f"{r['store_ms']:<12.2f} {r['recall_ms']:<12.2f} {r['memory_mb']:<12.1f}")
    
    return results

# Run standard benchmarks
benchmark_results = run_standard_benchmarks()
```

## Conclusion

This performance guide covers comprehensive optimization strategies for SDM:

1. **Benchmarking** - Systematic performance measurement
2. **Parameter Tuning** - Optimal configuration selection
3. **Memory Efficiency** - Compression and sparse storage
4. **Computational Speed** - Vectorization and parallelization
5. **Scaling** - Distributed and hierarchical approaches
6. **Hardware Optimization** - CPU and GPU acceleration
7. **Monitoring** - Real-time performance tracking
8. **Best Practices** - Guidelines for optimal performance

Key takeaways:
- Choose dimensions between 1000-2000 for balanced performance
- Use activation radius near critical distance (0.451 × dimension)
- Enable parallelization for large SDMs (>1000 locations)
- Consider binary storage for memory-constrained applications
- Monitor performance continuously in production
- Profile and optimize bottlenecks iteratively

The SDM implementation provides excellent performance for content-addressable storage with proper tuning.