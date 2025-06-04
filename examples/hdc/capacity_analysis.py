#!/usr/bin/env python3
"""
HDC Capacity Analysis

This example analyzes the capacity and performance characteristics
of different HDC configurations:
- Memory capacity estimation
- Noise tolerance testing
- Operation benchmarking
- Comparison of hypervector types
"""

import numpy as np
from cognitive_computing.hdc import (
    create_hdc,
    HDCConfig,
    measure_capacity,
    benchmark_operations,
    compare_hypervector_types,
    analyze_binding_properties,
    measure_associativity,
    estimate_required_dimension,
    plot_capacity_analysis,
)
import matplotlib.pyplot as plt
from tabulate import tabulate


def capacity_estimation_demo():
    """Demonstrate capacity estimation for different dimensions."""
    print("=== Capacity Estimation Demo ===\n")
    
    dimensions = [100, 1000, 5000, 10000]
    results = []
    
    for dim in dimensions:
        print(f"Analyzing dimension {dim}...")
        
        # Create HDC system
        config = HDCConfig(dimension=dim, hypervector_type="bipolar")
        hdc = create_hdc(config)
        
        # Measure capacity
        metrics = measure_capacity(
            hdc,
            num_items=min(1000, dim//10),
            noise_levels=[0.0, 0.1, 0.2, 0.3],
            similarity_threshold=0.1
        )
        
        results.append({
            "Dimension": dim,
            "Est. Capacity": metrics.capacity_results["estimated_capacity"],
            "Mean Similarity": f"{metrics.capacity_results['mean_similarity']:.4f}",
            "Max Similarity": f"{metrics.capacity_results['max_similarity']:.4f}",
            "Interference Rate": f"{metrics.capacity_results['interference_rate']:.2%}",
            "Noise Tol. (10%)": f"{metrics.noise_tolerance.get(0.1, 0):.2%}",
            "Noise Tol. (20%)": f"{metrics.noise_tolerance.get(0.2, 0):.2%}",
        })
    
    # Display results
    print("\nCapacity Analysis Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Plot the last analysis
    fig, axes = plot_capacity_analysis(metrics)
    plt.suptitle(f"Capacity Analysis for {dim}-dimensional Hypervectors")
    
    return metrics


def noise_tolerance_demo():
    """Demonstrate noise tolerance across different configurations."""
    print("\n=== Noise Tolerance Demo ===\n")
    
    # Test different hypervector types
    types = ["binary", "bipolar", "ternary"]
    dimension = 5000
    noise_levels = np.linspace(0, 0.5, 11)
    
    tolerance_curves = {}
    
    for hv_type in types:
        print(f"Testing {hv_type} hypervectors...")
        
        # Create HDC system
        config = HDCConfig(dimension=dimension, hypervector_type=hv_type)
        hdc = create_hdc(config)
        
        # Measure noise tolerance
        metrics = measure_capacity(
            hdc,
            num_items=100,
            noise_levels=noise_levels.tolist()
        )
        
        tolerance_curves[hv_type] = metrics.noise_tolerance
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for hv_type, tolerances in tolerance_curves.items():
        noise_pts = sorted(tolerances.keys())
        recovery_rates = [tolerances[n] for n in noise_pts]
        plt.plot(noise_pts, recovery_rates, marker='o', label=hv_type)
    
    plt.xlabel("Noise Level (fraction of bits flipped)")
    plt.ylabel("Recovery Rate")
    plt.title(f"Noise Tolerance Comparison ({dimension} dimensions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    return tolerance_curves


def operation_benchmarking_demo():
    """Benchmark HDC operations."""
    print("\n=== Operation Benchmarking Demo ===\n")
    
    dimensions = [1000, 5000, 10000]
    results = []
    
    for dim in dimensions:
        print(f"Benchmarking dimension {dim}...")
        
        # Create HDC system
        config = HDCConfig(dimension=dim)
        hdc = create_hdc(config)
        
        # Benchmark
        times = benchmark_operations(hdc, num_trials=100)
        
        results.append({
            "Dimension": dim,
            "Generate (ms)": f"{times['generate_hypervector']:.3f}",
            "Bind (ms)": f"{times['bind']:.3f}",
            "Bundle-5 (ms)": f"{times['bundle_5']:.3f}",
            "Similarity (ms)": f"{times['similarity']:.3f}",
            "Permute (ms)": f"{times['permute']:.3f}",
        })
    
    print("\nOperation Timing Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Plot scaling
    plt.figure(figsize=(10, 6))
    
    operations = ["generate_hypervector", "bind", "bundle_5", "similarity", "permute"]
    
    for op in operations:
        times_list = []
        for dim in dimensions:
            config = HDCConfig(dimension=dim)
            hdc = create_hdc(config)
            times = benchmark_operations(hdc, num_trials=50)
            times_list.append(times[op])
        
        plt.plot(dimensions, times_list, marker='o', label=op)
    
    plt.xlabel("Dimension")
    plt.ylabel("Time (ms)")
    plt.title("Operation Scaling with Dimension")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return results


def binding_properties_demo():
    """Analyze binding operation properties."""
    print("\n=== Binding Properties Analysis ===\n")
    
    types = ["binary", "bipolar"]
    dimensions = [1000, 5000, 10000]
    
    results = []
    
    for hv_type in types:
        for dim in dimensions:
            print(f"Analyzing {hv_type} binding at dimension {dim}...")
            
            # Analyze binding
            props = analyze_binding_properties(
                dimension=dim,
                hypervector_type=hv_type,
                num_samples=50
            )
            
            results.append({
                "Type": hv_type,
                "Dimension": dim,
                "Self-Inverse Error": f"{props['mean_self_inverse_error']:.6f}",
                "Max Error": f"{props['max_self_inverse_error']:.6f}",
                "Distance Preservation": f"{props['mean_distance_preservation']:.3f}±{props['std_distance_preservation']:.3f}",
            })
    
    print("\nBinding Properties Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    return results


def associativity_analysis_demo():
    """Analyze associativity of bundling."""
    print("\n=== Associativity Analysis ===\n")
    
    types = ["binary", "bipolar", "ternary"]
    dimensions = [1000, 5000, 10000]
    
    results = []
    
    for hv_type in types:
        for dim in dimensions:
            print(f"Testing {hv_type} associativity at dimension {dim}...")
            
            # Measure associativity
            assoc = measure_associativity(
                dimension=dim,
                hypervector_type=hv_type,
                num_trials=50
            )
            
            results.append({
                "Type": hv_type,
                "Dimension": dim,
                "Mean Error": f"{assoc['mean_associativity_error']:.4f}",
                "Max Error": f"{assoc['max_associativity_error']:.4f}",
                "Perfect Rate": f"{assoc['perfect_associations']:.1%}",
            })
    
    print("\nAssociativity Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    return results


def dimension_estimation_demo():
    """Demonstrate dimension estimation for different scenarios."""
    print("\n=== Dimension Estimation Demo ===\n")
    
    scenarios = [
        {"items": 10, "threshold": 0.1, "name": "Small vocabulary"},
        {"items": 100, "threshold": 0.1, "name": "Medium vocabulary"},
        {"items": 1000, "threshold": 0.1, "name": "Large vocabulary"},
        {"items": 10000, "threshold": 0.05, "name": "Very large vocabulary"},
        {"items": 100, "threshold": 0.01, "name": "High precision requirement"},
    ]
    
    results = []
    
    for scenario in scenarios:
        dim = estimate_required_dimension(
            num_items=scenario["items"],
            similarity_threshold=scenario["threshold"]
        )
        
        results.append({
            "Scenario": scenario["name"],
            "Items": scenario["items"],
            "Threshold": scenario["threshold"],
            "Required Dim": dim,
        })
    
    print("\nDimension Estimation Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    return results


def type_comparison_demo():
    """Compare different hypervector types."""
    print("\n=== Hypervector Type Comparison ===\n")
    
    print("Running comprehensive comparison (this may take a moment)...")
    comparison = compare_hypervector_types(
        dimension=5000,
        num_items=100
    )
    
    # Format results
    results = []
    for hv_type, metrics in comparison.items():
        results.append({
            "Type": hv_type,
            "Mean Similarity": f"{metrics['mean_similarity']:.4f}",
            "Est. Capacity": f"{metrics['estimated_capacity']:,}",
            "Noise Tol (10%)": f"{metrics['noise_tolerance_0.1']:.2%}",
            "Noise Tol (20%)": f"{metrics['noise_tolerance_0.2']:.2%}",
            "Generate (ms)": f"{metrics['generate_time_ms']:.3f}",
            "Bind (ms)": f"{metrics['bind_time_ms']:.3f}",
            "Bundle (ms)": f"{metrics['bundle_time_ms']:.3f}",
        })
    
    print("\nHypervector Type Comparison:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    return comparison


def main():
    """Run all capacity analysis demonstrations."""
    # Capacity estimation
    capacity_metrics = capacity_estimation_demo()
    
    # Noise tolerance
    noise_curves = noise_tolerance_demo()
    
    # Operation benchmarking
    benchmark_results = operation_benchmarking_demo()
    
    # Binding properties
    binding_results = binding_properties_demo()
    
    # Associativity
    assoc_results = associativity_analysis_demo()
    
    # Dimension estimation
    dim_estimates = dimension_estimation_demo()
    
    # Type comparison
    type_comparison = type_comparison_demo()
    
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    # Note: tabulate is optional, provide fallback
    try:
        from tabulate import tabulate
    except ImportError:
        def tabulate(data, headers="keys", tablefmt="grid"):
            """Simple fallback for tabulate."""
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # Print header
                keys = list(data[0].keys())
                print(" | ".join(keys))
                print("-" * (len(" | ".join(keys)) + 2))
                # Print rows
                for row in data:
                    values = [str(row[k]) for k in keys]
                    print(" | ".join(values))
            return ""
    
    main()