#!/usr/bin/env python3
"""
Noise Tolerance Analysis for Sparse Distributed Memory

This example demonstrates SDM's robust noise tolerance capabilities,
including various noise types, iterative denoising, and practical
applications in noisy environments.

Usage:
    python noise_tolerance.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
import seaborn as sns

# Import SDM components
from cognitive_computing.sdm import create_sdm, SDM, SDMConfig
from cognitive_computing.sdm.utils import (
    add_noise, generate_random_patterns, PatternEncoder,
    calculate_pattern_similarity, test_sdm_performance
)
from cognitive_computing.sdm.visualizations import plot_recall_accuracy


@dataclass
class NoiseTestResult:
    """Results from noise tolerance testing."""
    noise_type: str
    noise_level: float
    accuracy: float
    bit_error_rate: float
    recall_success_rate: float
    confidence: float


class NoiseTolerance:
    """Class for analyzing SDM noise tolerance."""
    
    def __init__(self, dimension: int, num_locations: int = None):
        """
        Initialize noise tolerance analyzer.
        
        Parameters
        ----------
        dimension : int
            Pattern dimension
        num_locations : int, optional
            Number of hard locations
        """
        self.dimension = dimension
        
        if num_locations is None:
            num_locations = int(np.sqrt(2 ** min(dimension, 20)))
        
        self.sdm = create_sdm(
            dimension=dimension,
            num_locations=num_locations
        )
        
        self.test_patterns = []
        self.test_data = []
        
    def store_test_patterns(self, num_patterns: int, sparsity: float = 0.5):
        """Store test patterns for noise analysis."""
        self.test_patterns, self.test_data = generate_random_patterns(
            num_patterns, self.dimension, sparsity=sparsity
        )
        
        for addr, data in zip(self.test_patterns, self.test_data):
            self.sdm.store(addr, data)
        
        print(f"Stored {num_patterns} test patterns with {sparsity:.0%} sparsity")
    
    def test_noise_type(self, noise_type: str, noise_levels: List[float],
                       samples_per_level: int = 50) -> List[NoiseTestResult]:
        """
        Test tolerance to specific noise type.
        
        Parameters
        ----------
        noise_type : str
            Type of noise: 'flip', 'swap', 'burst', 'salt_pepper'
        noise_levels : list
            Noise levels to test
        samples_per_level : int
            Samples per noise level
        
        Returns
        -------
        list
            List of NoiseTestResult
        """
        results = []
        
        for noise_level in noise_levels:
            accuracies = []
            bit_errors = []
            successes = []
            confidences = []
            
            # Test each pattern multiple times
            for i in range(min(samples_per_level, len(self.test_patterns))):
                pattern_idx = i % len(self.test_patterns)
                original_addr = self.test_patterns[pattern_idx]
                original_data = self.test_data[pattern_idx]
                
                # Add noise
                noisy_addr = add_noise(original_addr, noise_level, noise_type)
                
                # Recall with confidence
                recalled_data, confidence = self.sdm.recall_with_confidence(noisy_addr)
                
                if recalled_data is not None:
                    # Calculate accuracy
                    accuracy = np.mean(recalled_data == original_data)
                    accuracies.append(accuracy)
                    
                    # Calculate bit error rate
                    bit_error = 1 - accuracy
                    bit_errors.append(bit_error)
                    
                    successes.append(1)
                    confidences.append(np.mean(confidence))
                else:
                    successes.append(0)
            
            # Aggregate results
            result = NoiseTestResult(
                noise_type=noise_type,
                noise_level=noise_level,
                accuracy=np.mean(accuracies) if accuracies else 0,
                bit_error_rate=np.mean(bit_errors) if bit_errors else 1,
                recall_success_rate=np.mean(successes),
                confidence=np.mean(confidences) if confidences else 0
            )
            results.append(result)
            
            print(f"  {noise_type} noise {noise_level:.1%}: "
                  f"accuracy={result.accuracy:.3f}, "
                  f"success_rate={result.recall_success_rate:.3f}")
        
        return results
    
    def compare_noise_types(self, noise_levels: List[float] = None) -> Dict[str, List[NoiseTestResult]]:
        """Compare tolerance to different noise types."""
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        
        noise_types = ['flip', 'swap', 'burst', 'salt_pepper']
        all_results = {}
        
        print("\nComparing noise types:")
        for noise_type in noise_types:
            print(f"\nTesting {noise_type} noise:")
            results = self.test_noise_type(noise_type, noise_levels)
            all_results[noise_type] = results
        
        return all_results
    
    def visualize_noise_tolerance(self, results: Dict[str, List[NoiseTestResult]]):
        """Visualize noise tolerance results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Accuracy vs noise level for all types
        ax = axes[0, 0]
        for noise_type, type_results in results.items():
            levels = [r.noise_level for r in type_results]
            accuracies = [r.accuracy for r in type_results]
            ax.plot(levels, accuracies, 'o-', label=noise_type, linewidth=2)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Recall Accuracy')
        ax.set_title('Accuracy vs Noise Level by Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Plot 2: Success rate vs noise level
        ax = axes[0, 1]
        for noise_type, type_results in results.items():
            levels = [r.noise_level for r in type_results]
            success_rates = [r.recall_success_rate for r in type_results]
            ax.plot(levels, success_rates, 'o-', label=noise_type, linewidth=2)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Recall Success Rate')
        ax.set_title('Success Rate vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Plot 3: Bit error rate
        ax = axes[1, 0]
        for noise_type, type_results in results.items():
            levels = [r.noise_level for r in type_results]
            bit_errors = [r.bit_error_rate for r in type_results]
            ax.plot(levels, bit_errors, 'o-', label=noise_type, linewidth=2)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Bit Error Rate')
        ax.set_title('Bit Error Rate vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Plot 4: Confidence scores
        ax = axes[1, 1]
        for noise_type, type_results in results.items():
            levels = [r.noise_level for r in type_results]
            confidences = [r.confidence for r in type_results]
            ax.plot(levels, confidences, 'o-', label=noise_type, linewidth=2)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Recall Confidence vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.suptitle('SDM Noise Tolerance Analysis', y=1.02, fontsize=14)
        plt.show()


class IterativeDenoising:
    """Iterative denoising using SDM's associative properties."""
    
    def __init__(self, sdm: SDM):
        self.sdm = sdm
        
    def denoise_pattern(self, noisy_pattern: np.ndarray, 
                       max_iterations: int = 10,
                       convergence_threshold: float = 0.001) -> Tuple[np.ndarray, List[float]]:
        """
        Iteratively denoise a pattern.
        
        Parameters
        ----------
        noisy_pattern : np.ndarray
            Noisy input pattern
        max_iterations : int
            Maximum denoising iterations
        convergence_threshold : float
            Convergence threshold
        
        Returns
        -------
        denoised : np.ndarray
            Denoised pattern
        similarities : list
            Similarity scores per iteration
        """
        current = noisy_pattern.copy()
        previous = None
        similarities = []
        
        for iteration in range(max_iterations):
            # Recall from SDM
            recalled = self.sdm.recall(current)
            
            if recalled is None:
                break
            
            # Check convergence
            if previous is not None:
                similarity = calculate_pattern_similarity(recalled, previous, metric='hamming')
                similarities.append(similarity)
                
                if similarity > (1 - convergence_threshold):
                    print(f"  Converged after {iteration + 1} iterations")
                    return recalled, similarities
            
            previous = current.copy()
            current = recalled
        
        return current, similarities
    
    def analyze_denoising_performance(self, test_patterns: List[np.ndarray],
                                    noise_levels: List[float],
                                    noise_type: str = 'flip') -> Dict:
        """Analyze iterative denoising performance."""
        results = {
            'noise_levels': noise_levels,
            'improvements': [],
            'iterations_needed': [],
            'final_accuracies': []
        }
        
        for noise_level in noise_levels:
            improvements = []
            iterations = []
            final_accs = []
            
            for original in test_patterns[:20]:  # Test subset
                # Add noise
                noisy = add_noise(original, noise_level, noise_type)
                initial_accuracy = calculate_pattern_similarity(noisy, original, metric='hamming')
                
                # Denoise
                denoised, _ = self.denoise_pattern(noisy)
                final_accuracy = calculate_pattern_similarity(denoised, original, metric='hamming')
                
                improvement = final_accuracy - initial_accuracy
                improvements.append(improvement)
                final_accs.append(final_accuracy)
                
                # Count iterations (simplified)
                iterations.append(len(_) if _ else 0)
            
            results['improvements'].append(np.mean(improvements))
            results['iterations_needed'].append(np.mean(iterations))
            results['final_accuracies'].append(np.mean(final_accs))
        
        return results


def demonstrate_basic_noise_tolerance():
    """Demonstrate basic noise tolerance capabilities."""
    print("\n" + "="*60)
    print("BASIC NOISE TOLERANCE DEMONSTRATION")
    print("="*60)
    
    # Create noise analyzer
    analyzer = NoiseTolerance(dimension=1000)
    
    # Store patterns
    analyzer.store_test_patterns(num_patterns=100, sparsity=0.5)
    
    # Test single noise type in detail
    print("\n1. Detailed analysis of flip noise:")
    
    flip_levels = np.arange(0, 0.5, 0.02)
    flip_results = analyzer.test_noise_type('flip', flip_levels, samples_per_level=30)
    
    # Find critical noise level (where accuracy drops below 90%)
    critical_level = None
    for result in flip_results:
        if result.accuracy < 0.9:
            critical_level = result.noise_level
            break
    
    print(f"\n  Critical noise level (90% accuracy): {critical_level:.1%}")
    
    # Visualize single noise type
    plt.figure(figsize=(10, 6))
    
    levels = [r.noise_level for r in flip_results]
    accuracies = [r.accuracy for r in flip_results]
    success_rates = [r.recall_success_rate for r in flip_results]
    
    plt.plot(levels, accuracies, 'b-o', label='Accuracy', linewidth=2)
    plt.plot(levels, success_rates, 'r-s', label='Success Rate', linewidth=2)
    
    if critical_level:
        plt.axvline(critical_level, color='gray', linestyle='--', 
                   label=f'Critical Level ({critical_level:.1%})')
    
    plt.axhline(0.9, color='green', linestyle=':', alpha=0.5)
    plt.xlabel('Noise Level (Flip Probability)')
    plt.ylabel('Performance')
    plt.title('SDM Performance vs Flip Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
    
    # Test all noise types
    print("\n2. Comparing different noise types:")
    all_results = analyzer.compare_noise_types()
    analyzer.visualize_noise_tolerance(all_results)


def demonstrate_storage_method_comparison():
    """Compare noise tolerance of different storage methods."""
    print("\n" + "="*60)
    print("STORAGE METHOD NOISE TOLERANCE COMPARISON")
    print("="*60)
    
    dimension = 1000
    num_patterns = 100
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    # Create SDMs with different storage methods
    counter_config = SDMConfig(
        dimension=dimension,
        num_hard_locations=1000,
        activation_radius=451,
        storage_method='counters'
    )
    counter_sdm = SDM(counter_config)
    
    binary_config = SDMConfig(
        dimension=dimension,
        num_hard_locations=1000,
        activation_radius=451,
        storage_method='binary'
    )
    binary_sdm = SDM(binary_config)
    
    # Generate and store patterns
    addresses, data = generate_random_patterns(num_patterns, dimension)
    
    for addr, dat in zip(addresses, data):
        counter_sdm.store(addr, dat)
        binary_sdm.store(addr, dat)
    
    # Test noise tolerance
    print("\n1. Testing counter-based storage:")
    counter_results = []
    
    for noise_level in noise_levels:
        accuracies = []
        
        for i in range(50):
            idx = i % num_patterns
            noisy_addr = add_noise(addresses[idx], noise_level)
            recalled = counter_sdm.recall(noisy_addr)
            
            if recalled is not None:
                accuracy = np.mean(recalled == data[idx])
                accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        counter_results.append(avg_accuracy)
        print(f"  Noise {noise_level:.1%}: {avg_accuracy:.3f}")
    
    print("\n2. Testing binary storage:")
    binary_results = []
    
    for noise_level in noise_levels:
        accuracies = []
        
        for i in range(50):
            idx = i % num_patterns
            noisy_addr = add_noise(addresses[idx], noise_level)
            recalled = binary_sdm.recall(noisy_addr)
            
            if recalled is not None:
                accuracy = np.mean(recalled == data[idx])
                accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        binary_results.append(avg_accuracy)
        print(f"  Noise {noise_level:.1%}: {avg_accuracy:.3f}")
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    
    plt.plot(noise_levels, counter_results, 'b-o', label='Counter Storage', 
             linewidth=2, markersize=8)
    plt.plot(noise_levels, binary_results, 'r-s', label='Binary Storage', 
             linewidth=2, markersize=8)
    
    plt.xlabel('Noise Level')
    plt.ylabel('Recall Accuracy')
    plt.title('Noise Tolerance: Counter vs Binary Storage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Add annotations for key differences
    max_diff_idx = np.argmax(np.abs(np.array(counter_results) - np.array(binary_results)))
    max_diff_noise = noise_levels[max_diff_idx]
    plt.axvline(max_diff_noise, color='gray', linestyle=':', alpha=0.5)
    plt.text(max_diff_noise, 0.5, f'Max difference\nat {max_diff_noise:.1%}', 
             rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()


def demonstrate_iterative_denoising():
    """Demonstrate iterative denoising techniques."""
    print("\n" + "="*60)
    print("ITERATIVE DENOISING DEMONSTRATION")
    print("="*60)
    
    # Create SDM and store patterns
    sdm = create_sdm(dimension=1000)
    patterns, _ = generate_random_patterns(100, 1000)
    
    for pattern in patterns:
        sdm.store(pattern, pattern)  # Auto-associative memory
    
    denoiser = IterativeDenoising(sdm)
    
    # Test on single pattern with visualization
    print("\n1. Single pattern denoising example:")
    
    original = patterns[0]
    noise_level = 0.25
    noisy = add_noise(original, noise_level)
    
    # Track denoising progress
    current = noisy.copy()
    history = [current]
    
    for i in range(5):
        recalled = sdm.recall(current)
        if recalled is None:
            break
        history.append(recalled)
        current = recalled
    
    # Visualize denoising progress
    fig, axes = plt.subplots(1, len(history), figsize=(15, 3))
    
    for i, pattern in enumerate(history):
        accuracy = calculate_pattern_similarity(pattern, original, metric='hamming')
        
        # Reshape for visualization
        size = int(np.sqrt(len(pattern)))
        if size * size == len(pattern):
            img = pattern.reshape(size, size)
        else:
            img = pattern[:size*size].reshape(size, size)
        
        axes[i].imshow(img, cmap='binary')
        axes[i].set_title(f'Iter {i}\nAcc: {accuracy:.2%}')
        axes[i].axis('off')
    
    plt.suptitle(f'Iterative Denoising Progress (Initial noise: {noise_level:.0%})', 
                fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Analyze denoising performance
    print("\n2. Denoising performance analysis:")
    
    noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    results = denoiser.analyze_denoising_performance(
        patterns[:50], noise_levels, noise_type='flip'
    )
    
    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Improvement
    ax1.plot(results['noise_levels'], results['improvements'], 'g-o', linewidth=2)
    ax1.set_xlabel('Initial Noise Level')
    ax1.set_ylabel('Accuracy Improvement')
    ax1.set_title('Denoising Improvement')
    ax1.grid(True, alpha=0.3)
    
    # Iterations needed
    ax2.plot(results['noise_levels'], results['iterations_needed'], 'b-s', linewidth=2)
    ax2.set_xlabel('Initial Noise Level')
    ax2.set_ylabel('Average Iterations')
    ax2.set_title('Iterations to Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Final accuracy
    ax3.plot(results['noise_levels'], results['final_accuracies'], 'r-^', linewidth=2)
    ax3.set_xlabel('Initial Noise Level')
    ax3.set_ylabel('Final Accuracy')
    ax3.set_title('Final Accuracy After Denoising')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    print("\n  Results summary:")
    for i, noise in enumerate(results['noise_levels']):
        print(f"    Noise {noise:.1%}: improvement={results['improvements'][i]:.3f}, "
              f"iterations={results['iterations_needed'][i]:.1f}, "
              f"final_accuracy={results['final_accuracies'][i]:.3f}")


def demonstrate_parameter_effects():
    """Demonstrate how SDM parameters affect noise tolerance."""
    print("\n" + "="*60)
    print("PARAMETER EFFECTS ON NOISE TOLERANCE")
    print("="*60)
    
    # Test different activation radii
    print("\n1. Effect of activation radius:")
    
    dimension = 1000
    critical_distance = int(0.451 * dimension)
    radii = [
        int(critical_distance * 0.8),  # Below critical
        critical_distance,              # At critical
        int(critical_distance * 1.2)    # Above critical
    ]
    
    radius_results = {}
    
    for radius in radii:
        config = SDMConfig(
            dimension=dimension,
            num_hard_locations=1000,
            activation_radius=radius
        )
        sdm = SDM(config)
        
        # Store patterns
        addresses, data = generate_random_patterns(100, dimension)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Test noise tolerance
        noise_levels = [0.1, 0.2, 0.3]
        accuracies = []
        
        for noise_level in noise_levels:
            level_accs = []
            for i in range(30):
                idx = i % len(addresses)
                noisy = add_noise(addresses[idx], noise_level)
                recalled = sdm.recall(noisy)
                
                if recalled is not None:
                    acc = np.mean(recalled == data[idx])
                    level_accs.append(acc)
            
            accuracies.append(np.mean(level_accs) if level_accs else 0)
        
        radius_results[radius] = accuracies
        print(f"  Radius {radius}: {[f'{a:.3f}' for a in accuracies]}")
    
    # Test different pattern densities
    print("\n2. Effect of pattern sparsity:")
    
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    sparsity_results = {}
    
    for sparsity in sparsities:
        sdm = create_sdm(dimension=1000)
        
        # Store patterns with specific sparsity
        addresses, data = generate_random_patterns(100, 1000, sparsity=sparsity)
        for addr, dat in zip(addresses, data):
            sdm.store(addr, dat)
        
        # Test at fixed noise level
        noise_level = 0.2
        accuracies = []
        
        for i in range(50):
            idx = i % len(addresses)
            noisy = add_noise(addresses[idx], noise_level)
            recalled = sdm.recall(noisy)
            
            if recalled is not None:
                acc = np.mean(recalled == data[idx])
                accuracies.append(acc)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        sparsity_results[sparsity] = avg_accuracy
        print(f"  Sparsity {sparsity:.1f}: accuracy={avg_accuracy:.3f}")
    
    # Visualize parameter effects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Radius effect
    noise_labels = ['10% noise', '20% noise', '30% noise']
    x = np.arange(len(radii))
    width = 0.25
    
    for i, noise_label in enumerate(noise_labels):
        values = [radius_results[r][i] for r in radii]
        ax1.bar(x + i*width, values, width, label=noise_label)
    
    ax1.set_xlabel('Activation Radius')
    ax1.set_ylabel('Recall Accuracy')
    ax1.set_title('Noise Tolerance vs Activation Radius')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{r}\n({r/critical_distance:.1f}x critical)' for r in radii])
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Sparsity effect
    sparsity_values = list(sparsity_results.keys())
    accuracy_values = list(sparsity_results.values())
    
    ax2.plot(sparsity_values, accuracy_values, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Pattern Sparsity')
    ax2.set_ylabel('Recall Accuracy (20% noise)')
    ax2.set_title('Noise Tolerance vs Pattern Sparsity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()


def demonstrate_real_world_noise():
    """Demonstrate noise tolerance in real-world scenarios."""
    print("\n" + "="*60)
    print("REAL-WORLD NOISE SCENARIOS")
    print("="*60)
    
    # Create SDM for image-like patterns
    sdm = create_sdm(dimension=2048)  # 64x32 binary image
    encoder = PatternEncoder(2048)
    
    # Simulate storing image patterns
    print("\n1. Image corruption recovery:")
    
    # Create simple patterns (letters/shapes)
    patterns = {
        'horizontal_line': np.zeros((32, 64), dtype=np.uint8),
        'vertical_line': np.zeros((32, 64), dtype=np.uint8),
        'cross': np.zeros((32, 64), dtype=np.uint8),
        'box': np.zeros((32, 64), dtype=np.uint8)
    }
    
    # Define patterns
    patterns['horizontal_line'][16, :] = 1
    patterns['horizontal_line'][15:18, 20:44] = 1
    
    patterns['vertical_line'][:, 32] = 1
    patterns['vertical_line'][10:22, 31:34] = 1
    
    patterns['cross'] = patterns['horizontal_line'] | patterns['vertical_line']
    
    patterns['box'][10:22, 20:44] = 1
    patterns['box'][11:21, 21:43] = 0
    
    # Store patterns
    pattern_vectors = {}
    for name, pattern in patterns.items():
        vector = pattern.flatten()
        pattern_vectors[name] = vector
        sdm.store(vector, vector)
        print(f"  Stored pattern: {name}")
    
    # Test recovery from different corruptions
    print("\n2. Testing recovery from various corruptions:")
    
    test_pattern = patterns['cross'].copy()
    test_vector = test_pattern.flatten()
    
    corruptions = {
        'salt_pepper': lambda p: add_noise(p, 0.2, 'salt_pepper'),
        'burst': lambda p: add_noise(p, 0.15, 'burst'),
        'random_blocks': lambda p: corrupt_blocks(p, num_blocks=5, block_size=100)
    }
    
    fig, axes = plt.subplots(len(corruptions), 3, figsize=(10, len(corruptions)*3))
    
    for i, (corruption_name, corrupt_fn) in enumerate(corruptions.items()):
        # Apply corruption
        corrupted = corrupt_fn(test_vector)
        
        # Recall
        recovered = sdm.recall(corrupted)
        
        # Calculate metrics
        corruption_level = 1 - calculate_pattern_similarity(corrupted, test_vector, 'hamming')
        if recovered is not None:
            recovery_accuracy = calculate_pattern_similarity(recovered, test_vector, 'hamming')
        else:
            recovery_accuracy = 0
            recovered = np.zeros_like(test_vector)
        
        # Visualize
        axes[i, 0].imshow(test_vector.reshape(32, 64), cmap='binary')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(corrupted.reshape(32, 64), cmap='binary')
        axes[i, 1].set_title(f'{corruption_name}\n({corruption_level:.1%} corrupted)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(recovered.reshape(32, 64), cmap='binary')
        axes[i, 2].set_title(f'Recovered\n({recovery_accuracy:.1%} accurate)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Recovery from Different Corruption Types', y=1.02, fontsize=14)
    plt.show()
    
    # Communication channel simulation
    print("\n3. Communication channel with errors:")
    
    # Simulate sending patterns through noisy channel
    channel_error_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    message_patterns = generate_random_patterns(20, 1000)[0]
    
    # Store patterns
    message_sdm = create_sdm(dimension=1000)
    for pattern in message_patterns:
        message_sdm.store(pattern, pattern)
    
    # Test channel
    results = []
    for error_rate in channel_error_rates:
        successes = 0
        bit_errors = []
        
        for original in message_patterns:
            # Simulate channel errors
            received = add_noise(original, error_rate, 'flip')
            
            # Error correction using SDM
            corrected = message_sdm.recall(received)
            
            if corrected is not None:
                if np.array_equal(corrected, original):
                    successes += 1
                bit_error = 1 - calculate_pattern_similarity(corrected, original, 'hamming')
                bit_errors.append(bit_error)
        
        results.append({
            'channel_error': error_rate,
            'success_rate': successes / len(message_patterns),
            'avg_bit_error': np.mean(bit_errors) if bit_errors else 1.0
        })
    
    print("\n  Channel simulation results:")
    print("  " + "-"*50)
    print("  Channel Error | Success Rate | Avg Bit Error")
    print("  " + "-"*50)
    for r in results:
        print(f"  {r['channel_error']:>12.1%} | {r['success_rate']:>11.1%} | {r['avg_bit_error']:>12.3%}")


def corrupt_blocks(pattern: np.ndarray, num_blocks: int, block_size: int) -> np.ndarray:
    """Corrupt random blocks in a pattern."""
    corrupted = pattern.copy()
    
    for _ in range(num_blocks):
        start = np.random.randint(0, len(pattern) - block_size)
        corrupted[start:start + block_size] = np.random.randint(0, 2, block_size)
    
    return corrupted


def performance_vs_noise_analysis():
    """Analyze how performance metrics change with noise."""
    print("\n" + "="*60)
    print("PERFORMANCE VS NOISE ANALYSIS")
    print("="*60)
    
    # Create SDM
    sdm = create_sdm(dimension=1500, num_locations=2000)
    
    # Store patterns
    num_patterns = 200
    addresses, data = generate_random_patterns(num_patterns, 1500)
    
    for addr, dat in zip(addresses, data):
        sdm.store(addr, dat)
    
    # Test at different noise levels
    noise_levels = np.linspace(0, 0.4, 20)
    metrics = {
        'recall_time': [],
        'accuracy': [],
        'confidence': [],
        'activations': []
    }
    
    print("\nMeasuring performance metrics vs noise level...")
    
    for noise_level in noise_levels:
        times = []
        accuracies = []
        confidences = []
        activations = []
        
        # Test subset
        for i in range(50):
            idx = i % num_patterns
            noisy = add_noise(addresses[idx], noise_level)
            
            # Measure recall time
            start = time.time()
            recalled, confidence = sdm.recall_with_confidence(noisy)
            recall_time = time.time() - start
            times.append(recall_time)
            
            # Measure accuracy
            if recalled is not None:
                accuracy = np.mean(recalled == data[idx])
                accuracies.append(accuracy)
                confidences.append(np.mean(confidence))
                
                # Count activations
                activated = sdm._get_activated_locations(noisy)
                activations.append(len(activated))
        
        metrics['recall_time'].append(np.mean(times))
        metrics['accuracy'].append(np.mean(accuracies) if accuracies else 0)
        metrics['confidence'].append(np.mean(confidences) if confidences else 0)
        metrics['activations'].append(np.mean(activations) if activations else 0)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Recall time
    ax = axes[0, 0]
    ax.plot(noise_levels, np.array(metrics['recall_time']) * 1000, 'b-', linewidth=2)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Recall Time (ms)')
    ax.set_title('Recall Time vs Noise')
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(noise_levels, metrics['accuracy'], 'g-', linewidth=2)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Recall Accuracy')
    ax.set_title('Accuracy vs Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Confidence
    ax = axes[1, 0]
    ax.plot(noise_levels, metrics['confidence'], 'r-', linewidth=2)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Confidence vs Noise')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Activations
    ax = axes[1, 1]
    ax.plot(noise_levels, metrics['activations'], 'm-', linewidth=2)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Average Activations')
    ax.set_title('Location Activations vs Noise')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Performance Metrics vs Noise Level', y=1.02, fontsize=14)
    plt.show()
    
    # Find critical points
    critical_accuracy = next((n for n, a in zip(noise_levels, metrics['accuracy']) if a < 0.9), None)
    max_useful_noise = next((n for n, a in zip(noise_levels, metrics['accuracy']) if a < 0.5), None)
    
    print(f"\nCritical points:")
    print(f"  90% accuracy threshold: {critical_accuracy:.1%} noise")
    print(f"  50% accuracy threshold: {max_useful_noise:.1%} noise")
    print(f"  Activation increase: {metrics['activations'][-1]/metrics['activations'][0]:.1f}x")


def main():
    """Run all noise tolerance demonstrations."""
    print("SPARSE DISTRIBUTED MEMORY - NOISE TOLERANCE EXAMPLES")
    print("====================================================")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Run demonstrations
    demonstrate_basic_noise_tolerance()
    demonstrate_storage_method_comparison()
    demonstrate_iterative_denoising()
    demonstrate_parameter_effects()
    demonstrate_real_world_noise()
    performance_vs_noise_analysis()
    
    print("\n" + "="*60)
    print("Noise tolerance demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
