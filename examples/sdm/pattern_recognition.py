#!/usr/bin/env python3
"""
Pattern Recognition with Sparse Distributed Memory

This example demonstrates how to use SDM for pattern recognition tasks,
including digit recognition, face recognition simulation, and general
pattern classification with noise tolerance.

Usage:
    python pattern_recognition.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass

# Import SDM components
from cognitive_computing.sdm import create_sdm, SDM, SDMConfig
from cognitive_computing.sdm.utils import (
    add_noise, generate_random_patterns, PatternEncoder,
    evaluate_sdm_performance, calculate_pattern_similarity
)
from cognitive_computing.sdm.visualizations import plot_recall_accuracy


@dataclass
class RecognitionResult:
    """Results from pattern recognition."""
    pattern_id: int
    confidence: float
    similarity: float
    correct: bool


class PatternRecognizer:
    """Base class for pattern recognition using SDM."""
    
    def __init__(self, dimension: int, num_patterns: int, num_locations: int = None):
        """
        Initialize pattern recognizer.
        
        Parameters
        ----------
        dimension : int
            Dimension of pattern vectors
        num_patterns : int
            Expected number of patterns to store
        num_locations : int, optional
            Number of hard locations (default: auto-calculated)
        """
        self.dimension = dimension
        self.num_patterns = num_patterns
        
        # Auto-calculate locations if not specified
        if num_locations is None:
            num_locations = max(1000, num_patterns * 10)
        
        # Create SDM
        self.sdm = create_sdm(
            dimension=dimension,
            num_locations=num_locations,
            activation_radius=int(0.451 * dimension)
        )
        
        # Pattern storage
        self.patterns = {}
        self.pattern_labels = {}
        self.training_count = 0
        
    def train(self, pattern_id: int, variations: List[np.ndarray], label: np.ndarray):
        """
        Train on multiple variations of a pattern.
        
        Parameters
        ----------
        pattern_id : int
            Unique identifier for the pattern
        variations : list
            List of pattern variations (e.g., with noise)
        label : np.ndarray
            Label vector to associate with pattern
        """
        self.patterns[pattern_id] = variations[0]  # Store base pattern
        self.pattern_labels[pattern_id] = label
        
        # Store all variations
        for variation in variations:
            self.sdm.store(variation, label)
            self.training_count += 1
        
        print(f"Trained pattern {pattern_id} with {len(variations)} variations")
    
    def recognize(self, query_pattern: np.ndarray, top_k: int = 1) -> List[RecognitionResult]:
        """
        Recognize a query pattern.
        
        Parameters
        ----------
        query_pattern : np.ndarray
            Pattern to recognize
        top_k : int
            Number of top matches to return
        
        Returns
        -------
        list
            List of RecognitionResult objects
        """
        # Recall from SDM
        recalled_label = self.sdm.recall(query_pattern)
        
        if recalled_label is None:
            return []
        
        # Compare with known labels
        results = []
        for pattern_id, label in self.pattern_labels.items():
            similarity = np.mean(recalled_label == label)
            
            # Also compute pattern similarity if possible
            if pattern_id in self.patterns:
                pattern_sim = calculate_pattern_similarity(
                    query_pattern, self.patterns[pattern_id], metric='hamming'
                )
            else:
                pattern_sim = 0.0
            
            results.append(RecognitionResult(
                pattern_id=pattern_id,
                confidence=similarity,
                similarity=pattern_sim,
                correct=False  # Will be set by caller
            ))
        
        # Sort by confidence and return top k
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]
    
    def evaluate(self, test_patterns: List[Tuple[int, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate recognition performance.
        
        Parameters
        ----------
        test_patterns : list
            List of (pattern_id, pattern) tuples
        
        Returns
        -------
        dict
            Performance metrics
        """
        correct = 0
        total = 0
        confusion_matrix = {}
        
        for true_id, pattern in test_patterns:
            results = self.recognize(pattern, top_k=1)
            
            if results:
                pred_id = results[0].pattern_id
                if pred_id == true_id:
                    correct += 1
                    results[0].correct = True
                
                # Update confusion matrix
                if true_id not in confusion_matrix:
                    confusion_matrix[true_id] = {}
                if pred_id not in confusion_matrix[true_id]:
                    confusion_matrix[true_id][pred_id] = 0
                confusion_matrix[true_id][pred_id] += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confusion_matrix': confusion_matrix
        }


class DigitRecognizer(PatternRecognizer):
    """Specialized recognizer for digit patterns."""
    
    def __init__(self, dimension: int = 1000):
        super().__init__(dimension=dimension, num_patterns=10)
        self.encoder = PatternEncoder(dimension)
    
    def create_digit_pattern(self, digit: int, style: int = 0) -> np.ndarray:
        """
        Create a pattern representing a digit.
        
        In a real application, this would encode actual digit images.
        Here we create synthetic patterns that are consistent for each digit.
        """
        # Create base pattern using digit and style as seeds
        rng = np.random.RandomState(digit * 100 + style)
        
        # Generate pattern with digit-specific characteristics
        pattern = np.zeros(self.dimension, dtype=np.uint8)
        
        # Add digit-specific features
        # Each digit has unique "regions" that are active
        num_features = 50 + digit * 5
        feature_indices = rng.choice(self.dimension, num_features, replace=False)
        pattern[feature_indices] = 1
        
        # Add style variations
        if style > 0:
            style_indices = rng.choice(self.dimension, 20, replace=False)
            pattern[style_indices] = 1 - pattern[style_indices]
        
        return pattern
    
    def train_digits(self, variations_per_digit: int = 20, noise_level: float = 0.05):
        """Train on all digits with variations."""
        print("Training digit recognizer...")
        
        for digit in range(10):
            # Create label (one-hot encoding extended to match dimension)
            label = np.zeros(self.dimension, dtype=np.uint8)
            # Use first 10 positions for one-hot encoding
            label[digit] = 1
            
            # Generate variations
            variations = []
            for i in range(variations_per_digit):
                # Create base pattern with style variation
                base = self.create_digit_pattern(digit, style=i % 3)
                
                # Add noise
                if i > 0:  # Keep first variation clean
                    noisy = add_noise(base, noise_level, noise_type='flip')
                    variations.append(noisy)
                else:
                    variations.append(base)
            
            # Train
            self.train(digit, variations, label)
        
        print(f"Training complete: {self.training_count} patterns stored")
    
    def recognize_digit(self, pattern: np.ndarray) -> Tuple[int, float]:
        """
        Recognize a digit pattern.
        
        Returns
        -------
        digit : int
            Recognized digit (0-9)
        confidence : float
            Recognition confidence
        """
        results = self.recognize(pattern, top_k=1)
        
        if results:
            return results[0].pattern_id, results[0].confidence
        else:
            return -1, 0.0
    
    def visualize_confusion_matrix(self, confusion_matrix: Dict):
        """Visualize confusion matrix."""
        # Convert to numpy array
        matrix = np.zeros((10, 10))
        for true_digit in range(10):
            if true_digit in confusion_matrix:
                for pred_digit, count in confusion_matrix[true_digit].items():
                    matrix[true_digit, pred_digit] = count
        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_norm = matrix / (row_sums + 1e-10)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix_norm, cmap='Blues', aspect='auto')
        plt.colorbar(label='Proportion')
        plt.xlabel('Predicted Digit')
        plt.ylabel('True Digit')
        plt.title('Digit Recognition Confusion Matrix')
        
        # Add text annotations
        for i in range(10):
            for j in range(10):
                if matrix[i, j] > 0:
                    plt.text(j, i, f'{int(matrix[i, j])}', 
                            ha='center', va='center',
                            color='white' if matrix_norm[i, j] > 0.5 else 'black')
        
        plt.tight_layout()
        plt.show()


class ImagePatternRecognizer(PatternRecognizer):
    """Recognizer for image-like patterns (simulated)."""
    
    def __init__(self, dimension: int = 2000, num_classes: int = 5):
        super().__init__(dimension=dimension, num_patterns=num_classes)
        self.num_classes = num_classes
        self.class_names = [f"Class_{i}" for i in range(num_classes)]
        self.encoder = PatternEncoder(dimension)
    
    def create_image_features(self, class_id: int, variation: int = 0) -> np.ndarray:
        """
        Simulate image feature extraction.
        
        In practice, this would use CNN features or other image descriptors.
        """
        # Simulate feature vector
        rng = np.random.RandomState(class_id * 1000 + variation)
        
        # Each class has a characteristic feature distribution
        if class_id == 0:  # "Circle" class
            center = 0.2 + 0.1 * variation
            spread = 0.1
        elif class_id == 1:  # "Square" class
            center = 0.5 + 0.05 * variation
            spread = 0.15
        elif class_id == 2:  # "Triangle" class
            center = 0.8 + 0.1 * variation
            spread = 0.2
        else:
            center = 0.3 + 0.2 * class_id + 0.1 * variation
            spread = 0.1
        
        # Generate features with class-specific distribution
        features = rng.normal(center, spread, 100)
        features = np.clip(features, 0, 1)
        
        # Encode to binary
        pattern = self.encoder.encode_vector(features, method='threshold')
        
        return pattern
    
    def train_image_classes(self, samples_per_class: int = 30):
        """Train on simulated image classes."""
        print("Training image pattern recognizer...")
        
        for class_id in range(self.num_classes):
            # Create label vector (extended to match dimension)
            label = np.zeros(self.dimension, dtype=np.uint8)
            # Use first num_classes positions for one-hot encoding
            label[class_id] = 1
            
            # Generate training samples
            variations = []
            for i in range(samples_per_class):
                pattern = self.create_image_features(class_id, variation=i)
                
                # Add realistic noise (blur, distortion, etc. simulation)
                if i > 0:
                    if i % 3 == 0:
                        # Simulate motion blur
                        pattern = add_noise(pattern, 0.03, noise_type='burst')
                    elif i % 3 == 1:
                        # Simulate pixel noise
                        pattern = add_noise(pattern, 0.05, noise_type='salt_pepper')
                    else:
                        # Simulate compression artifacts
                        pattern = add_noise(pattern, 0.04, noise_type='flip')
                
                variations.append(pattern)
            
            self.train(class_id, variations, label)
            print(f"  Trained {self.class_names[class_id]} with {samples_per_class} samples")
    
    def recognize_image(self, pattern: np.ndarray, show_confidence: bool = True) -> str:
        """Recognize an image pattern and return class name."""
        results = self.recognize(pattern, top_k=3)
        
        if not results:
            return "Unknown"
        
        if show_confidence:
            print("\nTop 3 predictions:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {self.class_names[result.pattern_id]}: "
                      f"{result.confidence:.2%} confidence")
        
        return self.class_names[results[0].pattern_id]


def demonstrate_digit_recognition():
    """Demonstrate digit recognition with SDM."""
    print("\n" + "="*60)
    print("DIGIT RECOGNITION DEMONSTRATION")
    print("="*60)
    
    # Create and train recognizer
    recognizer = DigitRecognizer(dimension=1500)
    recognizer.train_digits(variations_per_digit=25, noise_level=0.03)
    
    # Test with clean patterns
    print("\n1. Testing with clean patterns:")
    test_patterns = []
    for digit in range(10):
        pattern = recognizer.create_digit_pattern(digit, style=10)  # New style
        test_patterns.append((digit, pattern))
    
    results_clean = recognizer.evaluate(test_patterns)
    print(f"Clean accuracy: {results_clean['accuracy']:.2%}")
    
    # Test with noisy patterns
    print("\n2. Testing with increasing noise levels:")
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
    noise_results = []
    
    for noise in noise_levels:
        noisy_patterns = []
        for digit in range(10):
            # Create 5 test samples per digit
            for _ in range(5):
                clean = recognizer.create_digit_pattern(digit, style=11)
                noisy = add_noise(clean, noise, noise_type='flip')
                noisy_patterns.append((digit, noisy))
        
        results = recognizer.evaluate(noisy_patterns)
        noise_results.append(results['accuracy'])
        print(f"  Noise {noise:.0%}: {results['accuracy']:.2%} accuracy")
    
    # Visualize results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot([0] + noise_levels, [results_clean['accuracy']] + noise_results, 
             'o-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level')
    plt.ylabel('Recognition Accuracy')
    plt.title('Digit Recognition vs Noise')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.subplot(1, 2, 2)
    # Show example patterns
    fig_examples = plt.gca()
    for i, noise in enumerate([0, 0.1, 0.2]):
        pattern = recognizer.create_digit_pattern(5, style=0)
        if noise > 0:
            pattern = add_noise(pattern, noise)
        
        # Reshape for visualization (assuming square-ish)
        size = int(np.sqrt(len(pattern)))
        if size * size == len(pattern):
            img = pattern.reshape(size, size)
        else:
            img = pattern[:size*size].reshape(size, size)
        
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(img, cmap='binary')
        ax.set_title(f'Noise: {noise:.0%}')
        ax.axis('off')
    
    plt.suptitle('Example Digit 5 Patterns', y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Show confusion matrix for moderate noise
    print("\n3. Confusion matrix for 10% noise:")
    test_patterns_conf = []
    for digit in range(10):
        for _ in range(10):
            clean = recognizer.create_digit_pattern(digit, style=12)
            noisy = add_noise(clean, 0.1)
            test_patterns_conf.append((digit, noisy))
    
    results_conf = recognizer.evaluate(test_patterns_conf)
    recognizer.visualize_confusion_matrix(results_conf['confusion_matrix'])
    
    # Memory usage
    stats = recognizer.sdm.get_memory_stats()
    print(f"\n4. Memory usage:")
    print(f"  Patterns stored: {recognizer.training_count}")
    print(f"  Locations used: {stats['locations_used']}/{stats['num_hard_locations']}")
    print(f"  Average location usage: {stats['avg_location_usage']:.1f}")


def demonstrate_image_recognition():
    """Demonstrate image pattern recognition."""
    print("\n" + "="*60)
    print("IMAGE PATTERN RECOGNITION DEMONSTRATION")
    print("="*60)
    
    # Create and train recognizer
    recognizer = ImagePatternRecognizer(dimension=3000, num_classes=5)
    recognizer.class_names = ["Circle", "Square", "Triangle", "Star", "Cross"]
    recognizer.train_image_classes(samples_per_class=40)
    
    # Test recognition
    print("\n1. Testing pattern recognition:")
    
    # Test each class with variations
    all_results = []
    for class_id in range(recognizer.num_classes):
        class_results = []
        
        print(f"\nTesting {recognizer.class_names[class_id]} patterns:")
        for variation in range(20, 25):  # Test variations not seen in training
            pattern = recognizer.create_image_features(class_id, variation)
            
            # Add random noise
            noise_level = np.random.uniform(0, 0.15)
            noisy_pattern = add_noise(pattern, noise_level)
            
            # Recognize
            predicted = recognizer.recognize_image(noisy_pattern, show_confidence=False)
            correct = predicted == recognizer.class_names[class_id]
            class_results.append(correct)
            
            if not correct:
                print(f"  Misclassified as {predicted} (noise: {noise_level:.2%})")
        
        accuracy = np.mean(class_results)
        all_results.append(accuracy)
        print(f"  {recognizer.class_names[class_id]} accuracy: {accuracy:.2%}")
    
    overall_accuracy = np.mean(all_results)
    print(f"\nOverall accuracy: {overall_accuracy:.2%}")
    
    # Visualize class accuracies
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    x = range(len(recognizer.class_names))
    plt.bar(x, all_results)
    plt.xticks(x, recognizer.class_names)
    plt.ylabel('Recognition Accuracy')
    plt.title('Per-Class Recognition Accuracy')
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for i, v in enumerate(all_results):
        plt.text(i, v + 0.02, f'{v:.0%}', ha='center')
    
    # Test cross-class similarity
    plt.subplot(2, 1, 2)
    similarity_matrix = np.zeros((recognizer.num_classes, recognizer.num_classes))
    
    for i in range(recognizer.num_classes):
        pattern_i = recognizer.create_image_features(i, 0)
        for j in range(recognizer.num_classes):
            pattern_j = recognizer.create_image_features(j, 0)
            similarity = calculate_pattern_similarity(pattern_i, pattern_j, metric='jaccard')
            similarity_matrix[i, j] = similarity
    
    im = plt.imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im, label='Jaccard Similarity')
    plt.xticks(range(len(recognizer.class_names)), recognizer.class_names, rotation=45)
    plt.yticks(range(len(recognizer.class_names)), recognizer.class_names)
    plt.title('Inter-Class Pattern Similarity')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate real-time recognition
    print("\n2. Real-time recognition simulation:")
    print("   Simulating continuous recognition with varying noise...")
    
    recognition_times = []
    for i in range(50):
        # Random class and noise
        class_id = np.random.randint(0, recognizer.num_classes)
        noise_level = np.random.uniform(0, 0.2)
        
        # Create pattern
        pattern = recognizer.create_image_features(class_id, variation=100+i)
        noisy_pattern = add_noise(pattern, noise_level)
        
        # Time recognition
        start_time = time.time()
        predicted = recognizer.recognize_image(noisy_pattern, show_confidence=False)
        recognition_time = time.time() - start_time
        recognition_times.append(recognition_time)
        
        if i % 10 == 0:
            print(f"   Sample {i}: {recognizer.class_names[class_id]} -> {predicted} "
                  f"({recognition_time*1000:.1f}ms)")
    
    avg_time = np.mean(recognition_times) * 1000
    print(f"\n   Average recognition time: {avg_time:.2f}ms")
    print(f"   Recognition rate: {1000/avg_time:.1f} patterns/second")


def demonstrate_custom_patterns():
    """Demonstrate recognition of custom pattern types."""
    print("\n" + "="*60)
    print("CUSTOM PATTERN RECOGNITION DEMONSTRATION")
    print("="*60)
    
    # Create recognizer for abstract patterns
    class CustomPatternRecognizer(PatternRecognizer):
        def __init__(self):
            super().__init__(dimension=2000, num_patterns=4)
            self.pattern_types = {
                0: "Periodic",
                1: "Random",
                2: "Gradient",
                3: "Symmetric"
            }
        
        def create_pattern(self, pattern_type: int, variation: int = 0) -> np.ndarray:
            """Create different types of abstract patterns."""
            pattern = np.zeros(self.dimension, dtype=np.uint8)
            
            if pattern_type == 0:  # Periodic
                period = 10 + variation % 5
                pattern[::period] = 1
                pattern[1::period*2] = 1
            
            elif pattern_type == 1:  # Random
                rng = np.random.RandomState(42 + variation)
                pattern = rng.randint(0, 2, self.dimension)
            
            elif pattern_type == 2:  # Gradient
                threshold = 0.3 + 0.1 * (variation % 5)
                gradient = np.linspace(0, 1, self.dimension)
                pattern[gradient > threshold] = 1
            
            elif pattern_type == 3:  # Symmetric
                half = self.dimension // 2
                rng = np.random.RandomState(100 + variation)
                half_pattern = rng.randint(0, 2, half)
                pattern[:half] = half_pattern
                pattern[half:2*half] = half_pattern[::-1]
            
            return pattern
    
    # Create and train
    recognizer = CustomPatternRecognizer()
    
    print("Training custom pattern recognizer...")
    for pattern_type in range(4):
        # Create label (extended to match dimension)
        label = np.zeros(recognizer.dimension, dtype=np.uint8)
        # Use first 4 positions for one-hot encoding
        label[pattern_type] = 1
        
        # Create variations
        variations = []
        for v in range(20):
            base = recognizer.create_pattern(pattern_type, v)
            # Add slight noise
            if v > 0:
                noisy = add_noise(base, 0.02)
                variations.append(noisy)
            else:
                variations.append(base)
        
        recognizer.train(pattern_type, variations, label)
        print(f"  Trained '{recognizer.pattern_types[pattern_type]}' pattern")
    
    # Test recognition
    print("\nTesting pattern type recognition:")
    
    # Visualize patterns and test
    fig, axes = plt.subplots(4, 3, figsize=(10, 8))
    
    for i, pattern_type in enumerate(range(4)):
        # Original pattern
        original = recognizer.create_pattern(pattern_type, 50)
        axes[i, 0].imshow(original.reshape(-1, 1), aspect='auto', cmap='binary')
        axes[i, 0].set_title(f'{recognizer.pattern_types[pattern_type]} (Original)')
        axes[i, 0].set_ylabel(f'Pattern {i}')
        
        # Noisy version
        noisy = add_noise(original, 0.1)
        axes[i, 1].imshow(noisy.reshape(-1, 1), aspect='auto', cmap='binary')
        axes[i, 1].set_title('10% Noise')
        
        # Recognition result
        results = recognizer.recognize(noisy, top_k=4)
        
        # Bar plot of confidences
        if results:
            pattern_names = [recognizer.pattern_types[r.pattern_id] for r in results]
            confidences = [r.confidence for r in results]
            
            axes[i, 2].barh(range(len(results)), confidences)
            axes[i, 2].set_yticks(range(len(results)))
            axes[i, 2].set_yticklabels(pattern_names)
            axes[i, 2].set_xlabel('Confidence')
            axes[i, 2].set_xlim(0, 1)
            
            # Highlight correct answer
            correct_idx = next((i for i, r in enumerate(results) 
                              if r.pattern_id == pattern_type), None)
            if correct_idx is not None:
                axes[i, 2].barh(correct_idx, confidences[correct_idx], 
                               color='green', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('Custom Pattern Recognition Results', y=1.02)
    plt.show()
    
    # Test with severely corrupted patterns
    print("\nTesting with severe corruption:")
    for pattern_type in range(4):
        pattern = recognizer.create_pattern(pattern_type, 60)
        
        # Add 25% noise
        corrupted = add_noise(pattern, 0.25)
        
        results = recognizer.recognize(corrupted, top_k=1)
        if results:
            predicted = recognizer.pattern_types[results[0].pattern_id]
            actual = recognizer.pattern_types[pattern_type]
            confidence = results[0].confidence
            
            status = "✓" if predicted == actual else "✗"
            print(f"  {status} {actual} -> {predicted} ({confidence:.2%} confidence)")


def performance_analysis():
    """Analyze performance characteristics of pattern recognition."""
    print("\n" + "="*60)
    print("PATTERN RECOGNITION PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Test different configurations
    dimensions = [500, 1000, 2000, 4000]
    pattern_counts = [5, 10, 20, 50]
    
    results = []
    
    for dim in dimensions:
        for num_patterns in pattern_counts:
            print(f"\nTesting: dimension={dim}, patterns={num_patterns}")
            
            # Create recognizer
            recognizer = PatternRecognizer(
                dimension=dim,
                num_patterns=num_patterns,
                num_locations=num_patterns * 100
            )
            
            # Generate and train patterns
            for i in range(num_patterns):
                # Create random pattern class
                base = np.random.randint(0, 2, dim)
                variations = [add_noise(base, 0.03) for _ in range(20)]
                variations[0] = base  # Include original
                
                label = np.zeros(dim, dtype=np.uint8)
                # Use first num_patterns positions for one-hot encoding
                label[i] = 1
                
                recognizer.train(i, variations, label)
            
            # Test performance
            test_patterns = []
            for i in range(num_patterns):
                for _ in range(10):
                    base = np.random.randint(0, 2, dim)
                    noisy = add_noise(base, 0.1)
                    test_patterns.append((i, noisy))
            
            # Time operations
            start_time = time.time()
            eval_results = recognizer.evaluate(test_patterns)
            total_time = time.time() - start_time
            
            # Store results
            results.append({
                'dimension': dim,
                'num_patterns': num_patterns,
                'accuracy': eval_results['accuracy'],
                'total_time': total_time,
                'time_per_pattern': total_time / len(test_patterns),
                'patterns_per_second': len(test_patterns) / total_time
            })
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy vs dimension
    ax = axes[0, 0]
    for num_pat in pattern_counts:
        data = [r for r in results if r['num_patterns'] == num_pat]
        dims = [r['dimension'] for r in data]
        accs = [r['accuracy'] for r in data]
        ax.plot(dims, accs, 'o-', label=f'{num_pat} patterns')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Recognition Accuracy')
    ax.set_title('Accuracy vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speed vs dimension
    ax = axes[0, 1]
    for num_pat in pattern_counts:
        data = [r for r in results if r['num_patterns'] == num_pat]
        dims = [r['dimension'] for r in data]
        speeds = [r['patterns_per_second'] for r in data]
        ax.plot(dims, speeds, 'o-', label=f'{num_pat} patterns')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Patterns/Second')
    ax.set_title('Recognition Speed vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy vs number of patterns
    ax = axes[1, 0]
    for dim in dimensions:
        data = [r for r in results if r['dimension'] == dim]
        nums = [r['num_patterns'] for r in data]
        accs = [r['accuracy'] for r in data]
        ax.plot(nums, accs, 'o-', label=f'dim={dim}')
    ax.set_xlabel('Number of Pattern Classes')
    ax.set_ylabel('Recognition Accuracy')
    ax.set_title('Accuracy vs Pattern Classes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = "Performance Summary:\n\n"
    summary_text += "Best configurations:\n"
    
    # Find best configs
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        summary_text += f"{i+1}. Dim={r['dimension']}, Patterns={r['num_patterns']}: "
        summary_text += f"{r['accuracy']:.1%} accuracy, {r['patterns_per_second']:.0f} pat/s\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all pattern recognition demonstrations."""
    print("SPARSE DISTRIBUTED MEMORY - PATTERN RECOGNITION EXAMPLES")
    print("========================================================")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_digit_recognition()
    demonstrate_image_recognition()
    demonstrate_custom_patterns()
    performance_analysis()
    
    print("\n" + "="*60)
    print("Pattern recognition demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
