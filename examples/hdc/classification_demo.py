#!/usr/bin/env python3
"""
HDC Classification Demo

This example demonstrates HDC-based classification:
- One-shot learning
- Adaptive classification with online updates
- Ensemble methods
- Performance evaluation
"""

import numpy as np
from cognitive_computing.hdc import (
    OneShotClassifier,
    AdaptiveClassifier,
    EnsembleClassifier,
    CategoricalEncoder,
    ScalarEncoder,
    measure_classifier_performance,
    plot_classifier_performance,
)
import matplotlib.pyplot as plt


def generate_synthetic_data():
    """Generate synthetic data for classification."""
    np.random.seed(42)
    
    # Class 1: Low values with noise
    class1_data = np.random.normal(2.0, 0.5, 50)
    class1_labels = ["low"] * 50
    
    # Class 2: High values with noise
    class2_data = np.random.normal(8.0, 0.5, 50)
    class2_labels = ["high"] * 50
    
    # Combine
    X = np.concatenate([class1_data, class2_data])
    y = class1_labels + class2_labels
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = [y[i] for i in indices]
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test


def one_shot_classification_demo():
    """Demonstrate one-shot classification."""
    print("=== One-Shot Classification Demo ===\n")
    
    # Create encoder for continuous values
    # Using thermometer encoding with binary vectors for better one-shot learning
    encoder = ScalarEncoder(
        dimension=5000,  # Larger dimension for better separation
        min_value=0.0,
        max_value=10.0,
        n_levels=100,  # More levels for finer granularity
        method="thermometer",
        hypervector_type="binary"
    )
    
    # Create classifier
    classifier = OneShotClassifier(
        dimension=5000,
        encoder=encoder,
        similarity_threshold=0.4  # Threshold for binary vectors
    )
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # Train with just a few examples per class
    # One-shot learning: use only first occurrence of each class
    seen_classes = set()
    X_oneshot = []
    y_oneshot = []
    
    for x, y in zip(X_train, y_train):
        if y not in seen_classes:
            X_oneshot.append(x)
            y_oneshot.append(y)
            seen_classes.add(y)
            
    print(f"Training with {len(X_oneshot)} examples (one per class)")
    classifier.train(X_oneshot, y_oneshot)
    
    # Test initial performance (will be poor with just one example)
    initial_train_score = classifier.score(X_train[:10], y_train[:10])
    initial_test_score = classifier.score(X_test, y_test)
    
    print(f"Initial train accuracy (10 samples): {initial_train_score:.3f}")
    print(f"Initial test accuracy: {initial_test_score:.3f}\n")
    
    # Add more examples incrementally
    print("Adding more training examples...")
    for x, y in zip(X_train[2:10], y_train[2:10]):
        classifier.add_example(x, y)
        
    # Re-evaluate
    new_test_score = classifier.score(X_test, y_test)
    print(f"Test accuracy after adding examples: {new_test_score:.3f}\n")
    
    return classifier, X_test, y_test


def adaptive_classification_demo():
    """Demonstrate adaptive classification."""
    print("=== Adaptive Classification Demo ===\n")
    
    # Create encoder
    encoder = ScalarEncoder(
        dimension=1000,
        min_value=0.0,
        max_value=10.0,
        n_levels=20,
        method="level"
    )
    
    # Create adaptive classifier
    classifier = AdaptiveClassifier(
        dimension=1000,
        encoder=encoder,
        learning_rate=0.1,
        momentum=0.9
    )
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # Initial training
    print("Initial training...")
    classifier.train(X_train[:20], y_train[:20])
    initial_score = classifier.score(X_test, y_test)
    print(f"Initial test accuracy: {initial_score:.3f}")
    
    # Online updates
    print("\nPerforming online updates...")
    for i in range(20, len(X_train)):
        # Get prediction
        pred = classifier.predict([X_train[i]])[0]
        
        # Update with true label
        classifier.update(X_train[i], y_train[i], pred)
        
        if (i - 20) % 10 == 0:
            current_score = classifier.score(X_test, y_test)
            print(f"  After {i-20+1} updates: accuracy = {current_score:.3f}")
    
    final_score = classifier.score(X_test, y_test)
    print(f"\nFinal test accuracy: {final_score:.3f}\n")
    
    return classifier, X_test, y_test


def ensemble_classification_demo():
    """Demonstrate ensemble classification."""
    print("=== Ensemble Classification Demo ===\n")
    
    # Create multiple encoders with different parameters
    encoders = [
        ScalarEncoder(1000, 0, 10, 10, "thermometer"),
        ScalarEncoder(1000, 0, 10, 20, "thermometer"),
        ScalarEncoder(1000, 0, 10, 15, "level"),
    ]
    
    # Create base classifiers
    base_classifiers = []
    for i, encoder in enumerate(encoders):
        clf = OneShotClassifier(
            dimension=1000,
            encoder=encoder,
            similarity_threshold=0.2 + i * 0.1
        )
        base_classifiers.append(clf)
    
    # Create ensemble
    ensemble = EnsembleClassifier(base_classifiers, voting="soft")
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # Train ensemble
    print("Training ensemble...")
    ensemble.train(X_train, y_train)
    
    # Compare individual vs ensemble performance
    print("\nIndividual classifier performance:")
    for i, clf in enumerate(base_classifiers):
        score = clf.score(X_test, y_test)
        print(f"  Classifier {i+1}: {score:.3f}")
    
    ensemble_score = ensemble.score(X_test, y_test)
    print(f"\nEnsemble performance: {ensemble_score:.3f}\n")
    
    return ensemble, X_test, y_test


def visualize_results(classifiers, X_test, y_test):
    """Visualize classification results."""
    print("=== Visualizing Results ===\n")
    
    fig, axes = plt.subplots(1, len(classifiers), figsize=(15, 5))
    
    for i, (name, clf) in enumerate(classifiers):
        # Measure performance
        metrics = measure_classifier_performance(clf, X_test, y_test)
        
        # Create subplot
        ax = axes[i] if len(classifiers) > 1 else axes
        
        # Plot accuracy bars
        accuracy_metrics = {k: v for k, v in metrics.items() 
                          if 'accuracy' in k and isinstance(v, (int, float))}
        
        if accuracy_metrics:
            labels = list(accuracy_metrics.keys())
            values = list(accuracy_metrics.values())
            
            bars = ax.bar(labels, values, alpha=0.7)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{name} Performance')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations."""
    # One-shot learning
    oneshot_clf, X_test, y_test = one_shot_classification_demo()
    
    # Adaptive learning
    adaptive_clf, _, _ = adaptive_classification_demo()
    
    # Ensemble learning
    ensemble_clf, _, _ = ensemble_classification_demo()
    
    # Visualize all results
    classifiers = [
        ("One-Shot", oneshot_clf),
        ("Adaptive", adaptive_clf),
        ("Ensemble", ensemble_clf)
    ]
    
    visualize_results(classifiers, X_test, y_test)
    
    print("Demo complete!")


if __name__ == "__main__":
    main()