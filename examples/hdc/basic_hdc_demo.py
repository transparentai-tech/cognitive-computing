#!/usr/bin/env python3
"""
Basic HDC Demo

This example demonstrates basic operations in Hyperdimensional Computing:
- Creating hypervectors
- Binding and bundling operations
- Similarity computations
- Associative memory storage and retrieval
"""

import numpy as np
from cognitive_computing.hdc import (
    create_hdc,
    HDCConfig,
    ItemMemory,
    plot_hypervector,
    plot_similarity_matrix,
    plot_binding_operation,
)
import matplotlib.pyplot as plt


def main():
    print("=== Basic HDC Demo ===\n")
    
    # Create HDC system with bipolar vectors
    config = HDCConfig(
        dimension=10000,
        hypervector_type="bipolar"
    )
    hdc = create_hdc(config)
    print(f"Created HDC system with {config.dimension}-dimensional {config.hypervector_type} vectors\n")
    
    # Generate random hypervectors
    print("1. Generating random hypervectors...")
    fruit = hdc.generate_hypervector()
    red = hdc.generate_hypervector()
    sweet = hdc.generate_hypervector()
    
    # Check orthogonality
    sim = hdc.similarity(fruit, red)
    print(f"   Similarity between 'fruit' and 'red': {sim:.4f}")
    print(f"   (Should be close to 0 for orthogonal vectors)\n")
    
    # Binding operation: Create composite concept
    print("2. Binding operation (creating composite concepts)...")
    apple = hdc.bind(fruit, hdc.bind(red, sweet))
    print("   Created 'apple' = fruit ⊗ red ⊗ sweet")
    
    # Unbinding: Extract components
    extracted_properties = hdc.bind(apple, fruit)  # Should recover red ⊗ sweet
    similarity_to_red_sweet = hdc.similarity(extracted_properties, hdc.bind(red, sweet))
    print(f"   Unbinding apple ⊗ fruit recovers red ⊗ sweet: similarity = {similarity_to_red_sweet:.4f}\n")
    
    # Bundling operation: Create superposition
    print("3. Bundling operation (creating sets)...")
    # Create more fruit vectors
    banana = hdc.bind(fruit, hdc.generate_hypervector())  # fruit + yellow + sweet
    orange = hdc.bind(fruit, hdc.bind(hdc.generate_hypervector(), sweet))  # fruit + orange_color + sweet
    
    # Bundle to create "fruits" concept
    fruits = hdc.bundle([apple, banana, orange])
    print("   Created 'fruits' = apple + banana + orange")
    
    # Check similarity to components
    print(f"   Similarity of 'fruits' to apple: {hdc.similarity(fruits, apple):.4f}")
    print(f"   Similarity of 'fruits' to banana: {hdc.similarity(fruits, banana):.4f}")
    print(f"   Similarity of 'fruits' to fruit property: {hdc.similarity(fruits, fruit):.4f}\n")
    
    # Associative memory
    print("4. Associative memory storage and retrieval...")
    memory = ItemMemory(dimension=config.dimension)
    
    # Store items
    memory.add("apple", apple)
    memory.add("banana", banana) 
    memory.add("orange", orange)
    memory.add("fruits", fruits)
    memory.add("red", red)
    memory.add("sweet", sweet)
    
    print(f"   Stored {len(memory)} items in associative memory")
    
    # Query with noisy vector
    noise_level = 0.1
    noisy_apple = hdc.generate_hypervector()
    # Create noisy version: 90% apple + 10% noise
    noisy_apple = np.sign(0.9 * apple + 0.1 * noisy_apple).astype(np.int8)
    
    print(f"\n   Querying with noisy apple (10% noise)...")
    results = memory.query(noisy_apple, top_k=3)
    for label, similarity in results:
        print(f"   - {label}: {similarity:.4f}")
    
    # Cleanup to exact match
    cleaned, label = memory.cleanup(noisy_apple)
    print(f"\n   Cleanup result: {label} (similarity: {hdc.similarity(apple, cleaned):.4f})")
    
    # Visualization
    print("\n5. Visualization...")
    
    # Plot similarity matrix
    vectors = [fruit, red, sweet, apple, banana, orange, fruits]
    labels = ["fruit", "red", "sweet", "apple", "banana", "orange", "fruits"]
    
    fig1, ax1 = plot_similarity_matrix(vectors, labels)
    plt.tight_layout()
    
    # Plot binding operation
    fig2, axes2 = plot_binding_operation(
        fruit, red, 
        hdc.bind(fruit, red),
        operation="bind",
        segment_size=100
    )
    
    # Show plots
    plt.show()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()