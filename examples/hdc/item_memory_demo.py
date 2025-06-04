#!/usr/bin/env python3
"""
HDC Item Memory Demo

This example demonstrates the associative item memory functionality:
- Storing and retrieving items
- Similarity-based queries
- Memory cleanup and noise tolerance
- Memory management (LRU eviction)
- Saving and loading memory
"""

import numpy as np
import tempfile
import os
from cognitive_computing.hdc import (
    HDC,
    HDCConfig,
    ItemMemory,
    CategoricalEncoder,
    RecordEncoder,
    ScalarEncoder,
    noise_hypervector,
)


def basic_memory_demo():
    """Demonstrate basic memory operations."""
    print("=== Basic Item Memory Demo ===\n")
    
    # Create HDC system
    config = HDCConfig(dimension=10000, hypervector_type="bipolar")
    hdc = HDC(config)
    
    # Create item memory
    memory = ItemMemory(dimension=config.dimension)
    
    # Generate and store items
    print("1. Storing items in memory...")
    items = {
        "apple": hdc.generate_hypervector(),
        "banana": hdc.generate_hypervector(),
        "orange": hdc.generate_hypervector(),
        "grape": hdc.generate_hypervector(),
        "watermelon": hdc.generate_hypervector(),
    }
    
    for label, vector in items.items():
        memory.add(label, vector)
        print(f"   Stored: {label}")
    
    print(f"\nMemory contains {len(memory)} items")
    
    # Query with exact match
    print("\n2. Querying with exact vectors...")
    query_result = memory.query(items["apple"])
    print(f"   Query with 'apple' vector:")
    for label, similarity in query_result:
        print(f"     {label}: {similarity:.4f}")
    
    # Query with noisy vector
    print("\n3. Querying with noisy vectors...")
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    
    for noise in noise_levels:
        noisy_apple = noise_hypervector(items["apple"], noise, "bipolar")
        results = memory.query(noisy_apple, top_k=3)
        
        print(f"\n   Noise level {noise*100:.0f}%:")
        for label, similarity in results:
            print(f"     {label}: {similarity:.4f}")
    
    # Cleanup demonstration
    print("\n4. Memory cleanup (finding closest match)...")
    very_noisy = noise_hypervector(items["banana"], 0.25, "bipolar")
    cleaned, label = memory.cleanup(very_noisy)
    
    if cleaned is not None:
        original_sim = hdc.similarity(items["banana"], very_noisy)
        cleaned_sim = hdc.similarity(items["banana"], cleaned)
        print(f"   Cleanup recovered: {label}")
        print(f"   Original similarity: {original_sim:.4f}")
        print(f"   After cleanup: {cleaned_sim:.4f}")
    
    return memory, items


def semantic_memory_demo():
    """Demonstrate semantic relationships in memory."""
    print("\n=== Semantic Memory Demo ===\n")
    
    # Create HDC system
    config = HDCConfig(dimension=10000, hypervector_type="bipolar")
    hdc = HDC(config)
    
    # Create semantic vectors using composition
    print("1. Creating semantic vectors...")
    
    # Basic properties
    fruit = hdc.generate_hypervector()
    vegetable = hdc.generate_hypervector()
    red = hdc.generate_hypervector()
    green = hdc.generate_hypervector()
    yellow = hdc.generate_hypervector()
    sweet = hdc.generate_hypervector()
    sour = hdc.generate_hypervector()
    
    # Composite items
    items = {
        "apple": hdc.bundle([fruit, hdc.bind(red, sweet)]),
        "lemon": hdc.bundle([fruit, hdc.bind(yellow, sour)]),
        "banana": hdc.bundle([fruit, hdc.bind(yellow, sweet)]),
        "tomato": hdc.bundle([hdc.bind(fruit, vegetable), red]),  # Fruit-vegetable hybrid
        "carrot": hdc.bundle([vegetable, hdc.bind(green, sweet)]),
        "broccoli": hdc.bundle([vegetable, green]),
    }
    
    # Store in memory
    memory = ItemMemory(dimension=config.dimension)
    for label, vector in items.items():
        memory.add(label, vector)
    
    print(f"   Stored {len(memory)} semantic items")
    
    # Query by properties
    print("\n2. Querying by semantic properties...")
    
    # Query for "red things"
    print("\n   Query: Things that are red")
    results = memory.query(red, top_k=3)
    for label, similarity in results:
        print(f"     {label}: {similarity:.4f}")
    
    # Query for "sweet fruits"
    print("\n   Query: Sweet fruits")
    sweet_fruit = hdc.bind(sweet, fruit)
    results = memory.query(sweet_fruit, top_k=3)
    for label, similarity in results:
        print(f"     {label}: {similarity:.4f}")
    
    # Query for "yellow things"
    print("\n   Query: Yellow things")
    results = memory.query(yellow, top_k=3)
    for label, similarity in results:
        print(f"     {label}: {similarity:.4f}")
    
    return memory


def structured_data_memory_demo():
    """Demonstrate memory with structured data."""
    print("\n=== Structured Data Memory Demo ===\n")
    
    # Create encoders for structured data
    age_encoder = ScalarEncoder(10000, 0, 100, 20, "thermometer")
    role_encoder = CategoricalEncoder(10000, ["student", "professor", "staff"])
    dept_encoder = CategoricalEncoder(10000)
    
    # Create record encoder
    record_encoder = RecordEncoder(
        dimension=10000,
        field_encoders={
            "age": age_encoder,
            "role": role_encoder,
            "department": dept_encoder
        }
    )
    
    # Create memory
    memory = ItemMemory(dimension=10000)
    
    # Store people records
    people = [
        {"id": "P001", "name": "Alice", "age": 22, "role": "student", "department": "CS"},
        {"id": "P002", "name": "Bob", "age": 45, "role": "professor", "department": "CS"},
        {"id": "P003", "name": "Carol", "age": 30, "role": "staff", "department": "Math"},
        {"id": "P004", "name": "Dave", "age": 23, "role": "student", "department": "CS"},
        {"id": "P005", "name": "Eve", "age": 50, "role": "professor", "department": "Math"},
        {"id": "P006", "name": "Frank", "age": 28, "role": "staff", "department": "CS"},
    ]
    
    print("1. Storing person records...")
    for person in people:
        vector = record_encoder.encode(person)
        memory.add(person["id"], vector)
        print(f"   Stored: {person['id']} - {person['name']} ({person['role']})")
    
    # Query by partial record
    print("\n2. Querying by partial attributes...")
    
    # Find CS students
    print("\n   Query: CS students")
    query_record = {"role": "student", "department": "CS"}
    query_vector = record_encoder.encode(query_record)
    results = memory.query(query_vector, top_k=4)
    
    for pid, similarity in results:
        person = next(p for p in people if p["id"] == pid)
        print(f"     {pid}: {person['name']} - {similarity:.4f}")
    
    # Find people around age 30
    print("\n   Query: People around age 30")
    query_record = {"age": 30}
    query_vector = record_encoder.encode(query_record)
    results = memory.query(query_vector, top_k=3)
    
    for pid, similarity in results:
        person = next(p for p in people if p["id"] == pid)
        print(f"     {pid}: {person['name']} (age {person['age']}) - {similarity:.4f}")
    
    return memory, people


def memory_management_demo():
    """Demonstrate memory management features."""
    print("\n=== Memory Management Demo ===\n")
    
    # Create memory with limited capacity
    max_items = 5
    memory = ItemMemory(
        dimension=1000,
        max_items=max_items
    )
    
    print(f"1. Memory with max capacity: {max_items} items")
    
    # Generate items
    hdc = HDC(HDCConfig(dimension=1000))
    
    # Add items beyond capacity
    print("\n2. Adding items (demonstrating LRU eviction)...")
    for i in range(8):
        label = f"item_{i}"
        vector = hdc.generate_hypervector()
        memory.add(label, vector)
        print(f"   Added {label}, memory size: {len(memory)}")
        if len(memory) == max_items:
            print(f"   (Memory at capacity)")
    
    print(f"\n   Final items in memory: {memory.labels}")
    
    # Access tracking
    print("\n3. Access tracking and statistics...")
    
    # Access some items
    memory.get("item_5")
    memory.get("item_5")
    memory.get("item_6")
    
    # Get statistics
    stats = memory.statistics()
    print("\n   Memory statistics:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    # Update existing item
    print("\n4. Updating items...")
    new_vector = hdc.generate_hypervector()
    memory.update("item_7", new_vector)
    print("   Updated item_7 with new vector")
    
    # Merge vectors
    print("\n5. Merging vectors...")
    merge_vector = hdc.generate_hypervector()
    memory.merge("item_6", merge_vector, weight=0.3)
    print("   Merged new vector into item_6 (weight=0.3)")
    
    return memory


def persistence_demo():
    """Demonstrate saving and loading memory."""
    print("\n=== Memory Persistence Demo ===\n")
    
    # Create and populate memory
    memory = ItemMemory(dimension=1000)
    hdc = HDC(HDCConfig(dimension=1000))
    
    # Add items
    items = {
        "cat": hdc.generate_hypervector(),
        "dog": hdc.generate_hypervector(),
        "bird": hdc.generate_hypervector(),
    }
    
    for label, vector in items.items():
        memory.add(label, vector)
    
    print(f"1. Created memory with {len(memory)} items")
    
    # Save to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as pickle
        pickle_path = os.path.join(tmpdir, "memory.pkl")
        memory.save(pickle_path, format="pickle")
        print(f"\n2. Saved memory to pickle: {os.path.basename(pickle_path)}")
        
        # Save as JSON (labels only)
        json_path = os.path.join(tmpdir, "memory.json")
        memory.save(json_path, format="json")
        print(f"   Saved memory to JSON: {os.path.basename(json_path)}")
        
        # Clear and reload
        memory.clear()
        print(f"\n3. Cleared memory, size: {len(memory)}")
        
        # Load from pickle
        memory.load(pickle_path, format="pickle")
        print(f"\n4. Loaded from pickle, size: {len(memory)}")
        print(f"   Items: {memory.labels}")
        
        # Verify content
        print("\n5. Verifying loaded content...")
        for label, original_vector in items.items():
            loaded_vector = memory.get(label)
            if loaded_vector is not None:
                similarity = np.dot(original_vector, loaded_vector) / len(original_vector)
                print(f"   {label}: similarity = {similarity:.4f}")
    
    return memory


def similarity_threshold_demo():
    """Demonstrate similarity threshold in queries."""
    print("\n=== Similarity Threshold Demo ===\n")
    
    # Create memory
    memory = ItemMemory(dimension=10000)
    hdc = HDC(HDCConfig(dimension=10000))
    
    # Create related items
    base = hdc.generate_hypervector()
    
    items = {
        "exact": base.copy(),
        "very_similar": noise_hypervector(base, 0.05, "bipolar"),
        "similar": noise_hypervector(base, 0.15, "bipolar"),
        "somewhat_similar": noise_hypervector(base, 0.25, "bipolar"),
        "different": hdc.generate_hypervector(),
    }
    
    # Store items
    for label, vector in items.items():
        memory.add(label, vector)
    
    print("1. Stored items with varying similarity to base vector")
    
    # Query with different thresholds
    thresholds = [0.9, 0.7, 0.5, 0.3]
    
    print("\n2. Querying with different similarity thresholds...")
    for threshold in thresholds:
        results = memory.query(base, threshold=threshold)
        print(f"\n   Threshold {threshold}:")
        if results:
            for label, similarity in results:
                print(f"     {label}: {similarity:.4f}")
        else:
            print("     No matches found")
    
    return memory


def main():
    """Run all item memory demonstrations."""
    # Basic operations
    memory1, items = basic_memory_demo()
    
    # Semantic memory
    memory2 = semantic_memory_demo()
    
    # Structured data
    memory3, people = structured_data_memory_demo()
    
    # Memory management
    memory4 = memory_management_demo()
    
    # Persistence
    memory5 = persistence_demo()
    
    # Similarity thresholds
    memory6 = similarity_threshold_demo()
    
    print("\n" + "="*50)
    print("All demonstrations complete!")
    print("="*50)


if __name__ == "__main__":
    main()