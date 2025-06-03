#!/usr/bin/env python3
"""
Sequence processing demonstration using HRR.

This example demonstrates:
1. Encoding ordered sequences
2. Position-based retrieval
3. Sequence completion and prediction
4. Working with variable-length sequences
5. Sequence similarity and matching
6. Temporal patterns and trajectories
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cognitive_computing.hrr import (
    create_hrr,
    HRR,
    SequenceEncoder,
    CleanupMemory,
    CleanupMemoryConfig,
    generate_random_vector,
    generate_unitary_vector,
    plot_similarity_matrix,
)


class SequenceProcessor:
    """A sequence processing system using HRR."""
    
    def __init__(self, dimension: int = 1024):
        """Initialize the sequence processor."""
        self.dimension = dimension
        self.hrr = create_hrr(dimension=dimension)
        self.encoder = SequenceEncoder(self.hrr)
        
        # Item memory for cleanup
        self.item_memory = CleanupMemory(
            CleanupMemoryConfig(threshold=0.3),
            dimension
        )
        
        # Storage for sequences
        self.stored_sequences: Dict[str, np.ndarray] = {}
        
    def add_item(self, name: str, vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Add an item to the vocabulary."""
        if vector is None:
            vector = generate_random_vector(self.dimension)
        self.item_memory.add_item(name, vector)
        return vector
    
    def encode_sequence(self, items: List[str], method: str = "position") -> np.ndarray:
        """Encode a sequence of items."""
        # Get vectors for items
        vectors = []
        for item in items:
            if not self.item_memory.has_item(item):
                self.add_item(item)
            vectors.append(self.item_memory.get_vector(item))
        
        # Encode the sequence
        return self.encoder.encode_sequence(vectors, method=method)
    
    def store_sequence(self, name: str, items: List[str], method: str = "position"):
        """Store a named sequence."""
        encoding = self.encode_sequence(items, method)
        self.stored_sequences[name] = encoding
        
    def retrieve_position(self, sequence: np.ndarray, position: int) -> Tuple[str, float]:
        """Retrieve item at specific position."""
        retrieved = self.encoder.decode_position(sequence, position)
        name, _, confidence = self.item_memory.cleanup(retrieved)
        return name, confidence
    
    def find_in_sequence(self, sequence: np.ndarray, item: str) -> List[Tuple[int, float]]:
        """Find positions where an item appears in sequence."""
        item_vector = self.item_memory.get_vector(item)
        positions = []
        
        # Check each position
        max_positions = 20  # Reasonable upper bound
        for pos in range(max_positions):
            retrieved = self.encoder.decode_position(sequence, pos)
            similarity = self.hrr.similarity(retrieved, item_vector)
            
            if similarity > 0.3:  # Threshold
                positions.append((pos, similarity))
                
        return positions


def demonstrate_basic_sequences():
    """Demonstrate basic sequence encoding and retrieval."""
    print("\n" + "="*60)
    print("1. BASIC SEQUENCE ENCODING")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create a simple sequence
    sequence = ["apple", "banana", "cherry", "date", "elderberry"]
    
    print(f"Original sequence: {' → '.join(sequence)}")
    
    # Encode with position method
    encoded = processor.encode_sequence(sequence, method="position")
    
    print("\nRetrieving items by position:")
    for i in range(len(sequence)):
        retrieved, confidence = processor.retrieve_position(encoded, i)
        status = "✓" if retrieved == sequence[i] else "✗"
        print(f"  Position {i}: {retrieved} (confidence: {confidence:.3f}) {status}")
    
    # Try retrieving beyond sequence length
    retrieved, confidence = processor.retrieve_position(encoded, len(sequence))
    print(f"  Position {len(sequence)}: {retrieved} (confidence: {confidence:.3f}) - beyond length")


def demonstrate_sequence_methods():
    """Compare different sequence encoding methods."""
    print("\n" + "="*60)
    print("2. SEQUENCE ENCODING METHODS")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Test sequence
    sequence = ["one", "two", "three", "four", "five"]
    
    # Test both methods
    methods = ["position", "chaining"]
    results = {}
    
    for method in methods:
        print(f"\nMethod: {method}")
        encoded = processor.encode_sequence(sequence, method=method)
        
        # Test retrieval
        correct = 0
        for i in range(len(sequence)):
            retrieved, confidence = processor.retrieve_position(encoded, i)
            if retrieved == sequence[i]:
                correct += 1
                
        accuracy = correct / len(sequence) * 100
        results[method] = accuracy
        print(f"  Retrieval accuracy: {accuracy:.1f}%")
        
        # Test noise tolerance
        noise = np.random.normal(0, 0.2, processor.dimension)
        noisy_encoded = encoded + noise
        
        correct_noisy = 0
        for i in range(len(sequence)):
            retrieved, _ = processor.retrieve_position(noisy_encoded, i)
            if retrieved == sequence[i]:
                correct_noisy += 1
                
        noise_accuracy = correct_noisy / len(sequence) * 100
        print(f"  Noisy retrieval accuracy: {noise_accuracy:.1f}%")


def demonstrate_sequence_completion():
    """Demonstrate sequence completion/prediction."""
    print("\n" + "="*60)
    print("3. SEQUENCE COMPLETION")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create pattern sequences
    patterns = [
        ["A", "B", "C", "D", "E"],
        ["A", "B", "C", "D", "F"],  # Different ending
        ["X", "Y", "Z", "W", "V"],
    ]
    
    # Store patterns
    print("Training patterns:")
    for i, pattern in enumerate(patterns):
        processor.store_sequence(f"pattern_{i}", pattern)
        print(f"  Pattern {i}: {' → '.join(pattern)}")
    
    # Create partial sequence for completion
    partial = ["A", "B", "C"]
    print(f"\nPartial sequence: {' → '.join(partial)}")
    
    # Encode partial sequence
    partial_encoded = processor.encode_sequence(partial)
    
    # Find most similar stored sequence
    print("\nFinding best matching pattern:")
    similarities = {}
    
    for name, stored in processor.stored_sequences.items():
        # Compare first few positions
        sim = 0
        for i in range(len(partial)):
            retrieved_partial = processor.encoder.decode_position(partial_encoded, i)
            retrieved_stored = processor.encoder.decode_position(stored, i)
            sim += processor.hrr.similarity(retrieved_partial, retrieved_stored)
        
        similarities[name] = sim / len(partial)
    
    # Find best match
    best_match = max(similarities.items(), key=lambda x: x[1])
    print(f"Best match: {best_match[0]} (similarity: {best_match[1]:.3f})")
    
    # Complete the sequence
    best_pattern_idx = int(best_match[0].split('_')[1])
    completion = patterns[best_pattern_idx][len(partial):]
    print(f"Predicted completion: {' → '.join(partial)} → {' → '.join(completion)}")


def demonstrate_variable_length_sequences():
    """Demonstrate handling of variable-length sequences."""
    print("\n" + "="*60)
    print("4. VARIABLE LENGTH SEQUENCES")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create sequences of different lengths
    sequences = {
        "short": ["cat", "dog"],
        "medium": ["red", "green", "blue", "yellow"],
        "long": ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    }
    
    print("Encoding sequences of different lengths:")
    
    for name, seq in sequences.items():
        # Encode sequence
        encoded = processor.encode_sequence(seq)
        processor.stored_sequences[name] = encoded
        
        print(f"\n{name} ({len(seq)} items): {' → '.join(seq)}")
        
        # Test retrieval
        retrieved_items = []
        for i in range(len(seq)):
            item, confidence = processor.retrieve_position(encoded, i)
            if confidence > 0.3:
                retrieved_items.append(item)
        
        print(f"  Retrieved: {' → '.join(retrieved_items)}")
        
        # Check for spurious retrievals beyond sequence length
        spurious, conf = processor.retrieve_position(encoded, len(seq) + 1)
        print(f"  Position {len(seq)+1} (beyond end): {spurious} (conf: {conf:.3f})")


def demonstrate_sequence_search():
    """Demonstrate searching for items in sequences."""
    print("\n" + "="*60)
    print("5. SEQUENCE SEARCH")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create a sequence with repeating elements
    sequence = ["A", "B", "C", "B", "D", "E", "B", "F"]
    encoded = processor.encode_sequence(sequence)
    
    print(f"Sequence: {' → '.join(sequence)}")
    
    # Search for items
    search_items = ["B", "D", "G"]
    
    print("\nSearching for items in sequence:")
    for item in search_items:
        if item not in ["A", "B", "C", "D", "E", "F"]:
            processor.add_item(item)  # Add if not in vocabulary
            
        positions = processor.find_in_sequence(encoded, item)
        
        if positions:
            pos_str = ", ".join([f"{p[0]} (sim: {p[1]:.3f})" for p in positions])
            print(f"  '{item}' found at positions: {pos_str}")
        else:
            print(f"  '{item}' not found in sequence")


def demonstrate_temporal_patterns():
    """Demonstrate encoding of temporal patterns."""
    print("\n" + "="*60)
    print("6. TEMPORAL PATTERNS")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create temporal patterns (e.g., daily activities)
    daily_pattern = [
        "wake", "breakfast", "work", "lunch", 
        "work", "exercise", "dinner", "relax", "sleep"
    ]
    
    weekend_pattern = [
        "wake", "breakfast", "relax", "lunch",
        "shopping", "relax", "dinner", "movie", "sleep"
    ]
    
    # Encode patterns
    daily_encoded = processor.encode_sequence(daily_pattern)
    weekend_encoded = processor.encode_sequence(weekend_pattern)
    
    processor.stored_sequences["weekday"] = daily_encoded
    processor.stored_sequences["weekend"] = weekend_encoded
    
    print("Daily patterns encoded:")
    print(f"  Weekday: {' → '.join(daily_pattern[:5])}...")
    print(f"  Weekend: {' → '.join(weekend_pattern[:5])}...")
    
    # Create a partial day and classify
    partial_days = [
        ["wake", "breakfast", "work"],
        ["wake", "breakfast", "relax"],
        ["lunch", "shopping", "relax"]
    ]
    
    print("\nClassifying partial sequences:")
    
    for partial in partial_days:
        partial_encoded = processor.encode_sequence(partial)
        
        # Compare with stored patterns
        weekday_sim = 0
        weekend_sim = 0
        
        for i in range(len(partial)):
            item_vec = processor.encoder.decode_position(partial_encoded, i)
            
            weekday_item = processor.encoder.decode_position(daily_encoded, i)
            weekend_item = processor.encoder.decode_position(weekend_encoded, i)
            
            weekday_sim += processor.hrr.similarity(item_vec, weekday_item)
            weekend_sim += processor.hrr.similarity(item_vec, weekend_item)
        
        weekday_sim /= len(partial)
        weekend_sim /= len(partial)
        
        classification = "weekday" if weekday_sim > weekend_sim else "weekend"
        print(f"  {' → '.join(partial)}")
        print(f"    Weekday similarity: {weekday_sim:.3f}")
        print(f"    Weekend similarity: {weekend_sim:.3f}")
        print(f"    Classification: {classification}")


def demonstrate_sequence_transformations():
    """Demonstrate sequence transformations and operations."""
    print("\n" + "="*60)
    print("7. SEQUENCE TRANSFORMATIONS")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create base sequence
    original = ["A", "B", "C", "D", "E"]
    
    # Create transformations
    reversed_seq = list(reversed(original))
    shifted_seq = original[1:] + [original[0]]  # Rotate left
    
    print(f"Original:  {' → '.join(original)}")
    print(f"Reversed:  {' → '.join(reversed_seq)}")
    print(f"Shifted:   {' → '.join(shifted_seq)}")
    
    # Encode sequences
    orig_encoded = processor.encode_sequence(original)
    rev_encoded = processor.encode_sequence(reversed_seq)
    shift_encoded = processor.encode_sequence(shifted_seq)
    
    # Compare sequences
    print("\nSequence similarities:")
    sequences = {
        "original": orig_encoded,
        "reversed": rev_encoded,
        "shifted": shift_encoded
    }
    
    for name1, seq1 in sequences.items():
        for name2, seq2 in sequences.items():
            if name1 < name2:  # Avoid duplicates
                sim = processor.hrr.similarity(seq1, seq2)
                print(f"  {name1} vs {name2}: {sim:.3f}")


def visualize_sequence_embedding():
    """Visualize sequence embeddings in 2D."""
    print("\n" + "="*60)
    print("8. SEQUENCE EMBEDDING VISUALIZATION")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create various sequences
    sequences = {
        "counting": ["one", "two", "three", "four", "five"],
        "alphabet": ["A", "B", "C", "D", "E"],
        "colors": ["red", "green", "blue", "yellow", "orange"],
        "count_reverse": ["five", "four", "three", "two", "one"],
        "alpha_partial": ["A", "B", "C"],
        "mixed": ["one", "A", "red", "two", "B"]
    }
    
    # Encode all sequences
    encoded_sequences = {}
    for name, seq in sequences.items():
        encoded_sequences[name] = processor.encode_sequence(seq)
    
    # Create similarity matrix
    fig = plot_similarity_matrix(encoded_sequences)
    plt.title("Sequence Similarity Matrix")
    plt.tight_layout()
    plt.show()
    
    print("Sequence embedding visualization complete")


def demonstrate_sequence_animation():
    """Animate sequence processing."""
    print("\n" + "="*60)
    print("9. SEQUENCE PROCESSING ANIMATION")
    print("="*60)
    
    processor = SequenceProcessor(dimension=1024)
    
    # Create a sequence
    sequence = ["Start", "Process", "Analyze", "Complete", "End"]
    
    # This would create an animation showing how items are encoded/decoded
    # For simplicity, we'll just show the concept
    print("Animation concept:")
    print("  - Shows items being encoded into positions")
    print("  - Visualizes the binding process")
    print("  - Demonstrates retrieval from different positions")
    
    # In a real implementation, you would use matplotlib animation
    # to show the sequence processing in action


def main():
    """Run all sequence processing demonstrations."""
    print("="*60)
    print("SEQUENCE PROCESSING WITH HRR DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_basic_sequences()
    demonstrate_sequence_methods()
    demonstrate_sequence_completion()
    demonstrate_variable_length_sequences()
    demonstrate_sequence_search()
    demonstrate_temporal_patterns()
    demonstrate_sequence_transformations()
    visualize_sequence_embedding()
    demonstrate_sequence_animation()
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()