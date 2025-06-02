# SDM Examples and Use Cases

This document provides practical examples demonstrating how to use Sparse Distributed Memory for various applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Advanced Configuration](#advanced-configuration)
4. [Pattern Recognition](#pattern-recognition)
5. [Sequence Memory](#sequence-memory)
6. [Associative Memory](#associative-memory)
7. [Data Encoding Examples](#data-encoding-examples)
8. [Noise Tolerance Testing](#noise-tolerance-testing)
9. [Decoder Comparison](#decoder-comparison)
10. [Performance Optimization](#performance-optimization)
11. [Visualization Examples](#visualization-examples)
12. [Real-World Applications](#real-world-applications)

---

## Getting Started

### Installation and Import

```python
# Import necessary modules
import numpy as np
from cognitive_computing.sdm import create_sdm, SDM, SDMConfig
from cognitive_computing.sdm.utils import (
    generate_random_patterns, add_noise, PatternEncoder,
    evaluate_sdm_performance, save_sdm_state, load_sdm_state
)
from cognitive_computing.sdm.visualizations import (
    plot_memory_distribution, plot_recall_accuracy,
    visualize_memory_contents
)

# Set random seed for reproducibility
np.random.seed(42)
```

### Quick Start Example

```python
# Create a simple SDM with default parameters
sdm = create_sdm(dimension=1000)

# Generate and store a pattern
address = np.random.randint(0, 2, 1000)
data = np.random.randint(0, 2, 1000)
sdm.store(address, data)

# Recall the pattern
recalled = sdm.recall(address)
accuracy = np.mean(recalled == data)
print(f"Recall accuracy: {accuracy:.2%}")

# Test with noisy address
noisy_address = add_noise(address, noise_level=0.1)
recalled_noisy = sdm.recall(noisy_address)
noisy_accuracy = np.mean(recalled_noisy == data)
print(f"Noisy recall accuracy (10% noise): {noisy_accuracy:.2%}")
```

---

## Basic Operations

### Creating SDM with Different Configurations

```python
# Example 1: Small SDM for testing
small_sdm = create_sdm(
    dimension=256,
    num_locations=100,
    activation_radius=100
)

# Example 2: Large SDM with custom configuration
config = SDMConfig(
    dimension=5000,
    num_hard_locations=10000,
    activation_radius=2200,
    storage_method="counters",  # Use counter-based storage
    counter_bits=16,            # 16-bit counters
    saturation_value=32767,     # Max counter value
    parallel=True,              # Enable parallel processing
    num_workers=8               # Use 8 threads
)
large_sdm = SDM(config)

# Example 3: Binary storage for memory efficiency
binary_config = SDMConfig(
    dimension=2000,
    num_hard_locations=5000,
    activation_radius=900,
    storage_method="binary"     # Use binary storage
)
binary_sdm = SDM(binary_config)
```

### Storing and Recalling Multiple Patterns

```python
# Create SDM
sdm = create_sdm(dimension=1000)

# Generate multiple patterns
num_patterns = 100
addresses, data_patterns = generate_random_patterns(
    num_patterns=num_patterns,
    dimension=1000,
    sparsity=0.5,  # 50% ones
    correlation=0.0  # No correlation between address and data
)

# Store all patterns
print("Storing patterns...")
for addr, data in zip(addresses, data_patterns):
    sdm.store(addr, data)

# Test recall on all patterns
print("\nTesting recall accuracy...")
accuracies = []
for i, (addr, original_data) in enumerate(zip(addresses, data_patterns)):
    recalled = sdm.recall(addr)
    if recalled is not None:
        accuracy = np.mean(recalled == original_data)
        accuracies.append(accuracy)

print(f"Average recall accuracy: {np.mean(accuracies):.2%}")
print(f"Perfect recalls: {sum(a == 1.0 for a in accuracies)}/{len(accuracies)}")

# Check memory statistics
stats = sdm.get_memory_stats()
print(f"\nMemory Statistics:")
print(f"  Locations used: {stats['locations_used']}/{stats['num_hard_locations']}")
print(f"  Average location usage: {stats['avg_location_usage']:.1f}")
print(f"  Max location usage: {stats['max_location_usage']}")
```

### Using Confidence Scores

```python
# Store a pattern
sdm = create_sdm(dimension=1000)
address = np.random.randint(0, 2, 1000)
data = np.random.randint(0, 2, 1000)
sdm.store(address, data)

# Recall with confidence
recalled_data, confidence = sdm.recall_with_confidence(address)

# Analyze confidence
high_confidence_bits = np.sum(confidence > 0.8)
low_confidence_bits = np.sum(confidence < 0.2)

print(f"High confidence bits (>0.8): {high_confidence_bits}")
print(f"Low confidence bits (<0.2): {low_confidence_bits}")
print(f"Average confidence: {np.mean(confidence):.3f}")

# Visualize confidence
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 3))
plt.plot(confidence)
plt.xlabel('Bit Position')
plt.ylabel('Confidence')
plt.title('Bit-wise Recall Confidence')
plt.show()
```

---

## Advanced Configuration

### Comparing Storage Methods

```python
# Create two SDMs with different storage methods
dimension = 1000
num_locations = 1000
activation_radius = 451

# Counter-based storage
counter_config = SDMConfig(
    dimension=dimension,
    num_hard_locations=num_locations,
    activation_radius=activation_radius,
    storage_method="counters",
    counter_bits=8
)
counter_sdm = SDM(counter_config)

# Binary storage
binary_config = SDMConfig(
    dimension=dimension,
    num_hard_locations=num_locations,
    activation_radius=activation_radius,
    storage_method="binary"
)
binary_sdm = SDM(binary_config)

# Test both with same patterns
test_patterns = 150
addresses, data = generate_random_patterns(test_patterns, dimension)

# Store in both SDMs
for addr, dat in zip(addresses, data):
    counter_sdm.store(addr, dat)
    binary_sdm.store(addr, dat)

# Compare performance
print("Counter-based Storage:")
counter_results = evaluate_sdm_performance(counter_sdm, test_patterns=50)
print(f"  Recall accuracy: {counter_results.recall_accuracy_mean:.2%}")
print(f"  Noise tolerance (20%): {counter_results.noise_tolerance[0.2]:.2%}")

print("\nBinary Storage:")
binary_results = evaluate_sdm_performance(binary_sdm, test_patterns=50)
print(f"  Recall accuracy: {binary_results.recall_accuracy_mean:.2%}")
print(f"  Noise tolerance (20%): {binary_results.noise_tolerance[0.2]:.2%}")

# Memory usage comparison
print("\nMemory Usage:")
counter_memory = num_locations * dimension * counter_config.counter_bits / 8 / 1024 / 1024
binary_memory = num_locations * dimension / 8 / 1024 / 1024
print(f"  Counter storage: {counter_memory:.1f} MB")
print(f"  Binary storage: {binary_memory:.1f} MB")
print(f"  Savings: {(1 - binary_memory/counter_memory)*100:.1f}%")
```

### Parallel Processing Performance

```python
import time

# Test dimensions
dimension = 5000
num_locations = 10000
test_patterns = 100

# Create SDMs with and without parallel processing
sequential_config = SDMConfig(
    dimension=dimension,
    num_hard_locations=num_locations,
    activation_radius=2250,
    parallel=False
)
sequential_sdm = SDM(sequential_config)

parallel_config = SDMConfig(
    dimension=dimension,
    num_hard_locations=num_locations,
    activation_radius=2250,
    parallel=True,
    num_workers=8
)
parallel_sdm = SDM(parallel_config)

# Generate test data
addresses, data = generate_random_patterns(test_patterns, dimension)

# Test sequential processing
start_time = time.time()
for addr, dat in zip(addresses, data):
    sequential_sdm.store(addr, dat)
sequential_store_time = time.time() - start_time

start_time = time.time()
for addr in addresses:
    _ = sequential_sdm.recall(addr)
sequential_recall_time = time.time() - start_time

# Test parallel processing
start_time = time.time()
for addr, dat in zip(addresses, data):
    parallel_sdm.store(addr, dat)
parallel_store_time = time.time() - start_time

start_time = time.time()
for addr in addresses:
    _ = parallel_sdm.recall(addr)
parallel_recall_time = time.time() - start_time

# Results
print("Performance Comparison:")
print(f"Sequential - Store: {sequential_store_time:.2f}s, Recall: {sequential_recall_time:.2f}s")
print(f"Parallel   - Store: {parallel_store_time:.2f}s, Recall: {parallel_recall_time:.2f}s")
print(f"Speedup    - Store: {sequential_store_time/parallel_store_time:.2f}x, "
      f"Recall: {sequential_recall_time/parallel_recall_time:.2f}x")
```

---

## Pattern Recognition

### Handwritten Digit Recognition

```python
# Simulate handwritten digit patterns (simplified)
def create_digit_pattern(digit, dimension=1000, noise=0.0):
    """Create a binary pattern representing a digit."""
    # Use consistent seed for each digit
    rng = np.random.RandomState(digit)
    base_pattern = rng.randint(0, 2, dimension)
    
    if noise > 0:
        base_pattern = add_noise(base_pattern, noise)
    
    return base_pattern

# Create SDM for digit recognition
sdm = create_sdm(dimension=1000)

# Train with multiple variations of each digit
print("Training digit recognizer...")
digits = list(range(10))
samples_per_digit = 20
digit_labels = {}

for digit in digits:
    # Create label vector (one-hot encoding)
    label = np.zeros(10, dtype=np.uint8)
    label[digit] = 1
    digit_labels[digit] = label
    
    # Store multiple noisy variations
    for i in range(samples_per_digit):
        # Create pattern with slight noise
        pattern = create_digit_pattern(digit, noise=0.05)
        sdm.store(pattern, label)

print(f"Stored {len(digits) * samples_per_digit} training patterns")

# Test recognition with noisy inputs
print("\nTesting digit recognition...")
test_noise_levels = [0.0, 0.1, 0.2, 0.3]
results = {noise: [] for noise in test_noise_levels}

for noise_level in test_noise_levels:
    correct = 0
    total = 0
    
    for digit in digits:
        for _ in range(10):  # Test each digit 10 times
            # Create noisy test pattern
            test_pattern = create_digit_pattern(digit, noise=noise_level)
            
            # Recall label
            recalled_label = sdm.recall(test_pattern)
            
            if recalled_label is not None:
                # Find predicted digit (highest activation)
                predicted = np.argmax(recalled_label)
                if predicted == digit:
                    correct += 1
            total += 1
    
    accuracy = correct / total
    results[noise_level].append(accuracy)
    print(f"  Noise {noise_level:.1f}: {accuracy:.2%} accuracy")

# Visualize results
plt.figure(figsize=(8, 6))
noise_levels = list(results.keys())
accuracies = [results[n][0] for n in noise_levels]
plt.plot(noise_levels, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Input Noise Level')
plt.ylabel('Recognition Accuracy')
plt.title('Digit Recognition Performance vs Noise')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.show()
```

### Face Recognition System

```python
# Simulate face recognition with binary feature vectors
class FaceRecognitionSDM:
    def __init__(self, feature_dimension=2000):
        self.sdm = create_sdm(dimension=feature_dimension)
        self.feature_dimension = feature_dimension
        self.person_database = {}
        self.encoder = PatternEncoder(feature_dimension)
        
    def extract_features(self, person_id, variation=0):
        """Simulate face feature extraction."""
        # In real application, this would use actual face detection/feature extraction
        rng = np.random.RandomState(hash(person_id) + variation)
        features = rng.rand(100)  # Simulate 100 facial measurements
        # Convert to binary using encoder
        return self.encoder.encode_vector(features, method='threshold')
    
    def enroll_person(self, person_id, name, num_samples=10):
        """Enroll a person with multiple face samples."""
        # Create unique ID pattern for this person
        id_pattern = self.encoder.encode_string(person_id)
        self.person_database[person_id] = {
            'name': name,
            'id_pattern': id_pattern
        }
        
        # Store multiple face samples
        for i in range(num_samples):
            face_features = self.extract_features(person_id, variation=i)
            self.sdm.store(face_features, id_pattern)
        
        print(f"Enrolled {name} with {num_samples} face samples")
    
    def recognize_face(self, person_id, variation=20, noise_level=0.0):
        """Recognize a face from features."""
        # Extract features (with new variation to simulate different photo)
        face_features = self.extract_features(person_id, variation)
        
        # Add noise to simulate poor conditions
        if noise_level > 0:
            face_features = add_noise(face_features, noise_level)
        
        # Recall from SDM
        recalled_id = self.sdm.recall(face_features)
        
        if recalled_id is None:
            return None, 0.0
        
        # Find best matching person
        best_match = None
        best_score = -1
        
        for pid, info in self.person_database.items():
            score = np.mean(recalled_id == info['id_pattern'])
            if score > best_score:
                best_score = score
                best_match = pid
        
        # Threshold for acceptance
        if best_score > 0.7:
            return self.person_database[best_match]['name'], best_score
        else:
            return None, best_score

# Create face recognition system
face_system = FaceRecognitionSDM(feature_dimension=2000)

# Enroll people
people = [
    ('person_001', 'Alice Johnson'),
    ('person_002', 'Bob Smith'),
    ('person_003', 'Carol Williams'),
    ('person_004', 'David Brown'),
    ('person_005', 'Eve Davis')
]

print("Enrolling people...")
for person_id, name in people:
    face_system.enroll_person(person_id, name, num_samples=15)

# Test recognition
print("\nTesting face recognition...")
print("-" * 50)

# Test with clean conditions
for person_id, name in people:
    recognized_name, confidence = face_system.recognize_face(person_id, variation=25, noise_level=0.0)
    status = "✓" if recognized_name == name else "✗"
    print(f"{status} {name}: Recognized as {recognized_name} (confidence: {confidence:.2%})")

# Test with noise
print("\nTesting with 15% noise...")
print("-" * 50)
for person_id, name in people:
    recognized_name, confidence = face_system.recognize_face(person_id, variation=30, noise_level=0.15)
    status = "✓" if recognized_name == name else "✗"
    print(f"{status} {name}: Recognized as {recognized_name} (confidence: {confidence:.2%})")

# Test unknown person
print("\nTesting unknown person...")
print("-" * 50)
recognized_name, confidence = face_system.recognize_face('person_999', variation=0, noise_level=0.0)
print(f"Unknown person: Recognized as {recognized_name} (confidence: {confidence:.2%})")
```

---

## Sequence Memory

### Learning and Recalling Sequences

```python
class SequenceMemory:
    def __init__(self, dimension=1000):
        self.sdm = create_sdm(dimension=dimension)
        self.dimension = dimension
        
    def learn_sequence(self, sequence):
        """Learn a sequence of patterns."""
        print(f"Learning sequence of length {len(sequence)}...")
        
        # Store transitions: current -> next
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_item = sequence[i + 1]
            self.sdm.store(current, next_item)
        
        # Store last -> first for circular sequences
        self.sdm.store(sequence[-1], sequence[0])
    
    def recall_sequence(self, start_pattern, length):
        """Recall a sequence starting from a pattern."""
        sequence = [start_pattern]
        current = start_pattern
        
        for i in range(length - 1):
            next_pattern = self.sdm.recall(current)
            if next_pattern is None:
                print(f"Sequence broken at position {i+1}")
                break
            sequence.append(next_pattern)
            current = next_pattern
        
        return sequence
    
    def test_sequence_recall(self, original_sequence, start_idx=0):
        """Test sequence recall accuracy."""
        start_pattern = original_sequence[start_idx]
        recalled_seq = self.recall_sequence(start_pattern, len(original_sequence))
        
        # Compare sequences
        matches = 0
        for i, (orig, recalled) in enumerate(zip(original_sequence, recalled_seq)):
            if np.array_equal(orig, recalled):
                matches += 1
            else:
                # Check similarity for partial matches
                similarity = np.mean(orig == recalled)
                print(f"  Position {i}: {similarity:.2%} similar")
        
        accuracy = matches / len(original_sequence)
        return accuracy, recalled_seq

# Create sequence memory
seq_memory = SequenceMemory(dimension=500)

# Create a sequence of patterns
sequence_length = 10
sequence = [np.random.randint(0, 2, 500) for _ in range(sequence_length)]

# Learn the sequence
seq_memory.learn_sequence(sequence)

# Test perfect recall
print("\nTesting sequence recall (no noise)...")
accuracy, recalled = seq_memory.test_sequence_recall(sequence)
print(f"Sequence recall accuracy: {accuracy:.2%}")

# Test with noisy start
print("\nTesting sequence recall (10% noise in start pattern)...")
noisy_start = add_noise(sequence[0], 0.1)
recalled_seq = seq_memory.recall_sequence(noisy_start, sequence_length)

# Check if sequence recovers
print("Checking sequence recovery:")
for i in range(min(len(sequence), len(recalled_seq))):
    similarity = np.mean(sequence[i] == recalled_seq[i])
    print(f"  Position {i}: {similarity:.2%} correct")
```

### Temporal Pattern Learning

```python
# Learn temporal patterns (e.g., musical notes, words in sentences)
class TemporalPatternSDM:
    def __init__(self, dimension=1500, context_window=3):
        self.sdm = create_sdm(dimension=dimension)
        self.dimension = dimension
        self.context_window = context_window
        self.encoder = PatternEncoder(dimension // context_window)
        
    def create_context_vector(self, items):
        """Create a context vector from multiple items."""
        encoded_items = []
        for item in items:
            if isinstance(item, str):
                encoded = self.encoder.encode_string(item)
            else:
                encoded = self.encoder.encode_integer(item)
            encoded_items.append(encoded)
        
        # Concatenate encoded items
        context = np.concatenate(encoded_items)
        
        # Ensure correct dimension
        if len(context) > self.dimension:
            context = context[:self.dimension]
        elif len(context) < self.dimension:
            context = np.pad(context, (0, self.dimension - len(context)))
        
        return context
    
    def learn_patterns(self, sequences):
        """Learn multiple temporal sequences."""
        for seq_name, sequence in sequences.items():
            print(f"Learning sequence: {seq_name}")
            
            # Use sliding window
            for i in range(len(sequence) - self.context_window):
                context = sequence[i:i+self.context_window]
                next_item = sequence[i+self.context_window]
                
                context_vector = self.create_context_vector(context)
                next_vector = self.create_context_vector([next_item])
                
                self.sdm.store(context_vector, next_vector)
    
    def predict_next(self, context):
        """Predict next item given context."""
        context_vector = self.create_context_vector(context)
        prediction = self.sdm.recall(context_vector)
        return prediction

# Example: Learning musical patterns
temporal_sdm = TemporalPatternSDM(dimension=1500, context_window=3)

# Define some musical sequences (simplified as numbers)
musical_sequences = {
    'scale': [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1],
    'arpeggio': [1, 3, 5, 8, 5, 3, 1, 3, 5, 8, 5, 3],
    'pattern': [1, 1, 5, 5, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1]
}

# Learn the patterns
temporal_sdm.learn_patterns(musical_sequences)

# Test prediction
print("\nTesting temporal pattern prediction...")
test_contexts = [
    ([1, 2, 3], "Scale ascending"),
    ([8, 7, 6], "Scale descending"),
    ([1, 3, 5], "Arpeggio"),
    ([5, 5, 6], "Pattern")
]

for context, description in test_contexts:
    prediction = temporal_sdm.predict_next(context)
    print(f"{description} - Context: {context} -> Prediction made")
```

---

## Associative Memory

### Word Association Network

```python
class WordAssociationSDM:
    def __init__(self, dimension=2000):
        self.sdm = create_sdm(dimension=dimension)
        self.encoder = PatternEncoder(dimension)
        self.word_patterns = {}
        
    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word_patterns:
            pattern = self.encoder.encode_string(word, method='hash')
            self.word_patterns[word] = pattern
        return self.word_patterns[word]
    
    def associate_words(self, word1, word2, strength=1):
        """Create association between two words."""
        pattern1 = self.add_word(word1)
        pattern2 = self.add_word(word2)
        
        # Store bidirectional association
        for _ in range(strength):
            self.sdm.store(pattern1, pattern2)
            self.sdm.store(pattern2, pattern1)
    
    def associate_concept(self, words, concept_pattern=None):
        """Associate multiple words with a concept."""
        if concept_pattern is None:
            # Create a pattern that combines all words
            combined = np.zeros(self.encoder.dimension)
            for word in words:
                pattern = self.add_word(word)
                combined = np.logical_or(combined, pattern).astype(np.uint8)
            concept_pattern = combined
        
        # Associate each word with the concept
        for word in words:
            pattern = self.add_word(word)
            self.sdm.store(pattern, concept_pattern)
        
        return concept_pattern
    
    def find_associations(self, word, threshold=0.6):
        """Find words associated with the given word."""
        if word not in self.word_patterns:
            return []
        
        pattern = self.word_patterns[word]
        recalled = self.sdm.recall(pattern)
        
        if recalled is None:
            return []
        
        # Find similar patterns
        associations = []
        for other_word, other_pattern in self.word_patterns.items():
            if other_word != word:
                similarity = np.mean(recalled == other_pattern)
                if similarity > threshold:
                    associations.append((other_word, similarity))
        
        return sorted(associations, key=lambda x: x[1], reverse=True)

# Create word association network
word_sdm = WordAssociationSDM(dimension=2000)

# Build semantic network
print("Building word association network...")

# Animal concepts
word_sdm.associate_words("cat", "kitten", strength=3)
word_sdm.associate_words("cat", "meow", strength=2)
word_sdm.associate_words("cat", "fur", strength=2)
word_sdm.associate_words("cat", "pet", strength=2)
word_sdm.associate_words("dog", "puppy", strength=3)
word_sdm.associate_words("dog", "bark", strength=2)
word_sdm.associate_words("dog", "fur", strength=2)
word_sdm.associate_words("dog", "pet", strength=2)

# Color associations
word_sdm.associate_words("red", "apple", strength=2)
word_sdm.associate_words("red", "rose", strength=2)
word_sdm.associate_words("red", "blood", strength=1)
word_sdm.associate_words("green", "grass", strength=2)
word_sdm.associate_words("green", "leaf", strength=2)
word_sdm.associate_words("blue", "sky", strength=2)
word_sdm.associate_words("blue", "ocean", strength=2)

# Concept grouping
animal_concept = word_sdm.associate_concept(["cat", "dog", "bird", "fish"])
color_concept = word_sdm.associate_concept(["red", "green", "blue", "yellow"])
pet_concept = word_sdm.associate_concept(["cat", "dog", "bird"])

# Test associations
print("\nTesting word associations...")
test_words = ["cat", "red", "pet", "fur"]

for word in test_words:
    associations = word_sdm.find_associations(word, threshold=0.5)
    print(f"\n'{word}' is associated with:")
    for assoc_word, score in associations[:5]:  # Top 5 associations
        print(f"  - {assoc_word}: {score:.2%}")
```

### Semantic Similarity Network

```python
class SemanticMemory:
    def __init__(self, dimension=3000):
        self.sdm = create_sdm(dimension=dimension)
        self.encoder = PatternEncoder(dimension)
        self.concepts = {}
        
    def encode_concept(self, name, attributes):
        """Encode a concept with its attributes."""
        # Create pattern combining all attributes
        attribute_patterns = []
        for attr in attributes:
            attr_pattern = self.encoder.encode_string(attr)
            attribute_patterns.append(attr_pattern)
        
        # Combine using superposition
        concept_pattern = np.zeros(self.encoder.dimension)
        for pattern in attribute_patterns:
            concept_pattern = np.logical_or(concept_pattern, pattern).astype(np.uint8)
        
        self.concepts[name] = {
            'pattern': concept_pattern,
            'attributes': attributes
        }
        
        # Store concept -> attributes mapping
        for attr_pattern in attribute_patterns:
            self.sdm.store(concept_pattern, attr_pattern)
        
        # Store attributes -> concept mapping
        for attr_pattern in attribute_patterns:
            self.sdm.store(attr_pattern, concept_pattern)
        
        return concept_pattern
    
    def find_similar_concepts(self, concept_name, top_k=5):
        """Find concepts similar to the given one."""
        if concept_name not in self.concepts:
            return []
        
        target_pattern = self.concepts[concept_name]['pattern']
        similarities = []
        
        for name, info in self.concepts.items():
            if name != concept_name:
                similarity = np.mean(target_pattern == info['pattern'])
                similarities.append((name, similarity, info['attributes']))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def infer_attributes(self, concept_name):
        """Infer attributes of a concept using SDM recall."""
        if concept_name not in self.concepts:
            return []
        
        concept_pattern = self.concepts[concept_name]['pattern']
        recalled = self.sdm.recall(concept_pattern)
        
        if recalled is None:
            return []
        
        # Decode recalled pattern to find attributes
        # In a real system, this would involve more sophisticated decoding
        return self.concepts[concept_name]['attributes']

# Create semantic memory
semantic_mem = SemanticMemory(dimension=3000)

# Define concepts with attributes
concepts = {
    'car': ['vehicle', 'wheels', 'engine', 'transportation', 'road'],
    'bicycle': ['vehicle', 'wheels', 'pedals', 'transportation', 'eco-friendly'],
    'airplane': ['vehicle', 'wings', 'engine', 'transportation', 'sky'],
    'boat': ['vehicle', 'hull', 'water', 'transportation', 'marine'],
    'train': ['vehicle', 'rails', 'engine', 'transportation', 'tracks'],
    'apple': ['fruit', 'red', 'sweet', 'tree', 'healthy'],
    'orange': ['fruit', 'orange', 'citrus', 'tree', 'vitamin-c'],
    'carrot': ['vegetable', 'orange', 'root', 'healthy', 'vitamin-a'],
    'broccoli': ['vegetable', 'green', 'tree-like', 'healthy', 'iron']
}

# Encode all concepts
print("Building semantic memory...")
for concept, attributes in concepts.items():
    semantic_mem.encode_concept(concept, attributes)

# Find similar concepts
print("\nFinding similar concepts...")
test_concepts = ['car', 'apple']

for concept in test_concepts:
    print(f"\nConcepts similar to '{concept}':")
    similar = semantic_mem.find_similar_concepts(concept, top_k=3)
    for name, similarity, attributes in similar:
        common_attrs = set(concepts[concept]) & set(attributes)
        print(f"  - {name}: {similarity:.2%} similar")
        print(f"    Common attributes: {common_attrs}")
```

---

## Data Encoding Examples

### Encoding Different Data Types

```python
# Create encoder
encoder = PatternEncoder(dimension=1000)

# 1. Integer encoding
print("Integer Encoding Examples:")
integers = [42, -17, 1000, 0]
for num in integers:
    pattern = encoder.encode_integer(num)
    print(f"  {num}: {np.sum(pattern)} ones out of {len(pattern)} bits")

# 2. Float encoding
print("\nFloat Encoding Examples:")
floats = [3.14159, -2.71828, 0.5, 100.0]
for num in floats:
    pattern = encoder.encode_float(num, precision=16)
    density = np.mean(pattern)
    print(f"  {num}: density = {density:.3f}")

# 3. String encoding
print("\nString Encoding Examples:")
strings = ["Hello", "SDM", "Cognitive Computing", "🧠"]
for text in strings:
    pattern = encoder.encode_string(text, method='hash')
    print(f"  '{text}': {np.sum(pattern)} ones")

# 4. Vector encoding
print("\nVector Encoding Examples:")
vectors = [
    np.array([0.1, 0.5, 0.9]),
    np.random.rand(10),
    np.linspace(0, 1, 20)
]
for i, vec in enumerate(vectors):
    pattern = encoder.encode_vector(vec, method='threshold')
    print(f"  Vector {i+1} (length {len(vec)}): {np.sum(pattern)} ones")
```

### Image Encoding Example

```python
def encode_image_features(image_features, dimension=2000):
    """
    Encode image features (e.g., from a CNN) into SDM pattern.
    """
    encoder = PatternEncoder(dimension)
    
    # Normalize features
    features_norm = (image_features - np.mean(image_features)) / np.std(image_features)
    
    # Use multiple encoding methods and combine
    threshold_encoding = encoder.encode_vector(features_norm, method='threshold')
    rank_encoding = encoder.encode_vector(features_norm, method='rank')
    
    # Combine encodings (could also concatenate)
    combined = np.logical_or(threshold_encoding, rank_encoding).astype(np.uint8)
    
    return combined

# Simulate image features
num_images = 5
feature_dim = 128  # e.g., from CNN layer

print("Encoding image features...")
for i in range(num_images):
    # Simulate features (in practice, these would come from a real image)
    features = np.random.randn(feature_dim)
    pattern = encode_image_features(features)
    
    print(f"Image {i+1}: encoded to {np.sum(pattern)} active bits")
```

### Structured Data Encoding

```python
class StructuredDataEncoder:
    def __init__(self, dimension=2000):
        self.dimension = dimension
        self.encoder = PatternEncoder(dimension // 4)  # Reserve space for each field
        
    def encode_record(self, record):
        """Encode a structured record (dict) into a pattern."""
        patterns = []
        
        for key, value in record.items():
            # Encode field name
            key_pattern = self.encoder.encode_string(key)
            
            # Encode value based on type
            if isinstance(value, str):
                value_pattern = self.encoder.encode_string(value)
            elif isinstance(value, (int, float)):
                value_pattern = self.encoder.encode_float(float(value))
            elif isinstance(value, list):
                # Encode list as concatenated patterns
                list_patterns = []
                for item in value[:3]:  # Limit to first 3 items
                    if isinstance(item, str):
                        list_patterns.append(self.encoder.encode_string(item))
                value_pattern = np.logical_or.reduce(list_patterns)
            else:
                value_pattern = np.zeros(self.encoder.dimension, dtype=np.uint8)
            
            # Combine key and value
            field_pattern = np.concatenate([key_pattern[:100], value_pattern[:100]])
            patterns.append(field_pattern)
        
        # Combine all fields
        full_pattern = np.concatenate(patterns)
        
        # Ensure correct dimension
        if len(full_pattern) > self.dimension:
            full_pattern = full_pattern[:self.dimension]
        elif len(full_pattern) < self.dimension:
            full_pattern = np.pad(full_pattern, (0, self.dimension - len(full_pattern)))
        
        return full_pattern

# Example: Customer records
struct_encoder = StructuredDataEncoder(dimension=2000)

# Create and encode sample records
customers = [
    {
        'id': 1001,
        'name': 'Alice Smith',
        'age': 28,
        'interests': ['reading', 'hiking', 'photography']
    },
    {
        'id': 1002,
        'name': 'Bob Johnson',
        'age': 35,
        'interests': ['cooking', 'gaming', 'music']
    },
    {
        'id': 1003,
        'name': 'Carol Davis',
        'age': 28,  # Same age as Alice
        'interests': ['reading', 'cooking', 'travel']  # Some overlap
    }
]

# Store in SDM
customer_sdm = create_sdm(dimension=2000)

print("Storing customer records...")
for customer in customers:
    pattern = struct_encoder.encode_record(customer)
    # Use ID as address and full record as data
    id_pattern = struct_encoder.encoder.encode_integer(customer['id'])
    customer_sdm.store(id_pattern, pattern)
    print(f"  Stored customer {customer['name']}")

# Query by ID
print("\nQuerying by customer ID...")
query_id = 1002
query_pattern = struct_encoder.encoder.encode_integer(query_id)
recalled = customer_sdm.recall(query_pattern)
if recalled is not None:
    print(f"  Retrieved record for ID {query_id}")
```

---

## Noise Tolerance Testing

### Comprehensive Noise Analysis

```python
def analyze_noise_tolerance(sdm, test_patterns=50):
    """Perform comprehensive noise tolerance analysis."""
    # Generate test data
    addresses, data = generate_random_patterns(test_patterns, sdm.config.dimension)
    
    # Store patterns
    for addr, dat in zip(addresses, data):
        sdm.store(addr, dat)
    
    # Test different noise types and levels
    noise_types = ['flip', 'swap', 'burst', 'salt_pepper']
    noise_levels = np.arange(0, 0.5, 0.05)
    
    results = {noise_type: {'levels': noise_levels, 'accuracies': []} 
               for noise_type in noise_types}
    
    print("Testing noise tolerance...")
    for noise_type in noise_types:
        print(f"\n{noise_type.capitalize()} noise:")
        accuracies = []
        
        for noise_level in noise_levels:
            # Test subset of patterns
            test_size = min(20, test_patterns)
            noise_accuracies = []
            
            for i in range(test_size):
                # Add noise
                noisy_addr = add_noise(addresses[i], noise_level, noise_type)
                
                # Recall
                recalled = sdm.recall(noisy_addr)
                if recalled is not None:
                    accuracy = np.mean(recalled == data[i])
                    noise_accuracies.append(accuracy)
            
            avg_accuracy = np.mean(noise_accuracies) if noise_accuracies else 0
            accuracies.append(avg_accuracy)
            
            if noise_level in [0.0, 0.1, 0.2, 0.3]:
                print(f"  {noise_level:.1f}: {avg_accuracy:.2%}")
        
        results[noise_type]['accuracies'] = accuracies
    
    return results

# Create SDM for testing
test_sdm = create_sdm(dimension=1000)

# Run analysis
noise_results = analyze_noise_tolerance(test_sdm, test_patterns=100)

# Visualize results
plt.figure(figsize=(10, 6))
for noise_type, data in noise_results.items():
    plt.plot(data['levels'], data['accuracies'], 'o-', label=noise_type, linewidth=2)

plt.xlabel('Noise Level')
plt.ylabel('Recall Accuracy')
plt.title('SDM Noise Tolerance by Noise Type')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.show()

# Find critical noise levels (where accuracy drops below 90%)
print("\nCritical noise levels (90% accuracy threshold):")
for noise_type, data in noise_results.items():
    accuracies = np.array(data['accuracies'])
    critical_idx = np.where(accuracies < 0.9)[0]
    if len(critical_idx) > 0:
        critical_level = data['levels'][critical_idx[0]]
        print(f"  {noise_type}: {critical_level:.2f}")
    else:
        print(f"  {noise_type}: >0.45")
```

### Iterative Denoising

```python
def iterative_recall(sdm, noisy_pattern, iterations=5):
    """Use iterative recall to denoise a pattern."""
    current = noisy_pattern.copy()
    history = [current]
    
    for i in range(iterations):
        recalled = sdm.recall(current)
        if recalled is None:
            break
        history.append(recalled)
        
        # Check for convergence
        if np.array_equal(recalled, current):
            print(f"  Converged after {i+1} iterations")
            break
        
        current = recalled
    
    return history

# Test iterative denoising
print("Testing iterative denoising...")
sdm = create_sdm(dimension=1000)

# Store clean patterns
num_patterns = 50
addresses, data = generate_random_patterns(num_patterns, 1000)
for addr, dat in zip(addresses, data):
    sdm.store(addr, dat)

# Test with noisy input
test_idx = 0
original = data[test_idx]
noise_level = 0.25
noisy = add_noise(addresses[test_idx], noise_level)

print(f"\nOriginal accuracy: {np.mean(sdm.recall(addresses[test_idx]) == original):.2%}")
print(f"Noisy accuracy ({noise_level} noise): {np.mean(sdm.recall(noisy) == original):.2%}")

# Apply iterative recall
print("\nIterative denoising:")
history = iterative_recall(sdm, noisy, iterations=5)

for i, pattern in enumerate(history):
    if pattern is not None:
        accuracy = np.mean(pattern == original)
        print(f"  Iteration {i}: {accuracy:.2%} accuracy")

# Visualize convergence
if len(history) > 1:
    iterations = range(len(history))
    accuracies = [np.mean(h == original) if h is not None else 0 for h in history]
    
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Iterative Denoising Convergence')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.show()
```

---

## Decoder Comparison

### Comprehensive Decoder Evaluation

```python
from cognitive_computing.sdm.address_decoder import create_decoder

def compare_decoders(dimension=1000, num_locations=1000, test_patterns=100):
    """Compare all decoder types."""
    
    # Create SDMs with different decoders
    decoder_configs = {
        'Hamming': {'decoder_type': 'hamming'},
        'Jaccard': {'decoder_type': 'jaccard', 'min_similarity': 0.3},
        'Random': {'decoder_type': 'random', 'num_hashes': 50},
        'Adaptive': {'decoder_type': 'adaptive', 'target_activations': 100},
        'Hierarchical': {'decoder_type': 'hierarchical', 'num_levels': 3},
        'LSH': {'decoder_type': 'lsh', 'num_tables': 10}
    }
    
    results = {}
    
    for name, dec_config in decoder_configs.items():
        print(f"\nTesting {name} decoder...")
        
        # Create SDM
        config = SDMConfig(
            dimension=dimension,
            num_hard_locations=num_locations,
            activation_radius=int(0.451 * dimension)
        )
        sdm = SDM(config)
        
        # Create custom decoder
        decoder = create_decoder(
            dec_config['decoder_type'],
            config,
            sdm.hard_locations,
            **{k: v for k, v in dec_config.items() if k != 'decoder_type'}
        )
        
        # Replace default decoder
        sdm._get_activated_locations = lambda addr: decoder.decode(addr)
        
        # Generate test data
        addresses, data = generate_random_patterns(test_patterns, dimension)
        
        # Store patterns
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
                accuracy = np.mean(recalled == data[i])
                accuracies.append(accuracy)
            
            # Noisy recall
            noisy_addr = add_noise(addresses[i], 0.15)
            recalled_noisy = sdm.recall(noisy_addr)
            if recalled_noisy is not None:
                noise_accuracy = np.mean(recalled_noisy == data[i])
                noise_accuracies.append(noise_accuracy)
        
        # Collect results
        results[name] = {
            'store_time': np.mean(store_times),
            'recall_time': np.mean(recall_times),
            'accuracy': np.mean(accuracies),
            'noise_accuracy': np.mean(noise_accuracies),
            'activations': decoder.expected_activations(),
            'memory_stats': sdm.get_memory_stats()
        }
    
    return results

# Run comparison
decoder_results = compare_decoders(dimension=1000, num_locations=1000, test_patterns=100)

# Display results
print("\n" + "="*80)
print("DECODER COMPARISON RESULTS")
print("="*80)
print(f"{'Decoder':<15} {'Store(ms)':<12} {'Recall(ms)':<12} {'Accuracy':<12} {'Noise Acc':<12} {'Activations':<12}")
print("-"*80)

for name, res in decoder_results.items():
    print(f"{name:<15} {res['store_time']*1000:<12.2f} {res['recall_time']*1000:<12.2f} "
          f"{res['accuracy']:<12.2%} {res['noise_accuracy']:<12.2%} {res['activations']:<12.0f}")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Performance comparison
ax = axes[0, 0]
decoders = list(decoder_results.keys())
store_times = [r['store_time']*1000 for r in decoder_results.values()]
recall_times = [r['recall_time']*1000 for r in decoder_results.values()]

x = np.arange(len(decoders))
width = 0.35
ax.bar(x - width/2, store_times, width, label='Store')
ax.bar(x + width/2, recall_times, width, label='Recall')
ax.set_ylabel('Time (ms)')
ax.set_title('Operation Times')
ax.set_xticks(x)
ax.set_xticklabels(decoders, rotation=45)
ax.legend()

# Accuracy comparison
ax = axes[0, 1]
accuracies = [r['accuracy'] for r in decoder_results.values()]
noise_accuracies = [r['noise_accuracy'] for r in decoder_results.values()]

ax.bar(x - width/2, accuracies, width, label='Clean')
ax.bar(x + width/2, noise_accuracies, width, label='15% Noise')
ax.set_ylabel('Recall Accuracy')
ax.set_title('Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(decoders, rotation=45)
ax.set_ylim(0, 1.1)
ax.legend()

# Activation counts
ax = axes[1, 0]
activations = [r['activations'] for r in decoder_results.values()]
ax.bar(decoders, activations)
ax.set_ylabel('Expected Activations')
ax.set_title('Activation Counts')
ax.set_xticklabels(decoders, rotation=45)

# Memory utilization
ax = axes[1, 1]
utilizations = [r['memory_stats']['locations_used'] / r['memory_stats']['num_hard_locations'] 
                for r in decoder_results.values()]
ax.bar(decoders, utilizations)
ax.set_ylabel('Utilization')
ax.set_title('Memory Utilization')
ax.set_xticklabels(decoders, rotation=45)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.show()
```

---

## Performance Optimization

### Capacity Optimization

```python
from cognitive_computing.sdm.memory import MemoryOptimizer

def find_optimal_configuration(target_capacity, dimension, test_patterns=50):
    """Find optimal SDM configuration for target capacity."""
    
    print(f"Finding optimal configuration for {target_capacity} patterns in {dimension}D...")
    
    # Estimate required locations
    estimated_locations = MemoryOptimizer.estimate_required_locations(
        dimension, target_capacity
    )
    
    # Find optimal radius
    optimal_radius = MemoryOptimizer.find_optimal_radius(
        dimension, estimated_locations
    )
    
    print(f"\nEstimated configuration:")
    print(f"  Locations needed: {estimated_locations}")
    print(f"  Optimal radius: {optimal_radius}")
    print(f"  Critical distance: {int(0.451 * dimension)}")
    
    # Test the configuration
    config = SDMConfig(
        dimension=dimension,
        num_hard_locations=estimated_locations,
        activation_radius=optimal_radius
    )
    sdm = SDM(config)
    
    # Store patterns up to target capacity
    addresses, data = generate_random_patterns(target_capacity, dimension)
    
    print(f"\nTesting configuration...")
    for i, (addr, dat) in enumerate(zip(addresses, data)):
        sdm.store(addr, dat)
        
        # Check performance periodically
        if (i + 1) % (target_capacity // 5) == 0:
            # Test random subset
            test_indices = np.random.choice(i + 1, min(test_patterns, i + 1), replace=False)
            accuracies = []
            
            for idx in test_indices:
                recalled = sdm.recall(addresses[idx])
                if recalled is not None:
                    accuracies.append(np.mean(recalled == data[idx]))
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            print(f"  After {i+1} patterns: {avg_accuracy:.2%} accuracy")
    
    # Final performance test
    final_results = evaluate_sdm_performance(sdm, test_patterns=min(100, target_capacity))
    
    return {
        'config': config,
        'performance': final_results,
        'memory_stats': sdm.get_memory_stats()
    }

# Test different capacity targets
capacity_targets = [100, 500, 1000, 2000]
dimension = 2000

optimization_results = {}
for capacity in capacity_targets:
    print(f"\n{'='*60}")
    result = find_optimal_configuration(capacity, dimension)
    optimization_results[capacity] = result

# Summary
print("\n" + "="*60)
print("OPTIMIZATION SUMMARY")
print("="*60)
print(f"{'Target':<10} {'Locations':<12} {'Radius':<10} {'Accuracy':<12} {'Utilization':<12}")
print("-"*60)

for capacity, result in optimization_results.items():
    config = result['config']
    perf = result['performance']
    stats = result['memory_stats']
    
    print(f"{capacity:<10} {config.num_hard_locations:<12} {config.activation_radius:<10} "
          f"{perf.recall_accuracy_mean:<12.2%} "
          f"{stats['locations_used']/config.num_hard_locations:<12.2%}")
```

### Memory-Efficient Implementation

```python
class MemoryEfficientSDM:
    """SDM implementation with reduced memory footprint."""
    
    def __init__(self, dimension, num_locations, activation_radius,
                 use_sparse=True, compression_ratio=0.1):
        self.dimension = dimension
        self.num_locations = num_locations
        self.activation_radius = activation_radius
        self.use_sparse = use_sparse
        self.compression_ratio = compression_ratio
        
        # Generate hard locations more efficiently
        if use_sparse:
            # Store only non-zero indices for sparse addresses
            self.hard_locations = self._generate_sparse_locations()
        else:
            # Standard dense storage
            self.hard_locations = np.random.randint(0, 2, 
                                                   (num_locations, dimension),
                                                   dtype=np.uint8)
        
        # Use smaller counter type
        self.counters = np.zeros((num_locations, dimension), dtype=np.int8)
        
    def _generate_sparse_locations(self):
        """Generate sparse representation of hard locations."""
        locations = []
        for i in range(self.num_locations):
            # Each location stores indices of 1-bits
            num_ones = int(self.dimension * 0.5)  # 50% density
            ones_indices = np.random.choice(self.dimension, num_ones, replace=False)
            locations.append(ones_indices)
        return locations
    
    def _hamming_distance_sparse(self, dense_vector, sparse_indices):
        """Compute Hamming distance between dense vector and sparse representation."""
        # Count ones in dense vector
        dense_ones = np.sum(dense_vector)
        
        # Count overlapping ones
        overlap = np.sum(dense_vector[sparse_indices])
        
        # Hamming distance = (dense_ones - overlap) + (sparse_ones - overlap)
        sparse_ones = len(sparse_indices)
        return (dense_ones - overlap) + (sparse_ones - overlap)
    
    def get_memory_usage(self):
        """Calculate memory usage in MB."""
        if self.use_sparse:
            # Sparse storage (assuming 50% density)
            sparse_memory = self.num_locations * self.dimension * 0.5 * 4 / 1024 / 1024  # 4 bytes per index
        else:
            sparse_memory = self.num_locations * self.dimension / 8 / 1024 / 1024  # 1 bit per element
        
        counter_memory = self.num_locations * self.dimension * 1 / 1024 / 1024  # 1 byte per counter
        
        return {
            'locations_mb': sparse_memory,
            'counters_mb': counter_memory,
            'total_mb': sparse_memory + counter_memory
        }

# Compare memory usage
dimensions = [1000, 2000, 5000, 10000]
num_locations = 5000

print("Memory Usage Comparison:")
print(f"{'Dimension':<12} {'Standard (MB)':<15} {'Efficient (MB)':<15} {'Savings':<10}")
print("-" * 60)

for dim in dimensions:
    # Standard SDM memory
    standard_memory = (num_locations * dim * (1 + 8)) / 8 / 1024 / 1024  # 1 bit address + 8 bit counter
    
    # Efficient SDM
    eff_sdm = MemoryEfficientSDM(dim, num_locations, int(0.451 * dim))
    eff_memory = eff_sdm.get_memory_usage()['total_mb']
    
    savings = (1 - eff_memory / standard_memory) * 100
    
    print(f"{dim:<12} {standard_memory:<15.1f} {eff_memory:<15.1f} {savings:<10.1f}%")
```

---

## Visualization Examples

### Memory State Dashboard

```python
def create_sdm_dashboard(sdm, test_data=None):
    """Create comprehensive dashboard for SDM analysis."""
    from cognitive_computing.sdm.memory import MemoryStatistics
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Get memory statistics
    stats = MemoryStatistics(sdm)
    memory_maps = stats.contents.get_memory_map()
    
    # 1. Usage heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    usage = memory_maps['usage_map']
    im = ax1.imshow(usage.reshape(-1, 1), aspect='auto', cmap='hot')
    ax1.set_title('Location Usage')
    ax1.set_ylabel('Location Index')
    plt.colorbar(im, ax=ax1)
    
    # 2. Distribution analysis
    ax2 = fig.add_subplot(gs[0, 1])
    pattern_dist = stats.contents.analyze_pattern_distribution(sample_size=500)
    
    labels = ['Min', 'Mean', 'Max']
    values = [pattern_dist['min_activation_count'],
              pattern_dist['mean_activation_count'],
              pattern_dist['max_activation_count']]
    
    ax2.bar(labels, values)
    ax2.set_title('Activation Distribution')
    ax2.set_ylabel('Number of Activations')
    
    # 3. Capacity gauge
    ax3 = fig.add_subplot(gs[0, 2])
    capacity = stats.contents.get_capacity_estimate()
    
    # Create gauge chart
    used = capacity['patterns_stored']
    total = capacity['theoretical_capacity']
    
    wedges, texts = ax3.pie([used, total - used], 
                           labels=['Used', 'Available'],
                           startangle=90,
                           counterclock=False)
    ax3.set_title(f'Capacity: {used}/{total}')
    
    # 4. Noise tolerance curve
    ax4 = fig.add_subplot(gs[1, :2])
    if test_data and len(sdm._stored_addresses) > 0:
        noise_analysis = stats.analyze_recall_quality(test_size=30)
        ax4.plot(noise_analysis['noise_levels'], 
                noise_analysis['recall_accuracies'], 
                'o-', linewidth=2)
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Recall Accuracy')
        ax4.set_title('Noise Tolerance')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
    
    # 5. Memory statistics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    stats_text = f"""Configuration:
    Dimension: {sdm.config.dimension}
    Locations: {sdm.config.num_hard_locations}
    Radius: {sdm.config.activation_radius}
    Method: {sdm.config.storage_method}
    
Performance:
    Patterns: {capacity['patterns_stored']}
    Error Rate: {capacity['average_recall_error']:.3f}
    SNR: {capacity['signal_to_noise_ratio_db']:.1f} dB
    
Memory:
    Used: {capacity['location_utilization']:.1%}
    Saturation: {capacity['capacity_used_estimate']:.1%}"""
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # 6. 3D visualization of memory state
    ax6 = fig.add_subplot(gs[2, :], projection='3d')
    
    # Sample locations for 3D plot
    sample_size = min(100, sdm.config.num_hard_locations)
    sample_indices = np.random.choice(sdm.config.num_hard_locations, 
                                    sample_size, replace=False)
    
    # Use PCA for 3D projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    
    if sdm.config.storage_method == 'counters':
        data = sdm.counters[sample_indices]
    else:
        data = sdm.binary_storage[sample_indices]
    
    coords_3d = pca.fit_transform(data)
    
    # Color by usage
    colors = usage[sample_indices]
    
    scatter = ax6.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                         c=colors, cmap='viridis', s=50, alpha=0.6)
    
    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.set_zlabel('PC3')
    ax6.set_title('Memory Contents (PCA Projection)')
    
    plt.colorbar(scatter, ax=ax6, label='Usage Count')
    
    plt.suptitle('SDM Analysis Dashboard', fontsize=16)
    return fig

# Create and populate SDM
sdm = create_sdm(dimension=1000)
addresses, data = generate_random_patterns(150, 1000)
for addr, dat in zip(addresses, data):
    sdm.store(addr, dat)

# Create dashboard
dashboard = create_sdm_dashboard(sdm, test_data=(addresses, data))
plt.show()
```

### Interactive Memory Explorer

```python
from cognitive_computing.sdm.visualizations import visualize_memory_contents
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_explorer(sdm):
    """Create interactive Plotly visualization for SDM exploration."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Memory Contents (t-SNE)', 'Activation Patterns',
                       'Usage Distribution', 'Performance Metrics'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]],
        row_heights=[0.6, 0.4]
    )
    
    # 1. 3D t-SNE visualization
    memory_fig = visualize_memory_contents(sdm, method='tsne', 
                                         interactive=True, num_samples=200)
    
    # Extract data from the returned figure
    for trace in memory_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # 2. Activation pattern for random address
    test_addr = np.random.randint(0, 2, sdm.config.dimension)
    activated = sdm._get_activated_locations(test_addr)
    
    activation_pattern = np.zeros(sdm.config.num_hard_locations)
    activation_pattern[activated] = 1
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(activation_pattern))),
            y=activation_pattern,
            mode='markers',
            marker=dict(
                size=3,
                color=activation_pattern,
                colorscale='Viridis'
            ),
            name='Activated'
        ),
        row=1, col=2
    )
    
    # 3. Usage histogram
    usage = sdm.location_usage
    fig.add_trace(
        go.Histogram(
            x=usage[usage > 0],
            nbinsx=30,
            name='Usage Count'
        ),
        row=2, col=1
    )
    
    # 4. Performance over time (simulated)
    if len(sdm._stored_addresses) > 0:
        # Simulate performance degradation
        pattern_counts = list(range(10, len(sdm._stored_addresses) + 1, 10))
        accuracies = []
        
        for count in pattern_counts:
            # Test subset
            test_indices = np.random.choice(count, min(20, count), replace=False)
            accs = []
            for idx in test_indices:
                recalled = sdm.recall(sdm._stored_addresses[idx])
                if recalled is not None:
                    acc = np.mean(recalled == sdm._stored_data[idx])
                    accs.append(acc)
            accuracies.append(np.mean(accs) if accs else 0)
        
        fig.add_trace(
            go.Scatter(
                x=pattern_counts,
                y=accuracies,
                mode='lines+markers',
                name='Recall Accuracy',
                line=dict(width=2)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Interactive SDM Explorer',
        showlegend=False,
        height=800,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Location Index", row=1, col=2)
    fig.update_yaxes(title_text="Activated", row=1, col=2)
    fig.update_xaxes(title_text="Usage Count", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Patterns Stored", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)
    
    return fig

# Create interactive explorer
sdm = create_sdm(dimension=1000)
addresses, data = generate_random_patterns(100, 1000)
for addr, dat in zip(addresses, data):
    sdm.store(addr, dat)

interactive_fig = create_interactive_explorer(sdm)
interactive_fig.show()
```

---

## Real-World Applications

### Document Similarity Search

```python
class DocumentSDM:
    """SDM-based document similarity search."""
    
    def __init__(self, dimension=5000, num_docs=10000):
        self.sdm = create_sdm(dimension=dimension, num_locations=num_docs)
        self.encoder = PatternEncoder(dimension)
        self.documents = {}
        self.doc_patterns = {}
        
    def extract_features(self, text):
        """Extract features from document text."""
        # Simple feature extraction (in practice, use TF-IDF, word2vec, etc.)
        words = text.lower().split()
        
        # Use top frequent words as features
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Encode as pattern
        patterns = []
        for word, freq in top_words:
            word_pattern = self.encoder.encode_string(word)
            # Weight by frequency
            weighted = word_pattern * min(freq / 10, 1.0)
            patterns.append(weighted)
        
        # Combine patterns
        if patterns:
            combined = np.mean(patterns, axis=0)
            return (combined > 0.5).astype(np.uint8)
        else:
            return np.zeros(self.encoder.dimension, dtype=np.uint8)
    
    def add_document(self, doc_id, title, text):
        """Add document to the index."""
        # Extract features
        features = self.extract_features(text)
        
        # Create document ID pattern
        id_pattern = self.encoder.encode_string(doc_id)
        
        # Store bidirectional mapping
        self.sdm.store(features, id_pattern)  # features -> ID
        self.sdm.store(id_pattern, features)  # ID -> features
        
        # Save document info
        self.documents[doc_id] = {
            'title': title,
            'text': text[:200] + '...' if len(text) > 200 else text
        }
        self.doc_patterns[doc_id] = features
        
        print(f"Added document: {title}")
    
    def search(self, query, top_k=5):
        """Search for similar documents."""
        # Extract query features
        query_features = self.extract_features(query)
        
        # Recall from SDM
        recalled = self.sdm.recall(query_features)
        
        if recalled is None:
            return []
        
        # Find similar documents
        similarities = []
        for doc_id, doc_pattern in self.doc_patterns.items():
            similarity = np.mean(recalled == self.encoder.encode_string(doc_id))
            similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for doc_id, score in similarities[:top_k]:
            if score > 0.3:  # Threshold
                doc_info = self.documents[doc_id].copy()
                doc_info['score'] = score
                results.append(doc_info)
        
        return results

# Example usage
doc_sdm = DocumentSDM(dimension=3000)

# Add sample documents
documents = [
    ("doc1", "Introduction to Machine Learning",
     "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from and make predictions on data."),
    
    ("doc2", "Deep Learning Fundamentals",
     "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars."),
    
    ("doc3", "Natural Language Processing",
     "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language."),
    
    ("doc4", "Computer Vision Applications",
     "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world using digital images."),
    
    ("doc5", "Reinforcement Learning",
     "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment to maximize reward."),
    
    ("doc6", "Data Science Best Practices",
     "Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from structured and unstructured data."),
    
    ("doc7", "Quantum Computing Basics",
     "Quantum computing is a type of computation that harnesses the phenomena of quantum mechanics to process information in fundamentally new ways."),
    
    ("doc8", "Blockchain Technology",
     "Blockchain is a distributed ledger technology that maintains a secure and decentralized record of transactions across a network of computers.")
]

# Index documents
print("Indexing documents...")
for doc_id, title, text in documents:
    doc_sdm.add_document(doc_id, title, text)

# Search examples
print("\n" + "="*60)
print("DOCUMENT SEARCH RESULTS")
print("="*60)

queries = [
    "machine learning algorithms",
    "artificial intelligence applications",
    "quantum mechanics computation",
    "data analysis methods"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 40)
    
    results = doc_sdm.search(query, top_k=3)
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (score: {result['score']:.2f})")
            print(f"   {result['text']}")
    else:
        print("No relevant documents found.")
```

### Anomaly Detection System

```python
class AnomalyDetectorSDM:
    """SDM-based anomaly detection for time series or patterns."""
    
    def __init__(self, dimension=1000, window_size=10):
        self.sdm = create_sdm(dimension=dimension)
        self.dimension = dimension
        self.window_size = window_size
        self.encoder = PatternEncoder(dimension // window_size)
        self.normal_patterns = []
        
    def encode_window(self, data_window):
        """Encode a data window into a binary pattern."""
        patterns = []
        
        for value in data_window:
            # Encode each value
            if isinstance(value, (int, float)):
                pattern = self.encoder.encode_float(float(value))
            else:
                pattern = self.encoder.encode_string(str(value))
            patterns.append(pattern)
        
        # Concatenate patterns
        combined = np.concatenate(patterns)
        
        # Ensure correct dimension
        if len(combined) > self.dimension:
            combined = combined[:self.dimension]
        elif len(combined) < self.dimension:
            combined = np.pad(combined, (0, self.dimension - len(combined)))
        
        return combined
    
    def train_normal(self, normal_data):
        """Train on normal data patterns."""
        print("Training on normal patterns...")
        
        # Use sliding window
        for i in range(len(normal_data) - self.window_size + 1):
            window = normal_data[i:i + self.window_size]
            pattern = self.encode_window(window)
            
            # Store pattern pointing to itself (normal)
            self.sdm.store(pattern, pattern)
            self.normal_patterns.append(pattern)
        
        print(f"Trained on {len(self.normal_patterns)} normal patterns")
    
    def detect_anomaly(self, data_window, threshold=0.7):
        """Detect if a data window is anomalous."""
        # Encode the window
        pattern = self.encode_window(data_window)
        
        # Recall from SDM
        recalled = self.sdm.recall(pattern)
        
        if recalled is None:
            # No similar pattern found - likely anomalous
            return True, 0.0
        
        # Calculate similarity to recalled pattern
        similarity = np.mean(pattern == recalled)
        
        # If similarity is low, it's anomalous
        is_anomaly = similarity < threshold
        
        return is_anomaly, similarity
    
    def scan_timeseries(self, timeseries, plot=True):
        """Scan a time series for anomalies."""
        anomalies = []
        similarities = []
        
        for i in range(len(timeseries) - self.window_size + 1):
            window = timeseries[i:i + self.window_size]
            is_anomaly, similarity = self.detect_anomaly(window)
            
            anomalies.append(is_anomaly)
            similarities.append(similarity)
            
            if is_anomaly:
                print(f"Anomaly detected at position {i}: similarity = {similarity:.3f}")
        
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            # Plot time series
            ax1.plot(timeseries)
            ax1.set_ylabel('Value')
            ax1.set_title('Time Series Data')
            
            # Mark anomalies
            anomaly_indices = [i + self.window_size // 2 for i, a in enumerate(anomalies) if a]
            if anomaly_indices:
                ax1.scatter(anomaly_indices, 
                          [timeseries[i] for i in anomaly_indices],
                          color='red', s=50, zorder=5, label='Anomaly')
                ax1.legend()
            
            # Plot similarities
            ax2.plot(range(self.window_size // 2, 
                          len(timeseries) - self.window_size // 2),
                    similarities)
            ax2.axhline(y=0.7, color='r', linestyle='--', label='Threshold')
            ax2.set_ylabel('Similarity')
            ax2.set_title('Pattern Similarity to Normal')
            ax2.legend()
            
            # Plot anomaly indicators
            ax3.fill_between(range(self.window_size // 2,
                                 len(timeseries) - self.window_size // 2),
                           anomalies, alpha=0.3, color='red')
            ax3.set_ylabel('Anomaly')
            ax3.set_xlabel('Time')
            ax3.set_title('Anomaly Detection Results')
            ax3.set_ylim(-0.1, 1.1)
            
            plt.tight_layout()
            plt.show()
        
        return anomalies, similarities

# Example: Detect anomalies in synthetic data
np.random.seed(42)

# Generate normal data (sine wave with noise)
t = np.linspace(0, 100, 1000)
normal_data = np.sin(t) + 0.1 * np.random.randn(len(t))

# Add anomalies
anomaly_data = normal_data.copy()
# Spike anomalies
anomaly_positions = [200, 400, 600, 800]
for pos in anomaly_positions:
    anomaly_data[pos:pos+5] += np.random.uniform(2, 4)

# Level shift anomaly
anomaly_data[700:750] += 1.5

# Create and train detector
detector = AnomalyDetectorSDM(dimension=1000, window_size=20)

# Train on clean normal data
detector.train_normal(normal_data[:500])

# Detect anomalies
print("\nScanning for anomalies...")
anomalies, similarities = detector.scan_timeseries(anomaly_data, plot=True)

# Summary
total_windows = len(anomalies)
num_anomalies = sum(anomalies)
print(f"\nSummary: Detected {num_anomalies} anomalous windows out of {total_windows}")
```

---

## Conclusion

These examples demonstrate the versatility and power of Sparse Distributed Memory for various applications:

1. **Pattern Recognition** - Robust recognition despite noise
2. **Sequence Learning** - Temporal pattern storage and recall
3. **Associative Memory** - Content-based retrieval and associations
4. **Data Encoding** - Flexible encoding of various data types
5. **Performance Optimization** - Techniques for efficiency
6. **Real-World Applications** - Document search, anomaly detection

SDM provides a biologically-inspired approach to memory that exhibits many desirable properties like noise tolerance, graceful degradation, and automatic generalization. The examples show how to leverage these properties for practical applications.

For more examples and up-to-date code, visit the [cognitive-computing repository](https://github.com/transparentai-tech/cognitive-computing).