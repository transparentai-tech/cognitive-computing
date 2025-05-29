"""
Basic usage examples for Sparse Distributed Memory.

This module provides simple, reusable examples of common SDM operations
that can be imported and used in other code.
"""

import numpy as np
from typing import List, Tuple, Optional

from cognitive_computing.sdm import SDM, SDMConfig, create_sdm
from cognitive_computing.sdm.utils import (
    add_noise,
    generate_random_patterns,
    PatternEncoder,
    calculate_pattern_similarity
)


def simple_store_recall_example(dimension: int = 1000) -> Tuple[float, float]:
    """
    Simple example of storing and recalling a pattern.
    
    Parameters
    ----------
    dimension : int
        Dimension of the SDM
        
    Returns
    -------
    perfect_accuracy : float
        Accuracy with perfect recall
    noisy_accuracy : float
        Accuracy with noisy recall (10% noise)
        
    Example
    -------
    >>> perfect, noisy = simple_store_recall_example()
    >>> print(f"Perfect: {perfect:.2%}, Noisy: {noisy:.2%}")
    """
    # Create SDM
    sdm = create_sdm(dimension=dimension)
    
    # Generate random pattern
    address = np.random.randint(0, 2, dimension)
    data = np.random.randint(0, 2, dimension)
    
    # Store pattern
    sdm.store(address, data)
    
    # Perfect recall
    recalled_perfect = sdm.recall(address)
    perfect_accuracy = np.mean(recalled_perfect == data)
    
    # Noisy recall
    noisy_address = add_noise(address, 0.1)
    recalled_noisy = sdm.recall(noisy_address)
    noisy_accuracy = np.mean(recalled_noisy == data) if recalled_noisy is not None else 0.0
    
    return perfect_accuracy, noisy_accuracy


def pattern_recognition_example(num_classes: int = 5,
                              samples_per_class: int = 10,
                              dimension: int = 1000) -> SDM:
    """
    Example of using SDM for pattern recognition.
    
    Parameters
    ----------
    num_classes : int
        Number of pattern classes
    samples_per_class : int
        Training samples per class
    dimension : int
        Pattern dimension
        
    Returns
    -------
    sdm : SDM
        Trained SDM classifier
        
    Example
    -------
    >>> sdm = pattern_recognition_example()
    >>> # Test with noisy pattern from class 0
    >>> test_pattern = create_class_pattern(0, dimension=1000)
    >>> noisy_test = add_noise(test_pattern, 0.15)
    >>> label = sdm.recall(noisy_test)
    >>> predicted_class = np.argmax(label[:5])  # First 5 bits are one-hot label
    """
    sdm = create_sdm(dimension=dimension)
    
    # Generate prototype patterns for each class
    prototypes = [np.random.randint(0, 2, dimension) for _ in range(num_classes)]
    
    # Store noisy variants of each prototype
    for class_idx, prototype in enumerate(prototypes):
        # Create one-hot encoded label
        label = np.zeros(dimension, dtype=np.uint8)
        label[class_idx] = 1
        
        # Store multiple noisy variants
        for _ in range(samples_per_class):
            noisy_pattern = add_noise(prototype, noise_level=0.05)
            sdm.store(noisy_pattern, label)
    
    return sdm


def sequence_memory_example(sequence_length: int = 10,
                          dimension: int = 1000) -> Tuple[SDM, List[np.ndarray]]:
    """
    Example of using SDM for sequence storage.
    
    Parameters
    ----------
    sequence_length : int
        Length of sequence to store
    dimension : int
        Pattern dimension
        
    Returns
    -------
    sdm : SDM
        SDM with stored sequence
    sequence : List[np.ndarray]
        Original sequence
        
    Example
    -------
    >>> sdm, original = sequence_memory_example()
    >>> # Recall sequence starting from first element
    >>> recalled_seq = recall_sequence(sdm, original[0], len(original))
    """
    sdm = create_sdm(dimension=dimension)
    
    # Generate random sequence
    sequence = [np.random.randint(0, 2, dimension) for _ in range(sequence_length)]
    
    # Store transitions
    for i in range(len(sequence) - 1):
        current = sequence[i]
        next_item = sequence[i + 1]
        sdm.store(current, next_item)
    
    # Store loop back to beginning
    sdm.store(sequence[-1], sequence[0])
    
    return sdm, sequence


def recall_sequence(sdm: SDM, start_pattern: np.ndarray, 
                   max_length: int = 20) -> List[np.ndarray]:
    """
    Recall a sequence from SDM.
    
    Parameters
    ----------
    sdm : SDM
        SDM containing sequence
    start_pattern : np.ndarray
        Starting pattern
    max_length : int
        Maximum sequence length to recall
        
    Returns
    -------
    sequence : List[np.ndarray]
        Recalled sequence
    """
    sequence = [start_pattern]
    current = start_pattern
    
    for _ in range(max_length - 1):
        next_pattern = sdm.recall(current)
        if next_pattern is None:
            break
        
        sequence.append(next_pattern)
        
        # Check if we've looped back to start
        if np.array_equal(next_pattern, start_pattern):
            break
            
        current = next_pattern
    
    return sequence


def associative_memory_example(num_pairs: int = 20,
                             dimension: int = 1000) -> Tuple[SDM, dict]:
    """
    Example of using SDM as associative memory.
    
    Parameters
    ----------
    num_pairs : int
        Number of association pairs
    dimension : int
        Pattern dimension
        
    Returns
    -------
    sdm : SDM
        SDM with stored associations
    pairs : dict
        Dictionary of stored pairs for testing
        
    Example
    -------
    >>> sdm, pairs = associative_memory_example()
    >>> # Recall association
    >>> key = list(pairs.keys())[0]
    >>> recalled_value = sdm.recall(key)
    """
    sdm = create_sdm(dimension=dimension)
    pairs = {}
    
    # Generate and store random pairs
    for _ in range(num_pairs):
        key = np.random.randint(0, 2, dimension)
        value = np.random.randint(0, 2, dimension)
        
        sdm.store(key, value)
        pairs[key.tobytes()] = value
    
    return sdm, pairs


def real_data_encoding_example() -> SDM:
    """
    Example of encoding and storing real-world data types.
    
    Returns
    -------
    sdm : SDM
        SDM with encoded real-world data
        
    Example
    -------
    >>> sdm = real_data_encoding_example()
    >>> # Encode and recall a number
    >>> encoder = PatternEncoder(dimension=1000)
    >>> query = encoder.encode_integer(42)
    >>> recalled = sdm.recall(query)
    """
    dimension = 1000
    sdm = create_sdm(dimension=dimension)
    encoder = PatternEncoder(dimension=dimension)
    
    # Store various data types
    
    # 1. Integers
    numbers = [42, 123, 456, 789, 1000]
    for num in numbers:
        encoded = encoder.encode_integer(num)
        # Store the encoding as both key and value
        sdm.store(encoded, encoded)
    
    # 2. Strings
    words = ["hello", "world", "sparse", "distributed", "memory"]
    for word in words:
        encoded = encoder.encode_string(word, method='hash')
        sdm.store(encoded, encoded)
    
    # 3. Float values
    floats = [3.14159, 2.71828, 1.41421, 0.57721]
    for f in floats:
        encoded = encoder.encode_float(f, precision=16)
        sdm.store(encoded, encoded)
    
    # 4. Vector data
    vectors = [
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ]
    for vec in vectors:
        encoded = encoder.encode_vector(vec, method='threshold')
        sdm.store(encoded, encoded)
    
    return sdm


def hetero_associative_example() -> Tuple[SDM, PatternEncoder]:
    """
    Example of hetero-associative memory (different types for key and value).
    
    Returns
    -------
    sdm : SDM
        SDM with hetero-associations
    encoder : PatternEncoder
        Encoder used for data
        
    Example
    -------
    >>> sdm, encoder = hetero_associative_example()
    >>> # Query with string, get number
    >>> query = encoder.encode_string("age")
    >>> result_encoded = sdm.recall(query)
    >>> # Would need to decode result_encoded to get actual number
    """
    dimension = 1000
    sdm = create_sdm(dimension=dimension)
    encoder = PatternEncoder(dimension=dimension)
    
    # Store string -> number associations
    associations = {
        "age": 25,
        "temperature": 72,
        "count": 100,
        "year": 2024
    }
    
    for key_str, value_num in associations.items():
        key_encoded = encoder.encode_string(key_str)
        value_encoded = encoder.encode_integer(value_num)
        sdm.store(key_encoded, value_encoded)
    
    # Store number -> string associations
    reverse_associations = {
        1: "first",
        2: "second",
        3: "third",
        10: "tenth"
    }
    
    for key_num, value_str in reverse_associations.items():
        key_encoded = encoder.encode_integer(key_num)
        value_encoded = encoder.encode_string(value_str)
        sdm.store(key_encoded, value_encoded)
    
    return sdm, encoder


def create_class_pattern(class_idx: int, dimension: int = 1000, 
                        base_seed: int = 42) -> np.ndarray:
    """
    Create a prototype pattern for a given class.
    
    Parameters
    ----------
    class_idx : int
        Class index
    dimension : int
        Pattern dimension
    base_seed : int
        Base random seed
        
    Returns
    -------
    pattern : np.ndarray
        Prototype pattern for the class
    """
    rng = np.random.RandomState(base_seed + class_idx)
    return rng.randint(0, 2, dimension)


def test_sdm_noise_tolerance(sdm: SDM, test_pattern: np.ndarray,
                           expected_pattern: np.ndarray,
                           noise_levels: List[float] = None) -> dict:
    """
    Test SDM recall accuracy at different noise levels.
    
    Parameters
    ----------
    sdm : SDM
        SDM to test
    test_pattern : np.ndarray
        Pattern to add noise to
    expected_pattern : np.ndarray
        Expected recall result
    noise_levels : List[float]
        Noise levels to test
        
    Returns
    -------
    results : dict
        Noise levels mapped to accuracies
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    results = {}
    
    for noise in noise_levels:
        if noise == 0:
            noisy_pattern = test_pattern
        else:
            noisy_pattern = add_noise(test_pattern, noise)
        
        recalled = sdm.recall(noisy_pattern)
        
        if recalled is not None:
            accuracy = calculate_pattern_similarity(recalled, expected_pattern, 'hamming')
        else:
            accuracy = 0.0
        
        results[noise] = accuracy
    
    return results


# Make key functions available at module level
__all__ = [
    'simple_store_recall_example',
    'pattern_recognition_example',
    'sequence_memory_example',
    'recall_sequence',
    'associative_memory_example',
    'real_data_encoding_example',
    'hetero_associative_example',
    'create_class_pattern',
    'test_sdm_noise_tolerance'
]