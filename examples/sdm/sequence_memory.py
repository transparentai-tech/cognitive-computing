#!/usr/bin/env python3
"""
Sequence Memory with Sparse Distributed Memory

This example demonstrates how to use SDM for storing and recalling sequences,
including temporal patterns, time series prediction, and sequence completion.

Usage:
    python sequence_memory.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
from collections import defaultdict

# Import SDM components
from cognitive_computing.sdm import create_sdm, SDM, SDMConfig
from cognitive_computing.sdm.utils import (
    add_noise, generate_random_patterns, PatternEncoder,
    calculate_pattern_similarity
)
from cognitive_computing.sdm.visualizations import plot_memory_distribution


@dataclass
class SequenceResult:
    """Results from sequence recall."""
    sequence: List[np.ndarray]
    confidence: List[float]
    complete: bool
    accuracy: float


class SequenceMemory:
    """Base class for sequence storage and recall using SDM."""
    
    def __init__(self, dimension: int, max_sequence_length: int = 100):
        """
        Initialize sequence memory.
        
        Parameters
        ----------
        dimension : int
            Dimension of each sequence element
        max_sequence_length : int
            Maximum expected sequence length
        """
        self.dimension = dimension
        self.max_sequence_length = max_sequence_length
        
        # Create SDM with appropriate capacity
        num_locations = max(1000, max_sequence_length * 50)
        self.sdm = create_sdm(
            dimension=dimension,
            num_locations=num_locations,
            activation_radius=int(0.451 * dimension)
        )
        
        # Track sequences
        self.stored_sequences = []
        self.sequence_count = 0
        
    def store_sequence(self, sequence: List[np.ndarray], circular: bool = False):
        """
        Store a sequence in memory.
        
        Parameters
        ----------
        sequence : list
            List of pattern vectors
        circular : bool
            If True, last element points to first
        """
        if len(sequence) < 2:
            raise ValueError("Sequence must have at least 2 elements")
        
        # Store transitions
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_item = sequence[i + 1]
            self.sdm.store(current, next_item)
        
        # Store circular transition if requested
        if circular:
            self.sdm.store(sequence[-1], sequence[0])
        
        self.stored_sequences.append(sequence)
        self.sequence_count += 1
        
        print(f"Stored sequence of length {len(sequence)} "
              f"({'circular' if circular else 'linear'})")
    
    def recall_sequence(self, start_pattern: np.ndarray, 
                       max_length: Optional[int] = None,
                       threshold: float = 0.8) -> SequenceResult:
        """
        Recall a sequence starting from a pattern.
        
        Parameters
        ----------
        start_pattern : np.ndarray
            Starting pattern
        max_length : int, optional
            Maximum length to recall
        threshold : float
            Similarity threshold for sequence continuation
        
        Returns
        -------
        SequenceResult
            Recalled sequence and metadata
        """
        if max_length is None:
            max_length = self.max_sequence_length
        
        sequence = [start_pattern]
        confidence_scores = [1.0]
        current = start_pattern
        
        for i in range(max_length - 1):
            # Recall next pattern
            next_pattern = self.sdm.recall(current)
            
            if next_pattern is None:
                # Sequence broken
                return SequenceResult(
                    sequence=sequence,
                    confidence=confidence_scores,
                    complete=False,
                    accuracy=0.0
                )
            
            # Check for loops (returned to previous pattern)
            similarity_to_previous = [
                calculate_pattern_similarity(next_pattern, prev, metric='hamming')
                for prev in sequence
            ]
            
            if any(sim > 0.95 for sim in similarity_to_previous):
                # Found a loop
                return SequenceResult(
                    sequence=sequence,
                    confidence=confidence_scores,
                    complete=True,
                    accuracy=1.0
                )
            
            # Check confidence (similarity to expected pattern structure)
            if i > 0:
                # Compare pattern statistics
                confidence = self._calculate_confidence(next_pattern, sequence)
                confidence_scores.append(confidence)
                
                if confidence < threshold:
                    # Low confidence, stop
                    return SequenceResult(
                        sequence=sequence,
                        confidence=confidence_scores,
                        complete=False,
                        accuracy=np.mean(confidence_scores)
                    )
            else:
                confidence_scores.append(1.0)
            
            sequence.append(next_pattern)
            current = next_pattern
        
        return SequenceResult(
            sequence=sequence,
            confidence=confidence_scores,
            complete=True,
            accuracy=np.mean(confidence_scores)
        )
    
    def _calculate_confidence(self, pattern: np.ndarray, 
                            previous_patterns: List[np.ndarray]) -> float:
        """Calculate confidence score for a pattern in sequence."""
        # Simple confidence based on pattern statistics
        # Could be made more sophisticated
        
        # Check if pattern has reasonable density
        density = np.mean(pattern)
        if density < 0.1 or density > 0.9:
            return 0.5
        
        # Check if pattern is too similar to recent patterns
        recent_similarities = [
            calculate_pattern_similarity(pattern, prev, metric='hamming')
            for prev in previous_patterns[-3:]
        ]
        
        if any(sim > 0.9 for sim in recent_similarities):
            return 0.6
        
        return 0.9
    
    def find_sequence(self, partial_sequence: List[np.ndarray], 
                     complete_length: int) -> Optional[SequenceResult]:
        """
        Find and complete a sequence from partial input.
        
        Parameters
        ----------
        partial_sequence : list
            Partial sequence to complete
        complete_length : int
            Expected total length
        
        Returns
        -------
        SequenceResult or None
            Completed sequence if found
        """
        if not partial_sequence:
            return None
        
        # Try to continue from the last element
        result = self.recall_sequence(
            partial_sequence[-1], 
            max_length=complete_length - len(partial_sequence) + 1
        )
        
        if result.complete or len(result.sequence) >= complete_length - len(partial_sequence):
            # Combine partial and recalled
            complete_sequence = partial_sequence[:-1] + result.sequence
            return SequenceResult(
                sequence=complete_sequence[:complete_length],
                confidence=[1.0] * (len(partial_sequence) - 1) + result.confidence,
                complete=result.complete,
                accuracy=result.accuracy
            )
        
        return None


class MusicalSequenceMemory(SequenceMemory):
    """Specialized sequence memory for musical patterns."""
    
    def __init__(self, dimension: int = 1000):
        super().__init__(dimension=dimension)
        self.encoder = PatternEncoder(dimension)
        
        # Musical note mapping
        self.notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']
        self.note_to_pattern = {}
        
        # Create patterns for each note
        for i, note in enumerate(self.notes):
            pattern = np.zeros(dimension, dtype=np.uint8)
            # Each note has a unique pattern
            start_idx = i * (dimension // len(self.notes))
            end_idx = start_idx + (dimension // len(self.notes))
            pattern[start_idx:end_idx:2] = 1
            pattern[start_idx+1:end_idx:3] = 1
            self.note_to_pattern[note] = pattern
    
    def store_melody(self, notes: List[str], name: str = ""):
        """Store a melody as a sequence."""
        patterns = [self.note_to_pattern[note] for note in notes]
        self.store_sequence(patterns, circular=True)
        
        if name:
            print(f"Stored melody '{name}': {' '.join(notes)}")
        else:
            print(f"Stored melody: {' '.join(notes)}")
    
    def recall_melody(self, starting_note: str, length: int = 16) -> List[str]:
        """Recall a melody starting from a note."""
        start_pattern = self.note_to_pattern[starting_note]
        result = self.recall_sequence(start_pattern, max_length=length)
        
        # Convert patterns back to notes
        notes = []
        for pattern in result.sequence:
            # Find closest note pattern
            best_note = None
            best_similarity = 0
            
            for note, note_pattern in self.note_to_pattern.items():
                similarity = calculate_pattern_similarity(
                    pattern, note_pattern, metric='hamming'
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_note = note
            
            notes.append(best_note if best_note else '?')
        
        return notes
    
    def visualize_melody_recall(self, starting_note: str, true_melody: List[str]):
        """Visualize melody recall compared to original."""
        recalled_notes = self.recall_melody(starting_note, len(true_melody))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Plot original melody
        note_positions = {note: i for i, note in enumerate(self.notes)}
        true_positions = [note_positions.get(note, -1) for note in true_melody]
        ax1.plot(true_positions, 'o-', linewidth=2, markersize=8, label='Original')
        ax1.set_ylim(-0.5, len(self.notes) - 0.5)
        ax1.set_yticks(range(len(self.notes)))
        ax1.set_yticklabels(self.notes)
        ax1.set_title('Original Melody')
        ax1.grid(True, alpha=0.3)
        
        # Plot recalled melody
        recalled_positions = [note_positions.get(note, -1) for note in recalled_notes]
        colors = ['green' if r == t else 'red' for r, t in zip(recalled_notes, true_melody)]
        
        for i, (pos, color) in enumerate(zip(recalled_positions, colors)):
            ax2.plot(i, pos, 'o', color=color, markersize=8)
        ax2.plot(recalled_positions, '-', linewidth=2, alpha=0.5)
        
        ax2.set_ylim(-0.5, len(self.notes) - 0.5)
        ax2.set_yticks(range(len(self.notes)))
        ax2.set_yticklabels(self.notes)
        ax2.set_xlabel('Position in Sequence')
        ax2.set_title('Recalled Melody (Green=Correct, Red=Error)')
        ax2.grid(True, alpha=0.3)
        
        # Calculate accuracy
        accuracy = sum(1 for r, t in zip(recalled_notes, true_melody) if r == t) / len(true_melody)
        plt.suptitle(f'Melody Recall Accuracy: {accuracy:.1%}', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        return recalled_notes, accuracy


class TextSequenceMemory(SequenceMemory):
    """Sequence memory for text/character sequences."""
    
    def __init__(self, dimension: int = 1500):
        super().__init__(dimension=dimension)
        self.encoder = PatternEncoder(dimension)
        self.char_to_pattern = {}
        self.pattern_to_char = {}
        
    def _get_char_pattern(self, char: str) -> np.ndarray:
        """Get or create pattern for a character."""
        if char not in self.char_to_pattern:
            # Create unique pattern for character
            pattern = self.encoder.encode_string(char + str(ord(char)))
            self.char_to_pattern[char] = pattern
            # Store reverse mapping (approximate)
            self.pattern_to_char[pattern.tobytes()] = char
        
        return self.char_to_pattern[char]
    
    def store_text(self, text: str):
        """Store text as character sequence."""
        patterns = [self._get_char_pattern(char) for char in text]
        self.store_sequence(patterns, circular=False)
        print(f"Stored text: '{text}' ({len(text)} characters)")
    
    def complete_text(self, prompt: str, max_length: int = 50) -> str:
        """Complete text from a prompt."""
        if not prompt:
            return ""
        
        # Convert prompt to patterns
        prompt_patterns = [self._get_char_pattern(char) for char in prompt]
        
        # Recall continuation
        result = self.recall_sequence(
            prompt_patterns[-1], 
            max_length=max_length - len(prompt) + 1
        )
        
        # Convert back to text
        completed = prompt[:-1]  # Remove last char (it's in the sequence)
        
        for pattern in result.sequence:
            # Find closest character
            pattern_bytes = pattern.tobytes()
            
            if pattern_bytes in self.pattern_to_char:
                completed += self.pattern_to_char[pattern_bytes]
            else:
                # Find most similar character
                best_char = '?'
                best_sim = 0
                
                for char, char_pattern in self.char_to_pattern.items():
                    sim = calculate_pattern_similarity(
                        pattern, char_pattern, metric='hamming'
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_char = char
                
                completed += best_char
        
        return completed
    
    def analyze_text_memory(self, test_prompts: List[str]):
        """Analyze text completion performance."""
        results = []
        
        for prompt in test_prompts:
            completed = self.complete_text(prompt, max_length=30)
            
            # Check if it makes sense (basic check)
            # In practice, you'd use more sophisticated metrics
            contains_prompt = prompt.lower() in completed.lower()
            reasonable_length = len(completed) > len(prompt)
            
            results.append({
                'prompt': prompt,
                'completed': completed,
                'success': contains_prompt and reasonable_length
            })
        
        return results


class TimeSeriesSequenceMemory(SequenceMemory):
    """Sequence memory for time series patterns."""
    
    def __init__(self, dimension: int = 2000, window_size: int = 10):
        super().__init__(dimension=dimension)
        self.window_size = window_size
        self.encoder = PatternEncoder(dimension // window_size)
        
    def encode_window(self, values: np.ndarray) -> np.ndarray:
        """Encode a window of time series values."""
        patterns = []
        
        for value in values:
            # Encode each value
            if isinstance(value, (int, float)):
                pattern = self.encoder.encode_float(float(value), precision=10)
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
    
    def store_time_series(self, series: np.ndarray, name: str = ""):
        """Store time series using sliding window."""
        if len(series) < self.window_size + 1:
            raise ValueError(f"Series must have at least {self.window_size + 1} values")
        
        # Create sliding windows
        patterns = []
        for i in range(len(series) - self.window_size):
            window = series[i:i + self.window_size]
            pattern = self.encode_window(window)
            patterns.append(pattern)
        
        # Store as sequence
        self.store_sequence(patterns, circular=False)
        
        if name:
            print(f"Stored time series '{name}': {len(series)} values, "
                  f"{len(patterns)} windows")
    
    def predict_next_values(self, recent_values: np.ndarray, 
                           num_predictions: int = 5) -> np.ndarray:
        """Predict next values in time series."""
        if len(recent_values) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} recent values")
        
        # Use last window
        last_window = recent_values[-self.window_size:]
        current_pattern = self.encode_window(last_window)
        
        predictions = []
        
        for _ in range(num_predictions):
            # Recall next pattern
            next_pattern = self.sdm.recall(current_pattern)
            
            if next_pattern is None:
                break
            
            # Decode to get next value (simplified - just use pattern statistics)
            # In practice, you'd have a more sophisticated decoder
            next_value = np.mean(next_pattern) * 10  # Scale factor
            predictions.append(next_value)
            
            # Slide window
            new_window = np.append(last_window[1:], next_value)
            current_pattern = self.encode_window(new_window)
            last_window = new_window
        
        return np.array(predictions)
    
    def visualize_time_series_prediction(self, series: np.ndarray, 
                                       test_start: int, num_predictions: int = 10):
        """Visualize time series prediction."""
        # Use data before test_start for context
        context = series[:test_start]
        actual = series[test_start:test_start + num_predictions]
        
        # Predict
        predictions = self.predict_next_values(context, num_predictions)
        
        # Visualize
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(range(len(context)), context, 'b-', linewidth=2, label='Historical')
        
        # Plot actual future values
        future_x = range(len(context), len(context) + len(actual))
        plt.plot(future_x, actual, 'g-', linewidth=2, label='Actual')
        
        # Plot predictions
        pred_x = range(len(context), len(context) + len(predictions))
        plt.plot(pred_x, predictions, 'r--', linewidth=2, label='Predicted')
        
        # Mark prediction start
        plt.axvline(x=test_start, color='gray', linestyle=':', alpha=0.7)
        plt.text(test_start, plt.ylim()[1] * 0.9, 'Prediction Start', 
                rotation=90, verticalalignment='bottom')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Prediction with SDM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate error metrics
        if len(predictions) >= len(actual):
            mse = np.mean((predictions[:len(actual)] - actual) ** 2)
            mae = np.mean(np.abs(predictions[:len(actual)] - actual))
            print(f"Prediction MSE: {mse:.4f}, MAE: {mae:.4f}")


def demonstrate_basic_sequences():
    """Demonstrate basic sequence storage and recall."""
    print("\n" + "="*60)
    print("BASIC SEQUENCE MEMORY DEMONSTRATION")
    print("="*60)
    
    # Create sequence memory
    seq_memory = SequenceMemory(dimension=1000)
    
    # Create and store simple sequences
    print("\n1. Storing simple number sequences:")
    
    # Fibonacci-like sequence
    fib_patterns = []
    fib_values = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    encoder = PatternEncoder(1000)
    
    for val in fib_values:
        pattern = encoder.encode_integer(val, bits=32)
        fib_patterns.append(pattern)
    
    seq_memory.store_sequence(fib_patterns, circular=False)
    print(f"  Stored Fibonacci sequence: {fib_values}")
    
    # Arithmetic sequence
    arith_patterns = []
    arith_values = [5, 10, 15, 20, 25, 30, 35, 40]
    
    for val in arith_values:
        pattern = encoder.encode_integer(val, bits=32)
        arith_patterns.append(pattern)
    
    seq_memory.store_sequence(arith_patterns, circular=False)
    print(f"  Stored arithmetic sequence: {arith_values}")
    
    # Test recall
    print("\n2. Testing sequence recall:")
    
    # Recall Fibonacci from first element
    print("\n  Recalling Fibonacci from first element:")
    result = seq_memory.recall_sequence(fib_patterns[0], max_length=10)
    
    # Decode recalled patterns
    recalled_values = []
    for pattern in result.sequence:
        # Find closest stored pattern
        best_val = 0
        best_sim = 0
        
        for val, stored_pattern in zip(fib_values, fib_patterns):
            sim = calculate_pattern_similarity(pattern, stored_pattern, metric='hamming')
            if sim > best_sim:
                best_sim = sim
                best_val = val
        
        recalled_values.append(best_val)
    
    print(f"  Recalled: {recalled_values}")
    print(f"  Accuracy: {result.accuracy:.2%}")
    
    # Test with noisy start
    print("\n3. Testing recall with noisy input:")
    noisy_start = add_noise(arith_patterns[2], 0.1)  # Start from 15 with noise
    result_noisy = seq_memory.recall_sequence(noisy_start, max_length=6)
    
    print(f"  Starting from noisy '15'")
    print(f"  Sequence completed: {result_noisy.complete}")
    print(f"  Confidence scores: {[f'{c:.2f}' for c in result_noisy.confidence]}")


def demonstrate_musical_sequences():
    """Demonstrate musical sequence learning."""
    print("\n" + "="*60)
    print("MUSICAL SEQUENCE MEMORY DEMONSTRATION")
    print("="*60)
    
    # Create musical memory
    music_memory = MusicalSequenceMemory(dimension=1200)
    
    # Store some melodies
    print("\n1. Storing musical sequences:")
    
    # Simple scales
    c_major_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']
    music_memory.store_melody(c_major_scale, "C Major Scale")
    
    # Simple melody patterns
    melody1 = ['C', 'E', 'G', 'E', 'C', 'E', 'G', 'E']
    music_memory.store_melody(melody1, "Arpeggio Pattern")
    
    melody2 = ['C', 'C', 'G', 'G', 'A', 'A', 'G', 'F', 'F', 'E', 'E', 'D', 'D', 'C']
    music_memory.store_melody(melody2, "Twinkle Twinkle")
    
    # Test recall
    print("\n2. Testing melody recall:")
    
    # Recall scale from C
    print("\n  Recalling from 'C':")
    recalled_scale = music_memory.recall_melody('C', length=8)
    print(f"  Recalled: {' '.join(recalled_scale)}")
    
    # Recall melody from middle
    print("\n  Recalling from 'G' (middle of pattern):")
    recalled_mid = music_memory.recall_melody('G', length=8)
    print(f"  Recalled: {' '.join(recalled_mid)}")
    
    # Visualize recall accuracy
    print("\n3. Visualizing melody recall:")
    music_memory.visualize_melody_recall('C', c_major_scale)
    
    # Test with noise
    print("\n4. Testing recall with noisy patterns:")
    
    # Add noise to a note pattern and try to recall
    clean_pattern = music_memory.note_to_pattern['E']
    noisy_pattern = add_noise(clean_pattern, 0.15)
    
    # Store noisy pattern in SDM and recall
    test_memory = MusicalSequenceMemory(dimension=1200)
    test_memory.store_melody(melody1, "Test Melody")
    
    # Override note pattern temporarily
    original_E = test_memory.note_to_pattern['E'].copy()
    test_memory.note_to_pattern['E'] = noisy_pattern
    
    recalled_noisy = test_memory.recall_melody('E', length=6)
    print(f"  Recalled from noisy 'E': {' '.join(recalled_noisy)}")
    
    # Restore
    test_memory.note_to_pattern['E'] = original_E


def demonstrate_text_sequences():
    """Demonstrate text sequence learning and completion."""
    print("\n" + "="*60)
    print("TEXT SEQUENCE MEMORY DEMONSTRATION")
    print("="*60)
    
    # Create text memory
    text_memory = TextSequenceMemory(dimension=2000)
    
    # Store some text sequences
    print("\n1. Storing text sequences:")
    
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be that is the question",
        "Once upon a time in a land far far away",
        "Machine learning is a subset of artificial intelligence",
        "Sparse distributed memory exhibits noise tolerance"
    ]
    
    for text in texts:
        text_memory.store_text(text)
    
    # Test text completion
    print("\n2. Testing text completion:")
    
    prompts = [
        "The quick",
        "To be or",
        "Once upon",
        "Machine learning",
        "Sparse distributed"
    ]
    
    for prompt in prompts:
        completed = text_memory.complete_text(prompt, max_length=30)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Completed: '{completed}'")
    
    # Test with partial matches
    print("\n3. Testing with partial/noisy prompts:")
    
    partial_prompts = [
        "quick brown",  # Middle of sequence
        "far far",      # End portion
        "learning is"   # Technical term
    ]
    
    results = text_memory.analyze_text_memory(partial_prompts)
    
    for result in results:
        print(f"\n  Prompt: '{result['prompt']}'")
        print(f"  Completed: '{result['completed']}'")
        print(f"  Success: {result['success']}")
    
    # Visualize character patterns
    print("\n4. Character pattern similarity matrix:")
    
    # Sample some characters
    chars = ['a', 'b', 'c', 'e', 'i', 'o', 'u', ' ', '.', ',']
    similarity_matrix = np.zeros((len(chars), len(chars)))
    
    for i, char1 in enumerate(chars):
        for j, char2 in enumerate(chars):
            pattern1 = text_memory._get_char_pattern(char1)
            pattern2 = text_memory._get_char_pattern(char2)
            similarity = calculate_pattern_similarity(pattern1, pattern2, metric='jaccard')
            similarity_matrix[i, j] = similarity
    
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Jaccard Similarity')
    plt.xticks(range(len(chars)), chars)
    plt.yticks(range(len(chars)), chars)
    plt.title('Character Pattern Similarity Matrix')
    plt.tight_layout()
    plt.show()


def demonstrate_time_series():
    """Demonstrate time series sequence prediction."""
    print("\n" + "="*60)
    print("TIME SERIES SEQUENCE MEMORY DEMONSTRATION")
    print("="*60)
    
    # Create time series memory
    ts_memory = TimeSeriesSequenceMemory(dimension=2000, window_size=5)
    
    # Generate and store time series
    print("\n1. Storing time series patterns:")
    
    # Sine wave
    t = np.linspace(0, 4*np.pi, 100)
    sine_series = np.sin(t) * 10 + 20
    ts_memory.store_time_series(sine_series, "Sine Wave")
    
    # Linear trend with noise
    trend_series = np.linspace(10, 50, 100) + np.random.normal(0, 2, 100)
    ts_memory.store_time_series(trend_series, "Linear Trend")
    
    # Step function
    step_series = np.concatenate([
        np.ones(25) * 10,
        np.ones(25) * 20,
        np.ones(25) * 15,
        np.ones(25) * 25
    ])
    ts_memory.store_time_series(step_series, "Step Function")
    
    # Test prediction
    print("\n2. Testing time series prediction:")
    
    # Predict sine wave continuation
    print("\n  Predicting sine wave:")
    ts_memory.visualize_time_series_prediction(sine_series, test_start=80, num_predictions=15)
    
    # Predict trend continuation
    print("\n  Predicting linear trend:")
    ts_memory.visualize_time_series_prediction(trend_series, test_start=80, num_predictions=15)
    
    # Test with different window sizes
    print("\n3. Effect of window size on prediction:")
    
    window_sizes = [3, 5, 10]
    prediction_errors = []
    
    for window_size in window_sizes:
        # Create new memory with different window
        ts_mem_test = TimeSeriesSequenceMemory(dimension=2000, window_size=window_size)
        ts_mem_test.store_time_series(sine_series, f"Window {window_size}")
        
        # Predict and calculate error
        predictions = ts_mem_test.predict_next_values(sine_series[:80], num_predictions=10)
        actual = sine_series[80:90]
        
        if len(predictions) >= len(actual):
            mse = np.mean((predictions[:len(actual)] - actual) ** 2)
            prediction_errors.append(mse)
            print(f"  Window size {window_size}: MSE = {mse:.4f}")
        else:
            prediction_errors.append(np.inf)
            print(f"  Window size {window_size}: Failed to predict enough values")


def demonstrate_sequence_analysis():
    """Analyze sequence memory performance characteristics."""
    print("\n" + "="*60)
    print("SEQUENCE MEMORY PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Test different sequence lengths
    print("\n1. Performance vs sequence length:")
    
    sequence_lengths = [5, 10, 20, 50, 100]
    results = []
    
    for length in sequence_lengths:
        # Create memory
        seq_mem = SequenceMemory(dimension=1000)
        
        # Generate random sequences
        num_sequences = 20
        sequences = []
        
        for _ in range(num_sequences):
            seq = generate_random_patterns(length, 1000)[0]
            sequences.append(seq)
            seq_mem.store_sequence(seq, circular=False)
        
        # Test recall
        recall_times = []
        accuracies = []
        
        for seq in sequences[:10]:  # Test subset
            start_time = time.time()
            result = seq_mem.recall_sequence(seq[0], max_length=length)
            recall_time = time.time() - start_time
            recall_times.append(recall_time)
            
            # Check accuracy
            if len(result.sequence) == length:
                accuracy = sum(
                    calculate_pattern_similarity(r, s, metric='hamming') > 0.9
                    for r, s in zip(result.sequence, seq)
                ) / length
                accuracies.append(accuracy)
        
        results.append({
            'length': length,
            'avg_recall_time': np.mean(recall_times),
            'avg_accuracy': np.mean(accuracies) if accuracies else 0
        })
        
        print(f"  Length {length}: {np.mean(recall_times)*1000:.1f}ms, "
              f"accuracy {np.mean(accuracies) if accuracies else 0:.2%}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    lengths = [r['length'] for r in results]
    times = [r['avg_recall_time'] * 1000 for r in results]
    accuracies = [r['avg_accuracy'] for r in results]
    
    ax1.plot(lengths, times, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Recall Time (ms)')
    ax1.set_title('Recall Time vs Sequence Length')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(lengths, accuracies, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Recall Accuracy')
    ax2.set_title('Accuracy vs Sequence Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    # Test branching sequences
    print("\n2. Testing branching sequences:")
    
    branch_mem = SequenceMemory(dimension=1000)
    encoder = PatternEncoder(1000)
    
    # Create branching structure
    # A -> B -> C
    #        -> D
    patterns = {
        'A': encoder.encode_string('A'),
        'B': encoder.encode_string('B'),
        'C': encoder.encode_string('C'),
        'D': encoder.encode_string('D'),
        'E': encoder.encode_string('E')
    }
    
    # Store branches
    branch_mem.sdm.store(patterns['A'], patterns['B'])
    branch_mem.sdm.store(patterns['B'], patterns['C'])
    branch_mem.sdm.store(patterns['B'], patterns['D'])  # This creates ambiguity
    branch_mem.sdm.store(patterns['C'], patterns['E'])
    branch_mem.sdm.store(patterns['D'], patterns['E'])
    
    # Test recall from A
    print("\n  Recalling from A (should show branching):")
    
    # Multiple recalls to see different paths
    for i in range(5):
        result = branch_mem.recall_sequence(patterns['A'], max_length=4)
        path = ['A']
        
        for j, pattern in enumerate(result.sequence[1:], 1):
            # Decode pattern
            for name, p in patterns.items():
                if calculate_pattern_similarity(pattern, p, metric='hamming') > 0.9:
                    path.append(name)
                    break
        
        print(f"    Attempt {i+1}: {' -> '.join(path)}")


def performance_comparison():
    """Compare different sequence memory configurations."""
    print("\n" + "="*60)
    print("SEQUENCE MEMORY CONFIGURATION COMPARISON")
    print("="*60)
    
    # Test configurations
    configs = [
        {'name': 'Small', 'dim': 500, 'sequences': 10, 'length': 10},
        {'name': 'Medium', 'dim': 1000, 'sequences': 20, 'length': 20},
        {'name': 'Large', 'dim': 2000, 'sequences': 50, 'length': 20},
        {'name': 'Long', 'dim': 1000, 'sequences': 10, 'length': 50}
    ]
    
    comparison_results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        
        # Create memory
        seq_mem = SequenceMemory(dimension=config['dim'])
        
        # Generate and store sequences
        sequences = generate_random_patterns(
            config['sequences'] * config['length'], 
            config['dim']
        )[0]
        
        # Reshape into sequences
        sequences = sequences.reshape(config['sequences'], config['length'], -1)
        
        store_time = 0
        for seq in sequences:
            start = time.time()
            seq_mem.store_sequence(list(seq), circular=False)
            store_time += time.time() - start
        
        # Test recall
        recall_times = []
        successes = 0
        
        for seq in sequences[:min(10, len(sequences))]:
            start = time.time()
            result = seq_mem.recall_sequence(seq[0], max_length=config['length'])
            recall_times.append(time.time() - start)
            
            if len(result.sequence) == config['length']:
                successes += 1
        
        # Memory usage estimate
        memory_mb = seq_mem.sdm.get_memory_stats()['num_hard_locations'] * config['dim'] * 9 / 8 / 1024 / 1024
        
        comparison_results.append({
            'config': config['name'],
            'store_time': store_time,
            'avg_recall_time': np.mean(recall_times),
            'success_rate': successes / len(recall_times),
            'memory_mb': memory_mb
        })
    
    # Display results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Config':<10} {'Store(s)':<10} {'Recall(ms)':<12} {'Success':<10} {'Memory(MB)':<10}")
    print("-"*60)
    
    for r in comparison_results:
        print(f"{r['config']:<10} {r['store_time']:<10.3f} "
              f"{r['avg_recall_time']*1000:<12.1f} {r['success_rate']:<10.1%} "
              f"{r['memory_mb']:<10.1f}")


def main():
    """Run all sequence memory demonstrations."""
    print("SPARSE DISTRIBUTED MEMORY - SEQUENCE MEMORY EXAMPLES")
    print("====================================================")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_basic_sequences()
    demonstrate_musical_sequences()
    demonstrate_text_sequences()
    demonstrate_time_series()
    demonstrate_sequence_analysis()
    performance_comparison()
    
    print("\n" + "="*60)
    print("Sequence memory demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
