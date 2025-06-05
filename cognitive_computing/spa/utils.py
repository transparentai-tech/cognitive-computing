"""
Utility functions for Semantic Pointer Architecture operations.

This module provides helper functions for working with semantic pointers,
vocabularies, and SPA systems including vector generation, similarity
analysis, and performance measurement.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
from scipy.fft import fft, ifft

from .core import SemanticPointer, Vocabulary, SPAConfig
from .modules import Module
from .actions import ActionSet
from .production import Production, ProductionSystem

logger = logging.getLogger(__name__)


def make_unitary(pointer: np.ndarray) -> np.ndarray:
    """
    Make a semantic pointer unitary for binding operations.
    
    A unitary vector preserves dot products under binding:
    (A*B)·(A*C) = B·C
    
    Parameters
    ----------
    pointer : array_like
        Input vector
        
    Returns
    -------
    array
        Unitary vector
    """
    # Ensure 1D
    pointer = np.asarray(pointer).ravel()
    
    # Take FFT
    fft_vec = fft(pointer)
    
    # Set all magnitudes to 1, preserve phase
    fft_unitary = fft_vec / (np.abs(fft_vec) + 1e-10)
    
    # Transform back
    unitary = ifft(fft_unitary).real
    
    # Normalize
    norm = np.linalg.norm(unitary)
    if norm == 0:
        # Handle zero vector case
        return np.ones_like(pointer) / np.sqrt(len(pointer))
    return unitary / norm


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between semantic pointers.
    
    Parameters
    ----------
    a : array_like
        First vector
    b : array_like
        Second vector
        
    Returns
    -------
    float
        Similarity in range [-1, 1]
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length, got {len(a)} and {len(b)}")
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def normalize_semantic_pointer(pointer: np.ndarray) -> np.ndarray:
    """
    Normalize a semantic pointer to unit length.
    
    Parameters
    ----------
    pointer : array_like
        Input vector
        
    Returns
    -------
    array
        Normalized vector
    """
    pointer = np.asarray(pointer).ravel()
    norm = np.linalg.norm(pointer)
    
    if norm == 0:
        logger.warning("Cannot normalize zero vector")
        return pointer
    
    return pointer / norm


def generate_pointers(vocab_size: int, dimensions: int, 
                      unitary: bool = False,
                      rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """
    Generate orthogonal semantic pointers for vocabulary.
    
    Parameters
    ----------
    vocab_size : int
        Number of pointers to generate
    dimensions : int
        Dimensionality of vectors
    unitary : bool
        Whether to make vectors unitary
    rng : RandomState, optional
        Random number generator
        
    Returns
    -------
    dict
        Mapping from names to vectors
    """
    if vocab_size > dimensions:
        logger.warning(f"Generating {vocab_size} pointers in {dimensions}D space - "
                      f"vectors will not be orthogonal")
    
    if rng is None:
        rng = np.random.RandomState()
    
    pointers = {}
    
    # Generate random vectors
    for i in range(vocab_size):
        vec = rng.randn(dimensions)
        vec = normalize_semantic_pointer(vec)
        
        if unitary:
            vec = make_unitary(vec)
        
        pointers[f"P{i:03d}"] = vec
    
    return pointers


def analyze_vocabulary(vocab: Vocabulary) -> Dict[str, Any]:
    """
    Analyze vocabulary statistics.
    
    Computes similarity statistics, clustering, and quality metrics
    for a vocabulary of semantic pointers.
    
    Parameters
    ----------
    vocab : Vocabulary
        Vocabulary to analyze
        
    Returns
    -------
    dict
        Analysis results including:
        - size: Number of pointers
        - dimensions: Vector dimensionality  
        - mean_similarity: Average pairwise similarity
        - max_similarity: Maximum pairwise similarity
        - similarity_matrix: Full similarity matrix
        - orthogonality: Measure of orthogonality
    """
    pointers = list(vocab.pointers.values())
    n_pointers = len(pointers)
    
    if n_pointers == 0:
        return {
            'size': 0,
            'dimensions': vocab.dimension,
            'mean_similarity': 0.0,
            'max_similarity': 0.0,
            'similarity_matrix': np.array([[]]),
            'orthogonality': 0.0  # No vectors means no orthogonality measure
        }
    
    # Extract vectors
    vectors = np.array([p.vector for p in pointers])
    
    # Compute similarity matrix
    sim_matrix = np.zeros((n_pointers, n_pointers))
    
    for i in range(n_pointers):
        for j in range(n_pointers):
            sim_matrix[i, j] = similarity(vectors[i], vectors[j])
    
    # Compute statistics
    # Exclude diagonal (self-similarity)
    mask = ~np.eye(n_pointers, dtype=bool)
    off_diagonal = sim_matrix[mask]
    
    mean_sim = np.mean(np.abs(off_diagonal))
    max_sim = np.max(np.abs(off_diagonal)) if len(off_diagonal) > 0 else 0.0
    
    # Orthogonality measure (lower is better)
    orthogonality = np.sqrt(np.mean(off_diagonal**2))
    
    return {
        'size': n_pointers,
        'dimensions': vocab.dimension,
        'mean_similarity': mean_sim,
        'max_similarity': max_sim,
        'similarity_matrix': sim_matrix,
        'orthogonality': orthogonality,
        'pointer_names': list(vocab.pointers.keys())
    }


def measure_binding_capacity(dimensions: int, n_pairs: int = 100,
                            n_trials: int = 10,
                            rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
    """
    Measure binding capacity for given dimensionality.
    
    Tests how well bound pairs can be recovered as the number
    of superposed pairs increases.
    
    Parameters
    ----------
    dimensions : int
        Vector dimensionality
    n_pairs : int
        Maximum number of pairs to test
    n_trials : int
        Number of trials to average
    rng : RandomState, optional
        Random number generator
        
    Returns
    -------
    dict
        Capacity measurements including threshold capacities
    """
    if rng is None:
        rng = np.random.RandomState()
    
    results = {
        'dimensions': dimensions,
        'capacities': {},
        'similarities': []
    }
    
    for n in range(1, min(n_pairs + 1, dimensions // 2), max(1, n_pairs // 20)):
        similarities = []
        
        for trial in range(n_trials):
            # Generate random vectors
            keys = [normalize_semantic_pointer(rng.randn(dimensions)) for _ in range(n)]
            values = [normalize_semantic_pointer(rng.randn(dimensions)) for _ in range(n)]
            
            # Create memory trace
            memory = np.zeros(dimensions)
            for k, v in zip(keys, values):
                bound = np.fft.ifft(np.fft.fft(k) * np.fft.fft(v)).real
                memory += bound
            
            memory = normalize_semantic_pointer(memory)
            
            # Test retrieval
            test_idx = rng.randint(n)
            retrieved = np.fft.ifft(np.fft.fft(memory) * np.fft.fft(keys[test_idx].conj())).real
            
            sim = similarity(retrieved, values[test_idx])
            similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        results['similarities'].append((n, mean_sim))
    
    # Find capacity at different thresholds
    thresholds = [0.9, 0.7, 0.5, 0.3]
    for threshold in thresholds:
        capacity = 0
        for n, sim in results['similarities']:
            if sim >= threshold:
                capacity = n
            else:
                break
        results['capacities'][f'threshold_{threshold}'] = capacity
    
    return results


def create_transformation_matrix(source_vocab: Vocabulary,
                                target_vocab: Vocabulary,
                                mapping: Dict[str, str]) -> np.ndarray:
    """
    Create transformation matrix between vocabularies.
    
    Creates a matrix T such that T @ source ≈ target for
    corresponding vocabulary items.
    
    Parameters
    ----------
    source_vocab : Vocabulary
        Source vocabulary
    target_vocab : Vocabulary  
        Target vocabulary
    mapping : dict
        Mapping from source names to target names
        
    Returns
    -------
    array
        Transformation matrix
    """
    if source_vocab.dimension != target_vocab.dimension:
        raise ValueError("Vocabularies must have same dimensions")
    
    # Collect mapped vectors
    source_vecs = []
    target_vecs = []
    
    for src_name, tgt_name in mapping.items():
        if src_name in source_vocab.pointers and tgt_name in target_vocab.pointers:
            source_vecs.append(source_vocab.pointers[src_name].vector)
            target_vecs.append(target_vocab.pointers[tgt_name].vector)
    
    if not source_vecs:
        raise ValueError("No valid mappings found")
    
    # Stack vectors
    S = np.array(source_vecs).T  # dimensions x n_mappings
    T = np.array(target_vecs).T
    
    # Solve T @ S = T_desired
    # T = T_desired @ S^+ (pseudoinverse)
    transform = T @ np.linalg.pinv(S)
    
    return transform


def estimate_module_capacity(module: Module, n_items: int = 100,
                           similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Estimate storage capacity of a module.
    
    Parameters
    ----------
    module : Module
        Module to test
    n_items : int
        Number of items to test
    similarity_threshold : float
        Threshold for successful retrieval
        
    Returns
    -------
    dict
        Capacity estimate and performance metrics
    """
    if not hasattr(module, 'store') or not hasattr(module, 'query'):
        return {
            'capacity': 0,
            'error': 'Module does not support storage operations'
        }
    
    rng = np.random.RandomState(42)
    stored_items = []
    retrieval_accuracies = []
    
    for i in range(n_items):
        # Generate random item
        item = normalize_semantic_pointer(rng.randn(module.dimensions))
        stored_items.append(item)
        
        # Store it
        try:
            module.store(f"item_{i}", item)
        except Exception as e:
            logger.warning(f"Storage failed at item {i}: {e}")
            break
        
        # Test retrieval of random stored item
        if i > 0 and i % 10 == 0:
            test_idx = rng.randint(i)
            query = stored_items[test_idx] + 0.1 * rng.randn(module.dimensions)
            query = normalize_semantic_pointer(query)
            
            try:
                result = module.query(query)
                if result is not None:
                    sim = similarity(result, stored_items[test_idx])
                    retrieval_accuracies.append(sim)
            except Exception as e:
                logger.warning(f"Query failed: {e}")
    
    # Find capacity where retrieval drops below threshold
    capacity = len(stored_items)
    if retrieval_accuracies:
        for i, acc in enumerate(retrieval_accuracies):
            if acc < similarity_threshold:
                capacity = (i + 1) * 10  # Account for sampling interval
                break
    
    return {
        'capacity': capacity,
        'items_stored': len(stored_items),
        'mean_retrieval_accuracy': np.mean(retrieval_accuracies) if retrieval_accuracies else 0.0,
        'final_accuracy': retrieval_accuracies[-1] if retrieval_accuracies else 0.0
    }


def analyze_production_system(system: ProductionSystem,
                            test_context: Dict[str, Any],
                            max_cycles: int = 50) -> Dict[str, Any]:
    """
    Analyze production system behavior.
    
    Parameters
    ----------
    system : ProductionSystem
        System to analyze
    test_context : dict
        Test context with modules and vocabulary
    max_cycles : int
        Maximum execution cycles
        
    Returns
    -------
    dict
        Analysis results
    """
    # Set test context
    modules = test_context.get('modules', {})
    vocab = test_context.get('vocab', None)
    system.set_context(modules, vocab, **{k: v for k, v in test_context.items() if k not in ['modules', 'vocab']})
    
    # Evaluate all productions
    evaluations = system.evaluate_all()
    
    # Run system
    system.reset()
    cycles = system.run(max_cycles)
    fired = system.get_fired_productions()
    
    # Analyze production patterns
    production_stats = {}
    for prod in system.productions:
        production_stats[prod.name] = {
            'fired_count': prod._fired_count,
            'final_strength': prod._strength,
            'priority': prod.priority
        }
    
    # Detect cycles
    cycle_detected = False
    cycle_length = 0
    if len(fired) > 2:
        # Simple cycle detection
        for length in range(2, min(10, len(fired) // 2 + 1)):
            if len(fired) >= 2 * length and fired[-length:] == fired[-2*length:-length]:
                cycle_detected = True
                cycle_length = length
                break
    
    return {
        'total_productions': len(system.productions),
        'active_productions': len(evaluations),
        'cycles_executed': cycles,
        'productions_fired': len(fired),
        'unique_productions_fired': len(set(fired)),
        'firing_sequence': fired,
        'production_stats': production_stats,
        'cycle_detected': cycle_detected,
        'cycle_length': cycle_length
    }


def optimize_action_thresholds(action_set: ActionSet,
                              test_states: List[Dict[str, np.ndarray]],
                              desired_outputs: List[str],
                              learning_rate: float = 0.1,
                              n_epochs: int = 100) -> Dict[str, float]:
    """
    Optimize action selection thresholds.
    
    Uses gradient-free optimization to adjust thresholds
    for better action selection.
    
    Parameters
    ----------
    action_set : ActionSet
        Actions to optimize
    test_states : list
        List of module state dictionaries
    desired_outputs : list
        Desired action names for each state
    learning_rate : float
        Learning rate
    n_epochs : int
        Training epochs
        
    Returns
    -------
    dict
        Optimized thresholds
    """
    if len(test_states) != len(desired_outputs):
        raise ValueError("test_states and desired_outputs must have same length")
    
    # Track threshold adjustments
    threshold_history = {rule.name: [] for rule in action_set.rules}
    accuracy_history = []
    
    for epoch in range(n_epochs):
        correct = 0
        
        for states, desired in zip(test_states, desired_outputs):
            # Evaluate all actions
            utilities = []
            for rule in action_set.rules:
                # Simple state matching (would be more complex in practice)
                utility = rule.evaluate(states)
                utilities.append((rule.name, utility))
            
            # Get selected action
            utilities.sort(key=lambda x: x[1], reverse=True)
            selected = utilities[0][0] if utilities and utilities[0][1] > 0 else None
            
            # Update thresholds based on error
            if selected == desired:
                correct += 1
            else:
                # Adjust thresholds
                for rule in action_set.rules:
                    if rule.name == desired:
                        # Should have fired - decrease threshold
                        if hasattr(rule, 'threshold'):
                            rule.threshold *= (1 - learning_rate)
                    elif rule.name == selected:
                        # Shouldn't have fired - increase threshold
                        if hasattr(rule, 'threshold'):
                            rule.threshold *= (1 + learning_rate)
        
        accuracy = correct / len(test_states)
        accuracy_history.append(accuracy)
        
        # Record thresholds
        for rule in action_set.rules:
            if hasattr(rule, 'threshold'):
                threshold_history[rule.name].append(rule.threshold)
    
    # Extract final thresholds
    final_thresholds = {}
    for rule in action_set.rules:
        if hasattr(rule, 'threshold'):
            final_thresholds[rule.name] = rule.threshold
    
    return {
        'thresholds': final_thresholds,
        'final_accuracy': accuracy_history[-1],
        'accuracy_history': accuracy_history,
        'threshold_history': threshold_history
    }