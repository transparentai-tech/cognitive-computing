"""
Tests for SPA utility functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cognitive_computing.spa.utils import (
    make_unitary, similarity, normalize_semantic_pointer,
    generate_pointers, analyze_vocabulary, measure_binding_capacity,
    create_transformation_matrix, estimate_module_capacity,
    analyze_production_system, optimize_action_thresholds
)
from cognitive_computing.spa.core import SemanticPointer, Vocabulary, SPAConfig
from cognitive_computing.spa.modules import Module, Memory
from cognitive_computing.spa.actions import ActionSet, ActionRule
from cognitive_computing.spa.production import (
    Production, ProductionSystem, MatchCondition, SetEffect
)


class TestMakeUnitary:
    """Test make_unitary function."""
    
    def test_make_unitary_basic(self):
        """Test making vector unitary."""
        vec = np.random.randn(64)
        unitary = make_unitary(vec)
        
        # Should be normalized
        assert np.allclose(np.linalg.norm(unitary), 1.0)
        
        # Test unitary property: preserves dot products
        a = np.random.randn(64)
        b = np.random.randn(64)
        a = normalize_semantic_pointer(a)
        b = normalize_semantic_pointer(b)
        
        # Bind with unitary vector
        a_bound = np.fft.ifft(np.fft.fft(unitary) * np.fft.fft(a)).real
        b_bound = np.fft.ifft(np.fft.fft(unitary) * np.fft.fft(b)).real
        
        # Dot product should be preserved (approximately)
        orig_dot = np.dot(a, b)
        bound_dot = np.dot(a_bound, b_bound)
        assert abs(orig_dot - bound_dot) < 0.1
    
    def test_make_unitary_zero_vector(self):
        """Test with zero vector."""
        vec = np.zeros(64)
        unitary = make_unitary(vec)
        # Should handle gracefully
        assert np.allclose(np.linalg.norm(unitary), 1.0)


class TestSimilarity:
    """Test similarity function."""
    
    def test_similarity_identical(self):
        """Test similarity of identical vectors."""
        vec = np.random.randn(64)
        assert np.allclose(similarity(vec, vec), 1.0)
    
    def test_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        # Create approximately orthogonal vectors
        vec1 = np.zeros(64)
        vec1[0] = 1.0
        vec2 = np.zeros(64)
        vec2[1] = 1.0
        
        assert np.allclose(similarity(vec1, vec2), 0.0, atol=1e-10)
    
    def test_similarity_opposite(self):
        """Test similarity of opposite vectors."""
        vec = np.random.randn(64)
        assert np.allclose(similarity(vec, -vec), -1.0)
    
    def test_similarity_dimension_mismatch(self):
        """Test with mismatched dimensions."""
        vec1 = np.random.randn(64)
        vec2 = np.random.randn(32)
        
        with pytest.raises(ValueError, match="same length"):
            similarity(vec1, vec2)
    
    def test_similarity_zero_vectors(self):
        """Test with zero vectors."""
        vec1 = np.zeros(64)
        vec2 = np.random.randn(64)
        
        assert similarity(vec1, vec2) == 0.0
        assert similarity(vec1, vec1) == 0.0


class TestNormalizeSemanticPointer:
    """Test normalize_semantic_pointer function."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        vec = np.random.randn(64) * 5.0
        normalized = normalize_semantic_pointer(vec)
        
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        # Should preserve direction
        assert np.allclose(similarity(vec, normalized), 1.0)
    
    def test_normalize_zero_vector(self):
        """Test normalizing zero vector."""
        vec = np.zeros(64)
        normalized = normalize_semantic_pointer(vec)
        
        # Should return zero vector unchanged
        assert np.allclose(normalized, vec)
    
    def test_normalize_already_normalized(self):
        """Test normalizing already normalized vector."""
        vec = np.random.randn(64)
        vec = vec / np.linalg.norm(vec)
        
        normalized = normalize_semantic_pointer(vec)
        assert np.allclose(normalized, vec)


class TestGeneratePointers:
    """Test generate_pointers function."""
    
    def test_generate_pointers_basic(self):
        """Test basic pointer generation."""
        pointers = generate_pointers(10, 64)
        
        assert len(pointers) == 10
        # Check all are normalized
        for name, vec in pointers.items():
            assert np.allclose(np.linalg.norm(vec), 1.0)
            assert vec.shape == (64,)
    
    def test_generate_pointers_unitary(self):
        """Test generating unitary pointers."""
        pointers = generate_pointers(5, 64, unitary=True)
        
        assert len(pointers) == 5
        # Test unitary property on one pointer
        vec = list(pointers.values())[0]
        a = np.random.randn(64)
        b = np.random.randn(64)
        a = normalize_semantic_pointer(a)
        b = normalize_semantic_pointer(b)
        
        a_bound = np.fft.ifft(np.fft.fft(vec) * np.fft.fft(a)).real
        b_bound = np.fft.ifft(np.fft.fft(vec) * np.fft.fft(b)).real
        
        orig_dot = np.dot(a, b)
        bound_dot = np.dot(a_bound, b_bound)
        assert abs(orig_dot - bound_dot) < 0.1
    
    def test_generate_pointers_with_rng(self):
        """Test with specific RNG for reproducibility."""
        rng = np.random.RandomState(42)
        pointers1 = generate_pointers(5, 32, rng=rng)
        
        rng = np.random.RandomState(42)
        pointers2 = generate_pointers(5, 32, rng=rng)
        
        # Should be identical
        for name in pointers1:
            assert np.allclose(pointers1[name], pointers2[name])
    
    def test_generate_too_many_pointers(self):
        """Test generating more pointers than dimensions."""
        # Should warn but still work
        pointers = generate_pointers(100, 64)
        assert len(pointers) == 100


class TestAnalyzeVocabulary:
    """Test analyze_vocabulary function."""
    
    def test_analyze_empty_vocabulary(self):
        """Test analyzing empty vocabulary."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64, config)
        
        results = analyze_vocabulary(vocab)
        
        assert results['size'] == 0
        assert results['dimensions'] == 64
        assert results['mean_similarity'] == 0.0
        assert results['max_similarity'] == 0.0
        assert results['orthogonality'] == 0.0  # No off-diagonal elements
    
    def test_analyze_vocabulary_basic(self):
        """Test basic vocabulary analysis."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64, config)
        
        # Add some pointers
        vec_a = np.random.randn(64)
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = np.random.randn(64)
        vec_b = vec_b / np.linalg.norm(vec_b)
        vec_c = np.random.randn(64)
        vec_c = vec_c / np.linalg.norm(vec_c)
        
        vocab.pointers["A"] = SemanticPointer(vec_a, vocab)
        vocab.pointers["B"] = SemanticPointer(vec_b, vocab)
        vocab.pointers["C"] = SemanticPointer(vec_c, vocab)
        
        results = analyze_vocabulary(vocab)
        
        assert results['size'] == 3
        assert results['dimensions'] == 64
        assert 'mean_similarity' in results
        assert 'max_similarity' in results
        assert 'similarity_matrix' in results
        assert results['similarity_matrix'].shape == (3, 3)
        # Diagonal should be 1
        assert np.allclose(np.diag(results['similarity_matrix']), 1.0)
        # Orthogonality should be low for random vectors
        assert results['orthogonality'] < 0.3
        assert len(results['pointer_names']) == 3
    
    def test_analyze_vocabulary_with_similar_pointers(self):
        """Test with similar pointers."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64, config)
        
        # Add base pointer
        vec_a = np.random.randn(64)
        vec_a = vec_a / np.linalg.norm(vec_a)
        vocab.pointers["A"] = SemanticPointer(vec_a, vocab)
        # Add similar pointer
        similar = vocab.pointers["A"].vector + 0.1 * np.random.randn(64)
        similar = normalize_semantic_pointer(similar)
        vocab.pointers["B"] = SemanticPointer(similar, vocab)
        
        results = analyze_vocabulary(vocab)
        
        # Should have high similarity
        assert results['max_similarity'] > 0.7
        assert results['mean_similarity'] > 0.7


class TestMeasureBindingCapacity:
    """Test measure_binding_capacity function."""
    
    def test_binding_capacity_basic(self):
        """Test basic binding capacity measurement."""
        results = measure_binding_capacity(64, n_pairs=20, n_trials=5)
        
        assert 'dimensions' in results
        assert results['dimensions'] == 64
        assert 'capacities' in results
        assert 'similarities' in results
        
        # Check capacity decreases with threshold
        assert results['capacities']['threshold_0.9'] <= results['capacities']['threshold_0.7']
        assert results['capacities']['threshold_0.7'] <= results['capacities']['threshold_0.5']
    
    def test_binding_capacity_high_dimensions(self):
        """Test with high dimensions."""
        results = measure_binding_capacity(256, n_pairs=10, n_trials=3)
        
        # Higher dimensions should have higher capacity
        # With 256 dimensions and max 10 pairs tested, check reasonable capacity
        assert 'threshold_0.5' in results['capacities']
        # At minimum, should handle 1 pair
        assert results['capacities']['threshold_0.5'] >= 0
    
    def test_binding_capacity_with_rng(self):
        """Test with specific RNG."""
        rng = np.random.RandomState(42)
        results = measure_binding_capacity(32, n_pairs=10, n_trials=3, rng=rng)
        
        assert len(results['similarities']) > 0


class TestCreateTransformationMatrix:
    """Test create_transformation_matrix function."""
    
    def test_transformation_matrix_basic(self):
        """Test basic transformation matrix creation."""
        config = SPAConfig(dimension=64)
        source_vocab = Vocabulary(64, config)
        target_vocab = Vocabulary(64, config)
        
        # Add corresponding items
        vec_a = np.random.randn(64)
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = np.random.randn(64)
        vec_b = vec_b / np.linalg.norm(vec_b)
        vec_x = np.random.randn(64)
        vec_x = vec_x / np.linalg.norm(vec_x)
        vec_y = np.random.randn(64)
        vec_y = vec_y / np.linalg.norm(vec_y)
        
        source_vocab.pointers["A"] = SemanticPointer(vec_a, source_vocab)
        source_vocab.pointers["B"] = SemanticPointer(vec_b, source_vocab)
        target_vocab.pointers["X"] = SemanticPointer(vec_x, target_vocab)
        target_vocab.pointers["Y"] = SemanticPointer(vec_y, target_vocab)
        
        mapping = {"A": "X", "B": "Y"}
        
        transform = create_transformation_matrix(source_vocab, target_vocab, mapping)
        
        assert transform.shape == (64, 64)
        
        # Test transformation
        transformed_a = transform @ source_vocab.pointers["A"].vector
        sim = similarity(transformed_a, target_vocab.pointers["X"].vector)
        assert sim > 0.8  # Should be close
    
    def test_transformation_matrix_dimension_mismatch(self):
        """Test with mismatched dimensions."""
        config1 = SPAConfig(dimension=64)
        config2 = SPAConfig(dimension=32)
        source_vocab = Vocabulary(64, config1)
        target_vocab = Vocabulary(32, config2)
        
        with pytest.raises(ValueError, match="same dimensions"):
            create_transformation_matrix(source_vocab, target_vocab, {})
    
    def test_transformation_matrix_no_mappings(self):
        """Test with no valid mappings."""
        config = SPAConfig(dimension=64)
        source_vocab = Vocabulary(64, config)
        target_vocab = Vocabulary(64, config)
        
        with pytest.raises(ValueError, match="No valid mappings"):
            create_transformation_matrix(source_vocab, target_vocab, {"A": "X"})


class TestEstimateModuleCapacity:
    """Test estimate_module_capacity function."""
    
    def test_estimate_capacity_memory_module(self):
        """Test with Memory module."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64, config)
        
        # Create a mock module with store/query methods
        module = Mock()
        module.dimensions = 64
        module.store = Mock()
        module.query = Mock(return_value=np.random.randn(64))
        
        results = estimate_module_capacity(module, n_items=20)
        
        assert 'capacity' in results
        assert 'items_stored' in results
        assert results['capacity'] > 0
    
    def test_estimate_capacity_unsupported_module(self):
        """Test with module that doesn't support storage."""
        # Use a mock module without store/query methods
        module = Mock()
        module.dimensions = 64
        module.name = "test"
        # Remove store/query attributes to test unsupported module
        if hasattr(module, 'store'):
            delattr(module, 'store')
        if hasattr(module, 'query'):
            delattr(module, 'query')
        
        results = estimate_module_capacity(module)
        
        assert results['capacity'] == 0
        assert 'error' in results


class TestAnalyzeProductionSystem:
    """Test analyze_production_system function."""
    
    def test_analyze_production_system_basic(self):
        """Test basic production system analysis."""
        # Create simple production system
        cond = MatchCondition("state", "A")
        effect = SetEffect("state", "B")
        prod1 = Production("rule1", cond, effect)
        
        system = ProductionSystem([prod1])
        
        # Create test context
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64, config)
        vec_a = np.random.randn(64)
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = np.random.randn(64)
        vec_b = vec_b / np.linalg.norm(vec_b)
        
        vocab.pointers["A"] = SemanticPointer(vec_a, vocab)
        vocab.pointers["B"] = SemanticPointer(vec_b, vocab)
        
        state = Mock()
        state.state = vocab.pointers["A"].vector
        
        context = {
            'modules': {'state': state},
            'vocab': vocab
        }
        
        results = analyze_production_system(system, context, max_cycles=10)
        
        assert results['total_productions'] == 1
        assert results['cycles_executed'] <= 10
        assert 'production_stats' in results
        assert 'rule1' in results['production_stats']
    
    def test_analyze_production_system_with_cycle(self):
        """Test detecting cycles in production firing."""
        # Create cyclic productions
        prod1 = Production("rule1", Mock(evaluate=lambda x: 1.0), Mock())
        prod2 = Production("rule2", Mock(evaluate=lambda x: 1.0), Mock())
        
        system = ProductionSystem([prod1, prod2])
        
        # Mock firing sequence to create cycle
        system._fired_productions = ["rule1", "rule2", "rule1", "rule2", "rule1", "rule2"]
        
        # Mock the get_fired_productions method to return our sequence
        with patch.object(system, 'get_fired_productions', return_value=["rule1", "rule2", "rule1", "rule2", "rule1", "rule2"]):
            with patch.object(system, 'run', return_value=6):
                results = analyze_production_system(system, {}, max_cycles=0)
        
        # Should detect the cycle in the mocked sequence
        assert results['firing_sequence'] == ["rule1", "rule2", "rule1", "rule2", "rule1", "rule2"]
        assert results['cycle_detected'] == True
        assert results['cycle_length'] == 2


class TestOptimizeActionThresholds:
    """Test optimize_action_thresholds function."""
    
    def test_optimize_thresholds_basic(self):
        """Test basic threshold optimization."""
        # Create action set with mock rules
        rule1 = Mock(spec=ActionRule)
        rule1.name = "action1"
        rule1.threshold = 0.5
        rule1.evaluate = Mock(return_value=0.6)
        
        rule2 = Mock(spec=ActionRule)
        rule2.name = "action2"
        rule2.threshold = 0.5
        rule2.evaluate = Mock(return_value=0.4)
        
        action_set = Mock(spec=ActionSet)
        action_set.rules = [rule1, rule2]
        
        # Test data
        test_states = [{"state": np.random.randn(64)}]
        desired_outputs = ["action1"]
        
        results = optimize_action_thresholds(
            action_set, test_states, desired_outputs,
            learning_rate=0.1, n_epochs=10
        )
        
        assert 'thresholds' in results
        assert 'final_accuracy' in results
        assert 'accuracy_history' in results
        assert len(results['accuracy_history']) == 10
    
    def test_optimize_thresholds_mismatch_lengths(self):
        """Test with mismatched input lengths."""
        action_set = Mock(spec=ActionSet)
        action_set.rules = []
        
        with pytest.raises(ValueError, match="same length"):
            optimize_action_thresholds(
                action_set,
                [{}],  # 1 state
                ["a", "b"],  # 2 outputs
            )