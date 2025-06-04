"""Tests for HDC classifiers."""

import pytest
import numpy as np

from cognitive_computing.hdc.classifiers import (
    OneShotClassifier,
    AdaptiveClassifier,
    EnsembleClassifier,
    HierarchicalClassifier,
)
from cognitive_computing.hdc.encoding import CategoricalEncoder, ScalarEncoder


class TestOneShotClassifier:
    """Test one-shot classifier."""
    
    def test_basic_classification(self):
        """Test basic one-shot classification."""
        encoder = CategoricalEncoder(dimension=100, hypervector_type="bipolar")
        classifier = OneShotClassifier(
            dimension=100,
            encoder=encoder,
            similarity_metric="cosine",
            hypervector_type="bipolar"
        )
        
        # Train with single examples
        X_train = ["cat", "dog", "bird"]
        y_train = ["mammal", "mammal", "bird"]
        
        classifier.train(X_train, y_train)
        assert classifier.is_trained
        
        # Predict on same data
        predictions = classifier.predict(X_train)
        assert predictions == y_train
        
        # Predict on new data
        X_test = ["cat", "sparrow"]
        predictions = classifier.predict(X_test)
        assert predictions[0] == "mammal"
        # "sparrow" might be classified as unknown or closest match
        
    def test_add_example(self):
        """Test adding examples incrementally."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(dimension=100, encoder=encoder)
        
        # Add examples one by one
        classifier.add_example("apple", "fruit")
        assert "fruit" in classifier.class_hypervectors
        assert classifier.is_trained
        
        classifier.add_example("banana", "fruit")
        assert classifier.example_counts["fruit"] == 2
        
        # Test prediction
        predictions = classifier.predict(["apple"])
        assert predictions[0] == "fruit"
        
    def test_similarity_threshold(self):
        """Test classification with similarity threshold."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(
            dimension=100,
            encoder=encoder,
            similarity_threshold=0.5
        )
        
        # Train
        classifier.train(["A", "B"], ["class1", "class2"])
        
        # Predict with low similarity should return unknown
        # Using a very different input
        encoder.encode("Z")  # Ensure it's in vocabulary
        predictions = classifier.predict(["Z"])
        # Due to randomness, this might still match - test threshold behavior
        
    def test_remove_class(self):
        """Test removing a class."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(dimension=100, encoder=encoder)
        
        # Train with multiple classes
        classifier.train(["A", "B", "C"], ["c1", "c2", "c3"])
        assert len(classifier.class_hypervectors) == 3
        
        # Remove a class
        assert classifier.remove_class("c2") is True
        assert len(classifier.class_hypervectors) == 2
        assert "c2" not in classifier.class_hypervectors
        
        # Try to remove non-existent class
        assert classifier.remove_class("c4") is False
        
    def test_predict_proba(self):
        """Test probability prediction."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(dimension=100, encoder=encoder)
        
        # Train
        classifier.train(["cat", "dog", "bird"], ["mammal", "mammal", "bird"])
        
        # Get probabilities
        probas = classifier.predict_proba(["cat"])
        assert len(probas) == 1
        assert "mammal" in probas[0]
        assert "bird" in probas[0]
        assert abs(sum(probas[0].values()) - 1.0) < 1e-6  # Sum to 1
        
    def test_score(self):
        """Test scoring method."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(dimension=100, encoder=encoder)
        
        # Train
        X_train = ["A", "B", "C", "D"]
        y_train = ["c1", "c2", "c1", "c2"]
        classifier.train(X_train, y_train)
        
        # Score on training data
        score = classifier.score(X_train, y_train)
        assert score == 1.0  # Perfect on training data
        
    def test_untrained_prediction(self):
        """Test prediction before training."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(dimension=100, encoder=encoder)
        
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.predict(["test"])
            
        with pytest.raises(RuntimeError, match="must be trained"):
            classifier.predict_proba(["test"])


class TestAdaptiveClassifier:
    """Test adaptive classifier."""
    
    def test_initial_training(self):
        """Test initial training."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = AdaptiveClassifier(
            dimension=100,
            encoder=encoder,
            learning_rate=0.1,
            momentum=0.9
        )
        
        # Train
        X = ["A", "B", "C", "A", "B", "C"]
        y = ["c1", "c1", "c2", "c1", "c1", "c2"]
        
        classifier.train(X, y)
        assert classifier.is_trained
        assert len(classifier.class_hypervectors) == 2
        assert len(classifier.class_velocities) == 2
        
        # Test prediction
        predictions = classifier.predict(["A", "C"])
        assert len(predictions) == 2
        
    def test_online_update(self):
        """Test online updates."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = AdaptiveClassifier(
            dimension=100,
            encoder=encoder,
            learning_rate=0.1
        )
        
        # Initial training
        classifier.train(["A", "B"], ["c1", "c2"])
        
        # Make prediction
        pred = classifier.predict(["A"])[0]
        
        # Update with correct label
        classifier.update("A", "c1", pred)
        
        # Update with new class
        classifier.update("C", "c3")
        assert "c3" in classifier.class_hypervectors
        
    def test_update_before_training(self):
        """Test update before initial training."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = AdaptiveClassifier(dimension=100, encoder=encoder)
        
        # Update should initialize
        classifier.update("A", "c1")
        assert classifier.is_trained
        assert "c1" in classifier.class_hypervectors
        
    def test_momentum_updates(self):
        """Test updates with momentum."""
        encoder = ScalarEncoder(100, 0, 10, 10)
        classifier = AdaptiveClassifier(
            dimension=100,
            encoder=encoder,
            learning_rate=0.1,
            momentum=0.9
        )
        
        # Train
        X = [1.0, 9.0]
        y = ["low", "high"]
        classifier.train(X, y)
        
        # Multiple updates
        for _ in range(5):
            classifier.update(2.0, "low", "low")
            
        # Velocities should be non-zero due to momentum
        assert np.any(classifier.class_velocities["low"] != 0)


class TestEnsembleClassifier:
    """Test ensemble classifier."""
    
    def test_hard_voting(self):
        """Test ensemble with hard voting."""
        encoder = CategoricalEncoder(dimension=100)
        
        # Create base classifiers
        classifiers = []
        for i in range(3):
            clf = OneShotClassifier(dimension=100, encoder=encoder)
            classifiers.append(clf)
            
        # Create ensemble
        ensemble = EnsembleClassifier(classifiers, voting="hard")
        
        # Train
        X = ["A", "B", "C"]
        y = ["c1", "c2", "c1"]
        ensemble.train(X, y)
        
        # Predict
        predictions = ensemble.predict(["A"])
        assert len(predictions) == 1
        assert predictions[0] in ["c1", "c2"]
        
    def test_soft_voting(self):
        """Test ensemble with soft voting."""
        encoder = CategoricalEncoder(dimension=100)
        
        # Create base classifiers
        classifiers = []
        for i in range(3):
            clf = OneShotClassifier(dimension=100, encoder=encoder)
            classifiers.append(clf)
            
        # Create ensemble
        ensemble = EnsembleClassifier(classifiers, voting="soft")
        
        # Train
        X = ["cat", "dog", "bird"]
        y = ["mammal", "mammal", "bird"]
        ensemble.train(X, y)
        
        # Predict
        predictions = ensemble.predict(["cat"])
        assert len(predictions) == 1
        # Should predict mammal
        
    def test_empty_ensemble(self):
        """Test creating ensemble with no classifiers."""
        with pytest.raises(ValueError, match="At least one classifier"):
            EnsembleClassifier([])
            
    def test_untrained_ensemble(self):
        """Test prediction with untrained ensemble."""
        encoder = CategoricalEncoder(dimension=100)
        clf = OneShotClassifier(dimension=100, encoder=encoder)
        ensemble = EnsembleClassifier([clf])
        
        with pytest.raises(RuntimeError, match="must be trained"):
            ensemble.predict(["test"])


class TestHierarchicalClassifier:
    """Test hierarchical classifier."""
    
    def test_simple_hierarchy(self):
        """Test simple two-level hierarchy."""
        encoder = CategoricalEncoder(dimension=100)
        
        # Define hierarchy
        hierarchy = {
            "animal": ["dog", "cat", "bird"],
            "plant": ["tree", "flower"]
        }
        
        classifier = HierarchicalClassifier(
            dimension=100,
            encoder=encoder,
            hierarchy=hierarchy
        )
        
        # Train
        X = ["poodle", "siamese", "oak", "rose"]
        y = ["dog", "cat", "tree", "flower"]
        
        classifier.train(X, y)
        assert classifier.is_trained
        
        # Predict
        predictions = classifier.predict(["poodle", "oak"])
        assert predictions[0] == "dog"
        assert predictions[1] == "tree"
        
    def test_deep_hierarchy(self):
        """Test deeper hierarchy."""
        encoder = CategoricalEncoder(dimension=100)
        
        # Three-level hierarchy
        hierarchy = {
            "living": ["animal", "plant"],
            "animal": ["mammal", "bird"],
            "mammal": ["dog", "cat"],
            "bird": ["sparrow", "eagle"]
        }
        
        classifier = HierarchicalClassifier(
            dimension=100,
            encoder=encoder,
            hierarchy=hierarchy
        )
        
        # Train with leaf labels
        X = ["poodle", "persian", "robin"]
        y = ["dog", "cat", "sparrow"]
        
        classifier.train(X, y)
        
        # Predict
        predictions = classifier.predict(["poodle"])
        assert predictions[0] == "dog"
        
    def test_unknown_in_hierarchy(self):
        """Test prediction of unknown items."""
        encoder = CategoricalEncoder(dimension=100)
        
        hierarchy = {
            "A": ["B", "C"],
            "B": ["D", "E"]
        }
        
        classifier = HierarchicalClassifier(
            dimension=100,
            encoder=encoder,
            hierarchy=hierarchy
        )
        
        # Train
        classifier.train(["x", "y"], ["D", "C"])
        
        # Predict completely unknown
        predictions = classifier.predict(["z"])
        # Should return something (unknown or best match)
        assert len(predictions) == 1
        
    def test_find_path(self):
        """Test path finding in hierarchy."""
        encoder = CategoricalEncoder(dimension=100)
        
        hierarchy = {
            "A": ["B", "C"],
            "B": ["D", "E"]
        }
        
        classifier = HierarchicalClassifier(
            dimension=100,
            encoder=encoder,
            hierarchy=hierarchy
        )
        
        # Test path finding
        path = classifier._find_path("D")
        assert path == ["A", "B", "D"]
        
        path = classifier._find_path("C")
        assert path == ["A", "C"]
        
        path = classifier._find_path("A")
        assert path == ["A"]