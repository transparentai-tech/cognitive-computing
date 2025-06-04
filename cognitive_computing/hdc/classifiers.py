"""
HDC-based classification algorithms.

This module implements various classifiers using hyperdimensional computing
principles for robust and efficient classification.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import logging

from cognitive_computing.hdc.core import HDC, HDCConfig
from cognitive_computing.hdc.encoding import Encoder
from cognitive_computing.hdc.operations import (
    bundle_hypervectors,
    similarity,
    normalize_hypervector,
)
from cognitive_computing.hdc.item_memory import ItemMemory

logger = logging.getLogger(__name__)


class HDClassifier(ABC):
    """Abstract base class for HDC classifiers."""
    
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        similarity_metric: str = "cosine",
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize HDC classifier.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        encoder : Encoder
            Encoder for input data
        similarity_metric : str
            Similarity metric for classification
        hypervector_type : str
            Type of hypervectors
        """
        self.dimension = dimension
        self.encoder = encoder
        self.similarity_metric = similarity_metric
        self.hypervector_type = hypervector_type
        self.class_hypervectors: Dict[str, np.ndarray] = {}
        self.is_trained = False
        
    @abstractmethod
    def train(self, X: List[any], y: List[str]) -> None:
        """
        Train the classifier.
        
        Parameters
        ----------
        X : List[any]
            Training data
        y : List[str]
            Training labels
        """
        pass
        
    @abstractmethod
    def predict(self, X: List[any]) -> List[str]:
        """
        Predict labels for new data.
        
        Parameters
        ----------
        X : List[any]
            Data to classify
            
        Returns
        -------
        List[str]
            Predicted labels
        """
        pass
        
    def predict_proba(self, X: List[any]) -> List[Dict[str, float]]:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : List[any]
            Data to classify
            
        Returns
        -------
        List[Dict[str, float]]
            Class probabilities for each sample
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
            
        probabilities = []
        
        for x in X:
            # Encode input
            hv = self.encoder.encode(x)
            
            # Calculate similarities to all classes
            similarities = {}
            for class_label, class_hv in self.class_hypervectors.items():
                sim = similarity(hv, class_hv, metric=self.similarity_metric)
                similarities[class_label] = sim
                
            # Convert to probabilities using softmax
            # First shift to make all positive
            min_sim = min(similarities.values())
            shifted = {k: v - min_sim + 1e-10 for k, v in similarities.items()}
            
            # Apply exponential
            exp_sims = {k: np.exp(v) for k, v in shifted.items()}
            total = sum(exp_sims.values())
            
            # Normalize
            probs = {k: v / total for k, v in exp_sims.items()}
            probabilities.append(probs)
            
        return probabilities
        
    def score(self, X: List[any], y: List[str]) -> float:
        """
        Calculate classification accuracy.
        
        Parameters
        ----------
        X : List[any]
            Test data
        y : List[str]
            True labels
            
        Returns
        -------
        float
            Accuracy score
        """
        predictions = self.predict(X)
        correct = sum(pred == true for pred, true in zip(predictions, y))
        return correct / len(y) if y else 0.0


class OneShotClassifier(HDClassifier):
    """
    One-shot HDC classifier.
    
    Learns from single examples per class and can be updated online.
    """
    
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        similarity_metric: str = "cosine",
        hypervector_type: str = "bipolar",
        similarity_threshold: float = 0.0
    ):
        """
        Initialize one-shot classifier.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        encoder : Encoder
            Encoder for input data
        similarity_metric : str
            Similarity metric
        hypervector_type : str
            Type of hypervectors
        similarity_threshold : float
            Minimum similarity for classification
        """
        super().__init__(dimension, encoder, similarity_metric, hypervector_type)
        self.similarity_threshold = similarity_threshold
        self.example_counts: Dict[str, int] = defaultdict(int)
        
    def train(self, X: List[any], y: List[str]) -> None:
        """Train on examples (can be called multiple times)."""
        for x, label in zip(X, y):
            self.add_example(x, label)
            
        self.is_trained = True
        
    def add_example(self, x: any, label: str) -> None:
        """
        Add a single training example.
        
        Parameters
        ----------
        x : any
            Training example
        label : str
            Class label
        """
        # Encode example
        hv = self.encoder.encode(x)
        
        if label in self.class_hypervectors:
            # Update existing class vector by bundling
            self.class_hypervectors[label] = bundle_hypervectors(
                [self.class_hypervectors[label], hv],
                hypervector_type=self.hypervector_type
            )
        else:
            # Create new class vector
            self.class_hypervectors[label] = hv
            
        self.example_counts[label] += 1
        self.is_trained = True
        
    def predict(self, X: List[any]) -> List[str]:
        """Predict labels for new data."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
            
        predictions = []
        
        for x in X:
            # Encode input
            hv = self.encoder.encode(x)
            
            # Find most similar class
            best_label = None
            best_similarity = -np.inf
            
            for class_label, class_hv in self.class_hypervectors.items():
                sim = similarity(hv, class_hv, metric=self.similarity_metric)
                
                if sim > best_similarity and sim >= self.similarity_threshold:
                    best_similarity = sim
                    best_label = class_label
                    
            # If no class meets threshold, predict None or default
            predictions.append(best_label if best_label else "unknown")
            
        return predictions
        
    def remove_class(self, label: str) -> bool:
        """
        Remove a class from the classifier.
        
        Parameters
        ----------
        label : str
            Class label to remove
            
        Returns
        -------
        bool
            True if removed, False if not found
        """
        if label in self.class_hypervectors:
            del self.class_hypervectors[label]
            del self.example_counts[label]
            return True
        return False


class AdaptiveClassifier(HDClassifier):
    """
    Adaptive HDC classifier with online learning.
    
    Updates class prototypes based on prediction feedback.
    """
    
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        similarity_metric: str = "cosine",
        hypervector_type: str = "bipolar",
        learning_rate: float = 0.1,
        momentum: float = 0.9
    ):
        """
        Initialize adaptive classifier.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        encoder : Encoder
            Encoder for input data
        similarity_metric : str
            Similarity metric
        hypervector_type : str
            Type of hypervectors
        learning_rate : float
            Learning rate for updates
        momentum : float
            Momentum for updates
        """
        super().__init__(dimension, encoder, similarity_metric, hypervector_type)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.class_velocities: Dict[str, np.ndarray] = {}
        
    def train(self, X: List[any], y: List[str]) -> None:
        """Initial training."""
        # Group by class
        class_examples = defaultdict(list)
        for x, label in zip(X, y):
            class_examples[label].append(x)
            
        # Create initial class vectors
        for label, examples in class_examples.items():
            # Encode all examples
            hvs = [self.encoder.encode(x) for x in examples]
            
            # Bundle to create class prototype
            self.class_hypervectors[label] = bundle_hypervectors(
                hvs,
                hypervector_type=self.hypervector_type
            )
            
            # Initialize velocity
            self.class_velocities[label] = np.zeros(self.dimension)
            
        self.is_trained = True
        
    def predict(self, X: List[any]) -> List[str]:
        """Predict labels."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
            
        predictions = []
        
        for x in X:
            hv = self.encoder.encode(x)
            
            # Find most similar class
            best_label = None
            best_similarity = -np.inf
            
            for class_label, class_hv in self.class_hypervectors.items():
                sim = similarity(hv, class_hv, metric=self.similarity_metric)
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_label = class_label
                    
            predictions.append(best_label)
            
        return predictions
        
    def update(self, x: any, true_label: str, predicted_label: Optional[str] = None) -> None:
        """
        Update classifier based on feedback.
        
        Parameters
        ----------
        x : any
            Input example
        true_label : str
            True class label
        predicted_label : str, optional
            Predicted label (will predict if not provided)
        """
        if not self.is_trained:
            # Initialize with first example
            hv = self.encoder.encode(x)
            self.class_hypervectors[true_label] = hv
            self.class_velocities[true_label] = np.zeros(self.dimension)
            self.is_trained = True
            return
            
        # Get prediction if not provided
        if predicted_label is None:
            predicted_label = self.predict([x])[0]
            
        # Encode input
        hv = self.encoder.encode(x)
        
        # Update true class (move towards example)
        if true_label in self.class_hypervectors:
            true_class_hv = self.class_hypervectors[true_label]
            gradient = hv - true_class_hv
            
            # Update with momentum
            self.class_velocities[true_label] = (
                self.momentum * self.class_velocities[true_label] +
                self.learning_rate * gradient
            )
            
            # Update class vector
            new_hv = true_class_hv + self.class_velocities[true_label]
            self.class_hypervectors[true_label] = normalize_hypervector(
                new_hv,
                self.hypervector_type
            )
        else:
            # New class
            self.class_hypervectors[true_label] = hv
            self.class_velocities[true_label] = np.zeros(self.dimension)
            
        # Update wrong prediction (move away from example)
        if predicted_label != true_label and predicted_label in self.class_hypervectors:
            pred_class_hv = self.class_hypervectors[predicted_label]
            gradient = pred_class_hv - hv
            
            self.class_velocities[predicted_label] = (
                self.momentum * self.class_velocities[predicted_label] +
                self.learning_rate * gradient * 0.5  # Smaller step for negative update
            )
            
            new_hv = pred_class_hv + self.class_velocities[predicted_label]
            self.class_hypervectors[predicted_label] = normalize_hypervector(
                new_hv,
                self.hypervector_type
            )


class EnsembleClassifier(HDClassifier):
    """
    Ensemble of HDC classifiers.
    
    Combines predictions from multiple HDC classifiers using voting.
    """
    
    def __init__(
        self,
        classifiers: List[HDClassifier],
        voting: str = "hard"
    ):
        """
        Initialize ensemble classifier.
        
        Parameters
        ----------
        classifiers : List[HDClassifier]
            List of base classifiers
        voting : str
            Voting method: "hard" or "soft"
        """
        if not classifiers:
            raise ValueError("At least one classifier required")
            
        # Get parameters from first classifier
        first = classifiers[0]
        super().__init__(
            first.dimension,
            first.encoder,
            first.similarity_metric,
            first.hypervector_type
        )
        
        self.classifiers = classifiers
        self.voting = voting
        
    def train(self, X: List[any], y: List[str]) -> None:
        """Train all classifiers in ensemble."""
        for classifier in self.classifiers:
            classifier.train(X, y)
            
        self.is_trained = True
        
    def predict(self, X: List[any]) -> List[str]:
        """Predict using ensemble voting."""
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction")
            
        if self.voting == "hard":
            return self._hard_voting(X)
        else:
            return self._soft_voting(X)
            
    def _hard_voting(self, X: List[any]) -> List[str]:
        """Hard voting (majority vote)."""
        # Get predictions from all classifiers
        all_predictions = []
        for classifier in self.classifiers:
            predictions = classifier.predict(X)
            all_predictions.append(predictions)
            
        # Vote for each sample
        ensemble_predictions = []
        for i in range(len(X)):
            votes = defaultdict(int)
            for predictions in all_predictions:
                votes[predictions[i]] += 1
                
            # Get most voted class
            best_label = max(votes, key=votes.get)
            ensemble_predictions.append(best_label)
            
        return ensemble_predictions
        
    def _soft_voting(self, X: List[any]) -> List[str]:
        """Soft voting (average probabilities)."""
        # Get probabilities from all classifiers
        all_probas = []
        for classifier in self.classifiers:
            probas = classifier.predict_proba(X)
            all_probas.append(probas)
            
        # Average probabilities for each sample
        ensemble_predictions = []
        for i in range(len(X)):
            # Collect all classes
            all_classes = set()
            for probas in all_probas:
                all_classes.update(probas[i].keys())
                
            # Average probabilities
            avg_probs = {}
            for class_label in all_classes:
                probs = [probas[i].get(class_label, 0.0) for probas in all_probas]
                avg_probs[class_label] = np.mean(probs)
                
            # Get class with highest average probability
            best_label = max(avg_probs, key=avg_probs.get)
            ensemble_predictions.append(best_label)
            
        return ensemble_predictions


class HierarchicalClassifier(HDClassifier):
    """
    Hierarchical HDC classifier.
    
    Classifies using a hierarchy of classes (e.g., animal -> mammal -> dog).
    """
    
    def __init__(
        self,
        dimension: int,
        encoder: Encoder,
        hierarchy: Dict[str, List[str]],
        similarity_metric: str = "cosine",
        hypervector_type: str = "bipolar"
    ):
        """
        Initialize hierarchical classifier.
        
        Parameters
        ----------
        dimension : int
            Hypervector dimension
        encoder : Encoder
            Encoder for input data
        hierarchy : Dict[str, List[str]]
            Class hierarchy (parent -> children)
        similarity_metric : str
            Similarity metric
        hypervector_type : str
            Type of hypervectors
        """
        super().__init__(dimension, encoder, similarity_metric, hypervector_type)
        self.hierarchy = hierarchy
        self.level_classifiers: Dict[str, HDClassifier] = {}
        
        # Build hierarchy levels
        self._build_hierarchy()
        
    def _build_hierarchy(self) -> None:
        """Build classifiers for each level of hierarchy."""
        # Find root nodes (no parents)
        all_children = set()
        for children in self.hierarchy.values():
            all_children.update(children)
            
        self.roots = set(self.hierarchy.keys()) - all_children
        
        # Create classifier for root level
        self.level_classifiers["root"] = OneShotClassifier(
            self.dimension,
            self.encoder,
            self.similarity_metric,
            self.hypervector_type
        )
        
        # Create classifiers for each parent node
        for parent in self.hierarchy:
            self.level_classifiers[parent] = OneShotClassifier(
                self.dimension,
                self.encoder,
                self.similarity_metric,
                self.hypervector_type
            )
            
    def train(self, X: List[any], y: List[str]) -> None:
        """Train hierarchical classifier."""
        # Group examples by their position in hierarchy
        level_data = defaultdict(lambda: ([], []))
        
        for x, label in zip(X, y):
            # Find path from root to leaf
            path = self._find_path(label)
            
            # Add to appropriate level classifiers
            if len(path) == 1:
                # This is a root node itself
                level_data["root"][0].append(x)
                level_data["root"][1].append(label)
            else:
                # First determine which root this belongs to
                root = path[0]
                if root in self.roots:
                    # Add to root classifier
                    level_data["root"][0].append(x)
                    level_data["root"][1].append(root)
                    
                # Add to parent classifiers
                for i in range(len(path) - 1):
                    parent = path[i]
                    child = path[i + 1]
                    level_data[parent][0].append(x)
                    level_data[parent][1].append(child)
                    
        # Train each level classifier
        for level, (X_level, y_level) in level_data.items():
            if X_level:
                self.level_classifiers[level].train(X_level, y_level)
                
        self.is_trained = True
        
    def _find_path(self, label: str) -> List[str]:
        """Find path from root to label in hierarchy."""
        # Simple implementation - assumes label is a leaf
        path = [label]
        
        # Work backwards to find parents
        current = label
        while True:
            parent = None
            for p, children in self.hierarchy.items():
                if current in children:
                    parent = p
                    break
                    
            if parent is None:
                break
                
            path.insert(0, parent)
            current = parent
            
        return path
        
    def predict(self, X: List[any]) -> List[str]:
        """Predict using hierarchical classification."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
            
        predictions = []
        
        for x in X:
            # Start at root
            current_level = "root"
            prediction = None
            
            # Traverse hierarchy
            while current_level in self.level_classifiers:
                # Predict at current level
                pred = self.level_classifiers[current_level].predict([x])[0]
                
                if pred == "unknown":
                    break
                    
                prediction = pred
                
                # Move to next level if this is a parent node
                if pred in self.hierarchy:
                    current_level = pred
                else:
                    # Reached a leaf
                    break
                    
            predictions.append(prediction if prediction else "unknown")
            
        return predictions