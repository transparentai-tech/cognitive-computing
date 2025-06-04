"""Tests for HDC utility functions."""

import pytest
import numpy as np
import tempfile
import os

from cognitive_computing.hdc.utils import (
    HDCPerformanceMetrics,
    measure_capacity,
    benchmark_operations,
    analyze_binding_properties,
    compare_hypervector_types,
    generate_similarity_matrix,
    measure_associativity,
    estimate_required_dimension,
    create_codebook,
    measure_classifier_performance,
)
from cognitive_computing.hdc.core import HDC, HDCConfig
from cognitive_computing.hdc.classifiers import OneShotClassifier
from cognitive_computing.hdc.encoding import CategoricalEncoder


class TestHDCPerformanceMetrics:
    """Test HDC performance metrics."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = HDCPerformanceMetrics(
            dimension=1000,
            hypervector_type="bipolar"
        )
        
        assert metrics.dimension == 1000
        assert metrics.hypervector_type == "bipolar"
        assert isinstance(metrics.capacity_results, dict)
        assert isinstance(metrics.noise_tolerance, dict)
        
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = HDCPerformanceMetrics(
            dimension=1000,
            hypervector_type="binary"
        )
        metrics.capacity_results["test"] = 0.5
        metrics.operation_times["bind"] = 1.2
        
        data = metrics.to_dict()
        assert data["dimension"] == 1000
        assert data["hypervector_type"] == "binary"
        assert data["capacity_results"]["test"] == 0.5
        assert data["operation_times"]["bind"] == 1.2
        
    def test_metrics_save_load(self):
        """Test saving and loading metrics."""
        metrics = HDCPerformanceMetrics(
            dimension=500,
            hypervector_type="ternary"
        )
        metrics.capacity_results["capacity"] = 100
        metrics.noise_tolerance[0.1] = 0.9
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.json")
            
            # Save
            metrics.save(path)
            assert os.path.exists(path)
            
            # Load
            loaded = HDCPerformanceMetrics.load(path)
            assert loaded.dimension == 500
            assert loaded.hypervector_type == "ternary"
            assert loaded.capacity_results["capacity"] == 100
            assert loaded.noise_tolerance[0.1] == 0.9


class TestCapacityMeasurement:
    """Test capacity measurement functions."""
    
    def test_measure_capacity(self):
        """Test basic capacity measurement."""
        config = HDCConfig(dimension=100, hypervector_type="bipolar")
        hdc = HDC(config)
        
        metrics = measure_capacity(
            hdc,
            num_items=50,
            noise_levels=[0.0, 0.1, 0.2]
        )
        
        assert metrics.dimension == 100
        assert metrics.hypervector_type == "bipolar"
        assert "mean_similarity" in metrics.capacity_results
        assert "estimated_capacity" in metrics.capacity_results
        assert 0.0 in metrics.noise_tolerance
        assert 0.1 in metrics.noise_tolerance
        assert 0.2 in metrics.noise_tolerance
        
    def test_similarity_distribution(self):
        """Test similarity distribution measurement."""
        config = HDCConfig(dimension=100, hypervector_type="binary")
        hdc = HDC(config)
        
        metrics = measure_capacity(hdc, num_items=20)
        
        assert "random_pairs" in metrics.similarity_distribution
        similarities = metrics.similarity_distribution["random_pairs"]
        assert len(similarities) > 0
        assert all(0 <= s <= 1 for s in similarities)


class TestBenchmarking:
    """Test benchmarking functions."""
    
    def test_benchmark_operations(self):
        """Test operation benchmarking."""
        config = HDCConfig(dimension=100)
        hdc = HDC(config)
        
        times = benchmark_operations(hdc, num_trials=10)
        
        assert "generate_hypervector" in times
        assert "bind" in times
        assert "bundle_5" in times
        assert "similarity" in times
        assert "permute" in times
        
        # All times should be positive
        assert all(t > 0 for t in times.values())
        
    def test_compare_hypervector_types(self):
        """Test comparing different hypervector types."""
        results = compare_hypervector_types(
            dimension=100,
            num_items=20
        )
        
        assert "binary" in results
        assert "bipolar" in results
        assert "ternary" in results
        
        for hv_type, metrics in results.items():
            assert "mean_similarity" in metrics
            assert "estimated_capacity" in metrics
            assert "noise_tolerance_0.1" in metrics
            assert "generate_time_ms" in metrics


class TestBindingAnalysis:
    """Test binding operation analysis."""
    
    def test_analyze_binding_properties(self):
        """Test binding property analysis."""
        results = analyze_binding_properties(
            dimension=100,
            hypervector_type="bipolar",
            num_samples=20
        )
        
        assert "mean_self_inverse_error" in results
        assert "max_self_inverse_error" in results
        assert "mean_distance_preservation" in results
        assert "std_distance_preservation" in results
        
        # Self-inverse should work well
        assert results["mean_self_inverse_error"] < 0.1
        
    def test_binding_with_binary(self):
        """Test binding analysis with binary vectors."""
        results = analyze_binding_properties(
            dimension=100,
            hypervector_type="binary",
            num_samples=20
        )
        
        assert results["mean_self_inverse_error"] < 1e-10  # Nearly perfect for XOR
        assert results["max_self_inverse_error"] < 1e-10


class TestAssociativity:
    """Test associativity testing."""
    
    def test_associativity(self):
        """Test bundling associativity."""
        results = measure_associativity(
            dimension=100,
            hypervector_type="bipolar",
            num_trials=20
        )
        
        assert "mean_associativity_error" in results
        assert "max_associativity_error" in results
        assert "perfect_associations" in results
        
        # Bundling should be somewhat associative (small dimension = more error)
        assert results["mean_associativity_error"] < 0.8
        
    def test_associativity_binary(self):
        """Test associativity with binary vectors."""
        results = measure_associativity(
            dimension=100,
            hypervector_type="binary",
            num_trials=20
        )
        
        # Binary bundling may not be perfectly associative due to thresholding
        assert results["mean_associativity_error"] < 0.5


class TestUtilityFunctions:
    """Test various utility functions."""
    
    def test_generate_similarity_matrix(self):
        """Test similarity matrix generation."""
        # Create test vectors
        vectors = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([1, 1, 0, 0], dtype=np.uint8),
            np.array([0, 0, 1, 1], dtype=np.uint8),
        ]
        labels = ["A", "B", "C"]
        
        sim_matrix, returned_labels = generate_similarity_matrix(
            vectors,
            labels,
            metric="cosine"
        )
        
        assert sim_matrix.shape == (3, 3)
        assert returned_labels == labels
        assert np.allclose(np.diag(sim_matrix), 1.0)  # Self-similarity
        assert np.allclose(sim_matrix, sim_matrix.T)  # Symmetric
        
    def test_generate_similarity_matrix_no_labels(self):
        """Test similarity matrix without labels."""
        vectors = [
            np.random.randint(0, 2, size=10, dtype=np.uint8)
            for _ in range(5)
        ]
        
        sim_matrix, labels = generate_similarity_matrix(vectors)
        
        assert sim_matrix.shape == (5, 5)
        assert len(labels) == 5
        assert labels == ["V0", "V1", "V2", "V3", "V4"]
        
    def test_estimate_required_dimension(self):
        """Test dimension estimation."""
        # Small number of items
        dim = estimate_required_dimension(
            num_items=10,
            similarity_threshold=0.1
        )
        assert dim >= 1000  # Minimum dimension
        assert dim % 100 == 0  # Multiple of 100
        
        # Large number of items
        dim_large = estimate_required_dimension(
            num_items=10000,
            similarity_threshold=0.05
        )
        assert dim_large > dim  # Should increase with items
        
    def test_create_codebook(self):
        """Test codebook creation."""
        codebook = create_codebook(
            num_symbols=10,
            dimension=100,
            hypervector_type="bipolar"
        )
        
        assert len(codebook) == 10
        assert all(key.startswith("symbol_") for key in codebook)
        assert all(vec.shape == (100,) for vec in codebook.values())
        
        # Check quasi-orthogonality
        vectors = list(codebook.values())
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = np.dot(vectors[i], vectors[j]) / 100
                assert abs(sim) < 0.3  # Low similarity


class TestClassifierPerformance:
    """Test classifier performance measurement."""
    
    def test_measure_classifier_performance(self):
        """Test classifier performance metrics."""
        # Create and train classifier
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(
            dimension=100,
            encoder=encoder
        )
        
        X_train = ["A", "B", "C", "D"]
        y_train = ["c1", "c1", "c2", "c2"]
        classifier.train(X_train, y_train)
        
        # Test data
        X_test = ["A", "C", "B", "D"]
        y_test = ["c1", "c2", "c1", "c2"]
        
        # Measure performance
        metrics = measure_classifier_performance(
            classifier,
            X_test,
            y_test,
            X_train,
            y_train
        )
        
        assert "test_accuracy" in metrics
        assert "train_accuracy" in metrics
        assert "overfitting_gap" in metrics
        assert "accuracy_c1" in metrics
        assert "accuracy_c2" in metrics
        
        # Perfect on training data
        assert metrics["train_accuracy"] == 1.0
        
    def test_confusion_pairs(self):
        """Test confusion pair tracking."""
        encoder = CategoricalEncoder(dimension=100)
        classifier = OneShotClassifier(
            dimension=100,
            encoder=encoder
        )
        
        # Train with clear separation
        X_train = ["cat", "dog", "bird"]
        y_train = ["mammal", "mammal", "bird"]
        classifier.train(X_train, y_train)
        
        # Test with some confusion
        X_test = ["cat", "dog", "bird", "fish"]
        y_test = ["mammal", "mammal", "bird", "bird"]
        
        metrics = measure_classifier_performance(
            classifier,
            X_test,
            y_test
        )
        
        assert "confusion_pairs" in metrics
        # May or may not have confusion depending on randomness