"""Tests for HDC visualization functions."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from cognitive_computing.hdc.visualizations import (
    plot_hypervector,
    plot_similarity_matrix,
    plot_binding_operation,
    plot_capacity_analysis,
    plot_classifier_performance,
    plot_hypervector_comparison,
    create_interactive_similarity_plot,
    save_plots,
)
from cognitive_computing.hdc.utils import HDCPerformanceMetrics
from cognitive_computing.hdc.operations import bind_hypervectors


@pytest.fixture
def sample_vectors():
    """Create sample hypervectors for testing."""
    np.random.seed(42)
    return [
        np.random.randint(0, 2, size=100, dtype=np.uint8),
        np.random.randint(0, 2, size=100, dtype=np.uint8),
        np.random.randint(0, 2, size=100, dtype=np.uint8),
    ]


@pytest.fixture
def sample_metrics():
    """Create sample performance metrics."""
    metrics = HDCPerformanceMetrics(
        dimension=1000,
        hypervector_type="bipolar"
    )
    
    # Add some data
    metrics.similarity_distribution["random_pairs"] = np.random.normal(0, 0.1, 100).tolist()
    metrics.noise_tolerance = {0.0: 1.0, 0.1: 0.9, 0.2: 0.7, 0.3: 0.4}
    metrics.operation_times = {
        "bind": 0.5,
        "bundle": 1.2,
        "similarity": 0.3
    }
    metrics.capacity_results = {
        "estimated_capacity": 10000,
        "mean_similarity": 0.001,
        "max_similarity": 0.15
    }
    
    return metrics


class TestBasicPlots:
    """Test basic plotting functions."""
    
    def test_plot_hypervector(self):
        """Test hypervector plotting."""
        hv = np.random.randint(-1, 2, size=1000, dtype=np.int8)
        
        fig, axes = plot_hypervector(hv, segment_size=50)
        
        assert fig is not None
        assert len(axes) == 2
        assert axes[0].get_title().startswith("Hypervector Visualization")
        
        plt.close(fig)
        
    def test_plot_similarity_matrix(self, sample_vectors):
        """Test similarity matrix plotting."""
        labels = ["A", "B", "C"]
        
        fig, ax = plot_similarity_matrix(
            sample_vectors,
            labels=labels,
            annotate=True
        )
        
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Hypervector Similarity Matrix"
        
        plt.close(fig)
        
    def test_plot_similarity_matrix_no_labels(self, sample_vectors):
        """Test similarity matrix without labels."""
        fig, ax = plot_similarity_matrix(sample_vectors)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
        
    def test_plot_binding_operation(self):
        """Test binding operation visualization."""
        a = np.array([1, 0, 1, 0] * 25, dtype=np.uint8)
        b = np.array([0, 1, 1, 0] * 25, dtype=np.uint8)
        result = bind_hypervectors(a, b, "binary")
        
        fig, axes = plot_binding_operation(a, b, result, segment_size=20)
        
        assert fig is not None
        assert len(axes) == 3
        assert "bind" in fig._suptitle.get_text().lower()
        
        plt.close(fig)


class TestAnalysisPlots:
    """Test analysis plotting functions."""
    
    def test_plot_capacity_analysis(self, sample_metrics):
        """Test capacity analysis plotting."""
        fig, axes = plot_capacity_analysis(sample_metrics)
        
        assert fig is not None
        assert axes.shape == (2, 2)
        
        # Check subplot titles
        assert "Similarity Distribution" in axes[0, 0].get_title()
        assert "Noise Tolerance" in axes[0, 1].get_title()
        assert "Operation Benchmarks" in axes[1, 0].get_title()
        assert "Summary Statistics" in axes[1, 1].get_title()
        
        plt.close(fig)
        
    def test_plot_classifier_performance(self):
        """Test classifier performance plotting."""
        metrics = {
            "test_accuracy": 0.85,
            "train_accuracy": 0.92,
            "accuracy_class1": 0.90,
            "accuracy_class2": 0.80,
            "confusion_pairs": {
                "class1_as_class2": 5,
                "class2_as_class1": 3
            }
        }
        
        fig, (ax1, ax2) = plot_classifier_performance(metrics)
        
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert "Classification Accuracy" in ax1.get_title()
        assert "Classification Confusions" in ax2.get_title()
        
        plt.close(fig)
        
    def test_plot_classifier_performance_no_confusion(self):
        """Test classifier performance without confusion data."""
        metrics = {
            "test_accuracy": 1.0,
            "train_accuracy": 1.0,
        }
        
        fig, axes = plot_classifier_performance(metrics)
        assert fig is not None
        
        plt.close(fig)


class TestComparisonPlots:
    """Test comparison plotting functions."""
    
    def test_plot_hypervector_comparison(self):
        """Test hypervector comparison plotting."""
        vectors = {
            "Original": np.random.randint(0, 2, size=200, dtype=np.uint8),
            "Noisy": np.random.randint(0, 2, size=200, dtype=np.uint8),
            "Random": np.random.randint(0, 2, size=200, dtype=np.uint8),
        }
        
        fig, axes = plot_hypervector_comparison(vectors, segment_size=50)
        
        assert fig is not None
        assert len(axes) == 3
        
        for i, label in enumerate(vectors.keys()):
            assert label in axes[i].get_title()
            
        plt.close(fig)
        
    def test_plot_single_hypervector_comparison(self):
        """Test comparison with single vector."""
        vectors = {
            "Single": np.random.randint(0, 2, size=100, dtype=np.uint8)
        }
        
        fig, axes = plot_hypervector_comparison(vectors)
        assert fig is not None
        
        plt.close(fig)


class TestInteractivePlots:
    """Test interactive plotting functions."""
    
    def test_create_interactive_similarity_plot(self, sample_vectors):
        """Test interactive similarity plot creation."""
        # This will return None if plotly is not available
        fig = create_interactive_similarity_plot(
            sample_vectors,
            labels=["V1", "V2", "V3"]
        )
        
        # If plotly is available, check the figure
        if fig is not None:
            assert hasattr(fig, 'data')
            assert hasattr(fig, 'layout')


class TestPlotSaving:
    """Test plot saving functionality."""
    
    def test_save_single_plot(self):
        """Test saving a single plot."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_plot"
            save_plots(fig, base_path, format="png", dpi=100)
            
            assert (Path(tmpdir) / "test_plot.png").exists()
            
        plt.close(fig)
        
    def test_save_multiple_plots(self):
        """Test saving multiple plots."""
        figs = []
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [i, i+1, i+2])
            figs.append(fig)
            
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_plots"
            save_plots(figs, base_path, format="pdf")
            
            for i in range(3):
                assert (Path(tmpdir) / f"test_plots_{i}.pdf").exists()
                
        for fig in figs:
            plt.close(fig)
            
    def test_save_with_directory_creation(self):
        """Test saving with automatic directory creation."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "subdir" / "test_plot"
            save_plots(fig, base_path)
            
            assert (Path(tmpdir) / "subdir" / "test_plot.png").exists()
            
        plt.close(fig)