"""
Tests for HRR visualization functions.

Tests plotting and visualization capabilities with mocked matplotlib.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

from cognitive_computing.hrr import create_hrr
from cognitive_computing.hrr.cleanup import CleanupMemory, CleanupMemoryConfig
from cognitive_computing.hrr.visualizations import (
    plot_similarity_matrix,
    plot_binding_accuracy,
    visualize_cleanup_space,
    plot_convolution_spectrum,
    animate_unbinding_process,
    plot_memory_capacity_curve,
    plot_crosstalk_analysis,
    create_performance_dashboard
)

# Use non-interactive backend for testing
matplotlib.use('Agg')


class TestSimilarityMatrix:
    """Test similarity matrix visualization."""
    
    def test_plot_similarity_matrix_basic(self):
        """Test basic similarity matrix plot."""
        # Create test vectors
        vectors = {
            "vec1": np.array([1, 0, 0, 0]),
            "vec2": np.array([0, 1, 0, 0]),
            "vec3": np.array([0, 0, 1, 0]),
            "vec4": np.array([1, 1, 0, 0]) / np.sqrt(2)
        }
        
        fig = plot_similarity_matrix(vectors, method="cosine")
        
        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar
        
        # Check title
        assert "Similarity Matrix" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_plot_similarity_matrix_dot_product(self):
        """Test similarity matrix with dot product."""
        vectors = {
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100)
        }
        
        fig = plot_similarity_matrix(vectors, method="dot", annot=False)
        
        assert fig is not None
        assert "dot" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_plot_similarity_matrix_custom_params(self):
        """Test similarity matrix with custom parameters."""
        vectors = {"v1": np.ones(50), "v2": -np.ones(50)}
        
        fig = plot_similarity_matrix(
            vectors,
            figsize=(10, 8),
            cmap="viridis",
            annot=True
        )
        
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8
        
        plt.close(fig)


class TestBindingAccuracy:
    """Test binding accuracy visualization."""
    
    def test_plot_binding_accuracy_single_point(self):
        """Test plotting with single data point."""
        hrr = create_hrr(dimension=512)
        
        test_results = {
            "n_pairs": 5,
            "mean_accuracy": 0.95,
            "mean_similarity": 0.85
        }
        
        fig = plot_binding_accuracy(hrr, test_results)
        
        assert fig is not None
        assert len(fig.axes) == 2
        
        # Check subplot titles
        titles = [ax.get_title() for ax in fig.axes]
        assert any("Capacity" in title for title in titles)
        assert any("Quality" in title for title in titles)
        
        plt.close(fig)
    
    def test_plot_binding_accuracy_multiple_points(self):
        """Test plotting with multiple data points."""
        hrr = create_hrr(dimension=1024)
        
        test_results = {
            "n_pairs": [1, 5, 10, 20, 50],
            "accuracies": [1.0, 0.95, 0.85, 0.65, 0.4],
            "mean_similarities": [0.98, 0.9, 0.8, 0.6, 0.3]
        }
        
        fig = plot_binding_accuracy(hrr, test_results, figsize=(12, 6))
        
        assert fig is not None
        assert fig.get_size_inches()[0] == 12
        
        # Check that data was plotted
        ax1, ax2 = fig.axes
        assert len(ax1.lines) >= 1  # Accuracy plot
        assert len(ax2.lines) >= 1  # Similarity plot
        
        plt.close(fig)


class TestCleanupSpaceVisualization:
    """Test cleanup memory space visualization."""
    
    def test_visualize_cleanup_space_empty(self):
        """Test visualization of empty cleanup memory."""
        cleanup = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        
        fig = visualize_cleanup_space(cleanup)
        
        assert fig is not None
        # Should show empty message
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    @patch('cognitive_computing.hrr.visualizations.PCA')
    def test_visualize_cleanup_space_pca(self, mock_pca):
        """Test cleanup space visualization with PCA."""
        # Mock PCA
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.randn(3, 2)
        mock_pca.return_value = mock_pca_instance
        
        # Create cleanup memory with items
        cleanup = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        for i in range(3):
            cleanup.add_item(f"item{i}", np.random.randn(100))
        
        fig = visualize_cleanup_space(cleanup, method="pca")
        
        assert fig is not None
        assert mock_pca.called
        
        # Check scatter plot was created
        ax = fig.axes[0]
        assert len(ax.collections) >= 1  # Scatter plot
        
        plt.close(fig)
    
    @patch('cognitive_computing.hrr.visualizations.TSNE')
    def test_visualize_cleanup_space_tsne(self, mock_tsne):
        """Test cleanup space visualization with t-SNE."""
        # Mock t-SNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.randn(3, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        cleanup = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        for i in range(3):
            cleanup.add_item(f"item{i}", np.random.randn(100))
        
        fig = visualize_cleanup_space(cleanup, method="tsne", figsize=(10, 10))
        
        assert fig is not None
        assert fig.get_size_inches()[0] == 10
        assert mock_tsne.called
        
        plt.close(fig)
    
    @patch('cognitive_computing.hrr.visualizations.PLOTLY_AVAILABLE', True)
    @patch('cognitive_computing.hrr.visualizations.go')
    def test_visualize_cleanup_space_interactive(self, mock_go):
        """Test interactive cleanup space visualization."""
        # Mock plotly
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()
        
        cleanup = CleanupMemory(CleanupMemoryConfig(), dimension=100)
        cleanup.add_item("test", np.random.randn(100))
        
        with patch('cognitive_computing.hrr.visualizations.PCA') as mock_pca:
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array([[1, 2]])
            mock_pca.return_value = mock_pca_instance
            
            fig = visualize_cleanup_space(cleanup, interactive=True)
        
        assert fig == mock_figure
        assert mock_go.Figure.called


class TestConvolutionSpectrum:
    """Test convolution spectrum visualization."""
    
    def test_plot_convolution_spectrum_basic(self):
        """Test basic convolution spectrum plot."""
        a = np.random.randn(128)
        b = np.random.randn(128)
        
        fig = plot_convolution_spectrum(a, b)
        
        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        # Check subplot titles
        titles = [ax.get_title() for ax in fig.axes]
        assert any("Input Spectra" in title for title in titles)
        assert any("Result Spectrum" in title for title in titles)
        
        plt.close(fig)
    
    def test_plot_convolution_spectrum_with_result(self):
        """Test convolution spectrum with pre-computed result."""
        from cognitive_computing.hrr.operations import CircularConvolution
        
        a = np.random.randn(64)
        b = np.random.randn(64)
        result = CircularConvolution.convolve(a, b)
        
        fig = plot_convolution_spectrum(a, b, result, figsize=(15, 10))
        
        assert fig is not None
        assert fig.get_size_inches()[0] == 15
        assert fig.get_size_inches()[1] == 10
        
        plt.close(fig)


class TestUnbindingAnimation:
    """Test unbinding process visualization."""
    
    def test_animate_unbinding_process_single(self):
        """Test unbinding animation with single key."""
        hrr = create_hrr(dimension=512, seed=42)
        
        # Create composite
        key = hrr.generate_vector()
        value = hrr.generate_vector()
        composite = hrr.bind(key, value)
        
        fig = animate_unbinding_process(hrr, composite, [key])
        
        assert fig is not None
        assert len(fig.axes) == 2  # 2 rows, 1 column
        
        plt.close(fig)
    
    def test_animate_unbinding_process_multiple(self):
        """Test unbinding animation with multiple keys."""
        hrr = create_hrr(dimension=1024, seed=42)
        
        # Create composite with multiple bindings
        keys = [hrr.generate_vector() for _ in range(3)]
        values = [hrr.generate_vector() for _ in range(3)]
        
        pairs = [hrr.bind(k, v) for k, v in zip(keys, values)]
        composite = hrr.bundle(pairs)
        
        fig = animate_unbinding_process(
            hrr, composite, keys,
            names=["Key A", "Key B", "Key C"],
            figsize=(15, 8)
        )
        
        assert fig is not None
        assert len(fig.axes) == 6  # 2 rows, 3 columns
        
        plt.close(fig)


class TestCapacityAnalysis:
    """Test capacity analysis visualizations."""
    
    def test_plot_memory_capacity_curve(self):
        """Test memory capacity curve plot."""
        dimensions = [256, 512, 1024, 2048, 4096]
        capacities = [10, 25, 55, 120, 250]
        theoretical = [d * 0.06 for d in dimensions]  # Hypothetical
        
        fig = plot_memory_capacity_curve(dimensions, capacities, theoretical)
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should have 2 lines (measured + theoretical)
        assert len(ax.lines) == 2
        
        # Check log scale
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_plot_memory_capacity_curve_no_theoretical(self):
        """Test capacity curve without theoretical values."""
        dimensions = [256, 512, 1024]
        capacities = [10, 25, 55]
        
        fig = plot_memory_capacity_curve(dimensions, capacities)
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should have only 1 line (measured)
        assert len(ax.lines) == 1
        
        plt.close(fig)


class TestCrosstalkAnalysis:
    """Test crosstalk analysis visualization."""
    
    def test_plot_crosstalk_analysis(self):
        """Test crosstalk analysis plot."""
        n_vectors = [2, 5, 10, 20, 50]
        crosstalk_levels = [0.5, 0.3, 0.2, 0.15, 0.1]
        
        fig = plot_crosstalk_analysis(n_vectors, crosstalk_levels)
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should have measured line and theoretical line
        assert len(ax.lines) >= 2
        
        # Check legend
        assert ax.get_legend() is not None
        
        plt.close(fig)


class TestPerformanceDashboard:
    """Test performance dashboard creation."""
    
    def test_create_performance_dashboard_basic(self):
        """Test basic performance dashboard."""
        perf_results = {
            'bind_time_mean': 0.001,
            'unbind_time_mean': 0.0015,
            'bundle_time_mean': 0.002,
            'operations_per_second': 500,
            'dimension': 1024,
            'storage_method': 'real',
            'normalized': True
        }
        
        fig = create_performance_dashboard(perf_results)
        
        assert fig is not None
        # Dashboard has multiple subplots
        assert len(fig.axes) >= 5
        
        plt.close(fig)
    
    def test_create_performance_dashboard_with_capacity(self):
        """Test dashboard with capacity curve."""
        perf_results = {
            'bind_time_mean': 0.001,
            'unbind_time_mean': 0.0015,
            'bundle_time_mean': 0.002,
            'operations_per_second': 500,
            'dimension': 2048,
            'storage_method': 'complex',
            'capacity_curve': {
                'n_items': [10, 20, 50, 100],
                'accuracies': [0.99, 0.95, 0.85, 0.7]
            }
        }
        
        fig = create_performance_dashboard(perf_results, figsize=(15, 10))
        
        assert fig is not None
        assert fig.get_size_inches()[0] == 15
        assert fig.get_size_inches()[1] == 10
        
        plt.close(fig)


class TestVisualizationHelpers:
    """Test visualization helper functions and edge cases."""
    
    def test_empty_vectors_similarity_matrix(self):
        """Test similarity matrix with empty vector dict."""
        fig = plot_similarity_matrix({})
        
        assert fig is not None
        # Should create empty matrix
        
        plt.close(fig)
    
    def test_single_vector_similarity_matrix(self):
        """Test similarity matrix with single vector."""
        vectors = {"only": np.random.randn(100)}
        
        fig = plot_similarity_matrix(vectors)
        
        assert fig is not None
        # Should show 1x1 matrix
        
        plt.close(fig)
    
    def test_figure_saving(self):
        """Test that figures can be saved."""
        import tempfile
        import os
        
        vectors = {"a": np.ones(10), "b": -np.ones(10)}
        fig = plot_similarity_matrix(vectors)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            fig.savefig(temp_path)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            os.unlink(temp_path)
            plt.close(fig)


class TestIntegrationWithHRR:
    """Test visualization integration with HRR system."""
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline with HRR."""
        # Create HRR system
        hrr = create_hrr(dimension=512, seed=42)
        
        # Generate some vectors
        vectors = {}
        for i in range(4):
            vectors[f"vec{i}"] = hrr.generate_vector()
        
        # Test similarity matrix
        fig1 = plot_similarity_matrix(vectors)
        assert fig1 is not None
        plt.close(fig1)
        
        # Test binding accuracy
        test_results = {
            "mean_accuracy": 0.9,
            "mean_similarity": 0.85
        }
        fig2 = plot_binding_accuracy(hrr, test_results)
        assert fig2 is not None
        plt.close(fig2)
        
        # Test convolution spectrum
        a = vectors["vec0"]
        b = vectors["vec1"]
        fig3 = plot_convolution_spectrum(a, b)
        assert fig3 is not None
        plt.close(fig3)