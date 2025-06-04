"""
Test suite for VSA visualization functions.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock

from cognitive_computing.vsa import (
    create_vsa, VSA,
    BinaryVector, BipolarVector, TernaryVector, ComplexVector, IntegerVector,
    generate_random_vector
)
from cognitive_computing.vsa.utils import (
    VSACapacityMetrics, VSAPerformanceMetrics,
    analyze_vector_distribution, compare_binding_methods,
    analyze_binding_capacity, benchmark_vsa_operations,
    estimate_memory_requirements
)

# Try to import visualization module
try:
    from cognitive_computing.vsa import visualizations
    HAS_VISUALIZATIONS = True
except ImportError:
    HAS_VISUALIZATIONS = False
    visualizations = None


@pytest.fixture
def vsa():
    """Create a VSA instance for testing."""
    return create_vsa(
        dimension=1000,
        vector_type='bipolar',
        vsa_type='custom',
        binding_method='multiplication'
    )


@pytest.fixture
def test_vectors():
    """Create test vectors."""
    vectors = []
    for vtype in [BinaryVector, BipolarVector, TernaryVector]:
        for _ in range(3):
            vectors.append(generate_random_vector(1000, vtype))
    return vectors


@pytest.fixture
def capacity_metrics():
    """Create sample capacity metrics."""
    metrics = []
    for dim in [1000, 5000]:
        for vtype in ['binary', 'bipolar']:
            for method in ['xor', 'multiplication']:
                metric = VSACapacityMetrics(
                    dimension=dim,
                    vector_type=vtype,
                    binding_method=method,
                    empirical_capacity=np.sqrt(dim) * 0.8,
                    theoretical_capacity=np.sqrt(dim),
                    max_reliable_bindings=int(np.sqrt(dim) * 0.7),
                    noise_tolerance=0.3,
                    similarity_threshold=0.5
                )
                metrics.append(metric)
    return metrics


@pytest.fixture
def performance_metrics():
    """Create sample performance metrics."""
    metrics = {}
    operations = ['bind', 'unbind', 'bundle', 'similarity']
    
    for op in operations:
        mean_time = 0.001 * (1 + operations.index(op))
        metrics[op] = VSAPerformanceMetrics(
            operation=op,
            dimension=1000,
            vector_type='bipolar',
            num_operations=1000,
            total_time=mean_time * 1000,
            mean_time=mean_time,
            std_time=0.0001,
            operations_per_second=1000 / (1 + operations.index(op))
        )
    
    return metrics


@pytest.mark.skipif(not HAS_VISUALIZATIONS, reason="Visualizations module not available")
class TestVSAVisualizations:
    """Test VSA visualization functions."""
    
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', False)
    def test_no_plotly_warning(self, test_vectors):
        """Test that functions warn when plotly is not available."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = visualizations.plot_vector_comparison(test_vectors[:2])
            assert result is None
            assert len(w) == 1
            assert "Plotly not available" in str(w[0].message)
    
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_comparison_heatmap(self, mock_go, test_vectors):
        """Test vector comparison heatmap."""
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Heatmap.return_value = Mock()
        
        # Test with binary vectors
        binary_vecs = [v for v in test_vectors if isinstance(v, BinaryVector)][:2]
        result = visualizations.plot_vector_comparison(
            binary_vecs,
            labels=["Vec1", "Vec2"],
            method="heatmap"
        )
        
        assert result == mock_fig
        mock_go.Figure.assert_called_once()
        mock_go.Heatmap.assert_called_once()
    
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_comparison_line(self, mock_go, test_vectors):
        """Test vector comparison line plot."""
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()
        
        binary_vecs = [v for v in test_vectors if isinstance(v, BinaryVector)][:2]
        result = visualizations.plot_vector_comparison(
            binary_vecs,
            method="line"
        )
        
        assert result == mock_fig
        assert mock_go.Scatter.call_count == 2  # One for each vector
    
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_comparison_radar(self, mock_go, test_vectors):
        """Test vector comparison radar plot."""
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatterpolar.return_value = Mock()
        
        binary_vecs = [v for v in test_vectors if isinstance(v, BinaryVector)][:2]
        result = visualizations.plot_vector_comparison(
            binary_vecs,
            method="radar"
        )
        
        assert result == mock_fig
        assert mock_go.Scatterpolar.call_count == 2
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_comparison_complex(self, mock_go, mock_subplots):
        """Test complex vector comparison."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Heatmap.return_value = Mock()
        
        # Create complex vectors
        complex_vecs = [
            generate_random_vector(100, ComplexVector),
            generate_random_vector(100, ComplexVector)
        ]
        
        result = visualizations.plot_vector_comparison(
            complex_vecs,
            method="heatmap"
        )
        
        assert result == mock_fig
        mock_subplots.assert_called_once()
        # Should create 2 heatmaps (magnitude and phase)
        assert mock_go.Heatmap.call_count == 2
    
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_comparison_errors(self, test_vectors):
        """Test error handling in vector comparison."""
        # No vectors
        with pytest.raises(ValueError, match="No vectors provided"):
            visualizations.plot_vector_comparison([])
        
        # Mixed vector types
        mixed_vecs = [
            generate_random_vector(100, BinaryVector),
            generate_random_vector(100, BipolarVector)
        ]
        with pytest.raises(ValueError, match="same type"):
            visualizations.plot_vector_comparison(mixed_vecs)
        
        # Unknown method
        with pytest.raises(ValueError, match="Unknown method"):
            visualizations.plot_vector_comparison(
                test_vectors[:2],
                method="invalid"
            )
    
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_similarity_matrix(self, mock_go, test_vectors):
        """Test similarity matrix plotting."""
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Heatmap.return_value = Mock()
        
        binary_vecs = [v for v in test_vectors if isinstance(v, BinaryVector)][:3]
        result = visualizations.plot_similarity_matrix(
            binary_vecs,
            labels=["A", "B", "C"]
        )
        
        assert result == mock_fig
        mock_go.Heatmap.assert_called_once()
        
        # Check heatmap data
        call_args = mock_go.Heatmap.call_args
        z_data = call_args[1]['z']
        assert z_data.shape == (3, 3)
        assert np.allclose(np.diag(z_data), 1.0)  # Self-similarity should be 1
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_binding_operation(self, mock_go, mock_subplots, vsa):
        """Test binding operation visualization."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()
        mock_go.Bar.return_value = Mock()
        
        vec1 = generate_random_vector(1000, BipolarVector)
        vec2 = generate_random_vector(1000, BipolarVector)
        
        result = visualizations.plot_binding_operation(
            vsa, vec1, vec2,
            labels=("X", "Y"),
            show_unbind=False
        )
        
        assert result == mock_fig
        mock_subplots.assert_called_once()
        # Should create scatter plots for vectors and bar plot for similarities
        assert mock_go.Scatter.call_count >= 3
        assert mock_go.Bar.call_count == 1
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_capacity_analysis(self, mock_go, mock_subplots, capacity_metrics):
        """Test capacity analysis plotting."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()
        mock_go.Bar.return_value = Mock()
        mock_go.Box.return_value = Mock()
        
        result = visualizations.plot_capacity_analysis(
            capacity_metrics,
            title="Capacity Test"
        )
        
        assert result == mock_fig
        mock_subplots.assert_called_once()
        # Should create various plots
        assert mock_go.Scatter.call_count >= 2  # Empirical and theoretical curves
        assert mock_go.Bar.call_count >= 1  # Noise tolerance or efficiency
        assert mock_go.Box.call_count >= 1  # Capacity by method
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_performance_comparison(self, mock_go, mock_subplots, performance_metrics):
        """Test performance comparison plotting."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Bar.return_value = Mock()
        mock_go.Box.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        
        result = visualizations.plot_performance_comparison(
            performance_metrics
        )
        
        assert result == mock_fig
        mock_subplots.assert_called_once()
        # Should create bar plots and box plots
        assert mock_go.Bar.call_count >= 2
        assert mock_go.Box.call_count >= len(performance_metrics)
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_distribution_binary(self, mock_go, mock_subplots):
        """Test vector distribution plotting for binary vectors."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Histogram.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        mock_go.Heatmap.return_value = Mock()
        
        binary_vecs = [generate_random_vector(100, BinaryVector) for _ in range(5)]
        
        result = visualizations.plot_vector_distribution(binary_vecs)
        
        assert result == mock_fig
        mock_subplots.assert_called_once()
        # Should create histogram and other plots
        assert mock_go.Histogram.call_count >= 1
        assert mock_go.Scatter.call_count >= 2  # Mean and variance plots
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_distribution_ternary(self, mock_go, mock_subplots):
        """Test vector distribution plotting for ternary vectors."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Histogram.return_value = Mock()
        mock_go.Heatmap.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        mock_go.Bar.return_value = Mock()
        
        ternary_vecs = [generate_random_vector(100, TernaryVector, sparsity=0.1) 
                        for _ in range(5)]
        
        result = visualizations.plot_vector_distribution(ternary_vecs)
        
        assert result == mock_fig
        # Should create sparsity-specific plots
        assert mock_go.Histogram.call_count >= 1
        assert mock_go.Heatmap.call_count >= 1  # Sparsity pattern
        assert mock_go.Bar.call_count >= 1  # Value counts
    
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_vector_distribution_complex(self, mock_go, mock_subplots):
        """Test vector distribution plotting for complex vectors."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Histogram.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        mock_go.Scatterpolar.return_value = Mock()
        
        complex_vecs = [generate_random_vector(100, ComplexVector) for _ in range(5)]
        
        result = visualizations.plot_vector_distribution(complex_vecs)
        
        assert result == mock_fig
        # Should create magnitude/phase specific plots
        assert mock_go.Histogram.call_count >= 2  # Magnitude and phase
        assert mock_go.Scatter.call_count >= 1  # Real vs imaginary
        assert mock_go.Scatterpolar.call_count >= 1  # Phase scatter
    
    @patch('cognitive_computing.vsa.visualizations.compare_binding_methods')
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_plot_binding_comparison(self, mock_go, mock_subplots, mock_compare):
        """Test binding method comparison plotting."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        mock_go.Bar.return_value = Mock()
        
        # Mock comparison results
        mock_compare.return_value = {
            'xor': {
                'mean_similarity': 0.95,
                'std_similarity': 0.02,
                'associativity': 1.0,
                'commutativity': 1.0,
                'bind_ops_per_second': 10000,
                'unbind_ops_per_second': 10000
            },
            'multiplication': {
                'mean_similarity': 0.90,
                'std_similarity': 0.03,
                'associativity': 0.95,
                'commutativity': 1.0,
                'bind_ops_per_second': 8000,
                'unbind_ops_per_second': 8000
            }
        }
        
        result = visualizations.plot_binding_comparison(
            dimension=1000,
            vector_type='binary',
            num_items=5
        )
        
        assert result == mock_fig
        mock_compare.assert_called_once_with(1000, 'binary', 5)
        # Should create multiple bar plots
        assert mock_go.Bar.call_count >= 4
    
    @patch('cognitive_computing.vsa.utils.benchmark_vsa_operations')
    @patch('cognitive_computing.vsa.utils.analyze_binding_capacity')
    @patch('cognitive_computing.vsa.utils.estimate_memory_requirements')
    @patch('cognitive_computing.vsa.utils.generate_random_vector')
    @patch('cognitive_computing.vsa.visualizations.make_subplots')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_create_vsa_dashboard(self, mock_subplots, mock_gen_vec,
                                  mock_memory, mock_capacity, mock_benchmark, vsa):
        """Test VSA dashboard creation."""
        # Create a more sophisticated mock that avoids Plotly's validation
        mock_fig = Mock()
        mock_fig.add_trace = Mock()  # Mock the add_trace method directly
        mock_subplots.return_value = mock_fig
        
        # Mock utility functions
        mock_gen_vec.return_value = generate_random_vector(1000, BipolarVector)
        
        mock_benchmark.return_value = {
            'bind': Mock(operations_per_second=1000),
            'unbind': Mock(operations_per_second=900),
            'bundle': Mock(operations_per_second=1100),
            'similarity': Mock(operations_per_second=1200)
        }
        
        mock_capacity.return_value = Mock(
            empirical_capacity=30,
            theoretical_capacity=31.6,
            max_reliable_bindings=25,
            noise_tolerance=0.3
        )
        
        mock_memory.return_value = {
            'basic_storage_mb': 10,
            'operation_overhead_mb': 5,
            'total_mb': 15
        }
        
        # Patch go module to return properly structured trace dicts instead of mocks
        with patch('cognitive_computing.vsa.visualizations.go') as mock_go:
            # Make trace constructors return dict-like objects that Plotly will accept
            mock_go.Table.return_value = {'type': 'table', 'cells': {}}
            mock_go.Histogram.return_value = {'type': 'histogram', 'x': []}
            mock_go.Heatmap.return_value = {'type': 'heatmap', 'z': [[]]}
            mock_go.Scatter.return_value = {'type': 'scatter', 'y': []}
            mock_go.Bar.return_value = {'type': 'bar', 'x': [], 'y': []}
            
            result = visualizations.create_vsa_dashboard(vsa)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()
            # Check that various trace types were created
            assert mock_go.Table.call_count >= 3  # Config, properties, summary
            assert mock_go.Histogram.call_count >= 1  # Distribution
            assert mock_go.Heatmap.call_count >= 1  # Similarity matrix
            assert mock_go.Bar.call_count >= 3  # Performance, capacity, memory
    
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_save_path_functionality(self, tmpdir, test_vectors):
        """Test save path functionality."""
        # Mock the figure's write_html method
        with patch('cognitive_computing.vsa.visualizations.go') as mock_go:
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            mock_go.Heatmap.return_value = Mock()
            
            save_path = str(tmpdir.join("test_plot.html"))
            binary_vecs = [v for v in test_vectors if isinstance(v, BinaryVector)][:2]
            
            visualizations.plot_vector_comparison(
                binary_vecs,
                save_path=save_path
            )
            
            mock_fig.write_html.assert_called_once_with(save_path)


@pytest.mark.skipif(not HAS_VISUALIZATIONS, reason="Visualizations module not available")
class TestIntegrationWithUtils:
    """Test integration between visualization and utility functions."""
    
    @patch('cognitive_computing.vsa.visualizations.go')
    @patch('cognitive_computing.vsa.visualizations.HAS_PLOTLY', True)
    def test_distribution_analysis_integration(self, mock_go, test_vectors):
        """Test that distribution analysis integrates properly."""
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        
        # Mock the required plotly objects properly
        mock_histogram = Mock()
        mock_scatter = Mock()
        mock_heatmap = Mock()
        mock_go.Histogram.return_value = mock_histogram
        mock_go.Scatter.return_value = mock_scatter
        mock_go.Heatmap.return_value = mock_heatmap
        
        # Mock subplots
        with patch('cognitive_computing.vsa.visualizations.make_subplots') as mock_subplots:
            mock_subplots.return_value = mock_fig
            
            # Should use analyze_vector_distribution internally
            with patch('cognitive_computing.vsa.visualizations.analyze_vector_distribution') as mock_analyze:
                mock_analyze.return_value = {
                    'mean': np.zeros(1000),
                    'std': np.ones(1000),
                    'histogram': np.random.randn(30),
                    'bin_edges': np.linspace(-3, 3, 31),
                    'sparsity': 0.0,
                    'negative_fraction': 0.5,
                    'positive_fraction': 0.5
                }
                
                binary_vecs = [v for v in test_vectors if isinstance(v, BinaryVector)][:3]
                visualizations.plot_vector_distribution(binary_vecs)
                
                mock_analyze.assert_called_once()
                assert len(mock_analyze.call_args[0][0]) == 3