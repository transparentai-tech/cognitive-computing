"""
Tests for SDM visualization functions.

This module tests all visualization functions, ensuring they:
- Create valid figures without errors
- Handle different parameter combinations
- Work with various SDM configurations
- Don't fail in headless environments
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import warnings

# Use non-interactive backend for testing
matplotlib.use('Agg')

from cognitive_computing.sdm.visualizations import (
    plot_memory_distribution,
    plot_activation_pattern,
    plot_recall_accuracy,
    visualize_memory_contents,
    plot_decoder_comparison,
    create_recall_animation,
    plot_theoretical_analysis
)
from cognitive_computing.sdm.core import SDM, SDMConfig
from cognitive_computing.sdm.utils import (
    generate_random_patterns,
    test_sdm_performance,
    add_noise
)
from cognitive_computing.sdm.address_decoder import (
    HammingDecoder,
    JaccardDecoder,
    RandomDecoder
)


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cleanup any created plots after each test."""
    yield
    plt.close('all')


@pytest.fixture
def test_sdm():
    """Create a test SDM with some patterns stored."""
    config = SDMConfig(
        dimension=256,
        num_hard_locations=50,
        activation_radius=115,
        seed=42
    )
    sdm = SDM(config)
    
    # Store some patterns
    addresses, data = generate_random_patterns(10, 256, seed=42)
    for addr, dat in zip(addresses, data):
        sdm.store(addr, dat)
    
    return sdm


@pytest.fixture
def empty_sdm():
    """Create an empty SDM."""
    config = SDMConfig(
        dimension=256,
        num_hard_locations=50,
        activation_radius=115,
        seed=42
    )
    return SDM(config)


class TestPlotMemoryDistribution:
    """Test memory distribution plotting."""
    
    def test_basic_plot(self, test_sdm):
        """Test basic memory distribution plot."""
        fig = plot_memory_distribution(test_sdm)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        # Check that title was set
        assert fig._suptitle is not None
        assert 'Memory Distribution' in fig._suptitle.get_text()
    
    def test_plot_with_empty_sdm(self, empty_sdm):
        """Test plotting with empty SDM."""
        fig = plot_memory_distribution(empty_sdm)
        
        assert fig is not None
        # Should still create plot, just with empty/zero data
    
    def test_plot_with_save(self, test_sdm):
        """Test saving plot to file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = plot_memory_distribution(test_sdm, save_path=tmp.name)
            
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_custom_figsize(self, test_sdm):
        """Test custom figure size."""
        fig = plot_memory_distribution(test_sdm, figsize=(20, 15))
        
        width, height = fig.get_size_inches()
        assert width == 20
        assert height == 15
    
    def test_binary_storage(self):
        """Test with binary storage SDM."""
        config = SDMConfig(
            dimension=128,
            num_hard_locations=30,
            activation_radius=57,
            storage_method="binary"
        )
        sdm = SDM(config)
        
        # Store a pattern
        addr = np.random.randint(0, 2, 128)
        data = np.random.randint(0, 2, 128)
        sdm.store(addr, data)
        
        fig = plot_memory_distribution(sdm)
        assert fig is not None


class TestPlotActivationPattern:
    """Test activation pattern plotting."""
    
    def test_basic_plot(self, test_sdm):
        """Test basic activation pattern plot."""
        address = np.random.randint(0, 2, 256)
        fig = plot_activation_pattern(test_sdm, address)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot
    
    def test_with_comparison_addresses(self, test_sdm):
        """Test plot with comparison addresses."""
        main_address = np.random.randint(0, 2, 256)
        comparison_addresses = [
            np.random.randint(0, 2, 256) for _ in range(3)
        ]
        
        fig = plot_activation_pattern(
            test_sdm, 
            main_address,
            comparison_addresses=comparison_addresses
        )
        
        assert fig is not None
    
    def test_save_plot(self, test_sdm):
        """Test saving activation pattern plot."""
        address = np.random.randint(0, 2, 256)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = plot_activation_pattern(
                test_sdm,
                address,
                save_path=tmp.name
            )
            
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
    
    def test_no_activations(self, test_sdm):
        """Test when no locations are activated."""
        # Create address far from all hard locations
        # This is unlikely but we handle it
        address = np.random.randint(0, 2, 256)
        
        # Temporarily set activation radius to 0
        original_radius = test_sdm.config.activation_radius
        test_sdm.config.activation_radius = 0
        
        fig = plot_activation_pattern(test_sdm, address)
        assert fig is not None
        
        # Restore radius
        test_sdm.config.activation_radius = original_radius


class TestPlotRecallAccuracy:
    """Test recall accuracy plotting."""
    
    def test_single_result_dict(self, test_sdm):
        """Test with single test result dictionary."""
        test_results = {
            'noise_tolerance': {
                0.0: 1.0,
                0.1: 0.9,
                0.2: 0.8,
                0.3: 0.6
            },
            'label': 'Test SDM'
        }
        
        fig = plot_recall_accuracy(test_results)
        
        assert fig is not None
        assert len(fig.axes) >= 1
    
    def test_multiple_results(self):
        """Test with multiple test results."""
        test_results = [
            {
                'noise_tolerance': {0.0: 1.0, 0.1: 0.9, 0.2: 0.8},
                'label': 'Config 1'
            },
            {
                'noise_tolerance': {0.0: 0.95, 0.1: 0.85, 0.2: 0.7},
                'label': 'Config 2'
            }
        ]
        
        fig = plot_recall_accuracy(test_results)
        assert fig is not None
    
    def test_with_performance_test_result(self, test_sdm):
        """Test with actual performance test results."""
        results = test_sdm_performance(test_sdm, test_patterns=10, progress=False)
        
        test_dict = {
            'noise_tolerance': results.noise_tolerance,
            'recall_accuracy_mean': results.recall_accuracy_mean,
            'write_time_mean': results.write_time_mean,
            'read_time_mean': results.read_time_mean
        }
        
        fig = plot_recall_accuracy(test_dict)
        assert fig is not None
    
    def test_with_capacity_curve(self):
        """Test with capacity curve data."""
        test_results = {
            'noise_tolerance': {0.0: 1.0, 0.1: 0.9},
            'capacity_curve': {
                'patterns': [10, 20, 30, 40, 50],
                'accuracy': [1.0, 0.95, 0.85, 0.7, 0.5]
            }
        }
        
        fig = plot_recall_accuracy(test_results)
        assert fig is not None
    
    def test_save_plot(self):
        """Test saving recall accuracy plot."""
        test_results = {
            'noise_tolerance': {0.0: 1.0, 0.1: 0.9, 0.2: 0.8}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = plot_recall_accuracy(test_results, save_path=tmp.name)
            
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)


class TestVisualizeMemoryContents:
    """Test memory contents visualization."""
    
    def test_static_pca(self, test_sdm):
        """Test static visualization with PCA."""
        fig = visualize_memory_contents(
            test_sdm,
            num_samples=30,
            method='pca',
            interactive=False
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot
    
    def test_static_tsne(self, test_sdm):
        """Test static visualization with t-SNE."""
        fig = visualize_memory_contents(
            test_sdm,
            num_samples=20,
            method='tsne',
            interactive=False
        )
        
        assert fig is not None
    
    def test_static_mds(self, test_sdm):
        """Test static visualization with MDS."""
        fig = visualize_memory_contents(
            test_sdm,
            num_samples=20,
            method='mds',
            interactive=False
        )
        
        assert fig is not None
    
    def test_color_by_options(self, test_sdm):
        """Test different coloring options."""
        for color_by in ['usage', 'saturation', 'cluster']:
            fig = visualize_memory_contents(
                test_sdm,
                num_samples=20,
                color_by=color_by,
                interactive=False
            )
            assert fig is not None
    
    @patch('cognitive_computing.sdm.visualizations.go.Figure')
    def test_interactive_plot(self, mock_plotly_figure, test_sdm):
        """Test interactive plotly visualization."""
        mock_fig = MagicMock()
        mock_plotly_figure.return_value = mock_fig
        
        fig = visualize_memory_contents(
            test_sdm,
            num_samples=30,
            method='pca',
            interactive=True
        )
        
        assert mock_plotly_figure.called
        assert fig is mock_fig
    
    def test_save_static_plot(self, test_sdm):
        """Test saving static plot."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = visualize_memory_contents(
                test_sdm,
                num_samples=20,
                interactive=False,
                save_path=tmp.name
            )
            
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
    
    def test_invalid_method(self, test_sdm):
        """Test with invalid dimensionality reduction method."""
        with pytest.raises(ValueError, match="Unknown method"):
            visualize_memory_contents(
                test_sdm,
                method='invalid',
                interactive=False
            )
    
    def test_binary_storage(self):
        """Test with binary storage SDM."""
        config = SDMConfig(
            dimension=128,
            num_hard_locations=30,
            activation_radius=57,
            storage_method="binary"
        )
        sdm = SDM(config)
        
        fig = visualize_memory_contents(
            sdm,
            num_samples=20,
            interactive=False
        )
        assert fig is not None


class TestPlotDecoderComparison:
    """Test decoder comparison plotting."""
    
    def test_basic_comparison(self):
        """Test basic decoder comparison."""
        # Create SDM instances with different decoders
        config = SDMConfig(
            dimension=256,
            num_hard_locations=50,
            activation_radius=115,
            seed=42
        )
        
        sdm_hamming = SDM(config)
        sdm_random = SDM(config)
        
        sdm_instances = {
            'Hamming': sdm_hamming,
            'Random': sdm_random
        }
        
        fig = plot_decoder_comparison(
            sdm_instances,
            test_size=20
        )
        
        assert fig is not None
        assert len(fig.axes) == 6  # 2x3 subplot
    
    def test_save_comparison(self):
        """Test saving decoder comparison plot."""
        config = SDMConfig(dimension=128, num_hard_locations=30, activation_radius=57)
        sdm_instances = {
            'Test1': SDM(config),
            'Test2': SDM(config)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = plot_decoder_comparison(
                sdm_instances,
                test_size=10,
                save_path=tmp.name
            )
            
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)


class TestCreateRecallAnimation:
    """Test recall animation creation."""
    
    @patch('cognitive_computing.sdm.visualizations.FuncAnimation')
    def test_basic_animation(self, mock_animation, test_sdm):
        """Test basic animation creation."""
        mock_anim = MagicMock()
        mock_animation.return_value = mock_anim
        
        address = test_sdm._stored_addresses[0]
        
        anim = create_recall_animation(
            test_sdm,
            address,
            noise_levels=[0.0, 0.1, 0.2],
            interval=100
        )
        
        assert mock_animation.called
        assert anim is mock_anim
    
    @patch('cognitive_computing.sdm.visualizations.FuncAnimation')
    def test_animation_with_empty_sdm(self, mock_animation, empty_sdm):
        """Test animation with empty SDM."""
        mock_anim = MagicMock()
        mock_animation.return_value = mock_anim
        
        address = np.random.randint(0, 2, 256)
        
        # Should store a pattern automatically
        anim = create_recall_animation(
            empty_sdm,
            address,
            noise_levels=[0.0, 0.1]
        )
        
        assert mock_animation.called
        assert empty_sdm.size == 1  # Pattern was stored
    
    @patch('cognitive_computing.sdm.visualizations.FuncAnimation')
    def test_animation_save(self, mock_animation, test_sdm):
        """Test saving animation."""
        mock_anim = MagicMock()
        mock_animation.return_value = mock_anim
        
        address = test_sdm._stored_addresses[0]
        
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            anim = create_recall_animation(
                test_sdm,
                address,
                noise_levels=[0.0, 0.1],
                save_path=tmp.name
            )
            
            # Check that save was called on the animation
            mock_anim.save.assert_called_once()
            
            # Note: File won't actually exist since we mocked the animation
            os.unlink(tmp.name) if os.path.exists(tmp.name) else None


class TestPlotTheoreticalAnalysis:
    """Test theoretical analysis plotting."""
    
    def test_basic_analysis(self):
        """Test basic theoretical analysis plot."""
        fig = plot_theoretical_analysis(
            dimension_range=(100, 500),
            num_points=5
        )
        
        assert fig is not None
        assert len(fig.axes) == 6  # 2x3 subplot
    
    def test_custom_range(self):
        """Test with custom dimension range."""
        fig = plot_theoretical_analysis(
            dimension_range=(500, 2000),
            num_points=10
        )
        
        assert fig is not None
    
    def test_save_analysis(self):
        """Test saving theoretical analysis."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = plot_theoretical_analysis(
                dimension_range=(100, 300),
                num_points=3,
                save_path=tmp.name
            )
            
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""
    
    def test_full_visualization_workflow(self, test_sdm):
        """Test complete visualization workflow."""
        # 1. Plot memory distribution
        fig1 = plot_memory_distribution(test_sdm)
        assert fig1 is not None
        
        # 2. Plot activation pattern
        addr = test_sdm._stored_addresses[0]
        fig2 = plot_activation_pattern(test_sdm, addr)
        assert fig2 is not None
        
        # 3. Test performance and plot
        results = test_sdm_performance(test_sdm, test_patterns=5, progress=False)
        test_dict = {'noise_tolerance': results.noise_tolerance}
        fig3 = plot_recall_accuracy(test_dict)
        assert fig3 is not None
        
        # 4. Visualize contents
        fig4 = visualize_memory_contents(test_sdm, num_samples=20, interactive=False)
        assert fig4 is not None
    
    def test_visualization_with_different_configs(self):
        """Test visualizations work with various SDM configurations."""
        configs = [
            SDMConfig(dimension=128, num_hard_locations=30, activation_radius=57, storage_method="counters"),
            SDMConfig(dimension=256, num_hard_locations=50, activation_radius=115, storage_method="binary"),
            SDMConfig(dimension=512, num_hard_locations=100, activation_radius=230, parallel=True)
        ]
        
        for config in configs:
            sdm = SDM(config)
            
            # Store a pattern
            addr = np.random.randint(0, 2, config.dimension)
            data = np.random.randint(0, 2, config.dimension)
            sdm.store(addr, data)
            
            # Try basic visualizations
            fig1 = plot_memory_distribution(sdm)
            assert fig1 is not None
            
            fig2 = plot_activation_pattern(sdm, addr)
            assert fig2 is not None
            
            plt.close('all')  # Clean up


@pytest.mark.parametrize("figsize", [(10, 8), (15, 10), (20, 15)])
class TestFigureSizes:
    """Test different figure sizes work correctly."""
    
    def test_memory_distribution_sizes(self, test_sdm, figsize):
        """Test memory distribution with different sizes."""
        fig = plot_memory_distribution(test_sdm, figsize=figsize)
        width, height = fig.get_size_inches()
        assert (width, height) == figsize
    
    def test_activation_pattern_sizes(self, test_sdm, figsize):
        """Test activation pattern with different sizes."""
        addr = np.random.randint(0, 2, 256)
        fig = plot_activation_pattern(test_sdm, addr, figsize=figsize)
        width, height = fig.get_size_inches()
        assert (width, height) == figsize


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_plot_with_none_sdm(self):
        """Test handling of None SDM."""
        with pytest.raises(AttributeError):
            plot_memory_distribution(None)
    
    def test_plot_with_invalid_address_shape(self, test_sdm):
        """Test handling of invalid address shape."""
        with pytest.raises((ValueError, IndexError)):
            # Wrong dimension address
            bad_address = np.random.randint(0, 2, 128)
            plot_activation_pattern(test_sdm, bad_address)
    
    def test_visualize_with_too_many_samples(self, test_sdm):
        """Test requesting more samples than locations."""
        # Should handle gracefully by using all available
        fig = visualize_memory_contents(
            test_sdm,
            num_samples=1000,  # More than num_hard_locations
            interactive=False
        )
        assert fig is not None
    
    @patch('cognitive_computing.sdm.visualizations.plt.savefig')
    def test_save_path_error(self, mock_savefig, test_sdm):
        """Test handling of save errors."""
        mock_savefig.side_effect = IOError("Cannot save")
        
        # Should not crash, just log error
        fig = plot_memory_distribution(test_sdm, save_path="/invalid/path.png")
        assert fig is not None