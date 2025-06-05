"""
Test suite for SPA visualization functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from cognitive_computing.spa.visualizations import (
    plot_similarity_matrix, plot_action_selection,
    plot_network_graph, visualize_production_flow,
    animate_state_evolution, plot_module_activity,
    plot_vocabulary_structure, plot_interactive_network,
    NETWORKX_AVAILABLE, PLOTLY_AVAILABLE
)
from cognitive_computing.spa.core import Vocabulary, SemanticPointer, SPA, SPAConfig
from cognitive_computing.spa.actions import BasalGanglia, Thalamus, ActionSet
from cognitive_computing.spa.networks import Network, Connection
from cognitive_computing.spa.production import ProductionSystem, Production, Condition, Effect
from cognitive_computing.spa.modules import State, Memory, Gate


class TestPlotSimilarityMatrix:
    """Test similarity matrix plotting."""
    
    def test_basic_similarity_matrix(self):
        """Test basic similarity matrix plot."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B") 
        vocab.create_pointer("C")
        
        fig, ax = plot_similarity_matrix(vocab)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Semantic Pointer Similarity Matrix"
        
        plt.close(fig)
    
    def test_similarity_matrix_subset(self):
        """Test plotting subset of vocabulary."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        vocab.create_pointer("D")
        vocab.create_pointer("E")
        
        fig, ax = plot_similarity_matrix(vocab, subset=["A", "B", "C"])
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_similarity_matrix_threshold(self):
        """Test similarity threshold filtering."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        
        fig, ax = plot_similarity_matrix(vocab, threshold=0.5)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_similarity_matrix_no_annotation(self):
        """Test without annotations."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        fig, ax = plot_similarity_matrix(vocab, annotate=False)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_similarity_matrix_empty_subset(self):
        """Test with empty subset."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        with pytest.raises(ValueError, match="No valid keys found"):
            plot_similarity_matrix(vocab, subset=["X", "Y"])
    
    def test_similarity_matrix_save(self, tmp_path):
        """Test saving figure."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        save_path = tmp_path / "similarity.png"
        fig, ax = plot_similarity_matrix(vocab, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotActionSelection:
    """Test action selection plotting."""
    
    def test_basic_action_selection(self):
        """Test basic action selection plot."""
        config = SPAConfig(dimension=32)
        action_set = Mock()
        action_set.actions = [Mock() for _ in range(3)]
        bg = BasalGanglia(action_set, config)
        
        # Create history
        history = np.random.rand(100, 3)
        
        fig, ax = plot_action_selection(bg, history)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Action Selection Dynamics"
        
        plt.close(fig)
    
    def test_action_selection_with_labels(self):
        """Test with action labels."""
        config = SPAConfig(dimension=32)
        action_set = Mock()
        action_set.actions = [Mock() for _ in range(3)]
        bg = BasalGanglia(action_set, config)
        
        history = np.random.rand(50, 3)
        labels = ["Move", "Stop", "Turn"]
        
        fig, ax = plot_action_selection(bg, history, action_labels=labels)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_action_selection_with_threshold(self):
        """Test with threshold line."""
        config = SPAConfig(dimension=32)
        action_set = Mock()
        action_set.actions = [Mock() for _ in range(2)]
        bg = BasalGanglia(action_set, config)
        
        history = np.random.rand(30, 2)
        
        fig, ax = plot_action_selection(bg, history, threshold=0.5)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_action_selection_save(self, tmp_path):
        """Test saving figure."""
        config = SPAConfig(dimension=32)
        action_set = Mock()
        action_set.actions = [Mock() for _ in range(2)]
        bg = BasalGanglia(action_set, config)
        
        history = np.random.rand(20, 2)
        save_path = tmp_path / "actions.png"
        
        fig, ax = plot_action_selection(bg, history, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotNetworkGraph:
    """Test network graph plotting."""
    
    def test_basic_network_graph(self):
        """Test basic network graph."""
        network = Network()
        
        # Add modules
        state1 = State("state1", 16)
        state2 = State("state2", 16)
        network.add_module(state1)
        network.add_module(state2)
        
        # Add connection
        conn = Connection(state1, state2, np.eye(16))
        network.add_connection(conn)
        
        if NETWORKX_AVAILABLE:
            fig, ax = plot_network_graph(network)
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
            assert ax.get_title() == "SPA Network Graph"
        else:
            # Test warning when NetworkX not available
            fig, ax = plot_network_graph(network)
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_network_graph_layouts(self):
        """Test different layout algorithms."""
        network = Network()
        state = State("state", 16)
        network.add_module(state)
        
        for layout in ["spring", "circular", "random"]:
            if NETWORKX_AVAILABLE:
                fig, ax = plot_network_graph(network, layout=layout)
                assert isinstance(fig, Figure)
                plt.close(fig)
    
    def test_network_graph_save(self, tmp_path):
        """Test saving network graph."""
        network = Network()
        state = State("state", 16)
        network.add_module(state)
        
        save_path = tmp_path / "network.png"
        fig, ax = plot_network_graph(network, save_path=str(save_path))
        
        if NETWORKX_AVAILABLE:
            assert save_path.exists()
        plt.close(fig)


class TestVisualizeProductionFlow:
    """Test production flow visualization."""
    
    def test_basic_production_flow(self):
        """Test basic production flow."""
        ps = ProductionSystem()
        
        # Add productions
        cond1 = Condition("state.A > 0.5", "A is active")
        effect1 = Effect("state.B = 1", "Activate B")
        prod1 = Production("Rule1", cond1, effect1)
        
        cond2 = Condition("state.B > 0.5", "B is active")
        effect2 = Effect("state.C = 1", "Activate C")
        prod2 = Production("Rule2", cond2, effect2)
        
        ps.add_production(prod1)
        ps.add_production(prod2)
        
        fig, ax = visualize_production_flow(ps)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Production System Flow"
        
        plt.close(fig)
    
    def test_production_flow_with_execution(self):
        """Test with executed productions highlighted."""
        ps = ProductionSystem()
        
        cond = Condition("True", "Always")
        effect = Effect("state.A = 1", "Set A")
        prod1 = Production("Rule1", cond, effect)
        prod2 = Production("Rule2", cond, effect)
        
        ps.add_production(prod1)
        ps.add_production(prod2)
        
        fig, ax = visualize_production_flow(ps, executed_productions=[prod1])
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_empty_production_system(self):
        """Test with empty production system."""
        ps = ProductionSystem()
        
        fig, ax = visualize_production_flow(ps)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_production_flow_save(self, tmp_path):
        """Test saving production flow."""
        ps = ProductionSystem()
        
        cond = Condition("True", "Always")
        effect = Effect("state.A = 1", "Set A")
        prod = Production("Rule", cond, effect)
        ps.add_production(prod)
        
        save_path = tmp_path / "production.png"
        fig, ax = visualize_production_flow(ps, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)


class TestAnimateStateEvolution:
    """Test state evolution animation."""
    
    def test_basic_state_animation(self):
        """Test basic state animation."""
        # Create state sequence
        states = [np.random.randn(32) for _ in range(10)]
        
        anim = animate_state_evolution(states)
        
        assert anim is not None
        assert hasattr(anim, 'save')
        
        plt.close('all')
    
    def test_state_animation_with_vocab(self):
        """Test animation with vocabulary."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        
        # Create states that evolve
        states = []
        for i in range(10):
            state = vocab["A"].v * (1 - i/10) + vocab["B"].v * (i/10)
            states.append(state)
        
        anim = animate_state_evolution(states, vocab=vocab, top_k=3)
        
        assert anim is not None
        
        plt.close('all')
    
    def test_state_animation_save_gif(self, tmp_path):
        """Test saving animation as GIF."""
        states = [np.random.randn(32) for _ in range(5)]
        
        save_path = tmp_path / "animation.gif"
        anim = animate_state_evolution(states, save_path=str(save_path))
        
        # Note: File creation depends on writer availability
        plt.close('all')
    
    def test_state_animation_parameters(self):
        """Test animation with custom parameters."""
        states = [np.random.randn(64) for _ in range(8)]
        
        anim = animate_state_evolution(
            states, 
            top_k=10,
            interval=200,
            figsize=(12, 10)
        )
        
        assert anim is not None
        
        plt.close('all')


class TestPlotModuleActivity:
    """Test module activity plotting."""
    
    def test_basic_module_activity(self):
        """Test basic module activity plot."""
        module = State("test_state", 16)
        activity = np.random.randn(100)
        
        fig, ax = plot_module_activity(module, activity)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "test_state Activity Over Time" in ax.get_title()
        
        plt.close(fig)
    
    def test_module_activity_multichannel(self):
        """Test with multi-channel activity."""
        module = State("multi_state", 16)
        activity = np.random.randn(50, 3)
        
        fig, ax = plot_module_activity(module, activity)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_legend() is not None
        
        plt.close(fig)
    
    def test_module_activity_time_window(self):
        """Test with time window."""
        module = State("windowed", 16)
        activity = np.random.randn(200)
        
        fig, ax = plot_module_activity(module, activity, time_window=(50, 150))
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_module_activity_save(self, tmp_path):
        """Test saving module activity."""
        module = State("save_test", 16)
        activity = np.random.randn(30)
        
        save_path = tmp_path / "activity.png"
        fig, ax = plot_module_activity(module, activity, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotVocabularyStructure:
    """Test vocabulary structure visualization."""
    
    def test_vocabulary_structure_pca(self):
        """Test PCA visualization."""
        vocab = Vocabulary(dimension=64)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        vocab.create_pointer("D")
        vocab.create_pointer("E")
        
        fig, ax = plot_vocabulary_structure(vocab, method="pca")
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "PCA" in ax.get_title()
        
        plt.close(fig)
    
    @pytest.mark.slow
    def test_vocabulary_structure_tsne(self):
        """Test t-SNE visualization."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("X")
        vocab.create_pointer("Y")
        vocab.create_pointer("Z")
        
        fig, ax = plot_vocabulary_structure(vocab, method="tsne")
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "TSNE" in ax.get_title()
        
        plt.close(fig)
    
    def test_vocabulary_structure_mds(self):
        """Test MDS visualization."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("P")
        vocab.create_pointer("Q")
        vocab.create_pointer("R")
        vocab.create_pointer("S")
        
        fig, ax = plot_vocabulary_structure(vocab, method="mds")
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "MDS" in ax.get_title()
        
        plt.close(fig)
    
    def test_vocabulary_structure_3d(self):
        """Test 3D visualization."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        
        fig, ax = plot_vocabulary_structure(vocab, method="pca", n_components=3)
        
        assert isinstance(fig, Figure)
        assert hasattr(ax, 'get_zlabel')  # Check for 3D axes
        
        plt.close(fig)
    
    def test_vocabulary_structure_invalid_method(self):
        """Test with invalid method."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        
        with pytest.raises(ValueError, match="Unknown method"):
            plot_vocabulary_structure(vocab, method="invalid")
    
    def test_vocabulary_structure_save(self, tmp_path):
        """Test saving vocabulary structure."""
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        vocab.create_pointer("C")
        
        save_path = tmp_path / "vocab_structure.png"
        fig, ax = plot_vocabulary_structure(vocab, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotInteractiveNetwork:
    """Test interactive network visualization."""
    
    @patch('cognitive_computing.spa.visualizations.PLOTLY_AVAILABLE', False)
    def test_interactive_network_no_plotly(self):
        """Test when Plotly not available."""
        network = Network()
        
        # Should return without error
        with pytest.warns(UserWarning, match="Plotly not available"):
            plot_interactive_network(network)
    
    @patch('cognitive_computing.spa.visualizations.NETWORKX_AVAILABLE', False)
    def test_interactive_network_no_networkx(self):
        """Test when NetworkX not available."""
        network = Network()
        
        if PLOTLY_AVAILABLE:
            with pytest.warns(UserWarning, match="NetworkX required"):
                plot_interactive_network(network)
    
    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_interactive_network_basic(self):
        """Test basic interactive network."""
        with patch('dash.Dash') as mock_dash:
            # Mock Dash app
            mock_app = MagicMock()
            mock_dash.return_value = mock_app
            
            network = Network()
            state = State("state", 16)
            network.add_module(state)
            
            plot_interactive_network(network, port=8051)
            
            # Check that Dash app was created
            mock_dash.assert_called_once()
            mock_app.run_server.assert_called_once_with(debug=False, port=8051)


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""
    
    def test_spa_model_visualization_pipeline(self):
        """Test visualizing complete SPA model."""
        # Create SPA model
        config = SPAConfig(dimension=32)
        spa = SPA(config)
        
        # Add vocabulary
        spa.vocabulary.create_pointer("MOVE")
        spa.vocabulary.create_pointer("STOP")
        spa.vocabulary.create_pointer("LEFT")
        spa.vocabulary.create_pointer("RIGHT")
        
        # Test vocabulary visualization
        fig1, ax1 = plot_similarity_matrix(spa.vocabulary)
        assert isinstance(fig1, Figure)
        plt.close(fig1)
        
        # Test vocabulary structure
        fig2, ax2 = plot_vocabulary_structure(spa.vocabulary)
        assert isinstance(fig2, Figure)
        plt.close(fig2)
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization functions."""
        # Test with invalid inputs
        with pytest.raises(AttributeError):
            plot_similarity_matrix(None)
        
        with pytest.raises(ValueError):
            vocab = Vocabulary(dimension=32)
            plot_vocabulary_structure(vocab, method="invalid_method")
    
    def test_matplotlib_backend_handling(self):
        """Test handling of matplotlib backend."""
        # This should work regardless of backend
        vocab = Vocabulary(dimension=32)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        fig, ax = plot_similarity_matrix(vocab)
        assert fig is not None
        plt.close(fig)


# Performance benchmarks
@pytest.mark.benchmark
class TestVisualizationPerformance:
    """Performance tests for visualization functions."""
    
    def test_large_vocabulary_visualization(self, benchmark_timer):
        """Test visualization with large vocabulary."""
        vocab = Vocabulary(dimension=128)
        
        # Add many pointers
        for i in range(50):
            vocab.create_pointer(f"ITEM_{i}")
        
        benchmark_timer.start("large_vocab_similarity")
        fig, ax = plot_similarity_matrix(vocab, annotate=False)
        plt.close(fig)
        benchmark_timer.end("large_vocab_similarity")
    
    def test_long_history_animation(self, benchmark_timer):
        """Test animation with long history."""
        states = [np.random.randn(64) for _ in range(100)]
        
        benchmark_timer.start("long_animation")
        anim = animate_state_evolution(states, interval=50)
        plt.close('all')
        benchmark_timer.end("long_animation")