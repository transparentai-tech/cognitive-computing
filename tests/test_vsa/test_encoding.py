"""
Tests for VSA encoding strategies.

Tests encoders for text, spatial data, temporal sequences, graphs,
and other data types.
"""

import pytest
import numpy as np
from typing import List, Dict
import networkx as nx

from cognitive_computing.vsa.encoding import (
    VSAEncoder, RandomIndexingEncoder, SequenceEncoder,
    SpatialEncoder, TemporalEncoder, LevelEncoder,
    GraphEncoder, create_encoder
)
from cognitive_computing.vsa.core import VSA, VSAConfig, create_vsa
from cognitive_computing.vsa.vectors import BipolarVector


class TestVSAEncoderBase:
    """Test base VSAEncoder functionality."""
    
    def test_abstract_base_class(self):
        """Test that VSAEncoder cannot be instantiated."""
        with pytest.raises(TypeError):
            VSAEncoder()
            
    def test_encoder_interface(self):
        """Test that all encoders implement required methods."""
        # Create VSA instance
        vsa = create_vsa(dimension=100)
        
        encoder_types = [
            RandomIndexingEncoder, SequenceEncoder, SpatialEncoder,
            TemporalEncoder, LevelEncoder, GraphEncoder
        ]
        
        for encoder_type in encoder_types:
            encoder = encoder_type(vsa)
            assert hasattr(encoder, 'encode')
            assert hasattr(encoder, 'decode')


class TestRandomIndexingEncoder:
    """Test random indexing encoder for text and symbols."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        return RandomIndexingEncoder(vsa, seed=42)
        
    def test_encode_symbols(self, encoder):
        """Test encoding individual symbols."""
        # Encode symbols
        cat_vec = encoder.encode("cat")
        dog_vec = encoder.encode("dog")
        
        assert isinstance(cat_vec, np.ndarray)
        assert len(cat_vec) == 1000
        
        # Same symbol should give same vector
        cat_vec2 = encoder.encode("cat")
        assert np.array_equal(cat_vec, cat_vec2)
        
        # Different symbols should give different vectors
        similarity = np.corrcoef(cat_vec, dog_vec)[0, 1]
        assert similarity < 0.3  # Should be nearly orthogonal
        
    def test_encode_sequence(self, encoder):
        """Test encoding sequences of symbols."""
        # Encode a sentence
        sentence = ["the", "cat", "sat", "on", "the", "mat"]
        encoded = encoder.encode(sentence)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 1000
        
        # Should have some similarity to each word
        for word in set(sentence):
            word_vec = encoder.encode(word)
            similarity = np.corrcoef(encoded, word_vec)[0, 1]
            assert similarity > 0  # Positive similarity
            
    def test_decode_symbol(self, encoder):
        """Test decoding to find closest symbol."""
        # Encode some symbols
        symbols = ["cat", "dog", "bird", "fish"]
        for symbol in symbols:
            encoder.encode(symbol)  # Register symbols
            
        # Decode should find exact match
        cat_vec = encoder.encode("cat")
        decoded = encoder.decode(cat_vec)
        assert decoded == "cat"
        
        # Decode noisy vector
        noise = np.random.RandomState(42).randn(1000) * 0.1
        noisy_cat = cat_vec + noise
        decoded_noisy = encoder.decode(noisy_cat)
        assert decoded_noisy == "cat"
        
    def test_decode_threshold(self, encoder):
        """Test decoding with similarity threshold."""
        # Encode symbols
        symbols = ["cat", "dog", "bird"]
        for symbol in symbols:
            encoder.encode(symbol)
            
        # Create random vector
        random_vec = np.random.RandomState(42).randn(1000)
        
        # Should return None if no similar symbol
        decoded = encoder.decode(random_vec, threshold=0.5)
        assert decoded is None
        
    def test_vocabulary_management(self, encoder):
        """Test vocabulary tracking."""
        # Initially empty
        assert len(encoder.vocabulary) == 0
        
        # Add symbols
        encoder.encode("apple")
        encoder.encode("banana")
        encoder.encode("apple")  # Duplicate
        
        assert len(encoder.vocabulary) == 2
        assert "apple" in encoder.vocabulary
        assert "banana" in encoder.vocabulary


class TestSequenceEncoder:
    """Test sequence encoding strategies."""
    
    @pytest.fixture
    def encoder(self):
        """Create sequence encoder."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        return SequenceEncoder(vsa, method="position")
        
    def test_positional_encoding(self, encoder):
        """Test position-based sequence encoding."""
        sequence = ["first", "second", "third"]
        encoded = encoder.encode(sequence)
        
        assert isinstance(encoded, np.ndarray)
        
        # Decode positions
        for i, item in enumerate(sequence):
            decoded = encoder.decode(encoded, position=i)
            assert decoded == item
            
    def test_chaining_encoding(self):
        """Test chaining-based sequence encoding."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = SequenceEncoder(vsa, method="chaining")
        
        sequence = ["A", "B", "C", "D"]
        encoded = encoder.encode(sequence)
        
        # Should be able to traverse sequence
        decoded_seq = encoder.decode(encoded, length=len(sequence))
        assert decoded_seq == sequence
        
    def test_temporal_encoding(self):
        """Test temporal sequence encoding with decay."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = SequenceEncoder(vsa, method="temporal", decay_rate=0.8)
        
        sequence = ["past", "present", "future"]
        encoded = encoder.encode(sequence)
        
        # Most recent item should have highest similarity
        similarities = []
        for item in sequence:
            item_vec = encoder._get_or_create_item_vector(item)
            sim = np.corrcoef(encoded, item_vec)[0, 1]
            similarities.append(sim)
            
        # "future" (most recent) should have highest similarity
        assert similarities[2] > similarities[1] > similarities[0]
        
    def test_empty_sequence(self, encoder):
        """Test encoding empty sequence."""
        with pytest.raises(ValueError, match="Empty sequence"):
            encoder.encode([])
            
    def test_sequence_with_repetitions(self, encoder):
        """Test sequence with repeated elements."""
        sequence = ["A", "B", "A", "C", "A"]
        encoded = encoder.encode(sequence)
        
        # Should handle repetitions
        assert isinstance(encoded, np.ndarray)
        
        # Multiple positions for "A"
        positions = encoder.decode_positions(encoded, "A")
        assert len(positions) == 3
        assert set(positions) == {0, 2, 4}


class TestSpatialEncoder:
    """Test spatial data encoding."""
    
    @pytest.fixture
    def encoder(self):
        """Create spatial encoder."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        return SpatialEncoder(vsa, dimensions=2)
        
    def test_encode_2d_position(self, encoder):
        """Test encoding 2D positions."""
        # Encode position
        pos1 = encoder.encode([0.5, 0.5])  # Center
        pos2 = encoder.encode([0.0, 0.0])  # Origin
        pos3 = encoder.encode([1.0, 1.0])  # Corner
        
        # Different positions should have different vectors
        sim12 = np.corrcoef(pos1, pos2)[0, 1]
        sim13 = np.corrcoef(pos1, pos3)[0, 1]
        sim23 = np.corrcoef(pos2, pos3)[0, 1]
        
        assert sim12 < 0.9
        assert sim13 < 0.9
        assert sim23 < 0.9
        
    def test_encode_3d_position(self):
        """Test encoding 3D positions."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = SpatialEncoder(vsa, dimensions=3)
        
        pos = encoder.encode([0.5, 0.5, 0.5])
        assert isinstance(pos, np.ndarray)
        assert len(pos) == 1000
        
    def test_spatial_similarity(self, encoder):
        """Test that nearby positions have similar encodings."""
        # Encode nearby positions
        pos1 = encoder.encode([0.5, 0.5])
        pos2 = encoder.encode([0.51, 0.49])  # Very close
        pos3 = encoder.encode([0.9, 0.1])   # Far away
        
        # Nearby positions should be more similar
        sim_close = np.corrcoef(pos1, pos2)[0, 1]
        sim_far = np.corrcoef(pos1, pos3)[0, 1]
        
        assert sim_close > sim_far
        
    def test_decode_position(self, encoder):
        """Test decoding positions."""
        # Encode known position
        original_pos = [0.3, 0.7]
        encoded = encoder.encode(original_pos)
        
        # Decode should recover position (approximately)
        decoded = encoder.decode(encoded)
        assert len(decoded) == 2
        assert abs(decoded[0] - 0.3) < 0.1
        assert abs(decoded[1] - 0.7) < 0.1
        
    def test_grid_encoding(self):
        """Test grid-based spatial encoding."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = SpatialEncoder(vsa, dimensions=2, grid_size=10)
        
        # Positions in same grid cell should have same encoding
        pos1 = encoder.encode([0.11, 0.11])
        pos2 = encoder.encode([0.19, 0.19])
        
        similarity = np.corrcoef(pos1, pos2)[0, 1]
        assert similarity > 0.9  # Should be very similar
        
    def test_normalize_positions(self, encoder):
        """Test position normalization."""
        # Positions outside [0, 1] should be clipped
        pos = encoder.encode([-0.5, 1.5])
        
        # Should be same as boundary positions
        boundary = encoder.encode([0.0, 1.0])
        similarity = np.corrcoef(pos, boundary)[0, 1]
        assert similarity > 0.99


class TestTemporalEncoder:
    """Test temporal sequence encoding."""
    
    @pytest.fixture
    def encoder(self):
        """Create temporal encoder."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        return TemporalEncoder(vsa, max_lag=10)
        
    def test_encode_time_series(self, encoder):
        """Test encoding time series data."""
        # Simple time series
        time_series = [1.0, 2.0, 3.0, 2.0, 1.0]
        encoded = encoder.encode(time_series)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 1000
        
    def test_encode_with_lags(self, encoder):
        """Test that encoder uses temporal lags."""
        # Repeating pattern
        pattern1 = [1, 2, 3, 4, 5]
        pattern2 = [5, 4, 3, 2, 1]
        
        enc1 = encoder.encode(pattern1)
        enc2 = encoder.encode(pattern2)
        
        # Different patterns should give different encodings
        similarity = np.corrcoef(enc1, enc2)[0, 1]
        assert similarity < 0.5
        
    def test_decode_next_value(self, encoder):
        """Test predicting next value in sequence."""
        # Train on simple pattern
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        encoder.encode(sequence)
        
        # Encode partial sequence
        partial = [7, 8, 9]
        encoded = encoder.encode(partial)
        
        # Decode should predict something close to 10
        predicted = encoder.decode(encoded)
        assert isinstance(predicted, float)
        
    def test_multivariate_time_series(self):
        """Test encoding multivariate time series."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = TemporalEncoder(vsa, max_lag=5, n_features=3)
        
        # 3-dimensional time series
        time_series = [
            [1.0, 0.5, 0.0],
            [0.8, 0.6, 0.2],
            [0.6, 0.7, 0.4],
            [0.4, 0.8, 0.6],
            [0.2, 0.9, 0.8]
        ]
        
        encoded = encoder.encode(time_series)
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 1000
        
    def test_empty_time_series(self, encoder):
        """Test encoding empty time series."""
        with pytest.raises(ValueError, match="Empty time series"):
            encoder.encode([])


class TestLevelEncoder:
    """Test level/scalar encoding."""
    
    @pytest.fixture 
    def encoder(self):
        """Create level encoder."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        return LevelEncoder(vsa, min_value=0.0, max_value=10.0, num_levels=11)
        
    def test_encode_scalar(self, encoder):
        """Test encoding scalar values."""
        # Encode different levels
        vec0 = encoder.encode(0.0)
        vec5 = encoder.encode(5.0)
        vec10 = encoder.encode(10.0)
        
        # Different levels should have graduated similarity
        sim05 = np.corrcoef(vec0, vec5)[0, 1]
        sim010 = np.corrcoef(vec0, vec10)[0, 1]
        sim510 = np.corrcoef(vec5, vec10)[0, 1]
        
        # Closer values should be more similar
        assert sim05 > sim010
        assert sim510 > sim010
        
    def test_encode_out_of_range(self, encoder):
        """Test encoding values outside range."""
        # Below minimum
        vec_low = encoder.encode(-5.0)
        vec_min = encoder.encode(0.0)
        similarity = np.corrcoef(vec_low, vec_min)[0, 1]
        assert similarity > 0.99  # Should clip to minimum
        
        # Above maximum
        vec_high = encoder.encode(15.0)
        vec_max = encoder.encode(10.0)
        similarity = np.corrcoef(vec_high, vec_max)[0, 1]
        assert similarity > 0.99  # Should clip to maximum
        
    def test_decode_level(self, encoder):
        """Test decoding to recover level."""
        # Encode known values
        for val in [0, 2.5, 5, 7.5, 10]:
            encoded = encoder.encode(val)
            decoded = encoder.decode(encoded)
            assert abs(decoded - val) < 1.0  # Within one level
            
    def test_thermometer_encoding(self):
        """Test thermometer-style encoding."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = LevelEncoder(
            vsa, min_value=0, max_value=10, 
            num_levels=11, method="thermometer"
        )
        
        # Higher values should include lower values
        vec3 = encoder.encode(3)
        vec7 = encoder.encode(7)
        
        # vec7 should have high similarity to vec3 (includes it)
        similarity = np.corrcoef(vec3, vec7)[0, 1]
        assert similarity > 0.5
        
    def test_circular_encoding(self):
        """Test circular level encoding (e.g., for angles)."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        encoder = LevelEncoder(
            vsa, min_value=0, max_value=360,
            num_levels=36, circular=True
        )
        
        # 0 and 360 degrees should be similar
        vec0 = encoder.encode(0)
        vec360 = encoder.encode(360)
        similarity = np.corrcoef(vec0, vec360)[0, 1]
        assert similarity > 0.9
        
        # 180 degrees apart should be dissimilar
        vec180 = encoder.encode(180)
        similarity = np.corrcoef(vec0, vec180)[0, 1]
        assert similarity < 0.3


class TestGraphEncoder:
    """Test graph structure encoding."""
    
    @pytest.fixture
    def encoder(self):
        """Create graph encoder."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        return GraphEncoder(vsa)
        
    def test_encode_node(self, encoder):
        """Test encoding individual nodes."""
        # Create simple graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        
        # Encode nodes
        node1 = encoder.encode_node(G, 1)
        node2 = encoder.encode_node(G, 2)
        
        assert isinstance(node1, np.ndarray)
        assert len(node1) == 1000
        
        # Adjacent nodes should be somewhat similar
        similarity = np.corrcoef(node1, node2)[0, 1]
        assert 0.3 < similarity < 0.9
        
    def test_encode_edge(self, encoder):
        """Test encoding edges."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        
        # Encode edge
        edge12 = encoder.encode_edge(G, 1, 2)
        edge23 = encoder.encode_edge(G, 2, 3)
        
        # Different edges should be different
        similarity = np.corrcoef(edge12, edge23)[0, 1]
        assert similarity < 0.5
        
    def test_encode_subgraph(self, encoder):
        """Test encoding subgraphs."""
        # Create graph with communities
        G = nx.karate_club_graph()
        
        # Encode subgraph (first 5 nodes)
        subgraph_nodes = list(G.nodes())[:5]
        subgraph_vec = encoder.encode_subgraph(G, subgraph_nodes)
        
        assert isinstance(subgraph_vec, np.ndarray)
        
    def test_encode_entire_graph(self, encoder):
        """Test encoding entire graph."""
        # Small test graph
        G = nx.cycle_graph(5)
        graph_vec = encoder.encode(G)
        
        assert isinstance(graph_vec, np.ndarray)
        assert len(graph_vec) == 1000
        
    def test_graph_similarity(self, encoder):
        """Test that similar graphs have similar encodings."""
        # Two similar graphs
        G1 = nx.cycle_graph(5)
        G2 = nx.cycle_graph(5)
        G2.add_edge(0, 2)  # Add one edge
        
        vec1 = encoder.encode(G1)
        vec2 = encoder.encode(G2)
        
        # Should be similar but not identical
        similarity = np.corrcoef(vec1, vec2)[0, 1]
        assert 0.5 < similarity < 0.95
        
    def test_directed_graph(self, encoder):
        """Test encoding directed graphs."""
        # Create directed graph
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        
        # Encode should handle directed edges
        vec = encoder.encode(G)
        assert isinstance(vec, np.ndarray)
        
        # Reverse direction should give different encoding
        G_rev = G.reverse()
        vec_rev = encoder.encode(G_rev)
        
        similarity = np.corrcoef(vec, vec_rev)[0, 1]
        assert similarity < 0.9  # Should be different


class TestEncoderFactory:
    """Test encoder factory function."""
    
    def test_create_encoder(self):
        """Test creating encoders via factory."""
        vsa = create_vsa(dimension=1000)
        
        # Random indexing
        enc1 = create_encoder("random_indexing", vsa)
        assert isinstance(enc1, RandomIndexingEncoder)
        
        # Sequence 
        enc2 = create_encoder("sequence", vsa, method="chaining")
        assert isinstance(enc2, SequenceEncoder)
        
        # Spatial
        enc3 = create_encoder("spatial", vsa, dimensions=3)
        assert isinstance(enc3, SpatialEncoder)
        
        # Temporal
        enc4 = create_encoder("temporal", vsa, max_lag=5)
        assert isinstance(enc4, TemporalEncoder)
        
        # Level
        enc5 = create_encoder("level", vsa, min_value=0, max_value=1)
        assert isinstance(enc5, LevelEncoder)
        
        # Graph
        enc6 = create_encoder("graph", vsa)
        assert isinstance(enc6, GraphEncoder)
        
    def test_invalid_encoder_type(self):
        """Test creating invalid encoder type."""
        vsa = create_vsa()
        
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_encoder("invalid", vsa)


class TestEncoderIntegration:
    """Integration tests for encoders."""
    
    def test_combine_encoders(self):
        """Test combining multiple encoders."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        
        # Create different encoders
        text_encoder = RandomIndexingEncoder(vsa)
        spatial_encoder = SpatialEncoder(vsa, dimensions=2)
        level_encoder = LevelEncoder(vsa, min_value=0, max_value=100)
        
        # Encode multi-modal data
        text_vec = text_encoder.encode("kitchen")
        spatial_vec = spatial_encoder.encode([0.2, 0.8])  # Top-left
        level_vec = level_encoder.encode(75.0)  # High temperature
        
        # Combine with binding
        combined = vsa.bundle([text_vec, spatial_vec, level_vec])
        
        # Should maintain some similarity to each component
        assert np.corrcoef(combined, text_vec)[0, 1] > 0
        assert np.corrcoef(combined, spatial_vec)[0, 1] > 0
        assert np.corrcoef(combined, level_vec)[0, 1] > 0
        
    def test_hierarchical_encoding(self):
        """Test hierarchical encoding with multiple levels."""
        vsa = create_vsa(dimension=1000, vector_type="bipolar", seed=42)
        
        # Encode document hierarchy
        word_encoder = RandomIndexingEncoder(vsa)
        seq_encoder = SequenceEncoder(vsa, method="position")
        
        # Words -> Sentence -> Paragraph
        words = ["The", "cat", "sat"]
        word_vecs = [word_encoder.encode(w) for w in words]
        sentence_vec = seq_encoder.encode(words)
        
        # Multiple sentences -> Paragraph
        sentences = [
            ["The", "cat", "sat"],
            ["On", "the", "mat"]
        ]
        paragraph_vec = vsa.bundle([
            seq_encoder.encode(sent) for sent in sentences
        ])
        
        # Should maintain hierarchical relationships
        # Paragraph similar to sentences
        sent1_vec = seq_encoder.encode(sentences[0])
        similarity = np.corrcoef(paragraph_vec, sent1_vec)[0, 1]
        assert similarity > 0.3