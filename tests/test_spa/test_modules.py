"""Tests for SPA cognitive modules."""

import pytest
import numpy as np
from cognitive_computing.spa import (
    SPAConfig, SemanticPointer, Vocabulary, create_vocabulary
)
from cognitive_computing.spa.modules import (
    Module, State, Memory, AssociativeMemory, Buffer, Gate,
    Compare, DotProduct, Connection
)


class ConcreteModule(Module):
    """Concrete implementation for testing abstract Module class."""
    
    def update(self, dt: float):
        """Simple update for testing."""
        self._state += dt * 0.1


class TestModule:
    """Test base Module functionality."""
    
    def test_module_creation(self):
        """Test module initialization."""
        module = ConcreteModule("test", 128)
        
        assert module.name == "test"
        assert module.dimensions == 128
        assert isinstance(module.vocab, Vocabulary)
        assert np.array_equal(module.state, np.zeros(128))
        
    def test_module_with_vocab(self):
        """Test module with provided vocabulary."""
        vocab = create_vocabulary(256)
        module = ConcreteModule("test", 256, vocab)
        
        assert module.vocab is vocab
        
    def test_state_property(self):
        """Test state getter/setter."""
        module = ConcreteModule("test", 64)
        
        # Set state
        new_state = np.random.randn(64)
        module.state = new_state
        
        # Get returns copy
        retrieved = module.state
        assert np.array_equal(retrieved, new_state)
        assert retrieved is not module._state
        
    def test_state_validation(self):
        """Test state dimension validation."""
        module = ConcreteModule("test", 64)
        
        with pytest.raises(ValueError, match="doesn't match"):
            module.state = np.ones(128)
            
    def test_semantic_pointer_interface(self):
        """Test semantic pointer get/set."""
        vocab = create_vocabulary(128)
        module = ConcreteModule("test", 128, vocab)
        
        # Set via semantic pointer
        sp = vocab.create_pointer("A")
        module.set_semantic_pointer(sp)
        
        assert np.array_equal(module.state, sp.vector)
        
        # Get as semantic pointer
        retrieved = module.get_semantic_pointer()
        assert isinstance(retrieved, SemanticPointer)
        assert np.array_equal(retrieved.vector, sp.vector)
        
    def test_connections(self):
        """Test module connections."""
        source = ConcreteModule("source", 128)
        target = ConcreteModule("target", 128)
        
        # Create connection
        conn = target.connect_from(source)
        
        assert "source->target" in target.inputs
        assert "source->target" in source.outputs
        assert target.inputs["source->target"] is conn
        
    def test_connection_with_transform(self):
        """Test connection with transformation matrix."""
        source = ConcreteModule("source", 64)
        target = ConcreteModule("target", 32)
        
        # Random projection
        transform = np.random.randn(32, 64)
        conn = target.connect_from(source, transform=transform)
        
        assert conn.transform is transform


class TestState:
    """Test State module."""
    
    def test_creation(self):
        """Test state module creation."""
        state = State("memory", 512)
        
        assert state.name == "memory"
        assert state.dimensions == 512
        assert state.feedback == 0.0
        
    def test_feedback_validation(self):
        """Test feedback parameter validation."""
        with pytest.raises(ValueError, match="Feedback must be"):
            State("test", 128, feedback=1.5)
            
    def test_update_no_input(self):
        """Test state update without input."""
        state = State("test", 64)
        initial = np.random.randn(64)
        state.state = initial
        
        # Update should decay toward zero
        state.update(0.01)
        assert np.linalg.norm(state.state) < np.linalg.norm(initial)
        
    def test_update_with_input(self):
        """Test state update with input connection."""
        source = ConcreteModule("source", 128)
        state = State("state", 128)
        
        # Set source state
        source.state = np.ones(128)
        
        # Connect
        state.connect_from(source)
        
        # Update
        state.update(0.01)
        
        # State should move toward input
        assert np.mean(state.state) > 0
        
    def test_feedback(self):
        """Test state with feedback."""
        state = State("test", 64, feedback=0.9)
        
        # Set initial state
        initial = np.random.randn(64)
        state.state = initial
        
        # Update multiple times
        for _ in range(10):
            state.update(0.01)
        
        # With high feedback, state should persist
        correlation = np.dot(state.state, initial) / (
            np.linalg.norm(state.state) * np.linalg.norm(initial)
        )
        assert correlation > 0.5


class TestMemory:
    """Test Memory module."""
    
    def test_creation(self):
        """Test memory creation."""
        memory = Memory("assoc", 256)
        
        assert memory.name == "assoc"
        assert memory.dimensions == 256
        assert memory.threshold == 0.3
        assert memory.capacity is None
        assert len(memory.keys) == 0
        
    def test_add_pair(self):
        """Test adding key-value pairs."""
        memory = Memory("test", 128)
        
        key = np.random.randn(128)
        value = np.random.randn(128)
        
        memory.add_pair(key, value)
        
        assert len(memory.keys) == 1
        assert np.array_equal(memory.keys[0], key)
        assert np.array_equal(memory.values[0], value)
        
    def test_add_semantic_pointers(self):
        """Test adding semantic pointer pairs."""
        vocab = create_vocabulary(128)
        memory = Memory("test", 128, vocab)
        
        key_sp = vocab.create_pointer("KEY")
        value_sp = vocab.create_pointer("VALUE")
        
        memory.add_pair(key_sp, value_sp)
        
        assert len(memory.keys) == 1
        
    def test_capacity_limit(self):
        """Test memory capacity limiting."""
        memory = Memory("test", 64, capacity=2)
        
        # Add three pairs
        for i in range(3):
            memory.add_pair(np.random.randn(64), np.random.randn(64))
        
        # Should only keep last 2
        assert len(memory.keys) == 2
        
    def test_recall_exact(self):
        """Test exact key recall."""
        memory = Memory("test", 128)
        
        key = np.random.randn(128)
        value = np.random.randn(128)
        memory.add_pair(key, value)
        
        # Exact recall
        recalled = memory.recall(key)
        assert recalled is not None
        assert np.array_equal(recalled, value)
        
    def test_recall_noisy(self):
        """Test noisy key recall."""
        memory = Memory("test", 128, threshold=0.7)
        
        key = np.random.randn(128)
        key = key / np.linalg.norm(key)  # Normalize
        value = np.random.randn(128)
        memory.add_pair(key, value)
        
        # Add very small noise to ensure similarity stays above 0.7
        noisy_key = key + 0.05 * np.random.randn(128)
        noisy_key = noisy_key / np.linalg.norm(noisy_key)
        
        recalled = memory.recall(noisy_key)
        assert recalled is not None
        assert np.allclose(recalled, value)
        
    def test_recall_no_match(self):
        """Test recall with no match."""
        memory = Memory("test", 128, threshold=0.9)
        
        memory.add_pair(np.random.randn(128), np.random.randn(128))
        
        # Orthogonal key unlikely to match
        recalled = memory.recall(np.random.randn(128))
        assert recalled is None
        
    def test_update_with_recall(self):
        """Test memory update triggering recall."""
        memory = Memory("test", 128)
        source = ConcreteModule("source", 128)
        
        # Add memory pair
        key = np.random.randn(128)
        value = np.ones(128)
        memory.add_pair(key, value)
        
        # Connect and set source to key
        memory.connect_from(source)
        source.state = key
        
        # Update should recall value
        memory.update(0.01)
        
        # State should be close to value
        assert np.mean(memory.state) > 0.5


class TestAssociativeMemory:
    """Test AssociativeMemory module."""
    
    def test_creation(self):
        """Test associative memory creation."""
        am = AssociativeMemory("test", 256, input_scale=2.0, output_scale=0.5)
        
        assert am.input_scale == 2.0
        assert am.output_scale == 0.5
        
    def test_recall_all(self):
        """Test recalling multiple matches."""
        am = AssociativeMemory("test", 128, threshold=0.3)
        
        # Add multiple pairs with similar keys
        base_key = np.random.randn(128)
        base_key = base_key / np.linalg.norm(base_key)
        
        for i in range(5):
            # Create variations of base key
            key = base_key + 0.2 * np.random.randn(128)
            key = key / np.linalg.norm(key)
            value = np.ones(128) * i
            am.add_pair(key, value)
        
        # Recall all similar
        matches = am.recall_all(base_key, top_n=3)
        
        assert len(matches) <= 3
        assert all(sim >= 0.3 for _, sim in matches)
        # Should be sorted by similarity
        sims = [sim for _, sim in matches]
        assert sims == sorted(sims, reverse=True)


class TestBuffer:
    """Test Buffer module."""
    
    def test_creation(self):
        """Test buffer creation."""
        buffer = Buffer("working", 256, gate_threshold=0.7, decay_rate=2.0)
        
        assert buffer.gate_threshold == 0.7
        assert buffer.decay_rate == 2.0
        assert buffer.gate == 1.0  # Default open
        
    def test_gate_control(self):
        """Test gate value control."""
        buffer = Buffer("test", 128)
        
        # Set gate
        buffer.gate = 0.5
        assert buffer.gate == 0.5
        
        # Clipping
        buffer.gate = 1.5
        assert buffer.gate == 1.0
        
        buffer.gate = -0.5
        assert buffer.gate == 0.0
        
    def test_update_gate_open(self):
        """Test buffer update with open gate."""
        buffer = Buffer("test", 128, gate_threshold=0.5)
        source = ConcreteModule("source", 128)
        
        buffer.gate = 1.0  # Open
        buffer.connect_from(source)
        source.state = np.ones(128)
        
        # Update should accept input
        buffer.update(0.01)
        assert np.mean(buffer.state) > 0
        
    def test_update_gate_closed(self):
        """Test buffer update with closed gate."""
        buffer = Buffer("test", 128, gate_threshold=0.5, decay_rate=5.0)
        
        # Set initial state
        buffer.state = np.ones(128)
        buffer.gate = 0.0  # Closed
        
        # Update should decay
        buffer.update(0.01)
        assert np.mean(buffer.state) < 1.0


class TestGate:
    """Test Gate module."""
    
    def test_creation(self):
        """Test gate creation."""
        gate = Gate("control", 256)
        
        assert gate.name == "control"
        assert gate._gate_signal == 1.0
        
    def test_scalar_gating(self):
        """Test scalar gate signal."""
        gate = Gate("test", 128)
        source = ConcreteModule("source", 128)
        
        # Connect without synapse for immediate response
        gate.connect_from(source, synapse=0)
        source.state = np.ones(128)
        
        # Half gating
        gate.set_gate(0.5)
        gate.update(0.01)
        
        assert np.allclose(gate.state, 0.5)
        
    def test_vector_gating(self):
        """Test vector gate signal."""
        gate = Gate("test", 128)
        source = ConcreteModule("source", 128)
        
        # Connect without synapse for immediate response
        gate.connect_from(source, synapse=0)
        source.state = np.ones(128)
        
        # Different gating per dimension
        gate_signal = np.linspace(0, 1, 128)
        gate.set_gate(gate_signal)
        gate.update(0.01)
        
        assert np.allclose(gate.state, gate_signal)


class TestCompare:
    """Test Compare module."""
    
    def test_creation(self):
        """Test compare creation."""
        compare = Compare("similarity", 256)
        
        assert compare.input_dimensions == 256
        assert compare.dimensions == 1  # Scalar output
        
    def test_similarity_computation(self):
        """Test similarity computation."""
        compare = Compare("test", 128)
        
        # Identical vectors
        vec = np.random.randn(128)
        compare.set_input_a(vec)
        compare.set_input_b(vec)
        compare.update(0.01)
        
        assert np.isclose(compare.state[0], 1.0)
        
        # Orthogonal vectors
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        compare2 = Compare("test2", 2)
        compare2.set_input_a(vec1)
        compare2.set_input_b(vec2)
        compare2.update(0.01)
        
        assert np.isclose(compare2.state[0], 0.0)
        
    def test_semantic_pointer_inputs(self):
        """Test using semantic pointers as inputs."""
        vocab = create_vocabulary(128)
        compare = Compare("test", 128, vocab)
        
        sp_a = vocab.create_pointer("A")
        sp_b = vocab.create_pointer("B")
        
        compare.set_input_a(sp_a)
        compare.set_input_b(sp_b)
        compare.update(0.01)
        
        # Different random vectors should have low similarity
        assert -0.3 < compare.state[0] < 0.3


class TestDotProduct:
    """Test DotProduct module."""
    
    def test_creation(self):
        """Test dot product creation."""
        dot = DotProduct("energy", 256)
        
        assert dot.input_dimensions == 256
        assert dot.dimensions == 1
        
    def test_dot_product_computation(self):
        """Test dot product computation."""
        dot = DotProduct("test", 3)
        
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])
        
        dot.set_inputs(vec1, vec2)
        dot.update(0.01)
        
        expected = 1*4 + 2*5 + 3*6  # = 32
        assert np.isclose(dot.state[0], expected)
        
    def test_multidimensional_output(self):
        """Test dot product with multidimensional output."""
        dot = DotProduct("test", 128, output_dimensions=10)
        
        vec1 = np.ones(128)
        vec2 = np.ones(128) * 2
        
        dot.set_inputs(vec1, vec2)
        dot.update(0.01)
        
        # Should broadcast to all dimensions
        assert np.allclose(dot.state, 256.0)  # 128 * 2


class TestConnection:
    """Test Connection functionality."""
    
    def test_creation(self):
        """Test connection creation."""
        source = ConcreteModule("source", 64)
        target = ConcreteModule("target", 64)
        
        conn = Connection(source, target, synapse=0.05)
        
        assert conn.source is source
        assert conn.target is target
        assert conn.synapse == 0.05
        assert conn.transform is None
        
    def test_transform_validation(self):
        """Test transform matrix validation."""
        source = ConcreteModule("source", 64)
        target = ConcreteModule("target", 32)
        
        # Wrong shape
        with pytest.raises(ValueError, match="Transform shape"):
            Connection(source, target, transform=np.random.randn(64, 32))
            
        # Correct shape
        transform = np.random.randn(32, 64)
        conn = Connection(source, target, transform=transform)
        assert conn.transform is transform
        
    def test_output_no_filter(self):
        """Test connection output without filtering."""
        source = ConcreteModule("source", 64)
        target = ConcreteModule("target", 64)
        
        source.state = np.ones(64) * 2
        conn = Connection(source, target, synapse=0)
        
        output = conn.get_output()
        assert np.array_equal(output, source.state)
        
    def test_output_with_transform(self):
        """Test connection output with transformation."""
        source = ConcreteModule("source", 64)
        target = ConcreteModule("target", 32)
        
        # Projection matrix
        transform = np.random.randn(32, 64)
        source.state = np.random.randn(64)
        
        conn = Connection(source, target, transform=transform, synapse=0)
        output = conn.get_output()
        
        expected = np.dot(transform, source.state)
        assert np.allclose(output, expected)
        
    def test_synaptic_filtering(self):
        """Test synaptic filtering."""
        source = ConcreteModule("source", 64)
        target = ConcreteModule("target", 64)
        
        conn = Connection(source, target, synapse=0.1)
        
        # Step input
        source.state = np.ones(64)
        
        # First output should be small
        output1 = conn.get_output(dt=0.001)
        assert np.mean(output1) < 0.1
        
        # After many steps, should approach input
        for _ in range(1000):
            output = conn.get_output(dt=0.001)
            
        assert np.mean(output) > 0.9