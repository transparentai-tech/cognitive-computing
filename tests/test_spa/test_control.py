"""Tests for SPA cognitive control mechanisms."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cognitive_computing.spa.control import (
    CognitiveControl, Routing, Gating, Sequencing
)
from cognitive_computing.spa.core import SPAConfig, Vocabulary
from cognitive_computing.spa.modules import Buffer, Gate, State
from cognitive_computing.spa.actions import ActionSet


class TestCognitiveControl:
    """Test cognitive control functionality."""
    
    def test_init(self):
        """Test cognitive control initialization."""
        config = SPAConfig(dimension=64)
        control = CognitiveControl(64, config)
        
        assert control.name == "CognitiveControl"
        assert control.dimensions == 64
        assert control.config == config
        assert isinstance(control.vocab, Vocabulary)
        assert control.current_task is None
        assert len(control.task_stack) == 0
        assert control.error_signal == 0.0
        assert control.conflict_level == 0.0
        
    def test_attention_control(self):
        """Test attention setting and retrieval."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        vocab.create_pointer("OBJECT", np.random.randn(64))
        
        control = CognitiveControl(64, config, vocab)
        
        # Set attention by name
        control.set_attention("OBJECT")
        attention = control.attention
        assert np.allclose(attention, vocab.parse("OBJECT").vector)
        
        # Set attention by vector
        vec = np.random.randn(64)
        control.set_attention(vec)
        assert np.allclose(control.attention, vec)
        
    def test_goal_setting(self):
        """Test goal state management."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        vocab.create_pointer("GOAL1", np.random.randn(64))
        
        control = CognitiveControl(64, config, vocab)
        
        # Set goal by name
        control.set_goal("GOAL1")
        goal = control.goal_state
        assert np.allclose(goal, vocab.parse("GOAL1").vector)
        
        # Set goal by vector
        vec = np.random.randn(64)
        control.set_goal(vec)
        assert np.allclose(control.goal_state, vec)
        
    def test_task_stack(self):
        """Test task switching and stack management."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        vocab.create_pointer("TASK1", np.random.randn(64))
        vocab.create_pointer("TASK2", np.random.randn(64))
        vocab.create_pointer("TASK3", np.random.randn(64))
        
        control = CognitiveControl(64, config, vocab)
        
        # Push tasks
        control.push_task("TASK1")
        assert control.current_task == "TASK1"
        assert len(control.task_stack) == 0
        
        control.push_task("TASK2")
        assert control.current_task == "TASK2"
        assert control.task_stack == ["TASK1"]
        
        control.push_task("TASK3")
        assert control.current_task == "TASK3"
        assert control.task_stack == ["TASK1", "TASK2"]
        
        # Pop tasks
        prev = control.pop_task()
        assert prev == "TASK3"
        assert control.current_task == "TASK2"
        assert control.task_stack == ["TASK1"]
        
        prev = control.pop_task()
        assert prev == "TASK2"
        assert control.current_task == "TASK1"
        assert len(control.task_stack) == 0
        
        prev = control.pop_task()
        assert prev == "TASK1"
        assert control.current_task is None
        assert len(control.task_stack) == 0
        
    def test_working_memory(self):
        """Test working memory buffer management."""
        config = SPAConfig(dimension=64)
        control = CognitiveControl(64, config)
        
        # Add buffers
        control.add_working_memory("buffer1")
        assert "buffer1" in control.working_memory
        assert control.working_memory["buffer1"].dimensions == 64
        
        control.add_working_memory("buffer2", 32)
        assert "buffer2" in control.working_memory
        assert control.working_memory["buffer2"].dimensions == 32
        
    def test_conflict_monitoring(self):
        """Test conflict level computation."""
        config = SPAConfig(dimension=64, threshold=0.3)
        control = CognitiveControl(64, config)
        
        # No conflict - single high utility
        utilities = np.array([0.8, 0.1, 0.05])
        control.update_conflict(utilities)
        assert control.conflict_level == pytest.approx(0.1 / 0.8)
        
        # High conflict - similar utilities
        utilities = np.array([0.8, 0.75, 0.1])
        control.update_conflict(utilities)
        assert control.conflict_level == pytest.approx(0.75 / 0.8)
        
        # No action above threshold
        utilities = np.array([0.2, 0.1, 0.05])
        control.update_conflict(utilities)
        assert control.conflict_level == 0.0
        
    def test_error_monitoring(self):
        """Test error signal computation."""
        config = SPAConfig(dimension=16, subdimensions=4)
        control = CognitiveControl(16, config)
        
        expected = np.zeros(16)
        expected[0] = 1.0
        actual = np.zeros(16)
        actual[0] = 0.8
        actual[1] = 0.2
        
        control.update_error(expected, actual)
        # Error should be norm of difference normalized
        diff = expected - actual
        expected_error = np.linalg.norm(diff) / 4.0  # sqrt(16) = 4
        assert control.error_signal == pytest.approx(expected_error)
        
    def test_update(self):
        """Test control update with decay."""
        config = SPAConfig(dimension=64, synapse=0.1)
        control = CognitiveControl(64, config)
        
        # Set initial values
        control.error_signal = 1.0
        control.conflict_level = 0.8
        dt = 0.01
        
        control.update(dt)
        
        # Check decay
        decay = 1.0 - dt / config.synapse
        assert control.error_signal == pytest.approx(1.0 * decay)
        assert control.conflict_level == pytest.approx(0.8 * decay)


class TestRouting:
    """Test dynamic routing functionality."""
    
    def test_init(self):
        """Test routing initialization."""
        config = SPAConfig(dimension=64)
        routing = Routing(64, config)
        
        assert routing.name == "Routing"
        assert routing.dimensions == 64
        assert len(routing.routes) == 0
        assert len(routing.default_routes) == 0
        
    def test_add_route(self):
        """Test adding routes between modules."""
        config = SPAConfig(dimension=64)
        routing = Routing(64, config)
        
        # Add route without gate
        gate = routing.add_route("vision", "motor")
        assert ("vision", "motor") in routing.routes
        assert isinstance(gate, Gate)
        assert gate.name == "vision_to_motor"
        
        # Add route with custom gate
        custom_gate = Gate("custom", 64, config)
        routing.add_route("memory", "output", custom_gate)
        assert routing.routes[("memory", "output")] == custom_gate
        
    def test_default_routes(self):
        """Test default routing."""
        config = SPAConfig(dimension=64)
        routing = Routing(64, config)
        
        routing.set_default_route("input", "processing")
        assert routing.default_routes["input"] == "processing"
        
    def test_route_signal(self):
        """Test signal routing."""
        config = SPAConfig(dimension=16, subdimensions=4)
        routing = Routing(16, config)
        
        # Add routes
        gate1 = routing.add_route("A", "B")
        gate2 = routing.add_route("A", "C")
        gate1.set_gate(1.0)
        gate2.set_gate(0.5)
        
        signal = np.zeros(16)
        signal[0] = 1.0
        
        # Route to specific target
        outputs = routing.route("A", signal, "B")
        assert "B" in outputs
        assert np.allclose(outputs["B"], signal)
        
        # Route to all targets
        outputs = routing.route("A", signal)
        assert "B" in outputs
        assert "C" in outputs
        assert np.allclose(outputs["B"], signal)
        assert np.allclose(outputs["C"], 0.5 * signal)
        
    def test_route_with_defaults(self):
        """Test routing with default routes."""
        config = SPAConfig(dimension=16, subdimensions=4)
        routing = Routing(16, config)
        
        routing.set_default_route("input", "output")
        signal = np.zeros(16)
        signal[0] = 1.0
        
        # Should use default when no specific routes
        outputs = routing.route("input", signal)
        assert "output" in outputs
        assert np.allclose(outputs["output"], signal)


class TestGating:
    """Test gating control functionality."""
    
    def test_init(self):
        """Test gating initialization."""
        config = SPAConfig(dimension=64)
        gating = Gating(64, config)
        
        assert gating.name == "Gating"
        assert gating.dimensions == 64
        assert len(gating.gates) == 0
        assert len(gating.gate_groups) == 0
        
    def test_add_gate(self):
        """Test adding gates."""
        config = SPAConfig(dimension=64)
        gating = Gating(64, config)
        
        # Add gate without object
        gate = gating.add_gate("gate1")
        assert "gate1" in gating.gates
        assert isinstance(gate, Gate)
        
        # Add gate with custom object
        custom_gate = Gate("custom", 64, config)
        gating.add_gate("gate2", custom_gate)
        assert gating.gates["gate2"] == custom_gate
        
    def test_gate_groups(self):
        """Test gate group management."""
        config = SPAConfig(dimension=64)
        gating = Gating(64, config)
        
        # Add gates
        gating.add_gate("motor1")
        gating.add_gate("motor2")
        gating.add_gate("motor3")
        
        # Create group
        gating.add_gate_group("motor", ["motor1", "motor2", "motor3"])
        assert "motor" in gating.gate_groups
        assert gating.gate_groups["motor"] == ["motor1", "motor2", "motor3"]
        
    def test_gate_control(self):
        """Test opening and closing gates."""
        config = SPAConfig(dimension=64)
        gating = Gating(64, config)
        
        # Add gates
        gate1 = gating.add_gate("gate1")
        gate2 = gating.add_gate("gate2")
        gating.add_gate_group("both", ["gate1", "gate2"])
        
        # Open single gate
        gating.open_gate("gate1", 0.7)
        assert gate1.control == 0.7
        assert gate2.control == 0.0
        
        # Open group
        gating.open_gate("both", 1.0)
        assert gate1.control == 1.0
        assert gate2.control == 1.0
        
        # Close single gate
        gating.close_gate("gate1")
        assert gate1.control == 0.0
        assert gate2.control == 1.0
        
        # Close group
        gating.close_gate("both")
        assert gate1.control == 0.0
        assert gate2.control == 0.0


class TestSequencing:
    """Test sequential behavior control."""
    
    def test_init(self):
        """Test sequencing initialization."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        seq = Sequencing(64, config, vocab)
        
        assert seq.name == "Sequencing"
        assert seq.dimensions == 64
        assert seq.vocab == vocab
        assert seq.current_sequence is None
        assert seq.sequence_index == 0
        assert not seq.paused
        
    def test_define_sequence(self):
        """Test sequence definition."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        seq = Sequencing(64, config, vocab)
        
        steps = ["step1", "step2", "step3"]
        seq.define_sequence("test_seq", steps)
        
        assert "test_seq" in seq.sequences
        assert seq.sequences["test_seq"] == steps
        
    def test_sequence_execution(self):
        """Test basic sequence execution."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        seq = Sequencing(64, config, vocab)
        
        # Define and start sequence
        steps = ["action1", "action2", "action3"]
        seq.define_sequence("seq1", steps)
        seq.start_sequence("seq1")
        
        assert seq.current_sequence == "seq1"
        assert seq.sequence_index == 0
        
        # Execute steps
        step = seq.next_step()
        assert step == "action1"
        assert seq.sequence_index == 1
        
        step = seq.next_step()
        assert step == "action2"
        assert seq.sequence_index == 2
        
        step = seq.next_step()
        assert step == "action3"
        assert seq.sequence_index == 3
        
        # Set max_loops to 0 to disable looping for this test
        seq.max_loops = 0
        
        # Sequence complete (no looping)
        step = seq.next_step()
        assert step is None
        assert seq.current_sequence is None
        
    def test_sequence_control(self):
        """Test pause, resume, and stop."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        vocab.create_pointer("seq1", np.random.randn(64))
        
        seq = Sequencing(64, config, vocab)
        
        steps = ["step1", "step2", "step3"]
        seq.define_sequence("seq1", steps)
        seq.start_sequence("seq1")
        
        # Execute first step
        step = seq.next_step()
        assert step == "step1"
        
        # Pause
        seq.pause_sequence()
        assert seq.paused
        step = seq.next_step()
        assert step is None  # No step while paused
        
        # Resume
        seq.resume_sequence()
        assert not seq.paused
        step = seq.next_step()
        assert step == "step2"
        
        # Stop
        seq.stop_sequence()
        assert seq.current_sequence is None
        assert seq.sequence_index == 0
        
    def test_sequence_interruption(self):
        """Test sequence interruption and stack."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        seq = Sequencing(64, config, vocab)
        seq.max_loops = 0  # Disable looping
        
        # Define sequences
        seq.define_sequence("seq1", ["a1", "a2", "a3"])
        seq.define_sequence("seq2", ["b1", "b2"])
        
        # Start first sequence
        seq.start_sequence("seq1")
        seq.next_step()  # Execute a1
        
        # Interrupt with second sequence
        seq.start_sequence("seq2")
        assert seq.current_sequence == "seq2"
        assert len(seq.interrupt_stack) == 1
        assert seq.interrupt_stack[0] == ("seq1", 1)
        
        # Complete second sequence
        seq.next_step()  # b1
        seq.next_step()  # b2
        seq.next_step()  # Complete seq2
        
        # Should resume seq1
        assert seq.current_sequence == "seq1"
        assert seq.sequence_index == 1
        step = seq.next_step()
        assert step == "a2"
        
    def test_sequence_looping(self):
        """Test sequence looping."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        vocab.create_pointer("loop", np.random.randn(64))
        
        seq = Sequencing(64, config, vocab)
        seq.max_loops = 2  # Allow 2 loops
        
        seq.define_sequence("loop", ["step1", "step2"])
        seq.start_sequence("loop")
        
        # First iteration
        assert seq.next_step() == "step1"
        assert seq.next_step() == "step2"
        
        # Second iteration (first loop)
        assert seq.loop_count == 0
        assert seq.next_step() == "step1"
        assert seq.loop_count == 1
        assert seq.next_step() == "step2"
        
        # Third iteration (second loop)
        assert seq.next_step() == "step1"
        assert seq.loop_count == 2
        assert seq.next_step() == "step2"
        
        # Should stop after max loops
        assert seq.next_step() is None
        assert seq.current_sequence is None
        
    def test_sequence_state_encoding(self):
        """Test sequence state vector encoding."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        
        seq = Sequencing(64, config, vocab)
        
        # Define and start sequence
        seq.define_sequence("test_seq", ["a", "b", "c"])
        seq.start_sequence("test_seq")
        
        # Initial state should be the created pointer for test_seq
        # The pointer was created automatically by _update_sequence_state
        assert "test_seq" in seq.vocab.pointers
        test_seq_ptr = seq.vocab.pointers["test_seq"].vector
        assert np.allclose(seq.sequence_state, test_seq_ptr)
        
        # After first step, should include position
        seq.next_step()
        seq._update_sequence_state()
        
        # Check that state is a combination of sequence and position
        # Both vectors are created automatically by the sequencing module
        assert "STEP_1" in seq.vocab.pointers
        seq_vec = seq.vocab.pointers["test_seq"].vector
        step_vec = seq.vocab.pointers["STEP_1"].vector
        
        # Verify the state is a weighted combination
        expected = 0.7 * seq_vec + 0.3 * step_vec
        assert np.allclose(seq.sequence_state, expected)
        
    def test_callable_steps(self):
        """Test sequences with callable steps."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        vocab.create_pointer("func_seq", np.random.randn(64))
        
        seq = Sequencing(64, config, vocab)
        
        # Track calls
        calls = []
        
        def func1():
            calls.append("func1")
            
        def func2():
            calls.append("func2")
            
        # Define sequence with callables
        seq.define_sequence("func_seq", ["action1", func1, "action2", func2])
        seq.start_sequence("func_seq")
        
        # Execute steps
        assert seq.next_step() == "action1"
        assert seq.next_step() == func1
        assert seq.next_step() == "action2"
        assert seq.next_step() == func2
        
    def test_unknown_sequence_error(self):
        """Test error on unknown sequence."""
        config = SPAConfig(dimension=64)
        vocab = Vocabulary(64)
        seq = Sequencing(64, config, vocab)
        
        with pytest.raises(ValueError, match="Unknown sequence"):
            seq.start_sequence("undefined")