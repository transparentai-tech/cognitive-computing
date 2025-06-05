"""Tests for SPA action selection mechanisms."""

import pytest
import numpy as np
from cognitive_computing.spa import (
    SPAConfig, SemanticPointer, Vocabulary, create_vocabulary
)
from cognitive_computing.spa.modules import State, Module
from cognitive_computing.spa.actions import (
    Action, ActionRule, ActionSet, BasalGanglia, Thalamus,
    Cortex, ActionSelection
)


class TestAction:
    """Test Action functionality."""
    
    def test_creation(self):
        """Test action creation."""
        condition_called = False
        effect_called = False
        
        def condition():
            nonlocal condition_called
            condition_called = True
            return 0.8
        
        def effect():
            nonlocal effect_called
            effect_called = True
        
        action = Action("test_action", condition, effect, priority=2.0)
        
        assert action.name == "test_action"
        assert action.priority == 2.0
        assert action._utility == 0.0
        assert action._selected_count == 0
        
        # Evaluate condition
        utility = action.evaluate()
        assert utility == 0.8
        assert condition_called
        assert action._utility == 0.8
        
        # Execute effect
        action.execute()
        assert effect_called
        assert action._selected_count == 1
        
    def test_condition_error_handling(self):
        """Test error handling in condition evaluation."""
        def bad_condition():
            raise ValueError("Test error")
        
        def effect():
            pass
        
        action = Action("bad_action", bad_condition, effect)
        
        # Should handle error gracefully
        utility = action.evaluate()
        assert utility == 0.0
        assert action._utility == 0.0
        
    def test_effect_error_handling(self):
        """Test error handling in effect execution."""
        def condition():
            return 1.0
        
        def bad_effect():
            raise ValueError("Test error")
        
        action = Action("bad_effect", condition, bad_effect)
        
        # Should propagate error
        with pytest.raises(ValueError, match="Test error"):
            action.execute()
            
    def test_representation(self):
        """Test string representation."""
        action = Action("test", lambda: 0.5, lambda: None)
        action.evaluate()
        
        repr_str = repr(action)
        assert "test" in repr_str
        assert "0.500" in repr_str
        assert "selected=0" in repr_str


class TestActionRule:
    """Test ActionRule compilation."""
    
    def test_simple_rule(self):
        """Test compiling a simple rule."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("CIRCLE")
        vocab.create_pointer("GRASP")
        
        vision = State("vision", 128, vocab)
        motor = State("motor", 128, vocab)
        modules = {"vision": vision, "motor": motor}
        
        # Set vision to CIRCLE
        vision.set_semantic_pointer(vocab["CIRCLE"])
        
        rule = ActionRule(
            "see_circle_grasp",
            "dot(vision, CIRCLE) > 0.5",
            "motor.set_semantic_pointer(GRASP)",
            modules,
            vocab
        )
        
        action = rule.compile()
        
        # Test condition
        utility = action.evaluate()
        assert utility > 0.5  # Should be ~1.0 for exact match
        
        # Test effect
        action.execute()
        assert np.allclose(motor.state, vocab["GRASP"].vector)
        
    def test_similarity_helper(self):
        """Test similarity computation in rules."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("A")
        vocab.create_pointer("B")
        
        state1 = State("state1", 128, vocab)
        state1.set_semantic_pointer(vocab["A"])
        
        modules = {"state1": state1}
        
        rule = ActionRule(
            "test_sim",
            "sim(state1, A) > 0.9",
            "pass",
            modules,
            vocab
        )
        
        action = rule.compile()
        utility = action.evaluate()
        assert utility > 0.9
        
    def test_complex_expression(self):
        """Test more complex condition expressions."""
        vocab = create_vocabulary(128)
        vocab.create_pointer("RED")
        vocab.create_pointer("SQUARE")
        
        color = State("color", 128, vocab)
        shape = State("shape", 128, vocab)
        
        color.set_semantic_pointer(vocab["RED"])
        shape.set_semantic_pointer(vocab["SQUARE"])
        
        modules = {"color": color, "shape": shape}
        
        rule = ActionRule(
            "red_square",
            "(sim(color, RED) > 0.5) and (sim(shape, SQUARE) > 0.5)",
            "pass",
            modules,
            vocab
        )
        
        action = rule.compile()
        utility = action.evaluate()
        assert utility == True  # Boolean condition
        
    def test_module_state_access(self):
        """Test accessing module states in expressions."""
        vocab = create_vocabulary(128)
        state = State("state", 128, vocab)
        modules = {"state": state}
        
        # Set state to specific values
        state.state = np.ones(128) * 0.5
        
        rule = ActionRule(
            "check_state",
            "dot(state.state, state.state)",  # Should be 64 (128 * 0.5^2)
            "pass",
            modules,
            vocab
        )
        
        action = rule.compile()
        utility = action.evaluate()
        assert np.isclose(utility, 32.0)  # 128 * 0.5 * 0.5


class TestActionSet:
    """Test ActionSet functionality."""
    
    def test_creation(self):
        """Test action set creation."""
        action_set = ActionSet()
        assert len(action_set) == 0
        
        # With initial actions
        actions = [
            Action("a1", lambda: 0.5, lambda: None),
            Action("a2", lambda: 0.8, lambda: None)
        ]
        action_set = ActionSet(actions)
        assert len(action_set) == 2
        
    def test_add_action(self):
        """Test adding actions."""
        action_set = ActionSet()
        
        action = Action("test", lambda: 1.0, lambda: None)
        action_set.add_action(action)
        
        assert len(action_set) == 1
        assert action_set.actions[0] is action
        
    def test_evaluate_all(self):
        """Test evaluating all actions."""
        values = [0.3, 0.7, 0.1]
        actions = [
            Action(f"a{i}", lambda v=v: v, lambda: None)
            for i, v in enumerate(values)
        ]
        
        action_set = ActionSet(actions)
        utilities = action_set.evaluate_all()
        
        assert np.array_equal(utilities, values)
        
    def test_select_max(self):
        """Test max selection method."""
        actions = [
            Action("low", lambda: 0.3, lambda: None),
            Action("high", lambda: 0.9, lambda: None),
            Action("medium", lambda: 0.6, lambda: None)
        ]
        
        action_set = ActionSet(actions)
        selected = action_set.select_action(method="max")
        
        assert selected is not None
        assert selected.name == "high"
        
    def test_select_none_when_all_negative(self):
        """Test no selection when all utilities are negative."""
        actions = [
            Action("a1", lambda: -0.5, lambda: None),
            Action("a2", lambda: -0.1, lambda: None)
        ]
        
        action_set = ActionSet(actions)
        selected = action_set.select_action(method="max")
        
        assert selected is None
        
    def test_select_softmax(self):
        """Test softmax selection method."""
        # Use deterministic values for testing
        np.random.seed(42)
        
        actions = [
            Action("low", lambda: 0.1, lambda: None),
            Action("high", lambda: 5.0, lambda: None),  # Much higher
            Action("medium", lambda: 0.5, lambda: None)
        ]
        
        action_set = ActionSet(actions)
        
        # Run multiple times to check probabilistic selection
        selections = []
        for _ in range(100):
            selected = action_set.select_action(method="softmax")
            if selected:
                selections.append(selected.name)
        
        # High utility action should be selected most often
        high_count = selections.count("high")
        assert high_count > 80  # Should dominate due to high utility
        
    def test_select_epsilon_greedy(self):
        """Test epsilon-greedy selection."""
        np.random.seed(42)
        
        actions = [
            Action("low", lambda: 0.1, lambda: None),
            Action("high", lambda: 0.9, lambda: None)
        ]
        
        action_set = ActionSet(actions)
        
        # Run multiple times
        selections = []
        for _ in range(100):
            selected = action_set.select_action(method="epsilon-greedy")
            if selected:
                selections.append(selected.name)
        
        # Should mostly select high, but sometimes low
        high_count = selections.count("high")
        assert 80 < high_count < 95  # ~90% high (epsilon = 0.1)


class TestBasalGanglia:
    """Test BasalGanglia functionality."""
    
    def test_creation(self):
        """Test basal ganglia creation."""
        config = SPAConfig(dimension=128)
        actions = [
            Action("a1", lambda: 0.5, lambda: None),
            Action("a2", lambda: 0.8, lambda: None)
        ]
        action_set = ActionSet(actions)
        
        bg = BasalGanglia(action_set, config)
        
        assert bg.name == "BasalGanglia"
        assert bg.dimensions == 2  # Two actions
        assert len(bg._utilities) == 2
        assert len(bg._activations) == 2
        
    def test_update_basic(self):
        """Test basic basal ganglia update."""
        config = SPAConfig(dimension=128, threshold=0.4)
        actions = [
            Action("below", lambda: 0.3, lambda: None),  # Below threshold
            Action("above", lambda: 0.7, lambda: None)   # Above threshold
        ]
        action_set = ActionSet(actions)
        
        bg = BasalGanglia(action_set, config)
        bg.update(0.01)
        
        # Only above-threshold action should be active
        assert bg._activations[0] == 0.0  # Below threshold
        assert bg._activations[1] > 0.0   # Above threshold
        
        # State should be normalized
        assert bg.state[1] == 1.0  # Highest activation normalized to 1
        
    def test_mutual_inhibition(self):
        """Test mutual inhibition between actions."""
        config = SPAConfig(dimension=128, threshold=0.3, mutual_inhibition=0.5)
        actions = [
            Action("a1", lambda: 0.6, lambda: None),
            Action("a2", lambda: 0.8, lambda: None),
            Action("a3", lambda: 0.5, lambda: None)
        ]
        action_set = ActionSet(actions)
        
        bg = BasalGanglia(action_set, config)
        bg.update(0.01)
        
        # All are above threshold, so mutual inhibition applies
        # a2 should win due to highest utility
        assert bg._activations[1] > bg._activations[0]
        assert bg._activations[1] > bg._activations[2]
        
        # a1 and a3 should be inhibited
        assert bg._activations[0] < 0.6  # Less than raw utility
        assert bg._activations[2] < 0.5  # Less than raw utility
        
    def test_get_selected_action(self):
        """Test getting selected action."""
        config = SPAConfig(dimension=128, threshold=0.3)
        actions = [
            Action("lose", lambda: 0.4, lambda: None),
            Action("win", lambda: 0.9, lambda: None)
        ]
        action_set = ActionSet(actions)
        
        bg = BasalGanglia(action_set, config)
        bg.update(0.01)
        
        selected = bg.get_selected_action()
        assert selected is not None
        assert selected.name == "win"
        
    def test_no_selection(self):
        """Test no action selected when all below threshold."""
        config = SPAConfig(dimension=128, threshold=0.5)
        actions = [
            Action("a1", lambda: 0.2, lambda: None),
            Action("a2", lambda: 0.3, lambda: None)
        ]
        action_set = ActionSet(actions)
        
        bg = BasalGanglia(action_set, config)
        bg.update(0.01)
        
        selected = bg.get_selected_action()
        assert selected is None
        assert np.allclose(bg.state, 0.0)


class TestThalamus:
    """Test Thalamus functionality."""
    
    def test_creation(self):
        """Test thalamus creation."""
        config = SPAConfig(dimension=128)
        action_set = ActionSet([Action("a1", lambda: 0.5, lambda: None)])
        bg = BasalGanglia(action_set, config)
        
        thalamus = Thalamus(bg, config)
        
        assert thalamus.name == "Thalamus"
        assert thalamus.dimensions == 1
        assert len(thalamus._gates) == 1
        
    def test_routing_gates(self):
        """Test thalamus routing gate creation."""
        config = SPAConfig(dimension=128, threshold=0.3, routing_inhibition=3.0)
        actions = [
            Action("weak", lambda: 0.4, lambda: None),
            Action("strong", lambda: 0.9, lambda: None)
        ]
        action_set = ActionSet(actions)
        bg = BasalGanglia(action_set, config)
        
        thalamus = Thalamus(bg, config)
        
        # Update BG first
        bg.update(0.01)
        
        # Update thalamus
        thalamus.update(0.01)
        
        # Strong action should get full gate
        assert thalamus.get_gate(1) == 1.0
        
        # Weak action should be inhibited
        assert thalamus.get_gate(0) < 0.3  # 1/(1+3) = 0.25
        
    def test_route_information(self):
        """Test routing information through thalamus."""
        config = SPAConfig(dimension=128)
        vocab = create_vocabulary(128)
        
        # Create modules
        source = State("source", 128, vocab)
        target = State("target", 128, vocab)
        
        # Set source state
        source.state = np.ones(128)
        
        # Create action system
        action_set = ActionSet([Action("route", lambda: 1.0, lambda: None)])
        bg = BasalGanglia(action_set, config)
        thalamus = Thalamus(bg, config)
        
        # Update to open gate
        bg.update(0.01)
        thalamus.update(0.01)
        
        # Route with full gate
        thalamus.route(source, target, 0)
        
        # Target should receive source state
        assert np.allclose(target.state, 1.0)
        
    def test_route_with_transform(self):
        """Test routing with transformation."""
        config = SPAConfig(dimension=128)
        vocab = create_vocabulary(128)
        
        source = State("source", 64, vocab)
        target = State("target", 32, vocab)
        
        source.state = np.ones(64)
        
        # Projection matrix
        transform = np.random.randn(32, 64) * 0.1
        
        action_set = ActionSet([Action("route", lambda: 1.0, lambda: None)])
        bg = BasalGanglia(action_set, config)
        thalamus = Thalamus(bg, config)
        
        bg.update(0.01)
        thalamus.update(0.01)
        
        # Clear target first
        target.state = np.zeros(32)
        
        # Route with transform
        thalamus.route(source, target, 0, transform)
        
        expected = np.dot(transform, source.state)
        assert np.allclose(target.state, expected)


class TestCortex:
    """Test Cortex functionality."""
    
    def test_creation(self):
        """Test cortex creation."""
        config = SPAConfig(dimension=128)
        vocab = create_vocabulary(128)
        
        modules = {
            "motor": State("motor", 128, vocab),
            "vision": State("vision", 128, vocab)
        }
        
        cortex = Cortex(modules, config)
        
        assert cortex.name == "Cortex"
        assert len(cortex.modules) == 2
        assert cortex.basal_ganglia is None
        assert cortex.thalamus is None
        
    def test_connect_control(self):
        """Test connecting control structures."""
        config = SPAConfig(dimension=128)
        
        action_set = ActionSet()
        bg = BasalGanglia(action_set, config)
        thalamus = Thalamus(bg, config)
        cortex = Cortex({}, config)
        
        cortex.connect_control(bg, thalamus)
        
        assert cortex.basal_ganglia is bg
        assert cortex.thalamus is thalamus
        
    def test_execute_selected_action(self):
        """Test executing selected action through cortex."""
        config = SPAConfig(dimension=128, threshold=0.3)
        
        # Track execution
        executed = False
        def effect():
            nonlocal executed
            executed = True
        
        actions = [
            Action("do_nothing", lambda: 0.1, lambda: None),
            Action("do_something", lambda: 0.9, effect)
        ]
        
        action_set = ActionSet(actions)
        bg = BasalGanglia(action_set, config)
        thalamus = Thalamus(bg, config)
        cortex = Cortex({}, config)
        
        cortex.connect_control(bg, thalamus)
        
        # Update all components
        bg.update(0.01)
        thalamus.update(0.01)
        cortex.update(0.01)
        
        assert executed  # Action should have been executed
        assert cortex.state[0] == 1.0  # Activity indicator
        
    def test_no_execution_below_gate(self):
        """Test no execution when gate is closed."""
        config = SPAConfig(dimension=128, threshold=0.8)  # High threshold
        
        executed = False
        def effect():
            nonlocal executed
            executed = True
        
        actions = [Action("weak", lambda: 0.5, effect)]  # Below threshold
        
        action_set = ActionSet(actions)
        bg = BasalGanglia(action_set, config)
        thalamus = Thalamus(bg, config)
        cortex = Cortex({}, config)
        
        cortex.connect_control(bg, thalamus)
        
        bg.update(0.01)
        thalamus.update(0.01)
        cortex.update(0.01)
        
        assert not executed  # Should not execute
        assert cortex.state[0] == 0.0  # No activity
        
    def test_module_management(self):
        """Test adding and getting modules."""
        config = SPAConfig(dimension=128)
        cortex = Cortex({}, config)
        
        module = State("test", 128)
        cortex.add_module("test", module)
        
        assert cortex.get_module("test") is module
        assert cortex.get_module("nonexistent") is None


class TestActionSelection:
    """Test integrated ActionSelection system."""
    
    def test_creation(self):
        """Test creating action selection system."""
        config = SPAConfig(dimension=128)
        action_sel = ActionSelection(config)
        
        assert isinstance(action_sel.action_set, ActionSet)
        assert isinstance(action_sel.basal_ganglia, BasalGanglia)
        assert isinstance(action_sel.thalamus, Thalamus)
        assert isinstance(action_sel.cortex, Cortex)
        
    def test_add_action(self):
        """Test adding actions to system."""
        config = SPAConfig(dimension=128)
        action_sel = ActionSelection(config)
        
        action = Action("test", lambda: 0.5, lambda: None)
        action_sel.add_action(action)
        
        assert len(action_sel.action_set) == 1
        assert action_sel.basal_ganglia.dimensions == 1
        
    def test_add_rule(self):
        """Test adding rule-based action."""
        config = SPAConfig(dimension=128)
        vocab = create_vocabulary(128)
        
        motor = State("motor", 128, vocab)
        vision = State("vision", 128, vocab)
        modules = {"motor": motor, "vision": vision}
        
        vocab.create_pointer("CIRCLE")
        vocab.create_pointer("GRASP")
        vision.set_semantic_pointer(vocab["CIRCLE"])
        
        action_sel = ActionSelection(config)
        
        action = action_sel.add_rule(
            "grasp_circle",
            "sim(vision, CIRCLE) > 0.5",
            "motor.set_semantic_pointer(GRASP)",
            modules,
            vocab
        )
        
        assert len(action_sel.action_set) == 1
        assert action.name == "grasp_circle"
        
        # Update system
        action_sel.update(0.01)
        
        # Check that action executed
        assert np.allclose(motor.state, vocab["GRASP"].vector)
        
    def test_full_system_update(self):
        """Test updating full action selection system."""
        config = SPAConfig(dimension=128, threshold=0.3)
        action_sel = ActionSelection(config)
        
        # Track updates
        bg_utility = 0.0
        def condition():
            return bg_utility
        
        executed = False
        def effect():
            nonlocal executed
            executed = True
        
        action = Action("test", condition, effect)
        action_sel.add_action(action)
        
        # Below threshold - no execution
        bg_utility = 0.2
        action_sel.update(0.01)
        assert not executed
        
        # Above threshold - should execute
        bg_utility = 0.8
        action_sel.update(0.01)
        assert executed
        
    def test_get_selected_action(self):
        """Test getting selected action from system."""
        config = SPAConfig(dimension=128, threshold=0.3)
        action_sel = ActionSelection(config)
        
        actions = [
            Action("low", lambda: 0.4, lambda: None),
            Action("high", lambda: 0.9, lambda: None)
        ]
        
        for action in actions:
            action_sel.add_action(action)
        
        action_sel.update(0.01)
        
        selected = action_sel.get_selected_action()
        assert selected is not None
        assert selected.name == "high"
        
    def test_utilities_and_activations(self):
        """Test accessing utilities and activations."""
        config = SPAConfig(dimension=128)
        action_sel = ActionSelection(config)
        
        action_sel.add_action(Action("a1", lambda: 0.5, lambda: None))
        action_sel.add_action(Action("a2", lambda: 0.8, lambda: None))
        
        action_sel.update(0.01)
        
        utilities = action_sel.utilities
        assert len(utilities) == 2
        assert utilities[0] == 0.5
        assert utilities[1] == 0.8
        
        activations = action_sel.activations
        assert len(activations) == 2
        assert activations[1] > activations[0]  # Higher utility wins